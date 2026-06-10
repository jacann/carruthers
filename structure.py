import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from glide.common_components import constants
from glide.common_components.utils import circular_mask
import matplotlib
from dask.diagnostics import ProgressBar
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable



from concurrent.futures import ProcessPoolExecutor
import os

from avg import get_filepaths
from bias_wavelet import remove_dark_stripes
from bias_wavelet import wavelet_destripe


# 1. This function runs completely independently on its own CPU core/thread
def process_frame_chunk(start_idx, end_idx, channel):
    # Re-open the input memory map inside each separate process (Crucial for thread safety)
    images = np.load(f"{channel}_images.npy", mmap_mode='r', allow_pickle=True)
    
    # Load ONLY this specific batch slice into this core's local memory
    sub_stack = images[start_idx:end_idx]
    
    # Execute the heavy CPU math functions
    w1_sub, _ = remove_dark_stripes(sub_stack, "WFI")
    ds_sub, _ = wavelet_destripe(w1_sub, tuning_factor=1.2)
    
    # Return the data along with its coordinate position
    return start_idx, end_idx, ds_sub


def get_images():

    paths = get_filepaths("/carrdata/L1A/", channel)
    with ProgressBar():
        ds = xr.open_mfdataset(paths, chunks={'time': 10}, preprocess=print_update)
        

    images = ds["images"].to_numpy()
    time = ds["time"].to_numpy()
    np.save(f'{channel}_images', images)
    np.save(f'{channel}_time', time)

def rebuild_ds_img_ds():
    
    # Open 1la darks without loading all into RAM
    raw_l1a_drks = np.load(f"{channel}_images.npy", mmap_mode='r', allow_pickle=True)
    total_frames = raw_l1a_drks.shape[0]
    
    # --- Parallel Tuning Configuration ---
    max_workers = 50
    chunk_size = 10       # Number of frames given to a single thread at one time
    
    # Pre-allocate an empty disk skeleton file matching the exact size of the final array
    out_mmap = np.memmap('temp_ds.npy', dtype=raw_l1a_drks.dtype, mode='w+', shape=raw_l1a_drks.shape)
    del raw_l1a_drks   # Clear the initial handle from main memory
    
    # Generate batch boundary indexes
    tasks = [
        (i, min(i + chunk_size, total_frames), channel) 
        for i in range(0, total_frames, chunk_size)
    ]
    
    print(f"Spawning {max_workers} worker processes to parse {len(tasks)} parallel jobs...")
    
    # 2. Launch the multi-processing engine across all threads
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunk combinations to the thread farm
        futures = [executor.submit(process_frame_chunk, *task) for task in tasks]
        
        for num, future in enumerate(futures):
            # Collect data chunks from workers as they finish computing
            start_idx, end_idx, ds_sub = future.result()
            
            # Write directly to the persistent disk array layout
            out_mmap[start_idx:end_idx] = ds_sub
            
            if num % 10 == 0:
                print(f"Flushing progress tracker: Finished {num}/{len(tasks)} jobs.")
                out_mmap.flush() # Securely write data arrays to physical drive sectors

    print("Saving final outputs")
    np.save(f"{channel}_ds_img.npy", out_mmap)
    del out_mmap # Close file 
    print("Destriped images successfully saved")
    


def update(frame, channel, im_stack, variant, im, title_text, day, cbar):
    # Update the pixel array data
    new_img = im_stack[frame]
    im.set_array(new_img)
    title_text.set_text(f"{channel} L1A DRK Daily {variant} Image\n" + day[frame].astype(str))

    # Dynamically adjust the color intensity thresholds
    im.set_clim(0, vmax=np.percentile(new_img, 99))
    cbar.update_normal()
    

    return im, title_text

def main(rebuild=False):
     # ----- Load custom dataset -----
    with xr.open_dataset(f'products/{channel}_FOV_AVG.nc') as ds:
        day = ds['day'].to_numpy()
        day_mean_img = ds['day_mean_img'].to_numpy()
        day_median_img = ds['day_median_img'].to_numpy()
    day = day.astype('datetime64[D]')

    if rebuild == True:
        rebuild_ds_img_ds()
    # Re-open final products
    destriped_images = np.load(f"{channel}_ds_img.npy", mmap_mode='r')
    time = np.load(f"{channel}_time.npy", allow_pickle=True)


    # math
    npix = constants.NPIX[channel]
    fov_mask = circular_mask(constants.NPIX[channel],constants.MASK_L1A_FOV_R[channel])

    # ----- Initalize plot -----
    fig, ax = plt.subplots( figsize=(10, 10), dpi=108)
    im = ax.imshow(day_mean_img[0], cmap='viridis', animated=True)
    title_text = ax.set_title("First Light")
    

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05) # Append a colorbar to the right side of the axis (5% width, 0.05 padding)
    cbar = fig.colorbar(im, ax=cax, extend='max')

    # setup ffmpeg writer
    

    for run_n, im_stack in enumerate([day_mean_img, day_median_img]):
        if run_n == 0:
            variant = 'mean'
            fps = 5
        elif run_n == 1:
            variant = 'median'
            fps = 5
        elif run_n == 2:
            variant = 'all'
            day = time
            fps = 30

        print(f"Animation run {run_n}")

        
        # Create the animation loop
        ani = FuncAnimation(
            fig=fig, 
            func=update,
            frames=day.shape[0], 
            fargs=(channel, im_stack, variant, im, title_text, day, cbar),
            interval=100, 
            blit=True
        )




        writer = FFMpegWriter(
            fps=fps, 
            codec='libx264', 
            extra_args=['-preset', 'ultrafast', '-crf', '28']
        )
        
        ani.save(f'animation/{channel}_daily_{variant}.mp4', writer=writer)


def print_update(ds):
    print(f"Opened {ds.encoding['source']}")
    return ds

# ----- CONFIGURATION -----
channel = "WFI"



#matplotlib.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
# -------------------------

main()







'''
# Use 'h264_nvenc' for NVIDIA or 'h264_amf' for AMD
gpu_writer = animation.FFMpegWriter(
    fps=5,
    codec="h264_nvenc",
    extra_args=["-pix_fmt", "yuv420p"],
)
'''