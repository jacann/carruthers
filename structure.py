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

  
def update_dual(frame, channel, mean_stack, med_stack, im_mean, im_med, 
                title_mean, title_med, day, cbar_mean, cbar_med):
        """Update both axes per frame for the side-by-side animation."""
        new_mean = mean_stack[frame]
        new_med  = med_stack[frame]

        im_mean.set_array(new_mean)
        im_med.set_array(new_med)

        date_str = day[frame].astype(str)
        title_mean.set_text(f"{channel} L1A Dark Daily Mean\n{date_str}")
        title_med.set_text(f"{channel} L1A Dark Daily Median\n{date_str}")

        im_mean.set_clim(0, np.percentile(new_mean, 75))
        im_med.set_clim(0,  np.percentile(new_med,  75))

        cbar_mean.update_normal(im_mean)
        cbar_med.update_normal(im_med)

        return im_mean, im_med, title_mean, title_med

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
    day_str = day.astype(str)

    fig, (ax_mean, ax_med) = plt.subplots(1, 2, figsize=(20, 10), dpi=108, constrained_layout=False)
    
    # ----- setup mean image ------
    im_mean    = ax_mean.imshow(day_mean_img[0], cmap='viridis', animated=True)
    title_mean = ax_mean.set_title(f"{channel} L1A Dark Daily Mean Image\n{day_str[0]}")
    divider_mean = make_axes_locatable(ax_mean)
    cax_mean     = divider_mean.append_axes("right", size="5%", pad=0.05)
    cbar_mean    = fig.colorbar(im_mean, cax=cax_mean, extend='max')
    # ----- setup median image ------
    im_med    = ax_med.imshow(day_median_img[0], cmap='viridis', animated=True)
    title_med = ax_med.set_title(f"{channel} L1A Dark Daily Median Image\n{day_str[0]}")
    divider_med = make_axes_locatable(ax_med)
    cax_med     = divider_med.append_axes("right", size="5%", pad=0.05)
    cbar_med    = fig.colorbar(im_med, cax=cax_med, extend='max')
    

    fig.tight_layout(pad=2.0) 

    variant = 'dual'
    ani = FuncAnimation(
        fig=fig,
        func=update_dual,
        frames=day.shape[0],
        fargs=(
            channel,
            day_mean_img, day_median_img,
            im_mean, im_med,
            title_mean, title_med,
            day,
            cbar_mean, cbar_med
        ),
        interval=1000 // fps,
        blit=True,
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
fps = 5



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

        # Create the animation loop
        ani = FuncAnimation(
            fig=fig, 
            func=update,
            frames=day.shape[0], 
            fargs=(channel, im_stack, variant, im, title_text, day, cbar),
            interval=100, 
            blit=True
        )



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
'''