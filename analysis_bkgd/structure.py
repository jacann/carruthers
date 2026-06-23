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
import cv2



from concurrent.futures import ProcessPoolExecutor
import os

from analysis_bkgd.avg import get_filepaths
from analysis_bkgd.bias_wavelet import remove_dark_stripes
from analysis_bkgd.bias_wavelet import wavelet_destripe


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

  


def opencv_fill(stack):
    filled_stack = np.copy(stack)
    thresh_per_frame = np.percentile(stack, 99.9, axis=(1, 2), keepdims=True)
    drk_imgs_hp_mask = stack > thresh_per_frame
    stack[drk_imgs_hp_mask] = np.nan

    filled_stack = np.copy(stack)
    
    for i in range(stack.shape[0]):
        slice_2d = stack[i, :, :]
        nan_mask = np.isnan(slice_2d)
        
        if not np.any(nan_mask):
            continue
            
        # OpenCV requires 8-bit unsigned integer masks where 255 marks missing data
        mask_8u = (nan_mask).astype(np.uint8) * 255
        
        # Temporarily fill NaNs with 0 because cv2.inpaint cannot read NaN floats
        slice_filled_zeros = np.nan_to_num(slice_2d, nan=0.0).astype(np.float32)
        
        # Inpaint using Navies-Stokes based method (or cv2.INPAINT_TELEA)
        inpainted = cv2.inpaint(slice_filled_zeros, mask_8u, inpaintRadius=3, flags=cv2.INPAINT_NS)
        
        filled_stack[i, :, :] = inpainted
        
    return filled_stack

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

        p_min = 90
        p_max = 99
        
        im_mean.set_clim(0, np.percentile(new_mean, p_max))
        im_med.set_clim(0, np.percentile(new_med, p_max))

        cbar_mean.update_normal(im_mean)
        cbar_med.update_normal(im_med)

        return im_mean, im_med, title_mean, title_med

def update_hot_focus(frame, channel, hot_vals, med_stack, line, im_med, 
            title_med, day, cbar_med):
        """Update both axes per frame for the side-by-side animation."""
        new_med  = med_stack[frame]


        #p1.plot(day[:frame+1], hot_vals[:frame+1])
        line.set_data(day[:frame+1], hot_vals[:frame+1])
        im_med.set_array(new_med)

        date_str = day[frame].astype(str)
        title_med.set_text(f"{channel} L1A Dark Daily Median\n{date_str}")

        p_min = 90
        p_max = 99.9
        
        #p1.set_clim(0, np.percentile(new_hot, p_max))
        im_med.set_clim(0, 30)

        cbar_med.update_normal(im_med)

        return line, im_med, title_med

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


    #day_mean_img = opencv_fill(day_mean_img)
    #day_median_img = opencv_fill(day_median_img)

    # crop images
    #day_median_img[:, 374, 474] = 0
    day_mean_img = day_mean_img[:, 350:400, 450:500]
    day_median_img = day_median_img[:, 350:400, 450:500]

    hot_vals = day_median_img[:, 24, 24]


    # ----- Initalize plot -----
    day_str = day.astype(str)

    fig, (ax1, ax_med) = plt.subplots(1, 2, figsize=(20, 10), dpi=108, constrained_layout=False)
    
    # ----- setup mean image ------
    #im_mean    = ax_mean.imshow(day_mean_img[0], cmap='viridis', animated=True)
    #title_mean = ax_mean.set_title(f"{channel} L1A Dark Daily Mean Image\n{day_str[0]}")
    #divider_mean = make_axes_locatable(ax_mean)
    #cax_mean     = divider_mean.append_axes("right", size="5%", pad=0.05)
    #cbar_mean    = fig.colorbar(im_mean, cax=cax_mean, extend='max')
    #ax1.plot(day[0], day_median_img[0, 24, 24])
    line, = ax1.plot([], [], animated=True)
    ax1.set_title("Value of WFI pixel at row 374, col 474")
    ax1.set_xlim(day[0], day[-1])
    ymin = np.nanmin(hot_vals)
    ymax = np.nanmax(hot_vals)*1.2
    print(np.nanmax(hot_vals))

    if ymin == ymax:
        ymax += 1

    ax1.set_ylim(ymin, ymax)

    # ----- setup median image ------
    im_med    = ax_med.imshow(day_median_img[0], cmap='viridis', animated=True, extent=[450, 500, 350, 400])
    title_med = ax_med.set_title(f"{channel} L1A Dark Daily Median Image\n{day_str[0]}")
    divider_med = make_axes_locatable(ax_med)
    cax_med     = divider_med.append_axes("right", size="5%", pad=0.05)
    cbar_med    = fig.colorbar(im_med, cax=cax_med, extend='max')
    

    fig.tight_layout(pad=2.0) 

    variant = 'med_crop_plot'
    ani = FuncAnimation(
        fig=fig,
        func=update_hot_focus,
        frames=day.shape[0],
        fargs=(
            channel, hot_vals, day_median_img, line, im_med, title_med, day, cbar_med),
        interval=1000 // fps,
        blit=True,
    )
    

    writer = FFMpegWriter(
        fps=fps, 
        codec='libx264', 
        extra_args=['-preset', 'ultrafast', '-crf', '28']
    )
    savepath = f'animation/{channel}_daily_{variant}.mp4'
    ani.save(savepath, writer=writer)
    print(f"File saved to {savepath}")


def print_update(ds):
    print(f"Opened {ds.encoding['source']}")
    return ds

# ----- CONFIGURATION -----
channel = "WFI"
fps = 4



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