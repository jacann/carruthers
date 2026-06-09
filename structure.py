import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from glide.common_components import constants
from glide.common_components.utils import circular_mask
import matplotlib

def update(frame, channel, im_stack):
    # Update the pixel array data
    new_img = day_mean_img[frame]
    im.set_array(new_img)
    title_text.set_text(f"{channel=}".split('=')[0] + f" L1A DRK Daily {im_stack} Image\n" + day[frame].astype(str))

    # Dynamically adjust the color intensity thresholds
    im.set_clim(0, vmax=np.percentile(new_img, 99.9))

    return im, title_text



# ----- CONFIGURATION -----
channel = "NFI"

#matplotlib.rcParams['animation.ffmpeg_path'] = '/home/jacob/miniconda3/envs/carruthers-sdc/bin/ffmpeg'


# -------------------------

# Load dataset
with xr.open_dataset(f'products/{channel}_FOV_AVG.nc') as ds:
    day = ds['day'].to_numpy()
    day_mean_img = ds['day_mean_img'].to_numpy()
    day_median_img = ds['day_median_img'].to_numpy()
# Convert dateimes to daily resolution
day = day.astype('datetime64[D]')

npix = constants.NPIX[channel]
fov_mask = circular_mask(constants.NPIX[channel],constants.MASK_L1A_FOV_R[channel])

# Create figure and axis
fig, ax = plt.subplots( figsize=(10, 10), dpi=100)

# Initialize the plot with a starting image
im = ax.imshow(day_mean_img[0], cmap='viridis', animated=True)
title_text = ax.set_title("First Light")




for i, im_stack in enumerate([day_mean_img, day_median_img]):

    # Create the animation loop
    ani = FuncAnimation(
        fig=fig, 
        func=update,
        frames=day.shape[0], 
        fargs=(channel, im_stack),
        interval=10000, 
        blit=True
    )
    if i == 0:
        ani.save('animation/daily_mean.mp4', writer='ffmepg', fps=5)
    elif i == 1:
        ani.save('animation/daily_median.mp4', writer='ffmepg', fps=5)











'''
# Use 'h264_nvenc' for NVIDIA or 'h264_amf' for AMD
gpu_writer = animation.FFMpegWriter(
    fps=5,
    codec="h264_nvenc",
    extra_args=["-pix_fmt", "yuv420p"],
)
'''