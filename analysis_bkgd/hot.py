#%%
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
from analysis_bkgd.bias_wavelet import remove_dark_stripes
import glob
import glide.science_data_processing.L1A as L1A
import pandas as pd


channel = 'WFI'


with xr.open_dataset(f'{channel}_L1A-DRK_ALL.nc', engine='netcdf4') as ds:
    images = ds['images'].to_numpy()
    time = ds['time'].to_numpy()
    n_frames = ds['n_frames'].to_numpy()
    t_int = ds['t_int'].to_numpy()

with xr.open_dataset(f'products/{channel}_FOV_AVG.nc', engine='netcdf4') as ds:
    fov_means = (ds['fov_mean_top'] + ds['fov_mean_bottom'])/2
    fov_time = ds['time']

sort_idx = np.argsort(time)
images = images[sort_idx]
n_frames = n_frames[sort_idx]
t_int = t_int[sort_idx]
time = time[sort_idx]


images_nfnorm = images / n_frames[:, np.newaxis, np.newaxis]
alex, _ = remove_dark_stripes(images_nfnorm, channel)
alex *= n_frames[:, np.newaxis, np.newaxis]
alex /= t_int[:, np.newaxis, np.newaxis]

hot_t = alex[:, 374, 474]


size=6
fig, ax = plt.subplots(1, 1, figsize=(size*2.5, size), constrained_layout=True)
ax.plot(time, hot_t, linewidth=1, zorder=3, label='pix r374 c474')
ax1 = plt.twinx(ax)
ax1.plot(fov_time, fov_means, label="WFI FOV Means", color='orange')
ax.set_title('WFI Pixel value over time (Row 374, Col 474)')
ax.set_ylabel('Pixel DN/sec')
ax1.set_ylabel('FOV Mean DN/sec-pix')

#ax.set_ylim(0, 500)
ax1.set_ylim(0, 20)
ax.legend(loc='upper left')
ax1.legend(loc='upper right')

#ax.set_yscale('log')


df = pd.DataFrame({'Value': hot_t})

# Calculate the 3-point moving average
df['3_day_MA'] = df['Value'].rolling(window=3).mean()

#ax.plot(time, df['3_day_MA'])

ax.set_xlim(np.datetime64('2025-11-01'), np.datetime64('2026-01-01'))





'''
fig, ax = plt.subplots(1, 3, figsize=(size*3, size), constrained_layout=True)
im = ax[0].imshow(im, vmax=np.percentile(im, 99))
cbar = fig.colorbar(im, ax=ax[0])

im = ax[1].imshow(im_alex, vmax=np.percentile(im_alex, 99))
cbar = fig.colorbar(im, ax=ax[1])

im = ax[2].imshow(im_jacob, vmax=np.percentile(im_jacob, 99))
cbar = fig.colorbar(im, ax=ax[2])

plt.show


'''

















#%%






'''
with xr.open_mfdataset(sorted(glob.glob("/carrdata/L1A/CARRUTHERS_GCI-" + channel + "_L1A-DRK" + "**" + "v1.0.nc"))) as ds:
    t_int = ds['t_int'].to_numpy()
    n_frames = ds['n_frames'].to_numpy()

np.save('t_int', t_int)
np.save('n_frames', n_frames)
    




im_num = 1000
im = images[im_num]

alex, _ = remove_dark_stripes(im, channel)





size=4
fig, ax = plt.subplots(1, 3, figsize=(size*3, size), constrained_layout=True)
im = ax[0].imshow(images[im_num], vmax=np.percentile(images[im_num], 99))
cbar = fig.colorbar(im, ax=ax[0])

im = ax[1].imshow(alex, vmax=np.percentile(alex, 99))
cbar = fig.colorbar(im, ax=ax[1])

plt.show



'''