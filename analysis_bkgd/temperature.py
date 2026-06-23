#%%
import numpy as np
import xarray as xr
import concurrent.futures
from functools import partial
import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


from glide.common_components.utils import mask_average
from glide.common_components.utils import circular_mask
from glide.common_components import constants
import glide.science_data_processing.L1A as L1A
import glide.common_components.view_geometry as view_geometry
from glide.common_components.stars import get_beta_angle
from analysis_bkgd.bias_wavelet import remove_dark_stripes

with xr.open_dataset('products/NFI_FOV_AVG.nc') as ds:
    fov_mean = (ds['fov_mean_top'] * ds['fov_mean_bottom'])/2
    time = ds['time']
    fw_temp = ds['fw_temp']

fig, ax = plt.subplots()

ax.set_title("Filter Wheel Temp & NFI FOV Mean over time")

ax.plot(time, fw_temp, label="FW Temp", marker='o', ms=2)
ax1 = plt.twinx(ax)
ax1.plot(time, fov_mean, color='orange', label="NFI FOV Mean", marker='o', ms=2)
#ax1.set_ylim(-1, 100)
#ax.set_ylim(17,19)
ax.set_xlim(np.datetime64('2025-11-10'), np.datetime64('2025-11-20'))
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.set
#%%