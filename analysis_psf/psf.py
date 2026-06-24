#%%
import numpy as np
import xarray as xr
import glob
import resources

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import glide.science_data_processing.L1A as L1A
from glide.common_components.constants import NPIX
def remove_dark_stripes(image, channel):
    """
    Loads known column offsets in L1A images for the channel specified and subtracts them from the image given.
    The columns are guaranteed to have zero median to match the behavior of wavelets.
    This removal, on its own, performs well on any non-sagged, dark-only row (i.e., dark images and CaF2/SrF2 images with no bright source in the row).

    Args:
        image (2D numpy array): the L1A image. Bias from electrically dark rows should already be removed. Dark rows should already be removed.
        channel (str): the channel, either 'NFI' or 'WFI'.

    Returns:
        image (2D numpy array): the L1A image with stripes removed.
        stripes (2D numpy array): the stripes that got removed.
    """
    loadpath = "./data/NFI_cols.npy"
    stripes = np.load(loadpath)

    # Guarantee zero median
    top_median = np.median(stripes[:NPIX[channel]//2, :])
    bot_median = np.median(stripes[NPIX[channel]//2:, :])
    stripes[:NPIX[channel]//2, :] -= top_median
    stripes[NPIX[channel]//2:, :] -= bot_median

    return image - stripes, stripes

channel = "NFI"
date = "20260420"
l1a_dir = "/data/L1A/"
str_fp = glob.glob(l1a_dir +"CARRUTHERS_GCI-NFI_L1A-STR_*_v1.1.nc")
# testing override
str_fp = l1a_dir + "L1A-STR/" + f"CARRUTHERS_GCI-NFI_L1A-STR_{date}_v1.1.nc"
drk_fp = l1a_dir + "L1A-DRK/" + f"CARRUTHERS_GCI-NFI_L1A-DRK_{date}_v1.1.nc"

# Load offset stripes
nfi_cols = np.load("./data/NFI_cols.npy")

# Load images
with xr.open_mfdataset(str_fp, data_vars=all) as ds:
    str_obj = L1A.L1A(ds)
    str_rbias = ds.residual_bias.to_numpy()
    str_ims = str_obj.images    # Load ims
    str_sc = str_obj.scrafts
    str_sc = np.array(str_sc, dtype=object)
    str_time = str_obj.time
    str_filters = str_obj.filters
    str_nfr = str_obj.n_frames


with xr.open_mfdataset(drk_fp, data_vars=all) as ds:
    drk_obj = L1A.L1A(ds)
    drk_rbias = ds.residual_bias.to_numpy()
    drk_ims = drk_obj.images
    drk_time = drk_obj.time
    drk_nfr = drk_obj.n_frames

# Filter to only CaF2/SrF2
filter_idx = (str_obj.filters == "CaF2") | (str_obj.filters == "SrF2")
str_ims = str_ims[filter_idx]
str_sc = str_sc[filter_idx]
str_nfr = str_nfr[filter_idx]
str_rbias = str_rbias[filter_idx]
str_time = str_time[filter_idx]
str_filters = str_filters[filter_idx]

# Guarantee temporal sorting
sort_idx = np.argsort(str_time)
str_ims = str_ims[sort_idx]
str_sc = str_sc[sort_idx]
str_nfr = str_nfr[sort_idx]
str_rbias = str_rbias[sort_idx]
str_filters = str_filters[sort_idx]
str_time = str_time[sort_idx]

# Background Subtration ----------
# 1 & 2. Subtract electrically dark rows and residual vertical stripes
str_ims /= str_nfr[:,np.newaxis,np.newaxis] # Divide by n_frames
str_ims += nfi_cols         # Add offset stripes to L1A
str_rbias -= nfi_cols       # Sub offset stripes from rbias
str_ims -= str_rbias        # Sub rbias from images
str_ims, _ = remove_dark_stripes(str_ims, channel)
# 3. Subtract closest-in-time dark image
drk_ims /= drk_nfr[:, np.newaxis, np.newaxis]
d_time = np.abs(str_time[:, np.newaxis] - drk_time)
drk_close_idx = np.argmin(d_time, axis=1)
str_ims = str_ims - drk_ims[drk_close_idx]
# 4 Subtract median of non-bright pixels (>1 DN/frame) in each image half independantly to remove any remaining dark current
half_npix = NPIX[channel]//2
str_ims_nb = str_ims.copy()
str_ims_nb[str_ims_nb >= 1] = np.nan
str_ims[:, :half_npix, :] -= np.nanmedian(str_ims_nb[:, :half_npix, :], axis=(1,2))[:, np.newaxis, np.newaxis]
str_ims[:, half_npix:, :] -= np.nanmedian(str_ims_nb[:, half_npix:, :], axis=(1,2))[:, np.newaxis, np.newaxis]


# Star ground-truth process ------
# Use DAOStarFinder to extrct star locations, roundness, and sharpness from the image
# Temove stars that are not round enough (|round|<0.5) and too sharp (0.3<sharpness<1)
# If a star is >50 pixels from others stars, fit a Voigt profile and use the Voigt profile as the ground-truth
# If a star is <50 pixels from all other stars, use a Voigt profile with /sigma and /gamma extrapolated from amplitude



# Image ground-truth process -----
# Start with image of all zeros
# Add all star ground-truths
# Add median back from background subtraction
# Add calibrated closest-in-time dark image

fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=1000, constrained_layout=True)
im = ax.imshow(str_ims[3], vmin=0, vmax=4)
fig.colorbar(im, ax=ax)
plt.show()

"""
n = 3
vmax = 2
vmin = 0.75
fig, ax = plt.subplots(n,1, figsize=(5,n*3), constrained_layout=True)
for i in range(n):
    #vmax = np.percentile(str_ims[i], 99)
    ax[i].imshow(str_ims[i], vmin=vmin, vmax=vmax)
    title_str = str_time[i].astype('datetime64[s]').astype(str) + " " + str_filters[i] 
    ax[i].set_title(title_str)
plt.show
"""




"""
n = 3
vmax = 2
vmin = 0.75
fig, ax = plt.subplots(n,1, figsize=(5,n*3), constrained_layout=True)
for i in range(n):
    #vmax = np.percentile(str_ims[i], 99)
    ax[i].imshow(str_ims[i], vmin=vmin, vmax=vmax)
    title_str = str_time[i].astype('datetime64[s]').astype(str) + " " + str_filters[i] 
    ax[i].set_title(title_str)
plt.show

min_im = np.mean([str_ims[0:1]], axis=0)
plt.imshow(min_im[0], vmin=vmin, vmax=vmax)
plt.title("MERGED")
plt.show()
"""
# Scatter plots ------------------
# Plot ground-truth vs. image that we got and see what happened. Helpful make individual plots per column (or 10 columns at a time) ... ground-truth vs. image vs. row-sum
#%%