import astropy.units as unit
import matplotlib.pyplot as plt
import numpy as np
import pywt
import xarray as xr
from astropy.constants.iau2015 import R_earth
from importlib import resources
from scipy.stats import median_abs_deviation

import glide.calibration.calibration_helpers as ch
import glide.science_data_processing.L1A as L1A
from glide.common_components.cam import CamMode, CamSpec
from glide.common_components.camera import Camera
from glide.common_components.constants import NPIX
from glide.common_components.spacecraft import SpaceCraft
from glide.validation.cam import load_lab_data
from glide.validation.scene import Scene
from glide.validation.instrument import Instrument
from glide.common_components.constants import *
from glide.common_components.utils import circular_mask
from glide.operations.activity_block_generation.activity_block import get_ephemeris

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
    loadpath = resources.files('glide') / f'calibration/data_files/{channel}_cols.npy'
    stripes = np.load(loadpath)

    # Guarantee zero median
    top_median = np.median(stripes[:NPIX[channel]//2, :])
    bot_median = np.median(stripes[NPIX[channel]//2:, :])
    stripes[:NPIX[channel]//2, :] -= top_median
    stripes[NPIX[channel]//2:, :] -= bot_median

    return image - stripes, stripes

def wavelet_destripe(image, wavelet='haar', levels=7, tuning_factor=1.5, log=False):
    """
    Removes vertical striping using Wavelet Domain Median Subtraction.
    Preserves global DC offset.
    Uses Haar wavelet since the target is vertical stripes.
    Levels indicate how deep to go. Will crash if 2^levels < npix.
    Tuning factor controls the amount of thresholding - the higher the threshold, the more coefficients are removed.

    If log is False, this algorithm works well in images where the columns are only slightly perturbed from the usual offset (all CaF2/SrF2, off-nadir LyaN/LyaX)
    If log is True, this algorithm works well in images where the column are heavily perturbed (on-nadir LyaN/LyaX)

    Args:
        image (2D numpy array): the L1A image. Bias from electrically dark rows should already be removed. Dark rows should already be removed.
        wavelet (str): the wavelet to use.
        levels (int): how many levels to go down. Will crash if image.shape[0] < 2^levels.
        tuning_factor (float): how many standard deviations within the median absolute deviation to remove.
        log (bool): whether to wavelet destripe in the normal domain or the log domain

    Returns:
        image (2D numpy array): the L1A image with stripes removed.
        stripes (2D numpy array): the stripes that got removed.
    """
    if log:
        # Add offset so it doesn't crash
        offset = np.abs(image.min()) + 1.0
        image = np.log(image + offset)
    # Decompose the image into Approximation (LL) and Details (LH, HL, HH)
    coeffs = pywt.swt2(image, wavelet=wavelet, level=levels)
    N = image.size

    modified_coeffs = []
    for cA, (cH, cV, cD) in coeffs:
        # Calculate the noise (sigma) for this level's vertical band
        level_sigma = median_abs_deviation(cV.ravel(), scale=1/0.6745)
        # Calculate the threshold for this level
        level_threshold = level_sigma * np.sqrt(2 * np.log(N)) * tuning_factor
        
        # Apply threshold
        cV_cleaned = pywt.threshold(cV, value=level_threshold, mode='hard')
        modified_coeffs.append((cA, (cH, cV_cleaned, cD)))
    
    final_image = pywt.iswt2(modified_coeffs, wavelet=wavelet)
    if log:
        final_image = np.exp(final_image) - offset
    recovered_stripes = final_image - image
    return final_image, recovered_stripes

# -------- EXAMPLE USAGE HERE ------------
def display_transforms(channel, im_type):
    # ------- Looking for the proper file and image to use based on channel and im_type; not relevant for pipeline implementation -----
    if channel == 'WFI' and im_type == 'oob_on_nadir':
        return
    yscale='linear'
    if im_type == 'dark':
        filename = f"CARRUTHERS_GCI-{channel}_L1A-DRK_20251221_v1.0.nc"
        if channel == 'NFI': vmax, vmin = 0.8, 0
        else: vmax, vmin = 3, 0
    elif im_type == 'oob_off_nadir':
        filename = f"CARRUTHERS_GCI-{channel}_L1A-STR_20251221_v1.0.nc"
        if channel == 'NFI': vmax, vmin = 0.8, 0
        else: vmax, vmin = 3, 0
    elif im_type == 'sci_off_nadir':
        filename = f"CARRUTHERS_GCI-{channel}_L1A-STR_20251221_v1.0.nc"
        if channel == 'NFI': vmax, vmin = 1.8, 0
        else: vmax, vmin = 9, 5
    elif im_type == 'oob_on_nadir':
        filename = f"CARRUTHERS_GCI-{channel}_L1A-OOB_20251221_v1.0.nc"
        if channel == 'NFI': vmax, vmin = 0.8, -1.3
    elif im_type == 'sci_on_nadir':
        filename = f"CARRUTHERS_GCI-{channel}_L1A-SCI_20251221_v1.0.nc"
        if channel == 'NFI': vmax, vmin = None, 1
        else: vmax, vmin = None, 1
        yscale='log'
    else:
        return
    
    if 'oob' in im_type:
        filters = ['CaF2', 'SrF2']
    else:
        filters = ['LyaN']
    
    filepath = resources.files('glide') / f'samba_data_processing/bias_transforms/{filename}'
    r_disk = R_earth.to_value(unit.km)
    with xr.open_dataset(filepath, engine='netcdf4') as data:
        L1A_obj = L1A.L1A(data)
        if im_type == 'dark':
            final_image = L1A_obj.images[0] / L1A_obj.n_frames[0]
        else:
            for ind in range(len(L1A_obj.scrafts)):
                if not L1A_obj.filters[ind] in filters:
                    continue
                final_image = L1A_obj.images[ind] / L1A_obj.n_frames[ind]
                tan_pts, los_TP_r, r_obs = L1A_obj.scrafts[ind].calc_tan_pts(sph_body='earth', camID=channel)
                earth_mask = los_TP_r <= 1.5 * r_disk # mask out 1.5Re to help?
                earth_mask_big = los_TP_r <= 2.5 * r_disk # guessing stuff now
                npix = L1A_obj.cam_specs[ind].npix
                earth_mask = earth_mask.reshape((npix, npix))
                earth_mask_big = earth_mask_big.reshape((npix, npix))
                earth_mask = np.where(earth_mask != 0, 1, 0)
                earth_mask_big = np.where(earth_mask_big != 0, 1, 0)
                break
    
    # --------- Example conditioning that the pipeline should use, and what functions should be called in each case with what setting -------
    # --------- Heather: Reference the slide deck, which has more precise detail
    if im_type == 'dark' or im_type == 'oob_off_nadir':
        clean_image, recovered_stripes = remove_dark_stripes(final_image, channel)
    elif im_type == 'oob_on_nadir':
        clean_image, recovered_stripes = wavelet_destripe(final_image, wavelet='haar', levels=7, tuning_factor=1.5)
    elif im_type == 'sci_off_nadir':
        im_1, stripes_1 = remove_dark_stripes(final_image, channel)
        clean_image, stripes_2 = wavelet_destripe(im_1, wavelet='haar', levels=7, tuning_factor=1.2)
        recovered_stripes = stripes_1 + stripes_2
    else: # sci_on_nadir is the only option remaining now
        if channel == 'NFI':
            im_1, stripes_1 = remove_dark_stripes(final_image, channel)
            clean_image, stripes_2 = wavelet_destripe(im_1, wavelet='haar', levels=5, tuning_factor=1, log=True)
            recovered_stripes = stripes_1 + stripes_2
        else:
            clean_image, recovered_stripes = wavelet_destripe(final_image, wavelet='haar', levels=4, tuning_factor=1.7, log=True)

    # -------- Plots for funsies
    def add_colorbar(im, ax):
        return fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig = plt.figure(figsize=(15, 12))
    
    # Original Image
    ax1 = plt.subplot(1, 3, 1)
    if yscale=='log': im1 = ax1.imshow(np.log(final_image), vmax = vmax, vmin = vmin)
    else: im1 = ax1.imshow(final_image, vmax = vmax, vmin = vmin)
    ax1.set_title(f"Raw L1A {channel} {im_type}, DN/frame")
    add_colorbar(im1, ax1)

    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.imshow(recovered_stripes, vmax = 1, vmin = 0)
    ax2.set_title(f"Recovered Stripes {channel} {im_type}, DN/frame")
    add_colorbar(im2, ax2)

    # Recovered image
    ax3 = plt.subplot(1, 3, 3)
    if yscale=='log': im3 = ax3.imshow(np.log(clean_image), vmax = vmax, vmin = vmin)
    else: im3 = ax3.imshow(clean_image, vmax = vmax, vmin = vmin)
    ax3.set_title(f"Recovered image {channel} {im_type}, DN/frame")
    add_colorbar(im3, ax3)

    plt.tight_layout()

# Run the below to use the function above and display stuff
# for channel in CHANNELS:
#     for im_type in ['dark', 'oob_off_nadir', 'sci_off_nadir', 'oob_on_nadir', 'sci_on_nadir']:
#         display_transforms(channel, im_type)
