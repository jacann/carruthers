#%%
import numpy as np
import xarray as xr
import glob
import resources
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches

from astropy.stats import sigma_clipped_stats
from astropy.table import Table, vstack as astropy_vstack
from astropy.modeling import models, fitting
from photutils.detection import DAOStarFinder
from scipy.special import voigt_profile
from scipy.optimize import curve_fit
from scipy.spatial import KDTree

import glide.science_data_processing.L1A as L1A
from glide.common_components.constants import NPIX

# TODO
# compare with glide-sdc starfinder method


def remove_dark_stripes(image, channel, data_dir=Path("/home/jacob/carruthers/analysis_psf/data/")):
    """
    Loads known column offsets in L1A images for the channel specified and subtracts them from the image given.
    The columns are guaranteed to have zero median to match the behavior of wavelets.
    This removal, on its own, performs well on any non-sagged, dark-only row (i.e., dark images and CaF2/SrF2 images with no bright source in the row).

    Args:
        image (2D numpy array): the L1A image. Bias from electrically dark rows should already be removed. Dark rows should already be removed.
        channel (str): the channel, either 'NFI' or 'WFI'.
        data_dir (Pathlib.path): Path to directory containing NFI_cols.npy and WFI_cols.npy
        

    Returns:
        image (2D numpy array): the L1A image with stripes removed.
        stripes (2D numpy array): the stripes that got removed.
    """
    if channel != "WFI" and channel != "NFI":
        raise ValueError("invalid channel selection, use 'NFI' or 'WFI'")

    loadpath = data_dir / f"{channel}_cols.npy"
    stripes = np.load(loadpath)

    # Guarantee zero median
    top_median = np.median(stripes[:NPIX[channel]//2, :])
    bot_median = np.median(stripes[NPIX[channel]//2:, :])
    stripes[:NPIX[channel]//2, :] -= top_median
    stripes[NPIX[channel]//2:, :] -= bot_median

    return image - stripes, stripes

def voigt_2d_model(xy, amplitude, x_c, y_c, sigma, gamma):
    """
    2D Radially symmetric Voigt profile.
    
    Args:
        xy: Tuple of flattened (x, y) coordinate arrays.
        amplitude: Scaling factor.
        x_c: True center x-coordinate.
        y_c: True center y-coordinate.
        sigma: Standard deviation of the Gaussian component.
        gamma: Half-width at half-maximum of the Lorentzian component.
    """
    x, y = xy
    r = np.sqrt((x - x_c)**2 + (y - y_c)**2)
    return amplitude * voigt_profile(r, sigma, gamma)

def fit_2d_voigt(image, pixel_x, pixel_y, box_size=25, mask=None, gamma_override=None):
    """
    Fits a 2D radially symmetric Voigt profile to a region of interest.
    Args:
        image (np.ndarray): 2D numpy array representing the image (background removed).
        pixel_x (int): Initial guess for the center x-coordinate.
        pixel_y (int): Initial guess for the center y-coordinate.
        box_size (int): Half-size of the bounding box.
    Returns:
        popt (list): The optimal parameters [amplitude, x_c, y_c, sigma, gamma].
        pcov (2D array): The estimated covariance of popt.
    """
    height, width = image.shape
    # Define bounding box limits
    min_x = max(0, int(pixel_x - box_size))
    max_x = min(width, int(pixel_x + box_size + 1))
    min_y = max(0, int(pixel_y - box_size))
    max_y = min(height, int(pixel_y + box_size + 1))
    # Extract ROI and create coordinate grids
    roi_values = image[min_y:max_y, min_x:max_x]
    x_coords = np.arange(min_x, max_x)
    y_coords = np.arange(min_y, max_y)
    xx, yy = np.meshgrid(x_coords, y_coords)
    if mask is None:
        roi_mask = np.zeros(roi_values.shape, dtype=bool)
    else:
        roi_mask = mask[min_y:max_y, min_x:max_x]
    # Flatten the grids and image data for curve_fit
    x_flat = xx[~roi_mask].flatten()
    y_flat = yy[~roi_mask].flatten()
    z_flat = roi_values[~roi_mask].flatten()
    # Set Initial Guesses (p0)
    # The voigt_profile integrates to 1, so the amplitude needs to account for the peak height 
    # and the rough area. A naive guess based on the max pixel usually works well enough to start.
    sigma_guess = 2.0
    gamma_guess = 2.0
    normalized_peak = voigt_profile(0, sigma_guess, gamma_guess)
    max_pixel_val = np.max(roi_values)
    amp_guess = max_pixel_val / normalized_peak
    initial_guesses = [amp_guess, pixel_x, pixel_y, sigma_guess, gamma_guess]
    # Set Bounds
    # Lower bounds: Amp > 0, Center within box, Widths > 0
    # Upper bounds: Amp = inf, Center within box, Widths = inf
    if gamma_override is None:
        gammalo = 0.001
        gammahi = np.inf
    else:
        gammalo = gamma_override
        gammahi = gamma_override

    lower_bounds = (0, min_x, min_y, 0.001, gammalo)
    upper_bounds = (np.inf, max_x, max_y, np.inf, gammahi)

    # Perform the Fit
    popt, pcov = curve_fit(
        voigt_2d_model, 
        (x_flat, y_flat), 
        z_flat, 
        p0=initial_guesses, 
        bounds=(lower_bounds, upper_bounds),
        maxfev=5000
    )
    return popt, pcov

def find_stars(ims, method="dao", sharp_range=(0.3,1.0), round_range=(-5,5)):
    if method == "dao":
        # glide-sdc compatible with photutils <= v2.3.0 due to old dependancies
        # Needs to be updated if we ever want to use photutils v3.0.0 or higher

        dao = DAOStarFinder(
                            threshold=2,                # works well with threshold=2 DN/sec
                            fwhm=2,                     # works well with fwhm=2, fwhm=3 creates many false positives
                            sharplo=sharp_range[0],     # built-in default is (0.2, 1.0)
                            sharphi=sharp_range[1],
                            roundlo=round_range[0],     # built-in default is (-1.0, 1.0)
                            roundhi=round_range[1],
                            exclude_border=True         # image edges are borked by preprocessing and can cause false positives
                            )
        stars = dao(ims)
        return stars
        
    elif method == "glide_stars":
        return # TODO implement
    else:
        raise ValueError("Invalid method selection")

def main(channel="NFI"):

    # testing params
    date = "20260629"
    ''' dates with CaF2/SrF2
    20251113, 20251114, 20251115, 20251116, 20251117, 20251118, 20251119, 20251120, 20251121, 20251122, 
    20251123, 20251124, 20251125, 20251126, 20251127, 20251128, 20251129, 20251130, 20251201, 20251202, 
    20251206, 20251207, 20251208, 20251209, 20251210, 20251211, 20251212, 20251213, 20251214, 20251215, 
    20251216, 20251217, 20251218, 20251219, 20251220, 20251221, 20251222, 20251223, 20251224, 20251225, 
    20251226, 20251227, 20251228, 20251229, 20251230, 20251231, 20260101, 20260102, 20260103, 20260104, 
    20260105, 20260108, 20260109, 20260110, 20260111, 20260112, 20260113, 20260114, 20260115, 20260116, 
    20260117, 20260118, 20260119, 20260120, 20260121, 20260122, 20260123, 20260124, 20260125, 20260126, 
    20260127, 20260303, 20260304, 20260305, 20260309, 20260310, 20260311, 20260316, 20260318, 20260320, 
    20260323, 20260325, 20260327, 20260330, 20260331, 20260406, 20260407, 20260413, 20260417, 20260418, 
    20260419, 20260420, 20260427, 20260504, 20260511, 20260518, 20260525, 20260601, 20260615, 20260622, 
    20260629

    20251113 -> weird stray light
    20260629 -> lots of stars!
    '''

    l1a_dir = "/data/L1A/"
    str_fp = glob.glob(l1a_dir +"CARRUTHERS_GCI-NFI_L1A-STR_*_v1.1.nc")

    # testing override
    str_fp = l1a_dir + "L1A-STR/" + f"CARRUTHERS_GCI-NFI_L1A-STR_{date}_v1.1.nc"
    drk_fp = l1a_dir + "L1A-DRK/" + f"CARRUTHERS_GCI-NFI_L1A-DRK_{date}_v1.1.nc"

    # Load offset stripes
    nfi_cols = np.load("/home/jacob/carruthers/analysis_psf/data/NFI_cols.npy")

    # ==========================
    #  LOAD AND PREPROCESS DATA
    # ==========================

    # Load images
    with xr.open_mfdataset(str_fp, data_vars=all) as ds:
        str_obj = L1A.L1A(ds)
        str_rbias = ds.residual_bias.to_numpy()
        str_ablock = ds["activity_block"].to_numpy()
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

    if len(str_ims) == 0 or len(drk_ims) == 0:
        raise ValueError("No CaF2/SrF2 ims found")

    # Guarantee temporal sorting
    sort_idx = np.argsort(str_time)
    str_ims = str_ims[sort_idx]
    str_sc = str_sc[sort_idx]
    str_nfr = str_nfr[sort_idx]
    str_rbias = str_rbias[sort_idx]
    str_filters = str_filters[sort_idx]
    str_time = str_time[sort_idx]

    # Background Subtration ----------
    # Subtract electrically dark rows and residual vertical stripes
    str_ims /= str_nfr[:,np.newaxis,np.newaxis] # Divide by n_frames
    str_ims += nfi_cols         # Add offset stripes to L1A
    str_rbias -= nfi_cols       # Sub offset stripes from rbias
    str_ims -= str_rbias        # Sub rbias from images
    str_ims, _ = remove_dark_stripes(str_ims, channel)
    # Subtract closest-in-time dark image
    drk_ims /= drk_nfr[:, np.newaxis, np.newaxis]
    d_time = np.abs(str_time[:, np.newaxis] - drk_time)
    drk_close_idx = np.argmin(d_time, axis=1)
    str_ims = str_ims - drk_ims[drk_close_idx]
    # Subtract median of non-bright pixels (>1 DN/frame) in each image half independantly to remove any remaining dark current
    half_npix = NPIX[channel]//2
    str_ims_nb = str_ims.copy()
    str_ims_nb[str_ims_nb >= 1] = np.nan
    # Store per-image half-medians so they can be added back into the ground truth
    bg_medians_top = np.nanmedian(str_ims_nb[:, :half_npix, :], axis=(1,2))
    bg_medians_bot = np.nanmedian(str_ims_nb[:, half_npix:, :], axis=(1,2))
    str_ims[:, :half_npix, :] -= bg_medians_top[:, np.newaxis, np.newaxis]
    str_ims[:, half_npix:, :] -= bg_medians_bot[:, np.newaxis, np.newaxis]


    # ===========================
    # FIND STARS & FIT TO VOIGT     (W/ DAOSTARFINDER) --- plot ID'd stars
    # ===========================
    
    # Find stars with DAO
    master_table = Table()                  # Start as astropy Table
    for i, im in enumerate(str_ims):
        sources = find_stars(im, method="dao")
        if sources is not None:
            # Add an index column as number of image
            sources['im_id'] = i
            if len(master_table) == 0:
                master_table = sources
            else:
                master_table = astropy_vstack([master_table, sources])
    sources = master_table.to_pandas()      # Convert to pandas DataFrame

    # Plot star-finding results
    def add_colorbar(im, ax):
        return fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="DN/sec", extend="max")
    for i, str_im in enumerate(str_ims):
        fig, ax = plt.subplots(1,1, figsize=(10,10), dpi=1000)
        #print(sources[sources["im_id"] == i].to_string())  # dump full df for im
        im_sources = sources[sources["im_id"] == i]
        im = ax.imshow(str_im, vmin=0, vmax=4)
        ax.set_title("   |   ".join([str_ablock[i], str_filters[i], str_time[i].astype("datetime64[s]").astype("str")]))
                    
        for idx, row in im_sources.iterrows():

            if row["peak"] > 10:        # classify based on peak 
                edgecolor="red"
                radius=10
            else:
                edgecolor='green'
                radius=10
            source_coords = (row["xcentroid"], row["ycentroid"])
            circle = patches.Circle(source_coords, radius=radius, edgecolor=edgecolor, facecolor='none', linewidth=0.5)
            ax.add_patch(circle)
        
        add_colorbar(im, ax)
        plt.show()

    # Evaluate distance to closest source for each source in each image
    sources['nn_dist'] = np.nan
    for i in sources['im_id'].unique():
        # Filter sources belonging to the current image
        im_mask = sources['im_id'] == i
        im_sources = sources[im_mask]
        if len(im_sources) > 1:
            coords = im_sources[['xcentroid', 'ycentroid']].values
            tree = KDTree(coords)       # nearest-neighbour lookup
            distances, indices = tree.query(coords, k=2)    # k=1 is star itself, k=2 is next closest star
            sources.loc[im_mask, 'nn_dist'] = distances[:, 1]
    


    nfit = 0
    nclose = 0

    for idx, source in sources.iterrows():
        #print(source)
        im_id = source["im_id"]
        im = str_ims[im_id.astype(int)]

        if source['nn_dist'] >= 50:
            # If a star is >=50 pixels from all other stars, fit a Voigt profile and use the Voigt profile as the ground-truth.
            popt, pcov = fit_2d_voigt(im, source["xcentroid"], source["ycentroid"], box_size=25) 
            nfit +=1
        elif source['nn_dist'] < 50:
            # If a star is <50 pixels from all other stars, use a Voigt profile with 𝜎 and 𝛾 extrapolated from amplitude (see slides).
            nclose +=1
            # TODO implement
        else:
            raise ValueError("something went wrong :/")
    print(nfit, nclose, nfit+nclose)

    # ===========================
    # PLOT RESULTS
    # ===========================

    if False:
        n = len(str_ims)
        fig, ax = plt.subplots(n,2, figsize=(20, 10*n), dpi=600)
        
        for i, str_im in enumerate(str_ims[0:3]):
            im00 = ax[i,0].imshow(str_ims[i], vmin=10, vmax=np.percentile(str_ims[i], 99))
            #im01 = ax[i,1].imshow(gt_ims[i], vmin=0, vmax=np.percentile(gt_ims[i], 99))
            title_str = str_time[i].astype('datetime64[s]').astype(str) + " " + str_filters[i] 
            ax[i,0].set_title(title_str)
            fig.colorbar(im00, ax=ax[i,0])
            #fig.colorbar(im01, ax=ax[i,1])
        plt.savefig("my_plot.svg", format="svg", bbox_inches="tight")



if __name__ == "__main__":
    main()
    



#%%