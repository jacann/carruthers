#%%
import numpy as np
import xarray as xr
import glob
import resources

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import glide.science_data_processing.L1A as L1A
from glide.common_components.constants import NPIX

from astropy.stats import sigma_clipped_stats
from astropy.modeling import models, fitting
from photutils.detection import DAOStarFinder
from scipy.special import voigt_profile

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


def fit_voigt_profile(image, x0, y0, box_half=15):
    """
    Fit a 2D Voigt profile to a star in the image centered near (x0, y0).

    The Voigt is approximated as a separable product of 1D Voigt profiles.
    sigma and gamma are shared between x and y axes.

    Args:
        image (2D numpy array): single background-subtracted image.
        x0 (float): initial column (x) centroid guess.
        y0 (float): initial row (y) centroid guess.
        box_half (int): half-width of the fitting box in pixels.

    Returns:
        dict with keys: amplitude, x0, y0, sigma, gamma, success (bool)
    """

    ny, nx = image.shape
    x0i, y0i = int(round(x0)), int(round(y0))

    # Extract cutout, clamped to image bounds
    y_lo = max(0, y0i - box_half)
    y_hi = min(ny, y0i + box_half + 1)
    x_lo = max(0, x0i - box_half)
    x_hi = min(nx, x0i + box_half + 1)

    cutout = image[y_lo:y_hi, x_lo:x_hi]
    if cutout.size == 0:
        return dict(amplitude=np.nan, x0=x0, y0=y0, sigma=np.nan, gamma=np.nan, success=False)

    yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi].astype(float)

    def voigt_2d(xx, yy, amplitude, cx, cy, sigma, gamma):
        """Separable 2D Voigt: V(x)*V(y), normalized so peak = amplitude."""
        v_peak = voigt_profile(0, sigma, gamma)
        vx = voigt_profile(xx - cx, sigma, gamma) / v_peak
        vy = voigt_profile(yy - cy, sigma, gamma) / v_peak
        return amplitude * vx * vy

    # Levenberg-Marquardt via scipy
    from scipy.optimize import curve_fit

    git 

    def model_flat(coords, amplitude, cx, cy, sigma, gamma):
        xx_f, yy_f = coords
        return voigt_2d(xx_f, yy_f, amplitude, cx, cy, sigma, gamma).ravel()

    p0 = [np.nanmax(cutout), x0, y0, 1.0, 0.5]
    bounds_lo = [0,    x_lo, y_lo, 0.1, 0.01]
    bounds_hi = [np.inf, x_hi, y_hi, 20., 20.]

    try:
        popt, _ = curve_fit(
            model_flat,
            (xx.ravel(), yy.ravel()),
            cutout.ravel(),
            p0=p0,
            bounds=(bounds_lo, bounds_hi),
            maxfev=5000,
        )
        amplitude, cx, cy, sigma, gamma = popt
        return dict(amplitude=amplitude, x0=cx, y0=cy, sigma=sigma, gamma=gamma, success=True)
    except Exception:
        return dict(amplitude=np.nanmax(cutout), x0=x0, y0=y0, sigma=1.5, gamma=0.5, success=False)


def extrapolate_voigt_params(sources, target_amplitude):
    """
    Extrapolate sigma and gamma from nearby well-fit stars as a function of amplitude.

    Falls back to median sigma/gamma if regression is poorly conditioned.

    Args:
        sources (list of dict): fitted Voigt parameter dicts with 'success' == True.
        target_amplitude (float): amplitude of the star to extrapolate for.

    Returns:
        sigma (float), gamma (float)
    """
    good = [s for s in sources if s["success"]]
    if len(good) == 0:
        return 1.5, 0.5  # safe defaults

    sigmas = np.array([s["sigma"] for s in good])
    gammas = np.array([s["gamma"] for s in good])
    amps   = np.array([s["amplitude"] for s in good])

    if len(good) >= 3:
        # Linear regression of sigma/gamma vs amplitude
        A = np.column_stack([amps, np.ones_like(amps)])
        try:
            sigma_coeffs, _, _, _ = np.linalg.lstsq(A, sigmas, rcond=None)
            gamma_coeffs, _, _, _ = np.linalg.lstsq(A, gammas, rcond=None)
            sigma = float(np.dot([target_amplitude, 1], sigma_coeffs))
            gamma = float(np.dot([target_amplitude, 1], gamma_coeffs))
            sigma = np.clip(sigma, 0.3, 20.0)
            gamma = np.clip(gamma, 0.01, 20.0)
            return sigma, gamma
        except Exception:
            pass

    return float(np.median(sigmas)), float(np.median(gammas))


def render_voigt_stamp(image_shape, params):
    """
    Render a 2D Voigt profile stamp into an array of the given shape.

    Args:
        image_shape (tuple): (ny, nx)
        params (dict): amplitude, x0, y0, sigma, gamma

    Returns:
        stamp (2D numpy array)
    """
    ny, nx = image_shape
    sigma = params["sigma"]
    gamma = params["gamma"]
    amplitude = params["amplitude"]
    cx = params["x0"]
    cy = params["y0"]

    v_peak = voigt_profile(0, sigma, gamma)

    yy, xx = np.ogrid[:ny, :nx]
    vx = voigt_profile(xx - cx, sigma, gamma) / v_peak
    vy = voigt_profile(yy - cy, sigma, gamma) / v_peak
    return amplitude * vx * vy


def find_and_fit_stars(image, fwhm=3.0, threshold_sigma=5.0, isolation_radius=50):
    """
    Detect stars with DAOStarFinder, quality-filter them, then fit or extrapolate
    Voigt profiles.

    Args:
        image (2D numpy array): background-subtracted image.
        fwhm (float): expected stellar FWHM in pixels for DAOStarFinder.
        threshold_sigma (float): detection threshold in units of image sigma.
        isolation_radius (float): minimum pixel distance to consider a star isolated.

    Returns:
        fitted_stars (list of dict): one entry per accepted star, with keys:
            x0, y0, amplitude, sigma, gamma, roundness, sharpness, isolated, success
    """
    # Robust image statistics for threshold
    _, _, std = sigma_clipped_stats(image, sigma=3.0)
    threshold = threshold_sigma * std

    dao = DAOStarFinder(fwhm=fwhm, threshold=threshold)
    sources = dao(image)

    if sources is None or len(sources) == 0:
        return []

    # Build list from DAOStarFinder table, applying quality filters manually.
    # DAOStarFinder returns roundness1 and roundness2 (two orthogonal axes);
    # we require both to be within (-0.5, 0.5).
    stars = []
    for row in sources:
        roundness1 = float(row["roundness1"])
        roundness2 = float(row["roundness2"])
        sharpness  = float(row["sharpness"])

        if abs(roundness1) > 0.5 or abs(roundness2) > 0.5:
            continue
        if not (0.3 < sharpness < 1.0):
            continue

        stars.append(dict(
            x0=float(row["xcentroid"]),
            y0=float(row["ycentroid"]),
            amplitude=float(row["peak"]),
            roundness=0.5 * (roundness1 + roundness2),  # scalar summary
            sharpness=sharpness,
            isolated=False,
            sigma=np.nan,
            gamma=np.nan,
            success=False,
        ))

    if len(stars) == 0:
        return []

    # Determine isolation
    coords = np.array([[s["x0"], s["y0"]] for s in stars])
    for i, s in enumerate(stars):
        dists = np.sqrt(np.sum((coords - coords[i]) ** 2, axis=1))
        dists[i] = np.inf  # exclude self
        s["isolated"] = bool(np.min(dists) > isolation_radius)

    # Fit isolated stars first
    for s in stars:
        if s["isolated"]:
            params = fit_voigt_profile(image, s["x0"], s["y0"])
            s.update(params)

    # Extrapolate for crowded stars
    for s in stars:
        if not s["isolated"]:
            sigma, gamma = extrapolate_voigt_params(
                [st for st in stars if st["isolated"]],
                s["amplitude"]
            )
            s["sigma"] = sigma
            s["gamma"] = gamma
            s["success"] = True  # extrapolated, not fitted

    return stars


# ── Main pipeline ────────────────────────────────────────────────────────────

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
# Store per-image half-medians so they can be added back into the ground-truth
bg_medians_top = np.nanmedian(str_ims_nb[:, :half_npix, :], axis=(1,2))   # (N,)
bg_medians_bot = np.nanmedian(str_ims_nb[:, half_npix:, :], axis=(1,2))   # (N,)
str_ims[:, :half_npix, :] -= bg_medians_top[:, np.newaxis, np.newaxis]
str_ims[:, half_npix:, :] -= bg_medians_bot[:, np.newaxis, np.newaxis]



str_ims[str_ims < 0] = 0




# Star ground-truth process ------
# Use DAOStarFinder to extract star locations, roundness, and sharpness from the image.
# Remove stars that are not round enough (|round| < 0.5) and too sharp (0.3 < sharpness < 1).
#   (DAOStarFinder's roundhi/roundlo/sharplo/sharphi parameters handle this internally.)
# If a star is >50 pixels from all other stars, fit a Voigt profile and use the
#   Voigt profile as the ground-truth.
# If a star is <50 pixels from all other stars, use a Voigt profile with sigma and
#   gamma extrapolated from amplitude using the fits of isolated stars in the same image.

n_images = str_ims.shape[0]
all_star_fits = []   # one list-of-dicts per image

for i in range(n_images):
    im = str_ims[i]
    fitted = find_and_fit_stars(
        im,
        fwhm=3.0,
        threshold_sigma=5.0,
        isolation_radius=50,
    )
    all_star_fits.append(fitted)


# Image ground-truth process -----
# Start with image of all zeros.
# Add all star ground-truths (rendered Voigt profiles).
# Add median back from background subtraction (per image half).
# Add calibrated closest-in-time dark image.

# Calibrate dark images (already divided by n_frames above)
drk_ims_calibrated = drk_ims  # already per-frame after division above

gt_ims = np.zeros_like(str_ims)   # (N, ny, nx)

for i in range(n_images):
    image_shape = str_ims.shape[1:]   # (ny, nx)
    gt = np.zeros(image_shape, dtype=np.float64)

    # Add all star Voigt profiles
    for star in all_star_fits[i]:
        gt += render_voigt_stamp(image_shape, star)

    # Add background medians back (per image half)
    gt[:half_npix, :] += bg_medians_top[i]
    gt[half_npix:, :] += bg_medians_bot[i]

    # Add the calibrated closest-in-time dark image
    gt += drk_ims_calibrated[drk_close_idx[i]]

    gt_ims[i] = gt


# Scatter plots ------------------
# Plot ground-truth vs. image that we got and see what happened. Helpful make individual plots per column (or 10 columns at a time) ... ground-truth vs. image vs. row-sum
n = len(str_ims)
fig, ax = plt.subplots(n,2, figsize=(10, 5*n), dpi=1000)

for i in range(n):
    im00 = ax[i,0].imshow(str_ims[i], vmin=0, vmax=np.percentile(str_ims[i], 99))
    im01 = ax[i,1].imshow(gt_ims[i], vmin=0, vmax=np.percentile(gt_ims[i], 99))
    title_str = str_time[i].astype('datetime64[s]').astype(str) + " " + str_filters[i] 
    ax[i,0].set_title(title_str)
    fig.colorbar(im00, ax=ax[i,0])
    fig.colorbar(im01, ax=ax[i,1])

plt.savefig('gt_all.png')

plt.show()
#(gt_ims)








#%%