import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import glob

from glide.common_components.utils import mask_average
from glide.common_components.utils import circular_mask
from glide.common_components import constants
import glide.science_data_processing.L1A as L1A
import glide.common_components.view_geometry as view_geometry


# Generate masks
imager = 'WFI'
npix = constants.NPIX[imager]
half_npix = npix // 2
fov_radius = constants.MASK_L1A_FOV_R[imager]
full_fov_mask = circular_mask(npix, fov_radius)



# 1. Get filepaths dynamically (returns a sorted list for proper time order)
#file_paths = sorted(glob.glob(f"/data/L1A/CARRUTHERS_GCI-{imager}_L1A-DRK_*_v1.0.nc"))

file_paths = sorted(glob.glob(f"/data/L1A/CARRUTHERS_GCI-{imager}_L1A-DRK_202602[1-3][0-9]_v1.0.nc"))
print(file_paths)
# 2. Open datasets with lazy loading & minimal metadata schema (Blazing fast!)
# We assume the time dimension doesn't need complex concatenation alignment.
ds_all = xr.open_mfdataset(
    file_paths, 
    combine="nested", 
    concat_dim="time", 
    chunks={"time": 10},  # Lazy loading using Dask (adjust chunk size as needed)
    parallel=True         # Speeds up parsing metadata across cores
)

# 3. Access your L1A wrapper
l1a_all = L1A.L1A(ds_all)


def process_all_science_images(l1a_sci, l1a_all, half_npix,
                                top_correction, bottom_correction,
                                gamma=0.1, max_half_window_hours=48,
                                time_step_minutes=30):
    """
    Run ICI-adaptive dark subtraction for every image in l1a_sci.

    For each science image:
      1. Find the nearest dark frame in l1a_all by timestamp.
      2. Run ici_window_size to select the optimal time window.
      3. Subtract the ICI median dark composite from the science image.

    Parameters
    ----------
    l1a_sci               : L1A  — science dataset
    l1a_all               : L1A  — dark frame dataset
    half_npix             : int  — row midpoint for bias split
    top_correction        : array (1, W)
    bottom_correction     : array (1, W)
    gamma                 : float
    max_half_window_hours : float
    time_step_minutes     : float

    Returns
    -------
    xr.Dataset with variables:
      sci_raw          (n_images, H, W) — normalised science images
      dark_subtracted  (n_images, H, W) — sci_raw minus ICI median dark
      ici_median_dark  (n_images, H, W) — the median dark used per image
      ici_half_window  (n_images,)      — optimal Timedelta in seconds
      ici_n_frames     (n_images,)      — number of dark frames used
      ici_dark_start   (n_images,)      — earliest dark timestamp used
      ici_dark_end     (n_images,)      — latest dark timestamp used
    coords:
      time             (n_images,)      — science image timestamps
    """
    dt_index = pd.DatetimeIndex(l1a_all.time)
    n_images = l1a_sci.images.shape[0]
    H, W     = l1a_sci.images.shape[1], l1a_sci.images.shape[2]

    sci_raw         = np.zeros((n_images, H, W), dtype=np.float32)
    dark_subtracted = np.zeros((n_images, H, W), dtype=np.float32)
    ici_median_dark = np.zeros((n_images, H, W), dtype=np.float32)
    ici_half_window = np.zeros(n_images,         dtype=np.float64)  # seconds
    ici_n_frames    = np.zeros(n_images,         dtype=np.int32)
    ici_dark_start  = np.empty(n_images,         dtype='datetime64[ns]')
    ici_dark_end    = np.empty(n_images,         dtype='datetime64[ns]')

    for i in range(n_images):
        # ── normalise science image ───────────────────────────
        sci_norm = l1a_sci.images[i, :, :] / l1a_all.n_frames[i]

        # ── nearest dark frame ────────────────────────────────
        target_dt       = pd.Timestamp(l1a_sci.time[i])
        nearest_idx_pos = dt_index.get_indexer([target_dt], method='nearest')[0]

        # ── ICI window ────────────────────────────────────────
        opt_half_td, _, med_img, ici_indices = ici_window_size(
            images                = l1a_all.images,
            n_frames              = l1a_all.n_frames,
            time                  = l1a_all.time,
            target_idx            = nearest_idx_pos,
            half_npix             = half_npix,
            top_correction        = top_correction,
            bottom_correction     = bottom_correction,
            gamma                 = gamma,
            max_half_window_hours = max_half_window_hours,
            time_step_minutes     = time_step_minutes,
        )

        ici_times = pd.DatetimeIndex(l1a_all.time[ici_indices])

        sci_raw        [i] = sci_norm
        ici_median_dark[i] = med_img
        dark_subtracted[i] = sci_norm - med_img
        ici_half_window[i] = opt_half_td.total_seconds()
        ici_n_frames   [i] = len(ici_indices)
        ici_dark_start [i] = ici_times.min().to_datetime64()
        ici_dark_end   [i] = ici_times.max().to_datetime64()

        print(f"  [{i+1}/{n_images}] {target_dt}  →  "
              f"±{opt_half_td}, {len(ici_indices)} frames")

    # ── pack into xarray Dataset ──────────────────────────────
    coords = {"time": ("time", l1a_sci.time)}
    dims2d = ("time", "y", "x")

    ds = xr.Dataset(
        {
            "sci_raw":         (dims2d, sci_raw),
            "dark_subtracted": (dims2d, dark_subtracted),
            "ici_median_dark": (dims2d, ici_median_dark),
            "ici_half_window": ("time", ici_half_window),
            "ici_n_frames":    ("time", ici_n_frames),
            "ici_dark_start":  ("time", ici_dark_start),
            "ici_dark_end":    ("time", ici_dark_end),
        },
        coords=coords,
        attrs={
            "description":         "ICI-adaptive dark-subtracted science images",
            "gamma":               gamma,
            "max_half_window_hrs": max_half_window_hours,
            "time_step_min":       time_step_minutes,
        },
    )

    return ds


ds_corrected = process_all_science_images(
    l1a_sci               = l1a_sci,
    l1a_all               = l1a_all,
    half_npix             = half_npix,
    top_correction        = top_correction,
    bottom_correction     = bottom_correction,
    gamma                 = gamma,
    max_half_window_hours = 48,
    time_step_minutes     = 30,
)

out_path = f"products/CARRUTHERS_GCI-{imager}_ICI_DARK_SUBTRACTED.nc"
ds_corrected.to_netcdf(out_path)
print(f"Saved → {out_path}")