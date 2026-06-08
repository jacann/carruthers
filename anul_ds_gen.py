# %%
import numpy as np
import xarray as xr
import glob
import time
import concurrent.futures
from functools import partial

from glide.common_components.utils import circular_mask
from glide.common_components import constants
import glide.science_data_processing.L1A as L1A
import glide.common_components.view_geometry as view_geometry
from glide.common_components.stars import get_beta_angle
from glide.calibration.bias_wavelet import wavelet_destripe
from glide.calibration.bias_wavelet import remove_dark_stripes


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def get_filenames(data_dir: str, pattern: str) -> list[str]:
    """Return sorted list of file paths matching *pattern* inside *data_dir*.

    Parameters
    ----------
    data_dir:
        Root directory to search (trailing slash optional).
    pattern:
        Glob pattern relative to *data_dir*, e.g.
        ``"CARRUTHERS_GCI-NFI_L1A-DRK**v1.0.nc"``.
    """
    if not data_dir.endswith("/"):
        data_dir += "/"
    filepaths = glob.glob(data_dir + pattern)
    filepaths.sort()
    return filepaths


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def build_annulus_mask(npix: int, inner_r: float, outer_r: float) -> np.ndarray:
    """Boolean mask selecting pixels in the annulus [inner_r, outer_r)."""
    inner_mask = circular_mask(npix, inner_r)
    outer_mask = circular_mask(npix, outer_r)
    return outer_mask & ~inner_mask


def get_annulus_median(image: np.ndarray, annulus_mask: np.ndarray) -> float:
    """Return the median of pixels selected by *annulus_mask* in *image*."""
    return float(np.median(image[annulus_mask]))


# ---------------------------------------------------------------------------
# Per-file processing (runs in worker process)
# ---------------------------------------------------------------------------

def process_file(
    filepath: str,
    channel: str,
    annulus_mask_top: np.ndarray,
    annulus_mask_bottom: np.ndarray,
    inner_radius: float,
    outer_radius: float,
) -> dict:
    """Open one L1A file, destripe every image, and compute annulus medians.

    Returns a dict with arrays of length *n_observations* (one per image in
    the file) plus scalar metadata.
    """
    with xr.open_dataset(filepath, engine="netcdf4") as ds:
        l1a_obj = L1A.L1A(ds)

    n_obs = len(l1a_obj.images)

    medians_top    = np.empty(n_obs)
    medians_bottom = np.empty(n_obs)
    medians_full   = np.empty(n_obs)
    cam_filter     = np.empty(n_obs, dtype='<U32')

    for i in range(n_obs):
        # Normalize by number of frames then destripe
        raw = l1a_obj.images[i] / float(l1a_obj.t_int[i])
        # Use log if image is science LyA, use False otherwise

        cam_mode = l1a_obj.cam_modes[i].mode
        cam_filter[i] = l1a_obj.cam_modes[i].filter

        if cam_mode == "science":
            if cam_filter[i] == "LyaN" or cam_filter[i] == "LyaX":
                log = True
            else:
                log = False
            destriped, _ = wavelet_destripe(raw, log=log)
        elif cam_mode == "dark":
            # for dark frames, use simpler subtraction using known offsets
            destriped, _ = remove_dark_stripes(raw, channel)
        else:
            raise ValueError(f"Unexpected camera mode '{cam_mode}' in file {filepath}") 


        medians_top[i]    = get_annulus_median(destriped, annulus_mask_top)
        medians_bottom[i] = get_annulus_median(destriped, annulus_mask_bottom)
        medians_full[i]   = get_annulus_median(destriped, annulus_mask_top | annulus_mask_bottom)

    times       = np.asarray(l1a_obj.time)
    n_frames    = np.asarray(l1a_obj.n_frames)
    t_ints      = np.asarray(l1a_obj.t_int)

    roll_angles = np.array(
        [scraft.moc_roll for scraft in l1a_obj.scrafts]
    ).flatten()

    beta_angles = np.array([
        get_beta_angle(
            scraft,
            *scraft.boresight_to_sky(channel, view_geometry.Star_frame),
        )
        for scraft in l1a_obj.scrafts
    ]).flatten()
    beta_angles = 180.0 - beta_angles  # convert to angle from the Sun

    # get filter info
    filter = l1a_obj.cam_modes

    return {
        "medians_top":    medians_top,
        "medians_bottom": medians_bottom,
        "medians_full":   medians_full,
        "times":          times,
        "n_frames":       n_frames,
        "t_ints":         t_ints,
        "roll_angles":    roll_angles,
        "beta_angles":    beta_angles,
        "cam_filter":     cam_filter
    }


# ---------------------------------------------------------------------------
# Parallel orchestration
# ---------------------------------------------------------------------------

def process_all_files(
    filepaths: list[str],
    channel: str,
    annulus_mask_top: np.ndarray,
    annulus_mask_bottom: np.ndarray,
    inner_radius: float,
    outer_radius: float,
) -> dict:
    """Process all files in parallel; return concatenated result arrays."""
    total = len(filepaths)
    results = []

    worker = partial(
        process_file,
        channel=channel,
        annulus_mask_top=annulus_mask_top,
        annulus_mask_bottom=annulus_mask_bottom,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=64) as executor:
        futures = {executor.submit(worker, fp): fp for fp in filepaths}
        done = 0
        for future in concurrent.futures.as_completed(futures):
            fp = futures[future]
            try:
                results.append(future.result())
                done += 1
                print(f"[{done}/{total}] Done: {fp}", flush=True)
            except Exception as exc:
                done += 1
                print(f"[{done}/{total}] FAILED: {fp} — {exc}", flush=True)

    if not results:
        raise RuntimeError("No files were processed successfully.")

    # Concatenate across files
    combined = {
        key: np.concatenate([r[key] for r in results])
        for key in results[0]
    }

    # Sort by time
    sort_idx = np.argsort(combined["times"])
    for key in combined:
        combined[key] = combined[key][sort_idx]

    return combined


# ---------------------------------------------------------------------------
# Save to NetCDF
# ---------------------------------------------------------------------------

def save_results(combined: dict, channel: str, inner_radius: float, outer_radius: float, output_path: str) -> None:
    times = combined["times"]

    ds = xr.Dataset(
        {
            "annulus_median_top": (["observation"], combined["medians_top"]),
            "annulus_median_bottom": (["observation"], combined["medians_bottom"]),
            "annulus_median_full": (["observation"], combined["medians_full"]),
            "n_frames":    (["observation"], combined["n_frames"]),
            "t_int":       (["observation"], combined["t_ints"]),
            "roll_angles": (["observation"], combined["roll_angles"]),
            "beta_angles": (["observation"], combined["beta_angles"]),
            "cam_filter":  (["observation"], combined["cam_filter"]),
            "time":        (["observation"], times),
        },
        coords={"observation": times},
    )

    ds["annulus_median_top"].attrs    = {"units": "DN frame-1", "long_name": "Annulus Median (top half)"}
    ds["annulus_median_bottom"].attrs = {"units": "DN frame-1", "long_name": "Annulus Median (bottom half)"}
    ds["annulus_median_full"].attrs   = {"units": "DN frame-1", "long_name": "Annulus Median (full annulus)"}
    ds["n_frames"].attrs  = {"units": "n",       "long_name": "Number of Frames"}
    ds["t_int"].attrs     = {"units": "s",        "long_name": "Integration Time"}
    ds["roll_angles"].attrs = {"units": "degrees", "long_name": "Spacecraft Roll Angle"}
    ds["beta_angles"].attrs = {"units": "degrees", "long_name": "Beta Angle (from Sun)"}
    ds.attrs = {
        "channel": channel,
        "inner_radius_px": inner_radius,
        "outer_radius_px": outer_radius,
        "description": "Wavelet-destriped annulus medians computed from L1A dark frames.",
    }
    try:
        ds.to_netcdf(output_path)
        print(f"Saved results to {output_path}")
    except Exception as e:
        print(f"Error saving results to {output_path}: {e}")
        print(f"Attempting to save to {output_path.replace('.nc', '_NEW.nc')} instead.")
        try:
            ds.to_netcdf(output_path.replace('.nc', '_NEW.nc'))
            print(f"Saved results to {output_path.replace('.nc', '_NEW.nc')}")
        except Exception as e2:
            print(f"Failed to save results to {output_path.replace('.nc', '_NEW.nc')}: {e2}")
            print("Results were not saved.")
    ds.close()
    


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- Configuration -------------------------------------------------------
    DATA_DIR = "/data/L1A/"

    # Glob pattern relative to DATA_DIR.  Use "**" for any infix characters.
    # Examples:
    #   All WFI dark frames:  "CARRUTHERS_GCI-WFI_L1A-DRK**v1.0.nc"
    #   All NFI dark frames:  "CARRUTHERS_GCI-NFI_L1A-DRK**v1.0.nc"
    #   All channels:          "CARRUTHERS_GCI-*_L1A-DRK**v1.0.nc"

    CHANNEL = "WFI"   # "WFI" or "NFI" — used for FOV radius lookup and boresight projection
    CAM_MODE = "SCI"  # "SCI" or "DRK" — determines destriping method and log usage

    FILENAME_PATTERN = f"CARRUTHERS_GCI-{CHANNEL}_L1A-{CAM_MODE}**v1.0.nc"

    # Annulus defined as [outer_radius - annulus_width, outer_radius]
    ANNULUS_WIDTH = 10   # pixels

    OUTPUT_PATH = f"products/{CHANNEL}_{CAM_MODE}_ANNULUS_MEDIANS.nc"
    # --------------------------------------------------------------------------

    t0 = time.perf_counter()

    filepaths = get_filenames(DATA_DIR, FILENAME_PATTERN)
    if not filepaths:
        print(f"No files found matching pattern '{FILENAME_PATTERN}' in '{DATA_DIR}'.")
        return
    print(f"Found {len(filepaths)} files.")

    # Build annulus masks once (shared across workers via pickling)
    npix         = constants.NPIX[CHANNEL]
    outer_radius = constants.MASK_L1A_FOV_R[CHANNEL]
    inner_radius = outer_radius - ANNULUS_WIDTH

    half = npix // 2
    row_idx = np.arange(npix)[:, np.newaxis]  # (npix, 1) broadcasts over columns

    base_annulus      = build_annulus_mask(npix, inner_radius, outer_radius)
    annulus_mask_top    = base_annulus & (row_idx < half)
    annulus_mask_bottom = base_annulus & (row_idx >= half)

    # Process files in parallel
    combined = process_all_files(
        filepaths,
        channel=CHANNEL,
        annulus_mask_top=annulus_mask_top,
        annulus_mask_bottom=annulus_mask_bottom,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
    )

    save_results(combined, CHANNEL, inner_radius, outer_radius, OUTPUT_PATH)

    print(f"Done in {time.perf_counter() - t0:.1f} s.")


if __name__ == "__main__":
    main()
# %%