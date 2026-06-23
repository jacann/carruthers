import numpy as np
import xarray as xr
import time
import os
import concurrent.futures
from functools import partial
import glob
import multiprocessing as mp

from glide.common_components.utils import mask_average
from glide.common_components.utils import circular_mask
from glide.common_components import constants
import glide.science_data_processing.L1A as L1A
import glide.common_components.view_geometry as view_geometry
from glide.common_components.stars import get_beta_angle
from glide.calibration.bias_wavelet import wavelet_destripe  # ADDED

def get_filenames(data_dir, imager):
    filepaths = glob.glob(data_dir + "CARRUTHERS_GCI-" + imager + "_L1A-DRK" + "**" + "v1.0.nc")
    filepaths.sort()
    return filepaths

def filter_time_range(data, datetimes, start_datetime, end_datetime):
    datetimes = np.asarray(datetimes.values if hasattr(datetimes, 'values') else datetimes)
    data = np.asarray(data.values if hasattr(data, 'values') else data)

    start_idx = np.searchsorted(datetimes, start_datetime, side='left')
    end_idx = np.searchsorted(datetimes, end_datetime, side='right')

    return data[start_idx:end_idx]

def filter_n_frames(data, n_frames, n_frames_min):
    n_frames_mask = n_frames >= n_frames_min

    if np.issubdtype(data.dtype, np.floating):
        data = data.copy()
        data[~n_frames_mask] = np.nan
    elif np.issubdtype(data.dtype, np.datetime64):
        data = data.copy()
        data[~n_frames_mask] = np.datetime64('NaT', 'ns')
    
    return data

# ADDED: helper to compute annulus median for a single 2D image
def get_annulus_median(image, annulus_mask):
    return float(np.median(image[annulus_mask]))

def retrieve_mcp_radiation(filepath, imager, mask_fov_top, mask_fov_bottom, top_col_biases, bottom_col_biases, half_npix,
                            annulus_mask_top, annulus_mask_bottom):  # ADDED: two new mask parameters
    with xr.open_dataset(filepath, engine='netcdf4') as data:
        l1a_obj = L1A.L1A(data)
    
    # load data from dataset
    images = l1a_obj.images.copy()
    n_frames = l1a_obj.n_frames
    t_int = l1a_obj.t_int
    time = l1a_obj.time

    # CHANGED: compute annulus medians instead of mask_average FOV means
    # Iterate over images, normalizing and destriping each before computing median
    n_obs = len(images)
    mcp_rad_top_uncorrected    = np.empty(n_obs)
    mcp_rad_bottom_uncorrected = np.empty(n_obs)

    for i in range(n_obs):
        raw = images[i] / float(n_frames[i])
        destriped = wavelet_destripe(raw, log=True)
        mcp_rad_top_uncorrected[i]    = get_annulus_median(destriped, annulus_mask_top)
        mcp_rad_bottom_uncorrected[i] = get_annulus_median(destriped, annulus_mask_bottom)

    # Subtract voltage biases
    top_correction = n_frames[:, np.newaxis, np.newaxis] * top_col_biases[np.newaxis, np.newaxis, :]
    bottom_correction = n_frames[:, np.newaxis, np.newaxis] * bottom_col_biases[np.newaxis, np.newaxis, :]

    images[:, :half_npix, :] -= top_correction
    images[:, half_npix:, :] -= bottom_correction

    # CHANGED: compute annulus medians on bias-corrected images
    mcp_rad_top    = np.empty(n_obs)
    mcp_rad_bottom = np.empty(n_obs)

    for i in range(n_obs):
        raw = images[i] / float(n_frames[i])
        destriped = wavelet_destripe(raw, log=True)
        mcp_rad_top[i]    = get_annulus_median(destriped, annulus_mask_top)
        mcp_rad_bottom[i] = get_annulus_median(destriped, annulus_mask_bottom)

    # calculate roll angles
    roll_angles = np.array([scraft.moc_roll for scraft in l1a_obj.scrafts]).flatten()

    # calculate beta angle
    beta_angle = np.array([
        get_beta_angle(scraft, 
        *scraft.boresight_to_sky(imager, view_geometry.Star_frame))
        for scraft in l1a_obj.scrafts
    ]).flatten()

    beta_angle = 180 - beta_angle

    # Save temperature proxy
    top_inds = slice(0, half_npix)
    bottom_inds = slice(half_npix, None)
    top_temp_proxy = np.mean(l1a_obj.bias[:, top_inds, :], axis=(1,2)) 
    bottom_temp_proxy = np.mean(l1a_obj.bias[:, bottom_inds, :], axis=(1,2))
    mean_temp_proxy = np.mean(np.vstack([top_temp_proxy, bottom_temp_proxy]), axis=0)
    temp_proxy = np.array([[mean_temp_proxy], [top_temp_proxy], [bottom_temp_proxy]])
    return mcp_rad_top, mcp_rad_bottom, mcp_rad_top_uncorrected, mcp_rad_bottom_uncorrected, time, n_frames, t_int, roll_angles, beta_angle, temp_proxy

def process_mcp_data(filepaths, imager, mask_fov_top, mask_fov_bottom,
                     top_col_biases, bottom_col_biases, half_npix,
                     annulus_mask_top, annulus_mask_bottom):  # ADDED: two new mask parameters

    total_files = len(filepaths)
    files_processed = 0
    results = []

    worker_func = partial(
        retrieve_mcp_radiation,
        imager=imager,
        mask_fov_top=mask_fov_top,
        mask_fov_bottom=mask_fov_bottom,
        top_col_biases=top_col_biases,
        bottom_col_biases=bottom_col_biases,
        half_npix=half_npix,
        annulus_mask_top=annulus_mask_top,       # ADDED
        annulus_mask_bottom=annulus_mask_bottom, # ADDED
    )

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(worker_func, fp): fp for fp in filepaths}

        for future in concurrent.futures.as_completed(futures):
            fp = futures[future]
            try:
                res = future.result()
                results.append(res)
                files_processed += 1
                print(f"Processed file {files_processed}/{total_files}: {fp}", flush=True)
            except Exception as e:
                print(f"File failed: {fp} with error: {e}", flush=True)

    # Unpack results
    fov_means_top = [res[0] for res in results]
    fov_means_bottom = [res[1] for res in results]
    fov_means_top_uncorrected = [res[2] for res in results]
    fov_means_bottom_uncorrected = [res[3] for res in results]
    times = [res[4] for res in results]
    n_frames = [res[5] for res in results]
    t_ints = [res[6] for res in results]
    roll_angles = [res[7] for res in results]
    beta_angles = [res[8] for res in results]
    temp_proxies = [res[9] for res in results]

    # Convert lists to arrays after collecting all data
    fov_means_top = np.concatenate(fov_means_top)
    fov_means_bottom = np.concatenate(fov_means_bottom)
    fov_means_top_uncorrected = np.concatenate(fov_means_top_uncorrected)
    fov_means_bottom_uncorrected = np.concatenate(fov_means_bottom_uncorrected)
    times = np.concatenate(times)
    n_frames = np.concatenate(n_frames)
    t_ints = np.concatenate(t_ints)
    roll_angles = np.concatenate(roll_angles)
    beta_angles = np.concatenate(beta_angles)
    temp_proxies = np.concatenate(temp_proxies, axis=2)
    temp_proxies = np.squeeze(temp_proxies, axis=1)
    temp_proxies = temp_proxies.T

    # sort data by time
    sort_idx = np.argsort(times)
    fov_means_top = fov_means_top[sort_idx]
    fov_means_bottom = fov_means_bottom[sort_idx]
    fov_means_top_uncorrected = fov_means_top_uncorrected[sort_idx]
    fov_means_bottom_uncorrected = fov_means_bottom_uncorrected[sort_idx]
    n_frames = n_frames[sort_idx]
    t_ints = t_ints[sort_idx]
    roll_angles = roll_angles[sort_idx]
    beta_angles = beta_angles[sort_idx]
    temp_proxies = temp_proxies[sort_idx]
    times = times[sort_idx]

    # Create xarray Dataset with all data
    # CHANGED: variable names updated to reflect annulus medians
    ds_output = xr.Dataset({
        'annulus_median_top': (['observation'], fov_means_top),
        'annulus_median_bottom': (['observation'], fov_means_bottom),
        'annulus_median_top_uncorrected': (['observation'], fov_means_top_uncorrected),
        'annulus_median_bottom_uncorrected': (['observation'], fov_means_bottom_uncorrected),
        'time': (['observation'], times),
        'n_frames': (['observation'], n_frames),
        't_int': (['observation'], t_ints),
        'roll_angles': (['observation'], roll_angles),
        'beta_angles': (['observation'], beta_angles),
        'temp_proxies': (['observation', 'sensor_region'], temp_proxies)
    },  coords={
        'observation': times,
        'sensor_region': ['mean', 'top_half', 'bottom_half']
    })

    # CHANGED: variable attributes updated to reflect annulus medians
    ds_output['annulus_median_top'].attrs = {'units': 'DN frame-1', 'long_name': 'Annulus Median Top Half'}
    ds_output['annulus_median_bottom'].attrs = {'units': 'DN frame-1', 'long_name': 'Annulus Median Bottom Half'}
    ds_output['annulus_median_top_uncorrected'].attrs = {'units': 'DN frame-1', 'long_name': 'Uncorrected Annulus Median Top Half'}
    ds_output['annulus_median_bottom_uncorrected'].attrs = {'units': 'DN frame-1', 'long_name': 'Uncorrected Annulus Median Bottom Half'}
    ds_output['n_frames'].attrs = {'long_name': 'Number of Frames', 'units': 'n'}
    ds_output['t_int'].attrs = {'long_name': 'Integration Time', 'units': 's'}
    ds_output['roll_angles'].attrs = {'long_name': 'Spacecraft Roll Angle', 'units': 'degrees'}
    ds_output['beta_angles'].attrs = {'long_name': 'Beta Angle', 'units': 'degrees'}
    ds_output['temp_proxies'].attrs = {'long_name': 'Temperature Proxies', 'description': 'Mean, top half, and bottom half temperature proxies calculated from bias values'}

    # CHANGED: output filename updated
    output_filepath = "products/" + imager + "_ANNULUS_MEDIAN.nc"
    ds_output.to_netcdf(output_filepath)
    print(f"Annulus median data saved to {output_filepath}")
    ds_output.close()
    return

def generate_masks(imager):
    npix = constants.NPIX[imager]
    fov_radius = constants.MASK_L1A_FOV_R[imager]

    full_fov_mask = circular_mask(npix, fov_radius)
    row_indices = np.arange(npix)[:, np.newaxis]
    mask_fov_top    = np.logical_and(full_fov_mask, row_indices < npix // 2)
    mask_fov_bottom = np.logical_and(full_fov_mask, row_indices >= npix // 2)

    return mask_fov_top, mask_fov_bottom

# ADDED: builds the annulus masks (ring just inside the FOV edge)
def generate_annulus_masks(imager, annulus_width=10):
    npix         = constants.NPIX[imager]
    outer_radius = constants.MASK_L1A_FOV_R[imager]
    inner_radius = outer_radius - annulus_width

    base_annulus = circular_mask(npix, outer_radius) & ~circular_mask(npix, inner_radius)
    row_indices  = np.arange(npix)[:, np.newaxis]
    annulus_mask_top    = base_annulus & (row_indices < npix // 2)
    annulus_mask_bottom = base_annulus & (row_indices >= npix // 2)

    return annulus_mask_top, annulus_mask_bottom


def main():
    for imager in ["WFI", "NFI"]:
        start_time = time.perf_counter()

        data_files_directory = "/data/L1A/"

        if imager == "WFI":
            top_col_biases = np.load('products/COL_BIAS_WFI_TOP.npy')
            bottom_col_biases = np.load('products/COL_BIAS_WFI_BOTTOM.npy')
        elif imager == "NFI":
            top_col_biases = np.load('products/COL_BIAS_NFI_TOP.npy')
            bottom_col_biases = np.load('products/COL_BIAS_NFI_BOTTOM.npy')
        else:
            print("Invalid imager. Use 'WFI' or 'NFI'.")
            return

        filepaths = get_filenames(data_files_directory, imager)

        if False:
            if imager == "WFI":
                filepaths = ["/data/L1A/CARRUTHERS_GCI-WFI_L1A-DRK_20251004_v1.0.nc", 
                            "/data/L1A/CARRUTHERS_GCI-WFI_L1A-DRK_20251005_v1.0.nc",]
            elif imager == "NFI":
                filepaths = ["/data/L1A/CARRUTHERS_GCI-NFI_L1A-DRK_20251013_v1.0.nc"]

        print(f"Found {len(filepaths)} {imager}_L1A-DRK files in {data_files_directory}")

        mask_fov_top, mask_fov_bottom = generate_masks(imager)
        annulus_mask_top, annulus_mask_bottom = generate_annulus_masks(imager)  # ADDED

        half_npix = int((constants.NPIX[imager])/2)
        process_mcp_data(filepaths, imager, mask_fov_top, mask_fov_bottom,
                        top_col_biases, bottom_col_biases, half_npix,
                        annulus_mask_top, annulus_mask_bottom)  # ADDED

        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"{imager} annulus median processing complete ({execution_time:.2f} seconds).")

    return

if __name__ == '__main__':
    main()