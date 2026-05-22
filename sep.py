# %%
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

def get_filenames(data_dir, imager):
    filepaths = glob.glob(data_dir + "CARRUTHERS_GCI-" + imager + "_L1A-DRK" + "**" + "v1.0.nc")
    filepaths.sort()
    return filepaths

def filter_time_range(data, datetimes, start_datetime, end_datetime):
    # Extract underlying numpy arrays from xarray DataArrays if needed
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

def load_and_filter_data(start_datetime_str,
                          end_datetime_str,
                            filter_neg,
                              beta_max):

    ds = xr.open_dataset("products/WFI_FOV_AVG.nc")
    wfi_fov_mean_top = ds['fov_mean_top']
    wfi_fov_mean_bottom = ds['fov_mean_bottom']
    wfi_fov_mean_top_uncorrected = ds['fov_mean_top_uncorrected']
    wfi_fov_mean_bottom_uncorrected = ds['fov_mean_bottom_uncorrected']
    wfi_time = ds['time']
    wfi_roll_angles = ds['roll_angles']
    wfi_beta_angles = ds['beta_angles']
    wfi_n_frames = ds['n_frames']
    wfi_temp_proxy = ds['temp_proxies']
    ds.close()

    ds = xr.open_dataset("products/NFI_FOV_AVG.nc")
    nfi_fov_mean_top = ds['fov_mean_top']
    nfi_fov_mean_bottom = ds['fov_mean_bottom']
    nfi_fov_mean_top_uncorrected = ds['fov_mean_top_uncorrected']
    nfi_fov_mean_bottom_uncorrected = ds['fov_mean_bottom_uncorrected']
    nfi_time = ds['time']
    nfi_roll_angles = ds['roll_angles']
    nfi_beta_angles = ds['beta_angles']
    nfi_n_frames = ds['n_frames']
    nfi_temp_proxy = ds['temp_proxies']
    ds.close()

    # Sort data by time just in case (should already be sorted, but to be safe)
    # Sort all arrays by time
    wfi_sort_idx = np.argsort(wfi_time)
    nfi_sort_idx = np.argsort(nfi_time)

    wfi_fov_mean_top = wfi_fov_mean_top[wfi_sort_idx]
    wfi_fov_mean_bottom = wfi_fov_mean_bottom[wfi_sort_idx]
    wfi_fov_mean_top_uncorrected = wfi_fov_mean_top_uncorrected[wfi_sort_idx]
    wfi_fov_mean_bottom_uncorrected = wfi_fov_mean_bottom_uncorrected[wfi_sort_idx]
    wfi_n_frames = wfi_n_frames[wfi_sort_idx]
    wfi_temp_proxy = wfi_temp_proxy[wfi_sort_idx]
    wfi_roll_angles = wfi_roll_angles[wfi_sort_idx]
    wfi_beta_angles = wfi_beta_angles[wfi_sort_idx]
    wfi_time = wfi_time[wfi_sort_idx]

    nfi_fov_mean_top = nfi_fov_mean_top[nfi_sort_idx]
    nfi_fov_mean_bottom = nfi_fov_mean_bottom[nfi_sort_idx]
    nfi_fov_mean_top_uncorrected = nfi_fov_mean_top_uncorrected[nfi_sort_idx]
    nfi_fov_mean_bottom_uncorrected = nfi_fov_mean_bottom_uncorrected[nfi_sort_idx]
    nfi_n_frames = nfi_n_frames[nfi_sort_idx]
    nfi_temp_proxy = nfi_temp_proxy[nfi_sort_idx]
    nfi_roll_angles = nfi_roll_angles[nfi_sort_idx]
    nfi_beta_angles = nfi_beta_angles[nfi_sort_idx]
    nfi_time = nfi_time[nfi_sort_idx]


    start_dt = np.datetime64(start_datetime_str)
    end_dt = np.datetime64(end_datetime_str)

    '''
    n_frames_min = 7000

    # Filter the number of frames
    wfi_fov_mean_top = filter_n_frames(wfi_fov_mean_top, wfi_n_frames, n_frames_min)
    wfi_fov_mean_bottom = filter_n_frames(wfi_fov_mean_bottom, wfi_n_frames, n_frames_min)
    wfi_time = filter_n_frames(wfi_time, wfi_n_frames, n_frames_min)
    wfi_roll_angles = filter_n_frames(wfi_roll_angles, wfi_n_frames, n_frames_min)
    wfi_beta_angles = filter_n_frames(wfi_beta_angles, wfi_n_frames, n_frames_min)
    nfi_fov_mean_top = filter_n_frames(nfi_fov_mean_top, nfi_n_frames, n_frames_min)
    nfi_fov_mean_bottom = filter_n_frames(nfi_fov_mean_bottom, nfi_n_frames, n_frames_min)
    nfi_time = filter_n_frames(nfi_time, nfi_n_frames, n_frames_min)
    nfi_roll_angles = filter_n_frames(nfi_roll_angles, nfi_n_frames, n_frames_min)
    nfi_beta_angles = filter_n_frames(nfi_beta_angles, nfi_n_frames, n_frames_min)

    '''

    # Filter the time range
    wfi_fov_mean_top    = filter_time_range(wfi_fov_mean_top,    wfi_time, start_dt, end_dt)
    wfi_fov_mean_top_uncorrected    = filter_time_range(wfi_fov_mean_top_uncorrected,    wfi_time, start_dt, end_dt)
    wfi_fov_mean_bottom = filter_time_range(wfi_fov_mean_bottom, wfi_time, start_dt, end_dt)
    wfi_fov_mean_bottom_uncorrected = filter_time_range(wfi_fov_mean_bottom_uncorrected, wfi_time, start_dt, end_dt)
    wfi_roll_angles     = filter_time_range(wfi_roll_angles,     wfi_time, start_dt, end_dt)
    wfi_beta_angles     = filter_time_range(wfi_beta_angles,     wfi_time, start_dt, end_dt)
    wfi_temp_proxy     = filter_time_range(wfi_temp_proxy,     wfi_time, start_dt, end_dt)
    wfi_time            = filter_time_range(wfi_time,            wfi_time, start_dt, end_dt)  # filter last

    nfi_fov_mean_top    = filter_time_range(nfi_fov_mean_top,    nfi_time, start_dt, end_dt)
    nfi_fov_mean_top_uncorrected    = filter_time_range(nfi_fov_mean_top_uncorrected,    nfi_time, start_dt, end_dt)
    nfi_fov_mean_bottom = filter_time_range(nfi_fov_mean_bottom, nfi_time, start_dt, end_dt)
    nfi_fov_mean_bottom_uncorrected = filter_time_range(nfi_fov_mean_bottom_uncorrected, nfi_time, start_dt, end_dt)    
    nfi_roll_angles     = filter_time_range(nfi_roll_angles,     nfi_time, start_dt, end_dt)
    nfi_beta_angles     = filter_time_range(nfi_beta_angles,     nfi_time, start_dt, end_dt)
    nfi_temp_proxy     = filter_time_range(nfi_temp_proxy,     nfi_time, start_dt, end_dt)
    nfi_time            = filter_time_range(nfi_time,            nfi_time, start_dt, end_dt)  # filter last

    if filter_neg == True:
        # Filter out values where corrected FOV Avgs are negative across all variables
        valid_indices_wfi = (wfi_fov_mean_top > 0) | (wfi_fov_mean_bottom > 0) 
        valid_indices_nfi = (nfi_fov_mean_top > 0) | (nfi_fov_mean_bottom > 0)
        wfi_fov_mean_top = wfi_fov_mean_top[valid_indices_wfi]
        wfi_fov_mean_bottom = wfi_fov_mean_bottom[valid_indices_wfi]
        nfi_fov_mean_top = nfi_fov_mean_top[valid_indices_nfi]
        nfi_fov_mean_bottom = nfi_fov_mean_bottom[valid_indices_nfi ]
        wfi_fov_mean_top_uncorrected = wfi_fov_mean_top_uncorrected[valid_indices_wfi]
        wfi_fov_mean_bottom_uncorrected = wfi_fov_mean_bottom_uncorrected[valid_indices_wfi]
        nfi_fov_mean_top_uncorrected = nfi_fov_mean_top_uncorrected[valid_indices_nfi]
        nfi_fov_mean_bottom_uncorrected = nfi_fov_mean_bottom_uncorrected[valid_indices_nfi]
        wfi_temp_proxy = wfi_temp_proxy[valid_indices_wfi]
        nfi_temp_proxy = nfi_temp_proxy[valid_indices_nfi]
        wfi_time = wfi_time[valid_indices_wfi]
        nfi_time = nfi_time[valid_indices_nfi]
        wfi_roll_angles = wfi_roll_angles[valid_indices_wfi]
        wfi_beta_angles = wfi_beta_angles[valid_indices_wfi]
        nfi_roll_angles = nfi_roll_angles[valid_indices_nfi]
        nfi_beta_angles = nfi_beta_angles[valid_indices_nfi]

    if beta_max < 360:
        # Filter out values where beta angles are above the specified maximum across all variables
        valid_indices_wfi = (wfi_beta_angles <= beta_max)
        valid_indices_nfi = (nfi_beta_angles <= beta_max)
        wfi_fov_mean_top = wfi_fov_mean_top[valid_indices_wfi]
        wfi_fov_mean_bottom = wfi_fov_mean_bottom[valid_indices_wfi]
        nfi_fov_mean_top = nfi_fov_mean_top[valid_indices_nfi]
        nfi_fov_mean_bottom = nfi_fov_mean_bottom[valid_indices_nfi ]
        wfi_fov_mean_top_uncorrected = wfi_fov_mean_top_uncorrected[valid_indices_wfi]
        wfi_fov_mean_bottom_uncorrected = wfi_fov_mean_bottom_uncorrected[valid_indices_wfi]
        nfi_fov_mean_top_uncorrected = nfi_fov_mean_top_uncorrected[valid_indices_nfi]
        nfi_fov_mean_bottom_uncorrected = nfi_fov_mean_bottom_uncorrected[valid_indices_nfi]
        wfi_temp_proxy = wfi_temp_proxy[valid_indices_wfi]
        nfi_temp_proxy = nfi_temp_proxy[valid_indices_nfi]
        wfi_time = wfi_time[valid_indices_wfi]
        nfi_time = nfi_time[valid_indices_nfi]
        wfi_roll_angles = wfi_roll_angles[valid_indices_wfi]
        wfi_beta_angles = wfi_beta_angles[valid_indices_wfi]
        nfi_roll_angles = nfi_roll_angles[valid_indices_nfi]
        nfi_beta_angles = nfi_beta_angles[valid_indices_nfi]



    # Calculate the average of the top and bottom FOV Avgs
    wfi_avg_fov_mean = np.mean([wfi_fov_mean_top, wfi_fov_mean_bottom], axis=0)
    nfi_avg_fov_mean = np.mean([nfi_fov_mean_top, nfi_fov_mean_bottom], axis=0)

    # Calculate the average of the uncorrected top and bottom FOV Avgs
    wfi_avg_fov_mean_uncorrected = np.mean([wfi_fov_mean_top_uncorrected, wfi_fov_mean_bottom_uncorrected], axis=0)
    nfi_avg_fov_mean_uncorrected = np.mean([nfi_fov_mean_top_uncorrected, nfi_fov_mean_bottom_uncorrected], axis=0)

    


    return (wfi_avg_fov_mean, nfi_avg_fov_mean, wfi_avg_fov_mean_uncorrected, nfi_avg_fov_mean_uncorrected, wfi_time, nfi_time, wfi_roll_angles, wfi_beta_angles, nfi_roll_angles, nfi_beta_angles, wfi_temp_proxy, nfi_temp_proxy)

def retrieve_mcp_radiation(filepath, imager, mask_fov_top, mask_fov_bottom, top_col_biases, bottom_col_biases, half_npix):
    with xr.open_dataset(filepath, engine='netcdf4') as data:
        l1a_obj = L1A.L1A(data)
    
    # load data from dataset
    images = l1a_obj.images.copy()
    n_frames = l1a_obj.n_frames
    t_int = l1a_obj.t_int
    time = l1a_obj.time

    # Save uncorrected images 
    mcp_rad_top_uncorrected = mask_average(images, mask_fov_top, t_int)[0]
    mcp_rad_bottom_uncorrected = mask_average(images, mask_fov_bottom, t_int)[0]

    # Subtract voltage biases
    top_correction = n_frames[:, np.newaxis, np.newaxis] * top_col_biases[np.newaxis, np.newaxis, :]
    bottom_correction = n_frames[:, np.newaxis, np.newaxis] * bottom_col_biases[np.newaxis, np.newaxis, :]

    images[:, :half_npix, :] -= top_correction
    images[:, half_npix:, :] -= bottom_correction

    # Calculate mean FOV radiation
    mcp_rad_top = mask_average(images, mask_fov_top, t_int)[0]
    mcp_rad_bottom = mask_average(images, mask_fov_bottom, t_int)[0]

    # calculate roll angles
    roll_angles = np.array([scraft.moc_roll for scraft in l1a_obj.scrafts]).flatten()

    # calculate beta angle, getting RA and Dec by projecting the boresight to the Star frame

    beta_angle = np.array([
        get_beta_angle(scraft, 
        *scraft.boresight_to_sky(imager, view_geometry.Star_frame))
        for scraft in l1a_obj.scrafts
    ]).flatten()

    beta_angle = 180 - beta_angle # convert to angle from the Sun instea of angle to the sun

    # Save temperature proxy
    top_inds = slice(0, half_npix)
    bottom_inds = slice(half_npix, None)
    top_temp_proxy = np.mean(l1a_obj.bias[:, top_inds, :], axis=(1,2)) 
    bottom_temp_proxy = np.mean(l1a_obj.bias[:, bottom_inds, :], axis=(1,2))
    mean_temp_proxy = np.mean(np.vstack([top_temp_proxy, bottom_temp_proxy]), axis=0)
    temp_proxy = np.array([[mean_temp_proxy], [top_temp_proxy], [bottom_temp_proxy]])
    return mcp_rad_top, mcp_rad_bottom, mcp_rad_top_uncorrected, mcp_rad_bottom_uncorrected, time, n_frames, t_int, roll_angles, beta_angle, temp_proxy

def process_mcp_data(filepaths, imager, mask_fov_top, mask_fov_bottom,
                     top_col_biases, bottom_col_biases, half_npix):

    total_files = len(filepaths)
    files_processed = 0
    results = []

    # Drop lock/counter from worker — track progress in main thread instead
    worker_func = partial(
        retrieve_mcp_radiation,
        imager=imager,
        mask_fov_top=mask_fov_top,
        mask_fov_bottom=mask_fov_bottom,
        top_col_biases=top_col_biases,
        bottom_col_biases=bottom_col_biases,
        half_npix=half_npix
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
    temp_proxies = np.concatenate(temp_proxies, axis=2) # shape (n_observations, 3) for mean, top, bottom proxies
    temp_proxies = np.squeeze(temp_proxies, axis=1)
    temp_proxies = temp_proxies.T # shape (n_observations, 3) for mean, top, bottom proxies
    

    # sort data by time
    # Sort all arrays by time
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
    times = times[sort_idx]  # sort last to maintain correct order with other variables

# Create xarray Dataset with all data
    ds_output = xr.Dataset({
        'fov_mean_top': (['observation'], fov_means_top),
        'fov_mean_bottom': (['observation'], fov_means_bottom),
        'fov_mean_top_uncorrected': (['observation'], fov_means_top_uncorrected),
        'fov_mean_bottom_uncorrected': (['observation'], fov_means_bottom_uncorrected),
        'time': (['observation'], times),
        'n_frames': (['observation'], n_frames),
        't_int': (['observation'], t_ints),
        'roll_angles': (['observation'], roll_angles),
        'beta_angles': (['observation'], beta_angles),
        'temp_proxies': (['observation', 'sensor_region'], temp_proxies)
    },  coords={
        'observation': times,  # Use datetime values as the coordinate
        'sensor_region': ['mean', 'top_half', 'bottom_half']  # Coordinate for temperature proxies
    })

    # Add Dataset variable attributes
    ds_output['fov_mean_top'].attrs = {'units': 'DN s-1 pixel-1', 'long_name': 'MCP Radiation Top Half'}
    ds_output['fov_mean_bottom'].attrs = {'units': 'DN s-1 pixel-1', 'long_name': 'MCP Radiation Bottom Half'}
    ds_output['fov_mean_top_uncorrected'].attrs = {'units': 'DN s-1 pixel-1', 'long_name': 'Uncorrected MCP Radiation Top Half'}
    ds_output['fov_mean_bottom_uncorrected'].attrs = {'units': 'DN s-1 pixel-1', 'long_name': 'Uncorrected MCP Radiation Bottom Half'}
    ds_output['n_frames'].attrs = {'long_name': 'Number of Frames', 'units': 'n'}
    ds_output['t_int'].attrs = {'long_name': 'Integration Time', 'units': 's'}
    ds_output['roll_angles'].attrs = {'long_name': 'Spacecraft Roll Angle', 'units': 'degrees'}
    ds_output['beta_angles'].attrs = {'long_name': 'Beta Angle', 'units': 'degrees'}
    ds_output['temp_proxies'].attrs = {'long_name': 'Temperature Proxies', 'description': 'Mean, top half, and bottom half temperature proxies calculated from bias values'}

    # Save the Dataset
    output_filepath = "products/" + imager + "_FOV_AVG.nc"
    ds_output.to_netcdf(output_filepath)
    print(f"MCP radiation data saved to {output_filepath}")
    ds_output.close()
    return

def generate_masks(imager):
    npix = constants.NPIX[imager]
    fov_radius = constants.MASK_L1A_FOV_R[imager]

    full_fov_mask = circular_mask(npix, fov_radius)
    row_indices = np.arange(npix)[:, np.newaxis]  # shape (npix, 1) — broadcasts over rows
    mask_fov_top    = np.logical_and(full_fov_mask, row_indices < npix // 2)
    mask_fov_bottom = np.logical_and(full_fov_mask, row_indices >= npix // 2)
    #mask_fov_top = np.logical_and(full_fov_mask, np.arange(npix) < npix // 2)
    #mask_fov_bottom = np.logical_and(full_fov_mask, np.arange(npix) >= npix // 2)

    return mask_fov_top, mask_fov_bottom


def main():
    for imager in ["WFI", "NFI"]:
        start_time = time.perf_counter()

        data_files_directory = "/data/L1A/"

        # Load bias files
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

        # FILE PATH OVERRIDE FOR TESTING
        if False:
            if imager == "WFI":
                filepaths = ["/data/L1A/CARRUTHERS_GCI-WFI_L1A-DRK_20251004_v1.0.nc", 
                            "/data/L1A/CARRUTHERS_GCI-WFI_L1A-DRK_20251005_v1.0.nc",]
            elif imager == "NFI":
                filepaths = ["/data/L1A/CARRUTHERS_GCI-NFI_L1A-DRK_20251013_v1.0.nc"]

        print(f"Found {len(filepaths)} {imager}_L1A-DRK files in {data_files_directory}")

        # Generate FOV masks
        mask_fov_top, mask_fov_bottom = generate_masks(imager)

        # Process MCP radiation data
        half_npix = int((constants.NPIX[imager])/2)
        process_mcp_data(filepaths, imager, mask_fov_top, mask_fov_bottom,
                        top_col_biases, bottom_col_biases, half_npix)

        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"{imager} radiation data processing complete ({execution_time:.2f} seconds).")

    return

if __name__ == '__main__':
    main()
# %%
