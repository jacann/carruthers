import numpy as np
import xarray as xr
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from glide.common_components.utils import mask_average
from glide.common_components.utils import circular_mask
from glide.common_components import constants

def get_filenames(directory):
    paths = []
    for filenames in os.listdir(directory):
        paths.append(os.path.join(directory, filenames))
    return paths

def retrieve_mcp_radiation(ds, mask_fov, top_col_biases, bottom_col_biases):
    # Load data from given dataset
    images = ds["images"].values.copy()
    n_frames = ds["n_frames"].values
    t_int = ds["t_int"].values
    time = ds["time"]
    file_id = ds["Logical_file_id"]
    ds.close()

    # Notify file processing
    print(f"Processing file: {file_id}")

    # Subtract voltage biases
    images[:, :256, :] -= n_frames[:, np.newaxis, np.newaxis] * top_col_biases[np.newaxis, np.newaxis, :]
    images[:, 256:, :] -= n_frames[:, np.newaxis, np.newaxis] * bottom_col_biases[np.newaxis, np.newaxis, :]

    # Calculate mean FOV radiation and images with non-fov area set to NaN
    mcp_rad, mcp_fov = mask_average(images, mask_fov, t_int)

    return mcp_rad, mcp_fov, time, n_frames, t_int, file_id

def process_mcp_data(filepaths, mask_fov_top, mask_fov_bottom):

    # Load bias files
    top_col_biases = np.load('column_bias_top.npy')
    bottom_col_biases = np.load('column_bias_bottom.npy')

    # Process images for sensor top half
    worker_func = partial(retrieve_mcp_radiation, mask_fov=mask_fov_top,
                          top_col_biases=top_col_biases, bottom_col_biases=bottom_col_biases)
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(worker_func, filepaths))

    # Unpack sensor results (top half)
    top_mcp_rads = [res[0] for res in results]
    top_mcp_fovs = [res[1] for res in results]

    # Unpack general results
    times = [res[2] for res in results]
    n_frames = [res[3] for res in results]
    t_ints = [res[4] for res in results]
    file_ids = [res[5] for res in results]

    # Process images for sensor top half
    worker_func = partial(retrieve_mcp_radiation, mask_fov=mask_fov_bottom,
                          top_col_biases=top_col_biases, bottom_col_biases=bottom_col_biases)
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(worker_func, filepaths))

    # Unpack sensor results (bottom half)
    bottom_mcp_rads = [res[0] for res in results]
    bottom_mcp_fovs = [res[1] for res in results]

    # Convert lists to arrays after collecting all data
    top_mcp_rads = np.concatenate(top_mcp_rads)
    top_mcp_fovs = np.concatenate(top_mcp_fovs)
    bottom_mcp_rads = np.concatenate(bottom_mcp_rads)
    bottom_mcp_fovs = np.concatenate(bottom_mcp_fovs)
    times = np.concatenate(times)
    n_frames = np.concatenate(n_frames)
    t_ints = np.concatenate(t_ints)
    file_ids = np.concatenate(file_ids)

    # Create xarray Dataset with all data
    ds_output = xr.Dataset({
        'mcp_rad_top': (['observation'], top_mcp_rads),
        'mcp_fov_top': (['observation'], top_mcp_fovs),
        'mcp_rad_bottom': (['observation'], bottom_mcp_rads),
        'mcp_fov_bottom': (['observation'], bottom_mcp_fovs),
        'time': (['observation'], times),
        'n_frames': (['observation'], n_frames),
        't_int': (['observation'], t_ints),
        'file_id': (['observation'], file_ids)
    })

    # Add Dataset variable attributes
    ds_output['mcp_rad_top'].attrs = {'units': 'DN s-1 pixel-1', 'long_name': 'MCP Radiation Top Half'}
    ds_output['mcp_fov_top'].attrs = {'units': 'DN s-1 pixel-1', 'long_name': 'MCP FOV Top Half'}
    ds_output['mcp_rad_bottom'].attrs = {'units': 'DN s-1 pixel-1', 'long_name': 'MCP Radiation Bottom Half'}
    ds_output['mcp_fov_bottom'].attrs = {'units': 'DN s-1 pixel-1', 'long_name': 'MCP FOV Bottom Half'}
    ds_output['time'].attrs = {'long_name': 'Capture Time', 'units': 'datetime64[ns]'}
    ds_output['n_frames'].attrs = {'long_name': 'Number of Frames', 'units': 'n'}
    ds_output['t_int'].attrs = {'long_name': 'Integration Time', 'units': 's'}

    # Save the Dataset
    ds_output.to_netcdf('products/mcp_rad_data.nc')
    print("MCP radiation data saved to products/mcp_rad_data.nc")
    ds_output.close()
    return

def generate_masks(imager):
    npix = constants.NPIX[imager]
    fov_radius = constants.MASK_L1A_FOV_R[imager]

    full_fov_mask = circular_mask(npix, fov_radius)
    mask_fov_top = np.logical_and(full_fov_mask, np.arange(npix) < npix // 2)
    mask_fov_bottom = np.logical_and(full_fov_mask, np.arange(npix) >= npix // 2)

    return mask_fov_top, mask_fov_bottom


def main(imager="WFI"):
    data_files_directory = 'C:/Users/Jacob/repos/carruthers/data/WFI_1A-DRK'

    filepaths = get_filenames(data_files_directory)
    print(f"Found {len(filepaths)} files in {data_files_directory}")

    # Generate FOV masks
    mask_fov_top, mask_fov_bottom = generate_masks(imager)

    # Process MCP radiation data
    process_mcp_data(filepaths, mask_fov_top, mask_fov_bottom)

    print("MCP radiation data processing complete.")

    return

if __name__ == '__main__':
    main()


