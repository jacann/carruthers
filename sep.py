# %%
import numpy as np
import xarray as xr
import time
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import glob

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

def retrieve_mcp_radiation(filepath, imager, mask_fov_top, mask_fov_bottom, top_col_biases, bottom_col_biases, half_npix):
    with xr.open_dataset(filepath, engine='netcdf4') as data:
        l1a_obj = L1A.L1A(data)
    
    # load data from dataset
    images = l1a_obj.images.copy()
    n_frames = l1a_obj.n_frames
    t_int = l1a_obj.t_int
    time = l1a_obj.time

    # Notify file processing
    print(f"Processing {filepath}")
    # Subtract voltage biases
    top_correction = n_frames[:, np.newaxis, np.newaxis] * top_col_biases[np.newaxis, np.newaxis, :]
    bottom_correction = n_frames[:, np.newaxis, np.newaxis] * bottom_col_biases[np.newaxis, np.newaxis, :]

    images[:, :half_npix, :] -= top_correction
    images[:, half_npix:, :] -= bottom_correction

    # Calculate mean FOV radiation and images with non-fov area set to NaN
    mcp_rad_top, mcp_fov_top = mask_average(images, mask_fov_top, t_int)
    mcp_rad_bottom, mcp_fov_bottom = mask_average(images, mask_fov_bottom, t_int)

    # calculate roll angles
    roll_angles = np.array([scraft.moc_roll for scraft in l1a_obj.scrafts]).flatten()

    # calculate beta angle, getting RA and Dec by projecting the boresight to the Star frame
    ra_inputs, dec_inputs = zip(*(scraft.boresight_to_sky(imager, view_geometry.Star_frame) for scraft in l1a_obj.scrafts))
    beta_angle = np.array([get_beta_angle(scraft, ra, dec) for scraft, ra, dec, in zip(l1a_obj.scrafts, ra_inputs, dec_inputs)]).flatten()


    return mcp_rad_top, mcp_fov_top, mcp_rad_bottom, mcp_fov_bottom, time, n_frames, t_int, roll_angles, beta_angle



def retrieve_mcp_radiation_OLD(filepath, mask_fov, top_col_biases, bottom_col_biases, half_npix):
    ds = xr.open_dataset(filepath)

    # Load data from given dataset
    images = ds["images"].values.copy()
    n_frames = ds["n_frames"].values
    t_int = ds["t_int"].values
    time = ds["time"]
    file_id = ds.attrs["Logical_file_id"]
    ds.close()

    # Notify file processing
    print(f"Processing file: {file_id}")
    
    # Subtract voltage biases
    top_correction = n_frames[:, np.newaxis, np.newaxis] * top_col_biases[np.newaxis, np.newaxis, :]
    bottom_correction = n_frames[:, np.newaxis, np.newaxis] * bottom_col_biases[np.newaxis, np.newaxis, :]
        
    images[:, :half_npix, :] -= top_correction
    images[:, half_npix:, :] -= bottom_correction

    # Calculate mean FOV radiation and images with non-fov area set to NaN
    mcp_rad, mcp_fov = mask_average(images, mask_fov, t_int)

    return mcp_rad, mcp_fov, time, n_frames, t_int, [file_id] * len(images)

def process_mcp_data(filepaths, imager, mask_fov_top, mask_fov_bottom, top_col_biases, bottom_col_biases, half_npix):

    # Process images for sensor top and bottom halves
    worker_func = partial(retrieve_mcp_radiation, imager=imager, mask_fov_top=mask_fov_top,
                          mask_fov_bottom=mask_fov_bottom, top_col_biases=top_col_biases,
                          bottom_col_biases=bottom_col_biases, half_npix=half_npix)
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(worker_func, filepaths))

    # Unpack results
    top_mcp_rads = [res[0] for res in results]
    top_mcp_fovs = [res[1] for res in results]
    bottom_mcp_rads = [res[2] for res in results]
    bottom_mcp_fovs = [res[3] for res in results]
    times = [res[4] for res in results]
    n_frames = [res[5] for res in results]
    t_ints = [res[6] for res in results]
    roll_angles = [res[7] for res in results]
    beta_angles = [res[8] for res in results]


    # Convert lists to arrays after collecting all data
    top_mcp_rads = np.concatenate(top_mcp_rads)
    top_mcp_fovs = np.concatenate(top_mcp_fovs)
    bottom_mcp_rads = np.concatenate(bottom_mcp_rads)
    bottom_mcp_fovs = np.concatenate(bottom_mcp_fovs)
    times = np.concatenate(times)
    n_frames = np.concatenate(n_frames)
    t_ints = np.concatenate(t_ints)
    roll_angles = np.concatenate(roll_angles)
    beta_angles = np.concatenate(beta_angles)

    # Create xarray Dataset with all data
    ds_output = xr.Dataset({
        'mcp_rad_top': (['observation'], top_mcp_rads),
        'mcp_fov_top': (['observation', 'rows', 'cols'], top_mcp_fovs),
        'mcp_rad_bottom': (['observation'], bottom_mcp_rads),
        'mcp_fov_bottom': (['observation', 'rows', 'cols'], bottom_mcp_fovs),
        'time': (['observation'], times),
        'n_frames': (['observation'], n_frames),
        't_int': (['observation'], t_ints),
        'roll_angles': (['observation'], roll_angles),
        'beta_angles': (['observation'], beta_angles)
    },  coords={
        'observation': times,  # Use datetime values as the coordinate
        'rows': np.arange(top_mcp_fovs.shape[1]),
        'cols': np.arange(top_mcp_fovs.shape[2])
    })

    # Add Dataset variable attributes
    ds_output['mcp_rad_top'].attrs = {'units': 'DN s-1 pixel-1', 'long_name': 'MCP Radiation Top Half'}
    ds_output['mcp_fov_top'].attrs = {'units': 'DN s-1 pixel-1', 'long_name': 'MCP FOV Top Half'}
    ds_output['mcp_rad_bottom'].attrs = {'units': 'DN s-1 pixel-1', 'long_name': 'MCP Radiation Bottom Half'}
    ds_output['mcp_fov_bottom'].attrs = {'units': 'DN s-1 pixel-1', 'long_name': 'MCP FOV Bottom Half'}
    ds_output['n_frames'].attrs = {'long_name': 'Number of Frames', 'units': 'n'}
    ds_output['t_int'].attrs = {'long_name': 'Integration Time', 'units': 's'}
    ds_output['roll_angles'].attrs = {'long_name': 'Spacecraft Roll Angle', 'units': 'degrees'}
    ds_output['beta_angles'].attrs = {'long_name': 'Beta Angle', 'units': 'degrees'}

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
    mask_fov_top = np.logical_and(full_fov_mask, np.arange(npix) < npix // 2)
    mask_fov_bottom = np.logical_and(full_fov_mask, np.arange(npix) >= npix // 2)

    return mask_fov_top, mask_fov_bottom


def main(imager="WFI"):
    start_time = time.perf_counter()

    data_files_directory = "/home/jacob/products/L1A/"

    # Load bias files
    top_col_biases = np.load('products/column_bias_top.npy')
    bottom_col_biases = np.load('products/column_bias_bottom.npy')

    filepaths = get_filenames(data_files_directory, imager)

    # FILE PATH OVERRIDE FOR TESTING
    filepaths = ["/home/jacob/products/L1A/CARRUTHERS_GCI-WFI_L1A-DRK_20251004_v1.0.nc"]

    print(f"Found {len(filepaths)} files in {data_files_directory}")

    # Generate FOV masks
    mask_fov_top, mask_fov_bottom = generate_masks(imager)

    # Process MCP radiation data
    half_npix = int((constants.NPIX[imager])/2)
    process_mcp_data(filepaths, imager, mask_fov_top, mask_fov_bottom,
                     top_col_biases, bottom_col_biases, half_npix)

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"MCP radiation data processing complete ({execution_time:.2f} seconds).")

    return

if __name__ == '__main__':
    main()
# %%
