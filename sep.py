import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import statsmodels.api as sm

import glide.calibration.radiation as r
from glide.common_components.utils import circular_mask
import glide.common_components.constants as constants


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def get_filenames(directory):
    paths = []
    for filenames in os.listdir(directory):
        paths.append(os.path.join(directory, filenames))
    return paths


def plot_scaling_factor_vs_mcp_radiation(output_file_path, mask_variant):
    # Open and load the radiation data
    radiation_dataset = xr.open_dataset(output_file_path)
    mcp_rads = radiation_dataset["mcp_rad"].values
    scaling_factors = radiation_dataset["scaling_factor"].values

    # Linear regression analysis
    MCP_Radiation = sm.add_constant(mcp_rads)  # Add intercept
    model = sm.OLS(scaling_factors, MCP_Radiation).fit()
    open(f'products/regression-summary-factor-vs-aps{mask_variant}.txt', 'w').write(
        model.summary().as_text()
    )

    # Plot data with a regression line
    plt.scatter(mcp_rads, scaling_factors, s=1, alpha=1, label='Data')
    plt.plot(mcp_rads, model.predict(MCP_Radiation), 'r-', linewidth=2, alpha=0.5, label=f'Fit: y={model.params[1]:.4e}x+{model.params[0]:.4e}')
    plt.xlabel('MCP Radiation (DN second$^{-1}$ pixel$^{-1}$)')
    plt.ylabel('Scaling Factor (MCP Rad/APS Rad)')
    plt.title(f'Scaling Factor vs. MCP Radiation\nSensor Region: {mask_variant}\nR²={model.rsquared:.4f}')
    plt.legend()
    plt.savefig(f'products/scaling_factor_vs_mcp_radiation-{mask_variant}.png', dpi=1000)
    plt.show()
    
    # Close the dataset
    radiation_dataset.close()

def plot_radiation_vs_time(radiation_dataset, mask_variant, start_datetime_str, end_datetime_str, n_frames_min):
    start_index = 0
    end_index = len(radiation_dataset["observation"])

    start_datetime = np.datetime64(start_datetime_str)
    end_datetime = np.datetime64(end_datetime_str)

    mcp_rads = radiation_dataset["mcp_rad"].values
    aps_rads = radiation_dataset["aps_rad"].values
    datetimes = radiation_dataset["observation"].values
    n_frames = radiation_dataset["n_frames"].values
    radiation_dataset.close()

    datetimes = datetimes.astype('datetime64[ns]')
    print(datetimes.shape)
    # find start index
    for i, dt in enumerate(datetimes):
        #print(i, type(dt), type(start_datetime))
        if dt >= start_datetime:
            start_index = i
            break

    mcp_rads = mcp_rads[start_index:]
    aps_rads = aps_rads[start_index:]
    datetimes = datetimes[start_index:]
    n_frames = n_frames[start_index:]

    # find end index
    for i, dt in enumerate(datetimes):
        if dt > end_datetime:
            end_index = i
            break

    mcp_rads = mcp_rads[:end_index]
    aps_rads = aps_rads[:end_index]
    datetimes = datetimes[:end_index]
    n_frames = n_frames[:end_index]

    for i, t_int in enumerate(n_frames):
        if t_int < n_frames_min:
            mcp_rads[i] = -1
            aps_rads[i] = -1
            datetimes[i] = np.datetime64('NaT')

    t_int_mask = n_frames >= n_frames_min

    # Apply mask
    aps_rads_filtered = aps_rads[t_int_mask]
    mcp_rads_filtered = mcp_rads[t_int_mask]
    datetimes_filtered = datetimes[t_int_mask]

    # Plot data
    plt.scatter(datetimes_filtered, mcp_rads_filtered, s=1, alpha=1, label='MCP Radiation')
    plt.scatter(datetimes_filtered, aps_rads_filtered, s=1, alpha=1, label='APS Radiation')

    plt.xlabel('Time')
    plt.ylabel(f'Radiation (DN second$^{-1}$ pixel$^{-1}$)')
    plt.title(f'MCP & APS Radiation vs. Time\nSensor Region: {mask_variant}')
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.savefig(f'products/radiation_vs_time-{mask_variant}.png', dpi=1000)
    plt.show()

def get_radiation_data(filepath, mask_fov, mask_cnr, top_col_biases, bottom_col_biases):
    ds = xr.open_dataset(filepath)
    images = ds["images"].values.copy()
    n_frames = ds["n_frames"].values
    t_int = ds["t_int"].values
    time = ds["time"]

    print(f"Processing file: {os.path.basename(filepath)}")

    # Remove voltage biases
    # Vectorized subtraction:
    # images shape is (n_obs, 512, 512)
    # n_frames shape is (n_obs,)
    # top_col_biases shape is (512,)
    # bottom_col_biases shape is (512,)

    # top half: images[:, :256, :]
    # bottom half: images[:, 256:, :]

    # We need (n_obs, 256, 512) - (n_obs, 1, 1) * (1, 1, 512)
    images[:, :256, :] -= n_frames[:, np.newaxis, np.newaxis] * top_col_biases[np.newaxis, np.newaxis, :]
    images[:, 256:, :] -= n_frames[:, np.newaxis, np.newaxis] * bottom_col_biases[np.newaxis, np.newaxis, :]

    # Ensure retrieve_radiation is thread/process safe or called within ProcessPool
    aps_rad, mcp_rad, scaling_factor, mcp_gain = r.retrieve_radiation(images, mask_fov, mask_cnr, t_int)
    ds.close()
    return aps_rad, mcp_rad, scaling_factor, mcp_gain, time, ds["n_frames"]

def process_radiation_data(data_files_directory, output_file_path, mask_fov, mask_cnr, mask_variant):
    filepaths = get_filenames(data_files_directory)

    # Load bias files
    top_col_biases = np.load('column_bias_top.npy')
    bottom_col_biases = np.load('column_bias_bottom.npy')

    # Use ProcessPoolExecutor for parallel processing and using 'partial' to fix mask arguments for the mapping function
    worker_func = partial(get_radiation_data, mask_fov=mask_fov, mask_cnr=mask_cnr,
                          top_col_biases=top_col_biases, bottom_col_biases=bottom_col_biases)

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(worker_func, filepaths))

    # Unpack results
    aps_rads = [res[0] for res in results]
    mcp_rads = [res[1] for res in results]
    scaling_factors = [res[2] for res in results]
    mcp_gains = [res[3] for res in results]
    times = [res[4] for res in results]
    n_frames = [res[5] for res in results]

    # Convert lists to arrays after collecting all data
    aps_rads = np.concatenate(aps_rads)
    mcp_rads = np.concatenate(mcp_rads)
    scaling_factors = np.concatenate(scaling_factors)
    mcp_gains = np.concatenate(mcp_gains)
    times = np.concatenate(times)
    n_frames = np.concatenate(n_frames)

    # Store aps_rad, mcp_rad, scaling_factor, mcp_gain, and source file data in xarray DataArray
    ds_output = xr.Dataset({
        'aps_rad': (['observation'], aps_rads),
        'mcp_rad': (['observation'], mcp_rads),
        'scaling_factor': (['observation'], scaling_factors),
        'mcp_gain': (['observation', 'rows', 'cols'], mcp_gains),
        'time': (['observation'], times),
        'n_frames': (['observation'], n_frames)
    }, coords={
        'observation': times,  # Use datetime values as the coordinate
        'rows': np.arange(mcp_gains.shape[1]),
        'cols': np.arange(mcp_gains.shape[2])
    })

    # Add variable attributes
    ds_output['aps_rad'].attrs = {'units': 'DN s-1 pixel-1', 'long_name': 'APS Radiation'}
    ds_output['mcp_rad'].attrs = {'units': 'DN s-1 pixel-1', 'long_name': 'MCP Radiation'}
    ds_output['scaling_factor'].attrs = {'long_name': 'APS/MCP Scaling Factor', 'units': '1'}
    ds_output['mcp_gain'].attrs = {'long_name': 'MCP Gain Map', 'units': '1'}
    ds_output['n_frames'].attrs = {'long_name': 'Number of Frames', 'units': 's'}
    #ds_output['time'].attrs = {'long_name': 'Capture Time', 'units': 'datetime64[ns]'}
    #ds_output['observation'].attrs = {'long_name': 'Capture Time', 'units': 'datetime64[ns]'}

    ds_output.attrs = {
        'mask_variant': mask_variant,
        'source_directory': data_files_directory,
        'n_observations': len(aps_rads),
        'created': (np.datetime64('now')).astype(str)
    }


    ds_output.to_netcdf(output_file_path)
    print(f"Radiation data for {len(aps_rads)} observations saved to {output_file_path}")
    plt.savefig(f'products/masks-{mask_variant}.png', dpi=1000)
    ds_output.close()

def plot_fov_and_cnr_masks(background, mask_fov, mask_cnr, mask_variant):
    # TESTING: Create colored arrays for overlay
    overlay_color1 = np.zeros((*background.shape, 4))  # RGBA
    overlay_color1[mask_fov] = [1, 0, 0, 1]  # Red (R, G, B, A) - full opacity
    overlay_color2 = np.zeros((*background.shape, 4))  # RGBA
    overlay_color2[mask_cnr] = [0, 1, 0, 1]  # Red (R, G, B, A) - full opacity

    # Display the original image
    plt.imshow(background, vmin=0, vmax=250000 / 30)
    # Overlay the mask using imshow with an alpha value
    plt.imshow(overlay_color1, alpha=0.5)
    plt.imshow(overlay_color2, alpha=0.5)
    plt.title(f'Image with FOV & Corner Mask Overlays\n Sensor Region: {mask_variant}')
    plt.savefig(f'products/fov_and_cnr_masks-{mask_variant}.png', dpi=1000)
    plt.show()

def plot_scaling_factor_vs_aps_radiation(output_file_path, mask_variant):
    # Open and load the radiation data
    radiation_dataset = xr.open_dataset(output_file_path)
    aps_rads = radiation_dataset["aps_rad"].values
    scaling_factors = radiation_dataset["scaling_factor"].values

    # Linear regression analysis
    APS_Radiation = sm.add_constant(aps_rads)  # Add intercept
    model = sm.OLS(scaling_factors, APS_Radiation).fit()
    open(f'products/regression-summary-factor-vs-aps{mask_variant}.txt', 'w').write(
        model.summary().as_text()
    )

    # Plot data with a regression line
    plt.scatter(aps_rads, scaling_factors, s=1, alpha=1, label='Data')
    plt.plot(aps_rads, model.predict(APS_Radiation), 'r-', linewidth=2, alpha=0.5,
             label=f'Fit: y={model.params[1]:.4e}x+{model.params[0]:.4e}')
    plt.xlabel('APS Radiation (DN second$^{-1}$ pixel$^{-1}$)')
    plt.ylabel('Scaling Factor')
    plt.title(f'APS Radiation vs Scaling Factor\nSensor Region: {mask_variant}\nR²={model.rsquared:.4f}')
    plt.legend()
    plt.savefig(f'products/scaling_factor_vs_aps_radiation-{mask_variant}.png', dpi=1000)
    plt.show()

    # Close the dataset
    radiation_dataset.close()

def generate_masks(imager, mask_variant):
    npix = constants.NPIX[imager]
    fov_radius = constants.MASK_L1A_FOV_R[imager]
    cnr_radius = constants.MASK_CNR_R[imager]

    mask_fov = circular_mask(npix, fov_radius)
    mask_cnr = circular_mask(npix, cnr_radius, do_inverse=True)
    if mask_variant == 'full':
        pass
    elif mask_variant == 'top':
        mask_fov[npix // 2:, :] = 0
        mask_cnr[npix // 2:, :] = 0
    elif mask_variant == 'bottom':
        mask_fov[:npix // 2, :] = 0
        mask_cnr[:npix // 2, :] = 0

    return mask_fov, mask_cnr

def main(use_saved_data = False):
    data_files_directory = 'C:/Users/Jacob/repos/carruthers/data/WFI_1A-DRK'

    # Generate FOV & CNR masks
    mask_variants = ['top', 'bottom']

    for mask_variant in mask_variants:
        data_file_path = f'products/radiation_data-{mask_variant}.nc'

        mask_fov, mask_cnr = generate_masks('WFI', mask_variant)

        print(f"Processing data for sensor_half: {mask_variant}")
        if not(use_saved_data):
            process_radiation_data(data_files_directory, data_file_path, mask_fov, mask_cnr, mask_variant) # Process radiation data

        #plot_scaling_factor_vs_mcp_radiation(data_file_path, mask_variant) # Plot scaling factor vs. MCP radiation and print regression summary
        #plot_scaling_factor_vs_aps_radiation(data_file_path, mask_variant)
        #plot_fov_and_cnr_masks(np.loadtxt('background.txt', dtype=float), mask_fov, mask_cnr, mask_variant)
        #plot_radiation_vs_time(data_file_path, mask_variant,'2025-11-06T00:00:00.000000','2025-11-11T00:00:00.000000')

if __name__ == '__main__':
    main()