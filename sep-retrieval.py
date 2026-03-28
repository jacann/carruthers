import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import statsmodels.api as sm

import glide.calibration.radiation as r
from glide.common_components.utils import circular_mask
from glide.common_components.constants import MASK_L1A_FOV_R
from glide.common_components.constants import MASK_CNR_R

def get_filenames(directory):
    paths = []
    for filenames in os.listdir(directory):
        paths.append(os.path.join(directory, filenames))
    return paths

def get_radiation_data(filepath, mask_fov, mask_cnr):
    ds = xr.open_dataset(filepath)
    ims = ds["images"]
    t_int = ds["t_int"].values
    # Ensure retrieve_radiation is thread/process safe or called within ProcessPool
    aps_rad, mcp_rad, scaling_factor, mcp_gain = r.retrieve_radiation(ims, mask_fov, mask_cnr, t_int)
    ds.close()
    return aps_rad, mcp_rad, scaling_factor, mcp_gain

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
    plt.ylabel('Scaling Factor')
    plt.title(f'MCP Radiation vs Scaling Factor\nSensor Region: {mask_variant}\nR²={model.rsquared:.4f}')
    plt.legend()
    plt.savefig(f'products/scaling_factor_vs_mcp_radiation-{mask_variant}.png', dpi=1000)
    plt.show()
    
    # Close the dataset
    radiation_dataset.close()

def process_radiation_data(data_files_directory, output_file_path, mask_fov, mask_cnr, mask_variant):
    filepaths = get_filenames(data_files_directory)

    # Use ProcessPoolExecutor for parallel processing and using 'partial' to fix mask arguments for the mapping function
    worker_func = partial(get_radiation_data, mask_fov=mask_fov, mask_cnr=mask_cnr)

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(worker_func, filepaths))

    # Unpack results
    aps_rads = [res[0] for res in results]
    mcp_rads = [res[1] for res in results]
    scaling_factors = [res[2] for res in results]
    mcp_gains = [res[3] for res in results]

    # Convert lists to arrays after collecting all data
    aps_rads = np.concatenate(aps_rads)
    mcp_rads = np.concatenate(mcp_rads)
    scaling_factors = np.concatenate(scaling_factors)
    mcp_gains = np.concatenate(mcp_gains)

    # Store aps_rad, mcp_rad, scaling_factor, mcp_gain, and source file data in xarray DataArray
    da_aps = xr.DataArray(aps_rads, dims='observation')
    da_mcp = xr.DataArray(mcp_rads, dims='observation')
    da_scaling = xr.DataArray(scaling_factors, dims='observation')
    da_gain = xr.DataArray(mcp_gains, dims=('observation', 'rows', 'cols'))

    ds_output = xr.Dataset({'aps_rad': da_aps, 'mcp_rad': da_mcp, 'scaling_factor': da_scaling, 'mcp_gain': da_gain})
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

def main(use_saved_data = True):
    data_files_directory = 'C:/Users/Jacob/repos/carruthers/data/WFI_1A-DRK'

    # Generate FOV & CNR masks
    npix = 512
    fov_radius = MASK_L1A_FOV_R['WFI']
    cnr_radius = MASK_CNR_R['WFI']

    mask_variants = ['full', 'top', 'bottom']
    for mask_variant in mask_variants:
        print(f"Processing data for sensor_half: {mask_variant}")

        data_file_path = f'products/radiation_data-{mask_variant}.nc'

        mask_fov = circular_mask(npix, fov_radius)
        mask_cnr = circular_mask(npix, cnr_radius, do_inverse=True)
        if mask_variant == 'full':
            pass
        elif mask_variant == 'top':
            mask_fov[npix//2:, :] = 0
            mask_cnr[npix//2:, :] = 0
        elif mask_variant == 'bottom':
            mask_fov[:npix//2, :] = 0
            mask_cnr[:npix//2, :] = 0

        if not(use_saved_data):
            process_radiation_data(data_files_directory, data_file_path, mask_fov, mask_cnr, mask_variant) # Process radiation data
        plot_scaling_factor_vs_mcp_radiation(data_file_path, mask_variant) # Plot scaling factor vs. MCP radiation and print regression summary
        plot_scaling_factor_vs_aps_radiation(data_file_path, mask_variant)
        plot_fov_and_cnr_masks(np.loadtxt('background.txt', dtype=float), mask_fov, mask_cnr, mask_variant)

if __name__ == '__main__':
    main()