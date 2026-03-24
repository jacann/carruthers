import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import glide.calibration.radiation as r
from glide.common_components.utils import circular_mask
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import statsmodels.api as sm


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

def main():
    # Generate FOV & CNR masks
    npix = 512
    fov_radius = 260
    cnr_radius = 300
    mask_fov = circular_mask(npix, fov_radius)
    mask_cnr = circular_mask(npix, cnr_radius, do_inverse=True)
    filepaths = get_filenames('data\\WFI_1A-DRK')

    # Use ProcessPoolExecutor for parallel processing
    # Using 'partial' to fix mask arguments for the mapping function
    worker_func = partial(get_radiation_data, mask_fov=mask_fov, mask_cnr=mask_cnr)

    # Note: On Windows, the 'if __name__ == "__main__":' guard is required for multiprocessing
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

    # Linear regression analysis
    X = sm.add_constant(mcp_rads)  # Add intercept
    model = sm.OLS(scaling_factors, X).fit()
    print(model.summary())

    # Plot data with regression line
    plt.scatter(mcp_rads, scaling_factors, s=1, alpha=0.5, label='Data')
    plt.plot(mcp_rads, model.predict(X), 'r-', linewidth=2, label=f'Fit: y={model.params[1]:.4e}x+{model.params[0]:.4e}')
    plt.xlabel('MCP Radiation (W/m^2)')
    plt.ylabel('Scaling Factor')
    plt.title(f'MCP Radiation vs Scaling Factor\nR²={model.rsquared:.4f}')
    plt.legend()
    plt.savefig('scaling_factor_vs_mcp_radiation.svg')
    plt.show()

    # Store aps_rad, mcp_rad, scaling_factor, mcp_gain, and source file data in xarray DataArray
    da_aps = xr.DataArray(aps_rads, dims='observation')
    da_mcp = xr.DataArray(mcp_rads, dims='observation')
    da_scaling = xr.DataArray(scaling_factors, dims='observation')
    da_gain = xr.DataArray(mcp_gains, dims=('observation', 'rows', 'cols'))

    ds_output = xr.Dataset({'aps_rad': da_aps, 'mcp_rad': da_mcp, 'scaling_factor': da_scaling, 'mcp_gain': da_gain})
    ds_output.to_netcdf('radiation_data.nc')

if __name__ == '__main__':
    main()