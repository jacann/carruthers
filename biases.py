import xarray as xr
import numpy as np


def plot_biases():
        # Plot column biases for each point in time and plot
    filenames = ["CARRUTHERS_GCI-WFI_L1A-DRK_20251004_v1.0.nc",
                "CARRUTHERS_GCI-WFI_L1A-DRK_20251005_v1.0.nc",
                "CARRUTHERS_GCI-WFI_L1A-DRK_20251006_v1.0.nc",
                "CARRUTHERS_GCI-WFI_L1A-DRK_20251007_v1.0.nc",
                "CARRUTHERS_GCI-WFI_L1A-DRK_20251008_v1.0.nc",
                "CARRUTHERS_GCI-WFI_L1A-DRK_20251009_v1.0.nc",
                "CARRUTHERS_GCI-WFI_L1A-DRK_20251010_v1.0.nc",
                "CARRUTHERS_GCI-WFI_L1A-DRK_20251011_v1.0.nc",
                "CARRUTHERS_GCI-WFI_L1A-DRK_20251012_v1.0.nc",
                "CARRUTHERS_GCI-WFI_L1A-DRK_20251013_v1.0.nc",
                "CARRUTHERS_GCI-WFI_L1A-DRK_20251014_v1.0.nc",
                "CARRUTHERS_GCI-WFI_L1A-DRK_20251015_v1.0.nc",]

    all_times = []
    all_medians = []

    for filename in filenames:
        ds = xr.open_dataset(f'/home/jacob/products/L1A/{filename}')
        
        images = ds['images'].values
        n_frames = ds['n_frames'].values
        times = ds['time'].values
        ds.close()

        # Apply mask and normalize
        n_min = 0000
        mask = n_frames > n_min
        if not np.any(mask): # Skip file if no frames match criteria
            continue
            
        norm_images = images[mask] / n_frames[mask, np.newaxis, np.newaxis]
        filtered_times = times[mask]

        # Calculate median for this file
        mcp_rad_top = np.median(norm_images[:, :512, :], axis=1)
        
        all_times.append(filtered_times)
        all_medians.append(mcp_rad_top)

    # Concatenate all data into single arrays
    final_times = np.concatenate(all_times)
    final_medians = np.concatenate(all_medians, axis=0)

    # Sort by time to ensure the line doesn't "zig-zag" if files are out of order
    sort_idx = np.argsort(final_times)
    final_times = final_times[sort_idx]
    final_medians = final_medians[sort_idx]

    # Single plot call for a continuous line
    plt.figure(figsize=(10, 14))
    plt.plot(final_times, final_medians, label='MCP Rad Top')
    plt.title(f'Median Bias Over Time (n_frames > {n_min})')
    plt.xlabel('Time')
    plt.ylabel('Median Column Bias')
    plt.show()

    return

def main():
    # Plot column biases for each point in time and plot
    filenames = ["CARRUTHERS_GCI-WFI_L1A-DRK_20251004_v1.0.nc",
                "CARRUTHERS_GCI-WFI_L1A-DRK_20251005_v1.0.nc",
                "CARRUTHERS_GCI-WFI_L1A-DRK_20251006_v1.0.nc",
                "CARRUTHERS_GCI-WFI_L1A-DRK_20251007_v1.0.nc",
                "CARRUTHERS_GCI-WFI_L1A-DRK_20251008_v1.0.nc",
                "CARRUTHERS_GCI-WFI_L1A-DRK_20251009_v1.0.nc",
                "CARRUTHERS_GCI-WFI_L1A-DRK_20251010_v1.0.nc",
                "CARRUTHERS_GCI-WFI_L1A-DRK_20251011_v1.0.nc",
                "CARRUTHERS_GCI-WFI_L1A-DRK_20251012_v1.0.nc",
                "CARRUTHERS_GCI-WFI_L1A-DRK_20251013_v1.0.nc",
                "CARRUTHERS_GCI-WFI_L1A-DRK_20251014_v1.0.nc",
                "CARRUTHERS_GCI-WFI_L1A-DRK_20251015_v1.0.nc",]


    all_times = []
    all_medians_top = []
    all_medians_bottom = []

    for filename in filenames:
        ds = xr.open_dataset(f'/home/jacob/products/L1A/{filename}')
        
        images = ds['images'].values
        n_frames = ds['n_frames'].values
        times = ds['time'].values
        ds.close()

        # Apply mask and normalize
        n_min = 0000
        mask = n_frames > n_min
        if not np.any(mask): # Skip file if no frames match criteria
            continue
            
        norm_images = images[mask] / n_frames[mask, np.newaxis, np.newaxis]
        filtered_times = times[mask]

        # Calculate median for this file
        mcp_rad_top = np.median(norm_images[:, :256, :], axis=1)
        mcp_rad_bottom = np.median(norm_images[:, 256:, :], axis=1)
        
        all_times.append(filtered_times)
        all_medians_top.append(mcp_rad_top)
        all_medians_bottom.append(mcp_rad_bottom)


    # Concatenate all data into single arrays
    final_times = np.concatenate(all_times)
    final_medians_top = np.concatenate(all_medians_top, axis=0)
    final_medians_bottom = np.concatenate(all_medians_bottom, axis=0)

    # Sort by time to ensure the line doesn't "zig-zag" if files are out of order
    sort_idx = np.argsort(final_times)
    final_times = final_times[sort_idx]
    final_medians_top = final_medians_top[sort_idx]
    final_medians_bottom = final_medians_bottom[sort_idx]


    print("TOP medians shape:", final_medians_top.shape)
    print("BOTTOM medians shape:", final_medians_bottom.shape)

    # Find 512 medians for each column across all images and save as numpy arrays
    top_med_medians = np.zeros((final_medians_top.shape[1]))
    bottom_med_medians = np.zeros((final_medians_bottom.shape[1]))
    #print(final_medians_top.shape)

    top_med_medians = np.median(final_medians_top, axis=0)
    bottom_med_medians = np.median(final_medians_bottom, axis=0)

    np.save('products/column_bias_top_v2.npy', top_med_medians)
    np.save('products/column_bias_bottom_v2.npy', bottom_med_medians)

if __name__ == '__main__':
    main()
