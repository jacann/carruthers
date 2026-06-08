
#%%
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.dates as mdates


def parse_ace_sis_file(filepath):
    """
    Parse a NOAA ACE SIS 5-minute proton flux file.
    
    Returns:
        timestamps   : numpy array of datetime objects (UTC), shape (N,)
        flux_10mev   : numpy array of >10 MeV proton flux, shape (N,)  [p/cm2-sec-ster]
        flux_30mev   : numpy array of >30 MeV proton flux, shape (N,)  [p/cm2-sec-ster]
        valid_mask   : boolean numpy array, True where status==0 (nominal), shape (N,)
    """
    timestamps, flux_10, flux_30, valid = [], [], [], []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            # Skip comment/header lines
            if not line or line.startswith("#") or line.startswith(":"):
                continue

            parts = line.split()
            if len(parts) < 10:
                continue

            yr, mo, da, hhmm = parts[0], parts[1], parts[2], parts[3]
            status_10  = int(parts[6])
            raw_10     = float(parts[7])
            status_30  = int(parts[8])
            raw_30     = float(parts[9])

            hh = int(hhmm) // 100
            mm = int(hhmm) % 100
            dt = datetime(int(yr), int(mo), int(da), hh, mm, tzinfo=timezone.utc)

            is_valid = (status_10 == 0) and (status_30 == 0)

            timestamps.append(dt)
            flux_10.append(raw_10 if is_valid else np.nan)
            flux_30.append(raw_30 if is_valid else np.nan)
            valid.append(is_valid)

    return (
        np.array(timestamps),
        np.array(flux_10, dtype=float),
        np.array(flux_30, dtype=float),
        np.array(valid,   dtype=bool),
    )

def load_ace_sis_files(filepaths):
    """
    Load and concatenate multiple ACE SIS files, sorted chronologically.

    Args:
        filepaths : list of .txt file paths

    Returns:
        timestamps : numpy array of datetime objects (UTC)
        flux_10mev : numpy array of >10 MeV proton flux (NaN where bad data)
        flux_30mev : numpy array of >30 MeV proton flux (NaN where bad data)
        valid_mask : boolean numpy array (True = nominal data)
    """
    all_ts, all_10, all_30, all_valid = [], [], [], []

    for fp in filepaths:
        ts, f10, f30, v = parse_ace_sis_file(fp)
        all_ts.append(ts)
        all_10.append(f10)
        all_30.append(f30)
        all_valid.append(v)

    timestamps  = np.concatenate(all_ts)
    flux_10mev  = np.concatenate(all_10)
    flux_30mev  = np.concatenate(all_30)
    valid_mask  = np.concatenate(all_valid)

    # Sort chronologically (files may be out of order)
    order = np.argsort(timestamps)
    return timestamps[order], flux_10mev[order], flux_30mev[order], valid_mask[order]

def get_filepaths(directory, search_string, recursive=False):
    """
    Get a list of file paths in a directory that contain a specific string.

    Args:
        directory (str or Path): The directory to search.
        search_string (str): The string to match in file names.
        recursive (bool): If True, search subdirectories as well.

    Returns:
        List[Path]: A list of matching file paths.
    """
    dir_path = Path(directory)
    if recursive:
        return list(dir_path.rglob(f"*{search_string}*"))
    else:
        return list(dir_path.glob(f"*{search_string}*"))

def load_carrdata():
    with xr.open_dataset("/home/jacob/carruthers/products/NFI_FOV_AVG.nc") as ds:
        nfi_fov_mean_top = ds["fov_mean_top"].values
        nfi_fov_mean_bot = ds["fov_mean_bottom"].values
        nfi_fov_mean_full = np.mean([nfi_fov_mean_top, nfi_fov_mean_bot], axis=0)
        time = ds["time"].values
    
    return nfi_fov_mean_full, time
# --- Example usage ---
if __name__ == "__main__":

    files = get_filepaths("/home/jacob/noncarr/ace_sis/", "ace_sis_5m", recursive=False)

    print(f"Found {len(files)} file(s) matching criteria:")
    ace_time, flux_10mev, flux_30mev, valid_mask = load_ace_sis_files(files)

    print(f"Loaded {len(ace_time)} records from {len(files)} file(s)")
    print(f"Time range : {ace_time[0]}  →  {ace_time[-1]}")
    print(f"Valid points: {valid_mask.sum()} / {len(valid_mask)}")
    print(f">10 MeV flux — mean: {np.nanmean(flux_10mev):.3f}, "
          f"max: {np.nanmax(flux_10mev):.3f} p/cm2-sec-ster")
    print(f">30 MeV flux — mean: {np.nanmean(flux_30mev):.3f}, "
          f"max: {np.nanmax(flux_30mev):.3f} p/cm2-sec-ster")
    
    print("\n\n\n")


    # load carrdata
    nfi_fov_mean_full, nfi_time = load_carrdata()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ace_time, flux_10mev, label=">10 MeV")
    ax.plot(ace_time, flux_30mev, label=">30 MeV")
    ax.set_xlabel("Time")
    ax.set_ylabel("Proton Flux (p/cm²/s/ster)")
    
    # set x axis to every day
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.title("ACE SIS Proton Flux Data")
    ax.legend()
    ax.set_ylim(0, None)

    ax1 = ax.twinx()
    ax1.scatter(nfi_time, nfi_fov_mean_full, color='red', label='Carruthers FOV Mean', s=10)
    ax1.set_ylim(0, 2.5)
    ax1.set_ylabel("Carruthers FOV Mean (DN/s-pixel)")
    ax1.legend(loc='upper center')

    start_dt = np.datetime64('2026-01-19T00:00:00')
    end_dt = np.datetime64('2026-01-23T00:00:00')
    ax.set_xlim(start_dt, end_dt)


    plt.show()
#%%
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def interp_channel(flux_col, goes_t_num, nfi_t_num):
    """Linearly interpolate a single flux channel; NaNs are handled via masking."""
    # Use only finite values for interpolation
    finite_mask = np.isfinite(flux_col)
    if finite_mask.sum() < 2:
        return np.full(nfi_t_num.shape, np.nan)
    return np.interp(
        nfi_t_num,
        goes_t_num[finite_mask],
        flux_col[finite_mask],
        left=np.nan,
        right=np.nan
    )

def dt_to_numeric(dt_array):
    """Convert array of datetimes/datetime64 to float seconds."""
    epoch = np.datetime64(0, 'ns')
    one_second = np.timedelta64(1, 's')
    return (np.array(dt_array, dtype='datetime64[ns]') - epoch) / one_second

start_dt_num = dt_to_numeric(start_dt)
end_dt_num = dt_to_numeric(end_dt)
ace_time_num = dt_to_numeric(ace_time)
nfi_time_num = dt_to_numeric(nfi_time)

i_flux_10mev = interp_channel(flux_10mev, ace_time_num, nfi_time_num)
i_flux_30mev = interp_channel(flux_30mev, ace_time_num, nfi_time_num)


# Filter all data to the same time range (start_dt to end_dt)
nfi_fov_mean_full = nfi_fov_mean_full[(nfi_time_num >= start_dt_num) & (nfi_time_num <= end_dt_num)]
nfi_time = nfi_time[(nfi_time_num >= start_dt_num) & (nfi_time_num <= end_dt_num)]
i_flux_10mev = i_flux_10mev[(ace_time_num >= start_dt_num) & (ace_time_num <= end_dt_num)]
i_flux_30mev = i_flux_30mev[(ace_time_num >= start_dt_num) & (ace_time_num <= end_dt_num)]
ace_time = ace_time[(ace_time_num >= start_dt_num) & (ace_time_num <= end_dt_num)]







X = np.column_stack((i_flux_10mev, i_flux_30mev))  # Input features
y = nfi_fov_mean_full

# Filter out rows where any feature or target is NaN
valid_rows = np.isfinite(X).all(axis=1) & np.isfinite(y)
X = X[valid_rows]
y = y[valid_rows]


# 2. Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize the model
model = LinearRegression()

# 4. Train the model using the training data
model.fit(X_train, y_train)

# 5. Predict on the test data
y_pred = model.predict(X_test)

# 6. Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 7. Print results
print(f"Intercept (b0): {model.intercept_:.4f}")
print(f"Coefficients (b1, b2, b3): {model.coef_}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared Score (R2): {r2:.4f}")

import matplotlib.pyplot as plt


# 4. Create the Actual vs. Predicted Plot
plt.figure(figsize=(8, 6))

# Plot the test data points
plt.scatter(y_test, y_pred, color='blue', alpha=0.7, edgecolors='k', label='Predicted vs Actual')

# Plot a perfect prediction reference line (y = x)
ideal_line = np.linspace(min(y_test), max(y_test), 100)
plt.plot(ideal_line, ideal_line, color='red', linestyle='--', linewidth=2, label='Perfect Prediction Line')

# Customize the chart
plt.title('Multiple Regression: Actual vs. Predicted Values', fontsize=14, fontweight='bold')
plt.xlabel('Original (Actual) Values', fontsize=12)
plt.ylabel('Predicted Values', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)

# Show the plot window
plt.show()


#%%