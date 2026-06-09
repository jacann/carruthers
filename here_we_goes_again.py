
# GOES Proton Flux Data Analysis

# %% --- IMPORTS  ---
import os
import requests
import netCDF4 as nc
import xarray as xr
import numpy as np
import cftime
import glob
import matplotlib.pyplot as plt
from matplotlib import colors, gridspec
import matplotlib.dates as mdates
from avg import filter_time_range
from avg import filter_n_frames
from plotting import *

from scipy.interpolate import interp1d

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

def interpolate_nan_channels(flux_array, time_numeric):
    """
    Linearly interpolate NaN values per channel along the time axis.
    flux_array: shape (n_times, n_channels)
    time_numeric: shape (n_times,) — numeric time values for interpolation
    """
    flux_filled = flux_array.copy()
    for i in range(flux_array.shape[1]):
        col = flux_array[:, i]
        nan_mask = ~np.isfinite(col)
        if nan_mask.all():
            continue  # entire channel is NaN, can't interpolate
        if not nan_mask.any():
            continue  # no NaNs, nothing to do
        valid = np.where(~nan_mask)[0]
        interp_func = interp1d(
            time_numeric[valid], col[valid],
            kind='linear',
            bounds_error=False,
            fill_value=np.nan  # leave edges as NaN if outside valid range
        )
        flux_filled[nan_mask, i] = interp_func(time_numeric[nan_mask])
    return flux_filled

def load_sgps(path_glob: str) -> xr.Dataset:
    ds = xr.open_mfdataset(
        path_glob,
        combine='nested',
        concat_dim='time',
        preprocess=_normalize_time_var,  # see below
    )
    return ds

def _normalize_time_var(ds: xr.Dataset) -> xr.Dataset:
    """Rename legacy time variable so open_mfdataset sees a consistent dim."""
    if 'L2_SciData_TimeStamp' in ds:
        ds = ds.rename({'L2_SciData_TimeStamp': 'time'})
    return ds

# ASSUME FILES HAVE BEEN PULLED FROM NOAA
#%% --- READ & GET RELAVENT DATA ---
sgps_data_path = "/home/jacob/noncarr/data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes18/l2/data/sgps-l2-avg5m/**/**/*.nc"

filepaths = glob.glob(sgps_data_path, recursive=True)



proton_diff_flux_west = []
proton_diff_flux_east = []
proton_int_flux = []
goes_time = []

for filepath in filepaths:
    # Load data from file
    sgps_data = nc.Dataset(filepath)

    # Get flux data
    proton_diff_flux_west.append(sgps_data['AvgDiffProtonFlux'][:, 0, :])
    proton_diff_flux_east.append(sgps_data['AvgDiffProtonFlux'][:, 1, :])
    proton_int_flux.append(sgps_data['AvgIntProtonFlux'][:])

    # Get timestamps
    # Two different time variable names may have been used
    try:
        goes_time.append(sgps_data['L2_SciData_TimeStamp'][:])
    except IndexError:
        goes_time.append(sgps_data['time'][:])

# Convert list of arrays into single array
proton_diff_flux_west = np.ma.concatenate(proton_diff_flux_west)
proton_diff_flux_east = np.ma.concatenate(proton_diff_flux_east)
proton_int_flux = np.ma.concatenate(proton_int_flux)
goes_time = np.ma.concatenate(goes_time)

# replace zeros with nans
proton_diff_flux_west = np.where(proton_diff_flux_west < 1.e-12, np.nan, proton_diff_flux_west)
proton_diff_flux_east = np.where(proton_diff_flux_east < 1.e-12, np.nan, proton_diff_flux_east)
proton_int_flux = np.where(proton_int_flux < 1.e-12, np.nan, proton_int_flux)

# Convert J2000 time to python datetime
goes_time = cftime.num2pydate(goes_time[:], sgps_data['time'].units)

# Sort all proton flux data by datetime
sort_indices = np.argsort(goes_time)

# Reorder all arrays using the sort indices
goes_time_og             = goes_time[sort_indices]
proton_diff_flux_west_og = proton_diff_flux_west[sort_indices]
proton_diff_flux_east_og = proton_diff_flux_east[sort_indices]
proton_int_flux_og       = proton_int_flux[sort_indices]


# Load FOV Avg data
with xr.open_dataset('products/WFI_FOV_AVG.nc') as ds:
    wfi_fov_mean_top_og = ds["fov_mean_top"].values
    wfi_fov_mean_bot_og = ds["fov_mean_bottom"].values
    wfi_fov_mean_dt_og = ds["observation"].values
    wfi_fov_mean_nfr_og = ds["n_frames"].values

with xr.open_dataset('products/NFI_FOV_AVG.nc') as ds:
    nfi_fov_mean_top_og = ds["fov_mean_top"].values
    nfi_fov_mean_bot_og = ds["fov_mean_bottom"].values
    nfi_fov_mean_dt_og = ds["observation"].values
    nfi_fov_mean_nfr_og = ds["n_frames"].values

nfi_fov_mean_full_og = (nfi_fov_mean_top_og + nfi_fov_mean_bot_og)/2
wfi_fov_mean_full_og = (wfi_fov_mean_top_og + wfi_fov_mean_bot_og)/2
wfi_fov_mean_dt_og = wfi_fov_mean_dt_og.astype('datetime64[ns]')
nfi_fov_mean_dt_og = nfi_fov_mean_dt_og.astype('datetime64[ns]')


# alert products
proton_diff_flux_mean_og = (proton_diff_flux_west_og + proton_diff_flux_east_og)/2
proton_int_flux_mean_og = (proton_int_flux_og[:, 0] + proton_int_flux_og[:, 1])/2   
int_10mev_og = np.nansum(proton_diff_flux_mean_og[:,5:], axis=1) + proton_int_flux_mean_og # only sum channels > 10 MeV
print(proton_diff_flux_mean_og)

#%% Filter data by time range

# FOV TIME SELECTION
start_datetime_str = '2025-11-11T00:00:00'
end_datetime_str = '2025-11-14T00:00:00'
n_frames_min = 0
nfi_storm_thresh = 0.0
wfi_storm_thresh = 0.0
int_10mev_thresh = threshold_10mev =  0.175
channel = "NFI"


print(f"Start: {start_datetime_str}\t End: {end_datetime_str}")

# Convert the strings to datetime objects
start_datetime = np.datetime64(start_datetime_str)
end_datetime = np.datetime64(end_datetime_str)

wfi_fov_mean_top = filter_time_range(wfi_fov_mean_top_og, wfi_fov_mean_dt_og, start_datetime, end_datetime)
wfi_fov_mean_bot = filter_time_range(wfi_fov_mean_bot_og, wfi_fov_mean_dt_og, start_datetime, end_datetime)
wfi_fov_mean_nfr = filter_time_range(wfi_fov_mean_nfr_og, wfi_fov_mean_dt_og, start_datetime, end_datetime)
wfi_fov_mean_dt = filter_time_range(wfi_fov_mean_dt_og, wfi_fov_mean_dt_og, start_datetime, end_datetime)

nfi_fov_mean_top = filter_time_range(nfi_fov_mean_top_og, nfi_fov_mean_dt_og, start_datetime, end_datetime)
nfi_fov_mean_bot = filter_time_range(nfi_fov_mean_bot_og, nfi_fov_mean_dt_og, start_datetime, end_datetime)
nfi_fov_mean_nfr = filter_time_range(nfi_fov_mean_nfr_og, nfi_fov_mean_dt_og, start_datetime, end_datetime)
nfi_fov_mean_dt = filter_time_range(nfi_fov_mean_dt_og, nfi_fov_mean_dt_og, start_datetime, end_datetime)

proton_diff_flux_west = filter_time_range(proton_diff_flux_west_og, goes_time_og, start_datetime, end_datetime)
proton_diff_flux_east = filter_time_range(proton_diff_flux_east_og, goes_time_og, start_datetime, end_datetime)
proton_int_flux       = filter_time_range(proton_int_flux_og, goes_time_og, start_datetime, end_datetime)
int_10mev           = filter_time_range(int_10mev_og, goes_time_og, start_datetime, end_datetime)
goes_time             = filter_time_range(goes_time_og, goes_time_og, start_datetime, end_datetime)

#identify time range for which int_10mev_og exeeds threshold
int_10mev_mask = int_10mev >= int_10mev_thresh
new_start_datetime = goes_time[int_10mev_mask][0]
print(f"New start datetime based on 10 MeV integral flux threshold: {new_start_datetime}")
new_end_datetime = goes_time[int_10mev_mask][-1]
print(f"New end datetime based on 10 MeV integral flux threshold: {new_end_datetime}")


# filter by new time range (    this is bad code :-(     )
start_datetime = np.datetime64(new_start_datetime)
end_datetime = np.datetime64(new_end_datetime)
print(type(start_datetime))
wfi_fov_mean_top = filter_time_range(wfi_fov_mean_top_og, wfi_fov_mean_dt_og, start_datetime, end_datetime)
wfi_fov_mean_bot = filter_time_range(wfi_fov_mean_bot_og, wfi_fov_mean_dt_og, start_datetime, end_datetime)
wfi_fov_mean_nfr = filter_time_range(wfi_fov_mean_nfr_og, wfi_fov_mean_dt_og, start_datetime, end_datetime)
wfi_fov_mean_dt = filter_time_range(wfi_fov_mean_dt_og, wfi_fov_mean_dt_og, start_datetime, end_datetime)

nfi_fov_mean_top = filter_time_range(nfi_fov_mean_top_og, nfi_fov_mean_dt_og, start_datetime, end_datetime)
nfi_fov_mean_bot = filter_time_range(nfi_fov_mean_bot_og, nfi_fov_mean_dt_og, start_datetime, end_datetime)
nfi_fov_mean_nfr = filter_time_range(nfi_fov_mean_nfr_og, nfi_fov_mean_dt_og, start_datetime, end_datetime)
nfi_fov_mean_dt = filter_time_range(nfi_fov_mean_dt_og, nfi_fov_mean_dt_og, start_datetime, end_datetime)



# Filter data by n_frames
wfi_fov_mean_top = filter_n_frames(wfi_fov_mean_top, wfi_fov_mean_nfr, n_frames_min)
wfi_fov_mean_bot = filter_n_frames(wfi_fov_mean_bot, wfi_fov_mean_nfr, n_frames_min)
wfi_fov_mean_dt  = filter_n_frames(wfi_fov_mean_dt, wfi_fov_mean_nfr, n_frames_min)
wfi_fov_mean_nfr = filter_n_frames(wfi_fov_mean_nfr, wfi_fov_mean_nfr, n_frames_min)

nfi_fov_mean_top = filter_n_frames(nfi_fov_mean_top, nfi_fov_mean_nfr, n_frames_min)
nfi_fov_mean_bot = filter_n_frames(nfi_fov_mean_bot, nfi_fov_mean_nfr, n_frames_min)
nfi_fov_mean_dt  = filter_n_frames(nfi_fov_mean_dt, nfi_fov_mean_nfr, n_frames_min)
nfi_fov_mean_nfr = filter_n_frames(nfi_fov_mean_nfr, nfi_fov_mean_nfr, n_frames_min)


# calculate full-sensor averages for each dark channel
wfi_fov_mean_full = (wfi_fov_mean_top + wfi_fov_mean_bot)/2
nfi_fov_mean_full = (nfi_fov_mean_top + nfi_fov_mean_bot)/2
 



# PLOT BASIC GRAPH

plt.figure(1, figsize=[12, 10], layout='constrained')
gridspec.GridSpec(8, 1)


# important variables
n_frames_min = 0

# Plot differential flux
NUM_DIFF_CHANNELS = 13

chan_colors = [[.8, 0., 0.], colors.to_rgba('orangered')[0:3], colors.to_rgba('orange')[0:3],
               [.95, .95, .0], colors.to_rgba('greenyellow')[0:3], colors.to_rgba('yellowgreen')[0:3],
               colors.to_rgba('green')[0:3], [0., .78, .7], [0., .6, .88], [0., 0., .9],
               colors.to_rgba('violet')[0:3], [0.8, .1, 1.0], [0.49411765, 0., 0.70980392]]

chan_labels = ['P1 (1.0-1.9 MeV)', 'P2A (1.9-2.3 MeV)', 'P2B (2.3-3.4 MeV)', 'P3 (3.4-6.5 MeV)',
               'P4 (6.5-12 MeV)', 'P5 (12-25 MeV)', 'P6 (25-40 MeV)', 'P7 (40-80 MeV)',
               'P8A (83-99 MeV)', 'P8B (99-118 MeV)', 'P8C (118-150 MeV)', 'P9 (150-275 MeV)',
               'P10 (275-500 MeV)']

# SGPS-X (west)
ax1 = plt.subplot2grid((8, 1), (0, 0), colspan=1, rowspan=2)
for i in range(NUM_DIFF_CHANNELS):
    plt.plot(goes_time, proton_diff_flux_west[:, i], color=chan_colors[i], label=chan_labels[i], marker='none', ms=0.5)
textstr = 'GOES-18 SGPS-X (west looking field-of-view)'
ax1.text(0.042, 0.97, textstr, transform=ax1.transAxes, fontsize=14, verticalalignment='top')
#ax1.xaxis.set_minor_locator(mdates.HourLocator())
ax1.xaxis.set_major_locator(mdates.DayLocator())
ax1.tick_params(which = 'minor', length=3)
ax1.tick_params(which = 'major', length=5)
ax1.set_xlim(goes_time[0], goes_time[-1])
ax1.tick_params(labelbottom=False)
plt.yscale('log')
plt.ylim([1.e-8, 1.e3])
plt.ylabel('protons/cm$^2$-s-sr-keV')

leg1 = ax1.legend(loc='upper right', bbox_to_anchor=(1.27, 1), prop={'size': 12})
leg1.set_in_layout(False)

# SGPS+X (east)
ax2 = plt.subplot2grid((8, 1), (2, 0), colspan=1, rowspan=2)
for i in range(NUM_DIFF_CHANNELS):
    plt.plot(goes_time, proton_diff_flux_east[:, i], color=chan_colors[i], label=chan_labels[i] ,marker='none', ms=0.5)
textstr = 'GOES-18 SGPS+X (east looking field-of-view)'
ax2.text(0.042, 0.97, textstr, transform=ax2.transAxes, fontsize=14, verticalalignment='top')
#ax2.xaxis.set_minor_locator(mdates.HourLocator())
ax2.xaxis.set_major_locator(mdates.DayLocator())
ax2.tick_params(which = 'minor', length=3)
ax2.tick_params(which = 'major', length=5)
ax2.set_xlim(goes_time[0], goes_time[-1])
ax2.tick_params(labelbottom=False)
plt.yscale('log')
plt.ylim([1.e-8, 1.e3])
plt.ylabel('protons/cm$^2$-s-sr-keV')

# Plot integral flux
ax3 = plt.subplot2grid((8, 1), (4, 0), colspan=1, rowspan=1)
plt.plot(goes_time, proton_int_flux[:, 1], color='grey', label='SGPS+X P11 (>500 MeV)')
plt.plot(goes_time, proton_int_flux[:, 0], color='k', label='SGPS-X P11 (>500 MeV)')
textstr = 'GOES-18 SGPS P11'
ax3.text(0.042, 0.94, textstr, transform=ax3.transAxes, fontsize=14, verticalalignment='top')
#ax3.xaxis.set_minor_locator(mdates.HourLocator())
ax3.xaxis.set_major_locator(mdates.DayLocator())
#ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax3.tick_params(which = 'minor', length=3)
ax3.tick_params(which = 'major', length=5)
ax3.set_xlim(goes_time[0], goes_time[-1])
ax3.tick_params(labelbottom=False)
#plt.xlabel('2026 (UT hours)')
plt.ylabel('protons/cm$^2$-s-sr')
plt.yscale('log')
ylims = ax3.get_ylim()
plt.ylim([ylims[0], 1.4*ylims[1]])

ax3.legend(loc='upper right', bbox_to_anchor=(1.28, 1), prop={'size': 10})


# plot calculated integral fluxes
ax3a = plt.subplot2grid((8, 1), (5, 0), colspan=1, rowspan=1)
plt.plot(goes_time, int_10mev, color='purple', label='Integral >10 MeV', linestyle='-')
textstr = 'GOES-18 SGPS Integral >10 MeV'
ax3a.text(0.042, 0.94, textstr, transform=ax3a.transAxes, fontsize=14, verticalalignment='top')
#ax3a.xaxis.set_minor_locator(mdates.HourLocator())
ax3a.xaxis.set_major_locator(mdates.DayLocator())
#ax3a.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax3a.tick_params(which = 'minor', length=3)
ax3a.tick_params(which = 'major', length=5)
ax3a.set_xlim(goes_time[0], goes_time[-1])
ax3a.tick_params(labelbottom=False)
plt.ylabel('protons/cm$^2$-s-sr')
plt.yscale('log')
ylims = ax3a.get_ylim()
plt.ylim([ylims[0], 1.4*ylims[1]])
# plot line for 10 MeV integral flux threshold
plt.axhline(threshold_10mev, color='purple', linestyle='--', label=f'10 MeV Integral Threshold\n{threshold_10mev} pfu')
ax3a.legend(loc='upper right', bbox_to_anchor=(1.28, 1), prop={'size': 10})


# Plot MCP radiation
ax4 = plt.subplot2grid((8, 1), (6, 0), colspan=1, rowspan=2)
#plt.plot(wfi_fov_mean_dt, wfi_fov_mean_full, label='WFI FOV Mean Full', ms=3, marker="o")
plt.plot(nfi_fov_mean_dt, nfi_fov_mean_full, label='STORM TIME CLASSIFIED\nNFI FOV Means', ms=3, marker="o", zorder=5, color='blue')
plt.plot(nfi_fov_mean_dt_og, nfi_fov_mean_full_og, label='QUIET TIME CLASSIFIED \n NFI FOV Means', ms=3, marker="o", color='orange', alpha=1)
plt.ylabel('NFI\n' + r'DN sec$^{-1}$ pixel$^{-1}$')
plt.ylim(None, 30)


ax4a = ax4.twinx()
plt.plot(wfi_fov_mean_dt, wfi_fov_mean_full, label='STORM TIME CLASSIFIED\nWFI FOV Means',  ms=3, marker="o", zorder=5, color = 'purple')
plt.plot(wfi_fov_mean_dt_og, wfi_fov_mean_full_og, label='QUIET TIME CLASSIFIED \n WFI FOV Means', ms=3, marker="o", color = 'red')   
# legend for WFI FOV means
ax4a.legend(loc='upper right', bbox_to_anchor=(1.25, 1), prop={'size': 8})


textstr = 'GCI DRK FOV Means'
ax4.text(0.042, 0.94, textstr, transform=ax4.transAxes, fontsize=14, verticalalignment='top')
ax4.xaxis.set_minor_locator(mdates.HourLocator())
ax4.xaxis.set_major_locator(mdates.DayLocator())
#ax4.xaxis.set_major_locator(mdates.DayLocator())
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%Y'))
#ax4.tick_params(which = 'minor', length=3)
ax4.tick_params(which = 'major', length=5)
ax4.set_xlim(goes_time[0], goes_time[-1])
plt.xlabel('Time')
#ax4.set_ylim(0, 3)

ax4.legend(loc='lower right', bbox_to_anchor=(1.25, 0.0), prop={'size': 8})

#fig.autofmt_xdate()

plt.ylabel('WFI\n' + 'DN\n' + r'sec$^{-1}$' + '\n' + r'pixel$^{-1}$')
#ylims = ax4.get_ylim()
#plt.ylim([ylims[0], 1.4*ylims[1]])

#ax4.set_ylim(0, 0.5)


plt.show()


# --- RIDGE REGRESSION ANALYSIS ---
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# --- Interpolate GOES data to NFI datetimes ---

# Convert datetimes to numeric (seconds since epoch) for interpolation

nfi_t_numeric = dt_to_numeric(nfi_fov_mean_dt)
wfi_t_numeric = dt_to_numeric(wfi_fov_mean_dt)

# goes_time is a list/array of Python datetimes — convert to datetime64[ns]
goes_t_numeric = dt_to_numeric(
    np.array(goes_time, dtype='datetime64[ns]')
)

# --- Build uniform 10-second GOES grid for smooth predicted time series ---
t_10s_numeric = np.arange(goes_t_numeric[0], goes_t_numeric[-1], 10.0)

# Interpolate all 28 channels onto 10s grid
west_10s = np.column_stack([
    interp_channel(proton_diff_flux_west[:, i], goes_t_numeric, t_10s_numeric)
    for i in range(NUM_DIFF_CHANNELS)
])
east_10s = np.column_stack([
    interp_channel(proton_diff_flux_east[:, i], goes_t_numeric, t_10s_numeric)
    for i in range(NUM_DIFF_CHANNELS)
])
p11_west_10s = interp_channel(proton_int_flux[:, 0], goes_t_numeric, t_10s_numeric).reshape(-1, 1)
p11_east_10s = interp_channel(proton_int_flux[:, 1], goes_t_numeric, t_10s_numeric).reshape(-1, 1)

X_full_raw = np.hstack([west_10s, east_10s, p11_west_10s, p11_east_10s])

# Convert 10s numeric timestamps back to datetime64 for plotting
epoch = np.datetime64(0, 'ns')
one_second = np.timedelta64(1, 's')
goes_time_10s = epoch + (t_10s_numeric * 1e9).astype('timedelta64[ns]')


# Build design matrix: 13 west channels + 13 east channels + P11 west + P11 east = 28 features
# Interpolate each GOES channel onto NFI timestamps


NUM_DIFF_CHANNELS = 13

# Interpolate all 28 channels
nfi_interp_west = np.column_stack([
    interp_channel(proton_diff_flux_west[:, i], goes_t_numeric, nfi_t_numeric)
    for i in range(NUM_DIFF_CHANNELS)
])  # shape: (n_nfi, 13)
wfi_interp_west = np.column_stack([
    interp_channel(proton_diff_flux_west[:, i], goes_t_numeric, wfi_t_numeric)
    for i in range(NUM_DIFF_CHANNELS)
])  # shape: (n_nfi, 13)

nfi_interp_east = np.column_stack([
    interp_channel(proton_diff_flux_east[:, i], goes_t_numeric, nfi_t_numeric)
    for i in range(NUM_DIFF_CHANNELS)
])  # shape: (n_nfi, 13)

wfi_interp_east = np.column_stack([
    interp_channel(proton_diff_flux_east[:, i], goes_t_numeric, wfi_t_numeric)
    for i in range(NUM_DIFF_CHANNELS)
])  # shape: (n_nfi, 13)

nfi_interp_p11_west = interp_channel(proton_int_flux[:, 0], goes_t_numeric, nfi_t_numeric).reshape(-1, 1)
nfi_interp_p11_east = interp_channel(proton_int_flux[:, 1], goes_t_numeric, nfi_t_numeric).reshape(-1, 1)
wfi_interp_p11_west = interp_channel(proton_int_flux[:, 0], goes_t_numeric, wfi_t_numeric).reshape(-1, 1)
wfi_interp_p11_east = interp_channel(proton_int_flux[:, 1], goes_t_numeric, wfi_t_numeric).reshape(-1, 1)


# Full design matrix: 28 columns
X_full_nfi = np.hstack([nfi_interp_west, nfi_interp_east, nfi_interp_p11_west, nfi_interp_p11_east])
X_full_wfi = np.hstack([wfi_interp_west, wfi_interp_east, wfi_interp_p11_west, wfi_interp_p11_east])
y_nfi = nfi_fov_mean_full  # shape: (n_nfi,)
y_wfi = wfi_fov_mean_full  # shape: (n_wfi,)

# --- Build feature labels ---
chan_labels_short = [f'P{["1","2A","2B","3","4","5","6","7","8A","8B","8C","9","10"][i]}' for i in range(NUM_DIFF_CHANNELS)]
feature_labels = (
    [f'W-{lbl}' for lbl in chan_labels_short] +
    [f'E-{lbl}' for lbl in chan_labels_short] +
    ['W-P11', 'E-P11']
)

# --- Drop rows with any NaN in X or y ---
if channel == "NFI":
    X_full = X_full_nfi
    y = y_nfi
elif channel == "WFI":
    X_full = X_full_wfi
    y = y_wfi   
else:
    raise ValueError("Invalid channel selection. Choose 'NFI' or 'WFI'.")

valid_mask = np.isfinite(X_full).all(axis=1) & np.isfinite(y)
valid_mask_raw = np.isfinite(X_full_raw).all(axis=1)
X_clean = X_full[valid_mask]
X_clean_raw = X_full_raw[valid_mask_raw]
y_clean = y[valid_mask]
t_clean = nfi_fov_mean_dt[valid_mask]
goes_time_clean = goes_time_10s[valid_mask_raw]



print(f"Samples available for regression: {valid_mask.sum()} / {len(y)}")


# --- Scale features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)
X_full_scaled = scaler.transform(X_clean_raw)

# --- RidgeCV: auto-selects best alpha via cross-validation ---
alphas = np.logspace(-3, 6, 200)
ridge_cv = RidgeCV(alphas=alphas, store_cv_results=True)
ridge_cv.fit(X_scaled, y_clean)

best_alpha = ridge_cv.alpha_
y_pred = ridge_cv.predict(X_scaled)
y_full_pred = ridge_cv.predict(X_full_scaled)
r2 = r2_score(y_clean, y_pred)
residuals = y_clean - y_pred

print(f"\nRidge Regression Results")
print(f"  Best alpha (regularization): {best_alpha:.4f}")
print(f"  R² (in-sample):              {r2:.4f}")
print(f"  Residual std:                {residuals.std():.4f}")

# --- Coefficient summary ---
coef_df_data = sorted(
    zip(feature_labels, ridge_cv.coef_),
    key=lambda x: abs(x[1]),
    reverse=True
)
print(f"\n  Top 10 predictors by |coefficient|:")
print(f"  {'Feature':<12}  {'Coef':>10}")
for feat, coef in coef_df_data[:10]:
    print(f"  {feat:<12}  {coef:>10.4f}")

# --- Diagnostic plots ---
fig_ridge, axes = plt.subplots(3, 1, figsize=(12, 10), layout='constrained')

# 1. Observed vs. predicted time series
ax = axes[0]
ax.scatter(t_clean, y_clean, color='steelblue', label=f'Observed {channel} FOV Mean', s=30, zorder=3)
ax.plot(goes_time_clean, y_full_pred,  color='tomato',    label='Ridge Predicted', lw=2, ls='-')
ax.set_ylabel(r'DN sec$^{-1}$ pixel$^{-1}$')
ax.set_title(f'{channel} FOV Mean — Observed vs. Ridge Predicted')
ax.xaxis.set_minor_locator(mdates.HourLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%m/%d\n%Y'))
ax.legend(prop={'size': 10})
ax.tick_params(labelbottom=True)

# 2. Scatter: observed vs predicted
ax = axes[1]
lims = [min(y_clean.min(), y_pred.min()), max(y_clean.max(), y_pred.max())]
ax.scatter(y_clean, y_pred, s=8, alpha=0.5, color='steelblue')
ax.plot(lims, lims, 'k--', lw=1, label='1:1 line')
ax.set_xlabel(r'Observed DN sec$^{-1}$ pixel$^{-1}$')
ax.set_ylabel(r'Predicted DN sec$^{-1}$ pixel$^{-1}$')
ax.set_title(f'Observed vs. Predicted  (R² = {r2:.3f}, residual std: {residuals.std():.4f},  α = {best_alpha:.2f}, intercept: {ridge_cv.intercept_:.4f})')
ax.legend(prop={'size': 10})

# 3. Coefficients bar chart
ax = axes[2]
feat_names, coefs = zip(*[(f, c) for f, c in zip(feature_labels, ridge_cv.coef_)])
colors_bar = ['tomato' if c > 0 else 'steelblue' for c in coefs]
bars = ax.bar(feat_names, coefs, color=colors_bar, edgecolor='none')
ax.axhline(0, color='k', lw=0.8)
ax.set_xlabel('Feature')
ax.set_ylabel('Coefficient (standardized)')
ax.set_title('Ridge Regression Coefficients (standardized GOES-18 Proton Fluxes)')
ax.tick_params(axis='x', rotation=45)

plt.show()

# --- MULTIPLE LINEAR REGRESSION ANALYSIS ---
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score as mlr_r2_score

# Reuse the same cleaned, log-transformed, scaled data from Ridge section above:
# X_scaled  → (n_samples, 28) standardized log-flux features
# y_clean   → NFI FOV mean (target)
# t_clean   → NFI timestamps for the clean samples
# goes_time_clean → GOES timestamps for the raw prediction overlay
# feature_labels  → list of 28 feature name strings



# %%
# --- Fit OLS Linear Regression ---
mlr = LinearRegression()
mlr.fit(X_scaled, y_clean)

y_mlr_pred       = mlr.predict(X_scaled)           # in-sample fit on NFI times
y_mlr_full_pred  = mlr.predict(X_full_scaled)      # prediction on full GOES time grid

r2_mlr     = mlr_r2_score(y_clean, y_mlr_pred)
residuals_mlr = y_clean - y_mlr_pred

print(f"\nMultiple Linear Regression Results")
print(f"  R² (in-sample):   {r2_mlr:.4f}")
print(f"  Residual std:     {residuals_mlr.std():.4f}")
print(f"  Intercept:        {mlr.intercept_:.4f}")

coef_mlr_sorted = sorted(
    zip(feature_labels, mlr.coef_),
    key=lambda x: abs(x[1]),
    reverse=True
)
print(f"\n  Top 10 predictors by |coefficient|:")
print(f"  {'Feature':<12}  {'Coef':>10}")
for feat, coef in coef_mlr_sorted[:10]:
    print(f"  {feat:<12}  {coef:>10.4f}")

# --- Diagnostic plots ---
fig_mlr, axes_mlr = plt.subplots(3, 1, figsize=(12, 10), layout='constrained')
fig_mlr.suptitle(f'Multiple Linear Regression — {channel} FOV Mean', fontsize=14)

# 1. Time series: observed vs MLR predicted
ax = axes_mlr[0]
ax.plot(goes_time_clean, y_mlr_full_pred, color='darkorange', label='MLR Predicted',
        lw=2, ls='-')
ax.scatter(t_clean, y_clean, color='steelblue', label='Observed NFI FOV Mean', s=30, zorder=3)
ax.set_ylabel(r'DN sec$^{-1}$ pixel$^{-1}$')
ax.set_title('Observed vs. MLR Predicted (time series)')
ax.xaxis.set_minor_locator(mdates.HourLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%m/%d\n%Y'))
ax.legend(prop={'size': 10})
ax.tick_params(labelbottom=True)

# 2. Scatter: observed vs predicted
ax = axes_mlr[1]
lims_mlr = [min(y_clean.min(), y_mlr_pred.min()),
            max(y_clean.max(), y_mlr_pred.max())]
ax.scatter(y_clean, y_mlr_pred, s=8, alpha=0.5, color='darkorange')
ax.plot(lims_mlr, lims_mlr, 'k--', lw=1, label='1:1 line')
ax.set_xlabel(r'Observed DN sec$^{-1}$ pixel$^{-1}$')
ax.set_ylabel(r'Predicted DN sec$^{-1}$ pixel$^{-1}$')
ax.set_title(f'Observed vs. Predicted  (R² = {r2_mlr:.3f})')
ax.legend(prop={'size': 10})

# 3. Coefficient bar chart
ax = axes_mlr[2]
feat_names_mlr, coefs_mlr = zip(*[(f, c) for f, c in zip(feature_labels, mlr.coef_)])
colors_mlr = ['tomato' if c > 0 else 'steelblue' for c in coefs_mlr]
ax.bar(feat_names_mlr, coefs_mlr, color=colors_mlr, edgecolor='none')
ax.axhline(0, color='k', lw=0.8)
ax.set_xlabel('Feature')
ax.set_ylabel('Coefficient (standardized)')
ax.set_title('MLR Coefficients')
ax.tick_params(axis='x', rotation=45)

plt.show()