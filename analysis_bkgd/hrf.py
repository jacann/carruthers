# hrf.py


# ----- Storm-time regression functions -----
def interp_and_clean_xy(goes_time, proton_diff_flux_west, proton_diff_flux_east, p11_west, p11_east, gci_fov_mean_full, gci_fov_mean_dt):
    """
    Builds X and y arrays for Ridge regresssion.

    GOES data are linearly interpolated to create an array of GOES data aligned with GCI times of image capture. 
    Then, any rows with NaNs are dropped. Output data still needs to be scaled before being the regression is run.

    Args:
        goes_time (1D numpy array): datetimes for GOES proton flux data at a 5-minute cadence.
        proton_diff_flux_west (2D numpy array): SGPS differential proton fluxes across 13 differential channels (west-facing)
        proton_diff_flux_east (2D numpy array): SGPS differential proton fluxes across 13 differential channels (east-facing)
        p11_west (1D numpy array): SGPS integral proton fluxes (>500 MeV) (west-facing)
        p11_east (1D numpy array): SGPS integral proton fluxes (>500 MeV) (east-facing)
        gci_fov_mean_full (1D numpy array): GCI Dark image FOV Means across the full sensor
        gci_fov_mean_dt (1D numpy array): datetimes of capture for GCI dark images
        
    Returns:
        X_clean (nx28 2D numpy array): interpolated and cleaned GOES data for ridge regression
        y_clean (1D numpy array): cleaned GCI data for ridge regresssion
        
    """
    nfi_t_numeric = dt_to_numeric(gci_fov_mean_dt)

    goes_t_numeric = dt_to_numeric(
        np.array(goes_time, dtype='datetime64[ns]')
    )

    # Build matrix for regression input: 13 west channels + 13 east channels + P11 west + P11 east = 28 features
    # Interpolate each GOES channel onto GCI datetimes

    gci_interp_west = np.column_stack([
        interp_channel(proton_diff_flux_west[:, i], goes_t_numeric, nfi_t_numeric)
        for i in range(NUM_GOES_DIFF_CHANNELS)
    ])  # shape: (n_gci, 13)

    gci_interp_east = np.column_stack([
        interp_channel(proton_diff_flux_east[:, i], goes_t_numeric, nfi_t_numeric)
        for i in range(NUM_GOES_DIFF_CHANNELS)
    ])  # shape: (n_gci, 13)

    p11_west = p11_west.ravel()
    p11_east = p11_east.ravel()
    nfi_interp_p11_west = interp_channel(p11_west, goes_t_numeric, nfi_t_numeric)
    nfi_interp_p11_east = interp_channel(p11_east, goes_t_numeric, nfi_t_numeric)

    # Full design matrix: 28 columns
    X_full = np.column_stack([gci_interp_west, gci_interp_east, nfi_interp_p11_west, nfi_interp_p11_east])
    y = gci_fov_mean_full  # shape: (n_gci,)

    # Drop rows with any NaN
    valid_mask = np.isfinite(X_full).all(axis=1) 
    X_clean = X_full[valid_mask]
    y_clean = y[valid_mask]
    t_clean = gci_fov_mean_dt[valid_mask]
    
    return X_clean, y_clean, t_clean

def fit_gci_to_goes(X_scaled, y_clean):
    '''
    Fits GOES data to GCI FOV means using a ridge regression with cross-validation, select results are output. 
    
    Args:
        X_scaled (nx28 2D numpy array): GOES data that is interpolated to GCI datetimes, cleaned of NaNs, and scaled.
        y_clean (1D numpy array): GCI FOV Means that have been cleaned for times where data were unavailable.
    '''

    # run RidgeCV and auto-select best alpha with cross-validation
    alphas = np.logspace(-3, 6, 200)
    ridge_cv = RidgeCV(alphas=alphas, store_cv_results=True)
    ridge_cv.fit(X_scaled, y_clean)

    best_alpha = ridge_cv.alpha_
    y_pred = ridge_cv.predict(X_scaled)
    r2 = r2_score(y_clean, y_pred)
    residuals = y_clean - y_pred

    print(f"\nRidge Regression Results")
    print(f"  Best alpha:       {best_alpha:.4f}")
    print(f"  R² (in-sample):   {r2:.4f}")
    print(f"  Residual std:     {residuals.std():.4f}")

    return ridge_cv, r2, residuals

# ----- Helper functions -----

def dt_to_numeric(dt_array):
    """Convert array of datetimes/datetime64 to floating point seconds."""
    epoch = np.datetime64(0, 'ns')
    one_second = np.timedelta64(1, 's')
    return (np.array(dt_array, dtype='datetime64[ns]') - epoch) / one_second

def interp_channel(flux_col, goes_t_num, nfi_t_num):
    """Linearly interpolate a single flux channel and handle NaNs with masking."""
    # Only interpolate finite values
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

# ----- EXAMPLE USAGE -----
def filter_time_range(data, datetimes, start_datetime, end_datetime):
    # Extract underlying numpy arrays from xarray DataArrays if needed
    datetimes = np.asarray(datetimes.values if hasattr(datetimes, 'values') else datetimes)
    data = np.asarray(data.values if hasattr(data, 'values') else data)

    start_idx = np.searchsorted(datetimes, start_datetime, side='left')
    end_idx = np.searchsorted(datetimes, end_datetime, side='right')

    return data[start_idx:end_idx]

def load_fov_mean_data(channel):
    '''
    Loads full-sensor L1A Dark FOV means and their corresponding datetimes from a custom dataset for visualization, irrelevant for pipeline?
    Example dataset in DVC only extends to the beginning of June.
    '''
    if channel == "WFI" or "NFI":
        loadpath = resources.files('glide') / f'calibration/data_files/{channel}_FOV_AVG.nc'
    else:
        raise ValueError("Invalid channel selection. Choose 'NFI' or 'WFI'.")
    
    # Load FOV Avg data
    with xr.open_dataset(loadpath) as ds:
        gci_fov_mean_top = ds["fov_mean_top"].values
        gci_fov_mean_bot = ds["fov_mean_bottom"].values
        gci_fov_mean_dt = ds["observation"].values

    gci_fov_mean_full = (gci_fov_mean_top + gci_fov_mean_bot)/2
    gci_fov_mean_dt = gci_fov_mean_dt.astype('datetime64[ns]')
    return gci_fov_mean_full, gci_fov_mean_dt

def plot_goes_data(goes_time, proton_diff_flux_west, proton_diff_flux_east, proton_int_flux, int_10mev, threshold_10mev, gci_fov_mean_dt, gci_fov_mean_full, gci_fov_mean_dt_og, gci_fov_mean_full_og):
    plt.figure(1, figsize=[12, 10], layout='constrained')
    gridspec.GridSpec(8, 1)

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
    ax3.xaxis.set_major_locator(mdates.DayLocator())
    ax3.tick_params(which = 'minor', length=3)
    ax3.tick_params(which = 'major', length=5)
    ax3.set_xlim(goes_time[0], goes_time[-1])
    ax3.tick_params(labelbottom=False)
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
    ax3a.xaxis.set_major_locator(mdates.DayLocator())
    ax3a.tick_params(which = 'minor', length=3)
    ax3a.tick_params(which = 'major', length=5)
    ax3a.set_xlim(goes_time[0], goes_time[-1])
    ax3a.tick_params(labelbottom=False)
    plt.ylabel('protons/cm$^2$-s-sr')
    plt.yscale('log')
    ylims = ax3a.get_ylim()
    plt.ylim([ylims[0], 1.4*ylims[1]])
    # plot line for >12 MeV integral flux threshold
    plt.axhline(threshold_10mev, color='purple', linestyle='--', label=f'10 MeV Integral Threshold\n{threshold_10mev} pfu')
    ax3a.legend(loc='upper right', bbox_to_anchor=(1.28, 1), prop={'size': 10})

    
    # Plot MCP radiation
    ax4 = plt.subplot2grid((8, 1), (6, 0), colspan=1, rowspan=2)
    plt.plot(gci_fov_mean_dt, gci_fov_mean_full, label='STORM TIME CLASSIFIED\nGCI FOV Means', ms=3, marker="o", zorder=5, color='blue')
    plt.plot(gci_fov_mean_dt_og, gci_fov_mean_full_og, label='QUIET TIME CLASSIFIED \n GCI FOV Means', ms=3, marker="o", color='orange', alpha=1)
    plt.ylabel( r'DN sec$^{-1}$ pixel$^{-1}$')
    plt.ylim(None, 30)


    textstr = 'GCI DRK FOV Means'
    ax4.text(0.042, 0.94, textstr, transform=ax4.transAxes, fontsize=14, verticalalignment='top')
    ax4.xaxis.set_minor_locator(mdates.HourLocator())
    ax4.xaxis.set_major_locator(mdates.DayLocator())
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%Y'))
    ax4.tick_params(which = 'major', length=5)
    ax4.set_xlim(goes_time[0], goes_time[-1])
    plt.xlabel('Time')
    ax4.set_ylim(0, 1.2 * np.max(gci_fov_mean_full))

    ax4.legend(loc='upper right', bbox_to_anchor=(1.28, 1), prop={'size': 10})

    plt.show()

def plot_model_results(channel, int_12mev_thresh=0.175):    
    # load goes data (Jan SEP event example)
    filenames = ["sci_sgps-l2-avg5m_g18_d20260118_v3-0-3.nc",
                 "sci_sgps-l2-avg5m_g18_d20260119_v3-0-3.nc",
                 "sci_sgps-l2-avg5m_g18_d20260120_v3-0-3.nc"]
    goes_time, proton_diff_flux_west, proton_diff_flux_east, proton_int_flux, int_12mev = load_goes_sgps_l2(filenames)

    # load FOV Means
    gci_fov_mean_full, gci_fov_mean_dt = load_fov_mean_data(channel)
    gci_fov_mean_full_og = gci_fov_mean_full    # Save original, unfiltered data
    gci_fov_mean_dt_og = gci_fov_mean_dt

    #identify time range for which int_10mev exeeds threshold        - pipeline should run for each storm interval, this is just quick and dirty for visualizaiton
    int_10mev_mask = int_12mev > int_12mev_thresh
    new_start_datetime = goes_time[int_10mev_mask][0]
    new_end_datetime = goes_time[int_10mev_mask][-1]
    print(f'Storm inverval start: {new_start_datetime}')
    print(f'Storm inverval end: {new_end_datetime}')
    
    # filter by new time range 
    start_datetime = np.datetime64(new_start_datetime)
    end_datetime = np.datetime64(new_end_datetime)

    gci_fov_mean_full = filter_time_range(gci_fov_mean_full, gci_fov_mean_dt, start_datetime, end_datetime)
    gci_fov_mean_dt = filter_time_range(gci_fov_mean_dt, gci_fov_mean_dt, start_datetime, end_datetime)

    # build X and y for ridge regression
    X_clean, y_clean, t_clean = interp_and_clean_xy(goes_time, proton_diff_flux_west, proton_diff_flux_east, proton_int_flux[:, 0], proton_int_flux[:, 1], gci_fov_mean_full, gci_fov_mean_dt)
    
    # scale goes data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    ridge_cv, r2, residuals = fit_gci_to_goes(X_scaled, y_clean)
    y_pred = ridge_cv.predict(X_scaled)

    # Convert goes time time to numeric for interpolation
    goes_t_numeric = dt_to_numeric(
        np.array(goes_time, dtype='datetime64[ns]'))
    
    # Interpolate GOES data to 10 second cadence for visualization of model performace for SCI images anytime between dark image capture times
    t_10s_numeric = np.arange(goes_t_numeric[0], goes_t_numeric[-1], 10.0)

    # Interpolate all 28 channels onto the 10 second grid
    west_10s = np.column_stack([
        interp_channel(proton_diff_flux_west[:, i], goes_t_numeric, t_10s_numeric)
        for i in range(NUM_GOES_DIFF_CHANNELS)
    ])
    east_10s = np.column_stack([
        interp_channel(proton_diff_flux_east[:, i], goes_t_numeric, t_10s_numeric)
        for i in range(NUM_GOES_DIFF_CHANNELS)
    ])
    p11_west_10s = interp_channel(proton_int_flux[:, 0], goes_t_numeric, t_10s_numeric).reshape(-1, 1)
    p11_east_10s = interp_channel(proton_int_flux[:, 1], goes_t_numeric, t_10s_numeric).reshape(-1, 1)

    # Convert 10s numeric timestamps back to datetime64 for plotting
    epoch = np.datetime64(0, 'ns')
    goes_time_10s = epoch + (t_10s_numeric * 1e9).astype('timedelta64[ns]')

    X_clean_vis, _ , goes_time_clean = interp_and_clean_xy(goes_time_10s, west_10s, east_10s, p11_west_10s, p11_east_10s, p11_east_10s, goes_time_10s) # using dummy y vars for vis
    
    # scale goes data with previously defined scaler (fit to original goes dataset)
    X_scaled_vis = scaler.transform(X_clean_vis)
    
    y_pred_vis  = ridge_cv.predict(X_scaled_vis)
    best_alpha = ridge_cv.alpha_

    # Create labels for plotting X
    chan_labels_short = [f'P{["1","2A","2B","3","4","5","6","7","8A","8B","8C","9","10"][i]}' for i in range(NUM_GOES_DIFF_CHANNELS)]
    feature_labels = (
        [f'W-{lbl}' for lbl in chan_labels_short] +
        [f'E-{lbl}' for lbl in chan_labels_short] +
        ['W-P11', 'E-P11']
    )

    # Plot GOES data and FOVS
    plot_goes_data(goes_time, proton_diff_flux_west, proton_diff_flux_east, proton_int_flux, int_12mev, int_12mev_thresh, gci_fov_mean_dt, gci_fov_mean_full, gci_fov_mean_dt_og, gci_fov_mean_full_og)


# ----- Create Ridge diagnostic plots -----
    fig_ridge, axes = plt.subplots(3, 1, figsize=(12, 10), layout='constrained')

    # 1. Observed vs. predicted time series
    ax = axes[0]
    ax.scatter(t_clean, y_clean, color='steelblue', label=f'Observed {channel} FOV Mean', s=30, zorder=3)
    ax.plot(goes_time_clean, y_pred_vis,  color='tomato',    label='Ridge Predicted', lw=2, ls='-')
    ax.set_ylabel(r'DN sec$^{-1}$ pixel$^{-1}$')
    ax.set_title(f'{channel} FOV Mean — Observed vs. Ridge Predicted')
    ax.xaxis.set_minor_locator(mdates.HourLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%m/%d\n%Y'))
    ax.legend(prop={'size': 10})
    ax.tick_params(labelbottom=True)

    # 2. Observed vs predicted scatter plot
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



# Run line below to run a storm-time regression on the January SEP event, plot GOES data, plot MCP FOV avgs, and plot model results
#plot_model_results('NFI')
# %%
