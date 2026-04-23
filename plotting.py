import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def filter_time_range(data, datetimes, start_datetime, end_datetime):
    # convert ot numpy arrays
    datetimes = np.asanyarray(datetimes)
    data = np.asanyarray(data)

    # find start/end indices
    start_idx = np.searchsorted(datetimes, start_datetime, side='left')
    end_idx = np.searchsorted(datetimes, end_datetime, side='right')

    return data[start_idx:end_idx]

def filter_n_frames(data, n_frames, n_frames_min):

    n_frames_mask = n_frames >= n_frames_min
    if type(data) == np.float64:
        data[n_frames_mask == 0] = np.nan
    if type(data) == np.datetime64:
        data[n_frames_mask == 0] = np.datetime64('NaT')

    return data

def plot_data_vs_time(mcp_rad_top, mcp_rad_bottom, datetimes_filtered, n_frames_min):

    # Plot data
    fig, ax = plt.subplots()
    fig.set_size_inches((12, 8))
    ax.scatter(datetimes_filtered, mcp_rad_top, s=1, alpha=1, label='MCP Radiation (Top)')
    ax.scatter(datetimes_filtered, mcp_rad_bottom, s=1, alpha=1, label='MCP Radiation (Bottom)')

    ax.set_xlabel('Time')
    ax.set_ylabel(f'Mean MCP Radiation (DN)')
    ax.set_title(f'MCP Radiation vs. Time  |  n_frames >= {n_frames_min}')
    ax.legend()

    #ax.set_xticks([datetimes_filtered[0], datetimes_filtered[-1]])
    #weeks = mdates.WeekdayLocator()
    #ax.xaxis.set_major_locator(weeks)

    fig.autofmt_xdate()
    textstr = ''
    ax.text(0.042, 0.94, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top')
    ax.xaxis.set_minor_locator(mdates.HourLocator())
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
    ax.tick_params(which='minor', length=3)
    ax.tick_params(which='major', length=5)
    ax.set_xlim(datetimes_filtered[0], datetimes_filtered[-1])
    plt.yscale('log')
    ylims = ax.get_ylim()
    plt.ylim([ylims[0], 1.4 * ylims[1]])

    fig.show()
