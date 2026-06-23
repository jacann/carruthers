# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import glob

from glide.common_components.utils import mask_average
from glide.common_components.utils import circular_mask
from glide.common_components import constants
import glide.science_data_processing.L1A as L1A
from glide.calibration.bias_wavelet import wavelet_destripe



def get_annulus_median(image, annulus):
    # Construct annulus mask
    npix = image.shape[1]
    inner_mask = circular_mask(npix, annulus[0])
    outer_mask = circular_mask(npix, annulus[1])
    annulus_mask = outer_mask & ~inner_mask    
    # Apply mask to the image
    annulus_values = image[annulus_mask]
    annulus_median = np.median(annulus_values)
    
    if False:
        # show histogram of annulus values
        # for vissualizations 
        plt.hist(annulus_values, bins=250)
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of Annulus Pixel Values')
        # indicate median value on histogram
        median_value = np.median(annulus_values)
        plt.axvline(median_value, color='r', linestyle='dashed', linewidth
    =2, label=f'Median: {median_value:.2f}')
        plt.legend()
        plt.show()

    return annulus_median

def test_medians():
    imager = "NFI"
    img_idx = 1

    fpaths = "/data/carruthers/CARRUTHERS_GCI-NFI_L1A-DRK v1.0.nc"
    # process each file and append the median value in the annulus to a list using parrellel processing techniques

    for fpath in fpaths:
        print(f"Processing file: {fpath}")

    with xr.open_dataset(fpaths) as ds:
        l1a_obj = L1A.L1A(ds)


    for img_idx in range(len(l1a_obj.images)):
        image = l1a_obj.images[img_idx] / l1a_obj.n_frames[img_idx]  # Normalize by number of frames
        image = wavelet_destripe(image, log=True)

        outer_radius = constants.MASK_L1A_FOV_R[imager]
        inner_radius = outer_radius - 10

        median_value = get_annulus_median(image, (inner_radius, outer_radius))
        print(f"Median pixel value in the annulus: {median_value}")

def filter_filter(ds, lyan=True, lyax=False):
    if lyan and lyax:
        filter_mask = np.isin(ds["cam_filter"], ["LyaN", "LyaX"])
    elif lyan:
        filter_mask = (ds["cam_filter"] == ("LyaN"))
    elif lyax:
        filter_mask = (ds["cam_filter"] == ("LyaX"))
    else:
        return ds["annulus_median_full"], ds["time"]
    return ds["annulus_median_full"][filter_mask], ds["time"][filter_mask]
    

def main():
    with xr.open_dataset("products/WFI_DRK_ANNULUS_MEDIANS.nc") as ds:
        wfi_drk_ds = ds
    with xr.open_dataset("products/WFI_SCI_ANNULUS_MEDIANS.nc") as ds:
        wfi_sci_ds = ds
    with xr.open_dataset("products/NFI_DRK_ANNULUS_MEDIANS.nc") as ds:
        nfi_drk_ds = ds
    with xr.open_dataset("products/NFI_SCI_ANNULUS_MEDIANS.nc") as ds:
        nfi_sci_ds = ds

    # filter sci datasets
    wfi_sci_amf_lyan, wfi_sci_time_lyan = filter_filter(wfi_sci_ds, lyan=True, lyax=False)
    wfi_sci_amf_lyax, wfi_sci_time_lyax = filter_filter(wfi_sci_ds, lyan=False, lyax=True)
    nfi_sci_amf_lyan, nfi_sci_time_lyan = filter_filter(nfi_sci_ds, lyan=True, lyax=False)
    nfi_sci_amf_lyax, nfi_sci_time_lyax = filter_filter(nfi_sci_ds, lyan=False, lyax=True)

    # plot the top bottom and full medians for one ds
    plt.figure(figsize=(10, 6))
    plt.scatter(wfi_drk_ds["time"], wfi_drk_ds["annulus_median_full"], label="WFI SCI Full", s=2)
    plt.legend(markerscale=5)
    plt.show()

    # Create the primary axes explicitly
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Plot on primary y-axis (ax1)
    p1 = ax1.scatter(wfi_sci_time_lyan, wfi_sci_amf_lyan, label="WFI SCI LyaN", s=2, color="red", alpha=0.5)
    p1a = ax1.scatter(wfi_sci_time_lyax, wfi_sci_amf_lyax, label="WFI SCI LyaX", s=2, color="brown", alpha=0.5)
    p2 = ax1.scatter(nfi_sci_time_lyan, nfi_sci_amf_lyan, label="NFI SCI LyaN", s=2, color="green", alpha=0.5)
    p2a = ax1.scatter(nfi_sci_time_lyax, nfi_sci_amf_lyax, label="NFI SCI LyaX", s=2, color="darkolivegreen", alpha=0.5)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("SCI Annulus Median Pixel Value")  # Label for left axis

    # Create and plot on secondary y-axis (ax2)
    ax2 = ax1.twinx()
    p3 = ax2.scatter(wfi_drk_ds.time, wfi_drk_ds["annulus_median_full"], label="WFI DRK", s=2, color="orange", alpha=0.5)
    ax2.set_ylabel("WFI DRK Annulus Median Pixel Value")  # Label for right axis
    ax2.set_ylim(0,2.5)

    # Create and plot on a tertiary y-axis (ax3) by creating a new axes object
    ax3 = ax1.twinx()
    p4 = ax3.scatter(nfi_drk_ds.time, nfi_drk_ds["annulus_median_full"], label="NFI DRK", s=2, color="blue", alpha=0.5)
    ax3.set_ylabel("NFI DRK Annulus Median Pixel Value")  # Label for tertiary axis
    ax3.set_ylim(0,0.3)
    # Offset the tertiary axis to the right
    ax3.spines["right"].set_position(("outward", 60))  #

    # Combine all plots into a single legend
    plots = [p1, p1a, p2, p2a, p3, p4]
    labels = [p.get_label() for p in plots]
    ax1.legend(plots, labels, loc="lower right", markerscale=5)

    ax1.grid(True)  # Grid lines align to the primary axis
    plt.title("Annulus Median Pixel Value Over Time")
    plt.show()

     


if __name__ == "__main__":
    main()
# %%
