import xarray as xr
import matplotlib.pyplot as gplt

filename = 'data/WFI_1A-DRK/CARRUTHERS_GCI-WFI_L1A-DRK_20260119_v1.0.nc'
ds = xr.open_dataset(filename)
images = ds["images"]

gplt.imshow(images[0,:,:], vmin=0, vmax=250000/10)