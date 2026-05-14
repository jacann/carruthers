# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glide.science_data_processing.L1A as L1A
import glide.common_components.view_geometry as view_geometry
from glide.common_components.stars import get_beta_angle

# 1. Load the dataset and initialize the L1A object
file_path = '/home/jacob/products/L1A/CARRUTHERS_GCI-WFI_L1A-DRK_20251004_v1.0.nc'
with xr.open_dataset(file_path, engine='netcdf4') as data:
    l1a_obj = L1A.L1A(data)

# 3. Get the roll angle 
roll_angles = [scraft.moc_roll for scraft in l1a_obj.scrafts]
# 4. Calculate the beta angle
# Get RA and Dec by projecting the boresight to the Star frame
ra_inputs, dec_inputs = zip(*(scraft.boresight_to_sky('WFI', view_geometry.Star_frame) for scraft in l1a_obj.scrafts))

# Use those outputs to retrieve the beta angle
beta_angle = [get_beta_angle(scraft, ra_inputs, dec_inputs) for scraft in l1a_obj.scrafts]
print(len(beta_angle))
print(l1a_obj.n_images)
# You can also access other items via the index
ind = 0
image = l1a_obj.images[ind]
n_frames = l1a_obj.n_frames[ind]
image = image/n_frames

min = np.min(image)
max = np.max(image)
print(f"Image min: {min}, Image max: {max}")


print(f"Image shape: {image.shape}")
print(image)

filter = l1a_obj.filters[ind]
print(f"Filter: {filter}")

# 90% max of image
max = np.percentile(image, 99)
min = np.percentile(image, 1)

# %%
plt.imshow(image, vmin=min, vmax=max)
plt.colorbar()
plt.show()

# %%
