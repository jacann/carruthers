# %%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from glide.common_components.utils import circular_mask, mask_average
from glide.common_components import constants

filepath = '/home/jacob/products/L1A/CARRUTHERS_GCI-WFI_L1A-DRK_20251106_v1.0.nc'
imager = 'WFI'

half_npix = 256
top_col_biases = np.load('/home/jacob/carruthers/products/column_bias_top_v2.npy')
bottom_col_biases = np.load('/home/jacob/carruthers/products/column_bias_bottom_v2.npy')
npix = constants.NPIX[imager]
fov_radius = constants.MASK_L1A_FOV_R[imager]
mask_fov = circular_mask(npix, fov_radius)

plt.plot(top_col_biases, label='Top Column Biases')
plt.plot(bottom_col_biases, label='Bottom Column Biases')
plt.xlabel('Column')
plt.ylabel('Bias')
plt.title('Column Biases for Top and Bottom Halves')
plt.legend()
plt.show()

ds = xr.open_dataset(filepath)

# Load data from given dataset
images = ds["images"].values.copy()
n_frames = ds["n_frames"].values
t_int = ds["t_int"].values
time = ds["time"]
file_id = ds.attrs["Logical_file_id"]
ds.close()

# Notify file processing
print(f"Processing file: {file_id}")

# normalize image array
images = images / n_frames[:, np.newaxis, np.newaxis]
top_correction = n_frames[:, np.newaxis, np.newaxis] * top_col_biases[np.newaxis, np.newaxis, :]
bottom_correction = n_frames[:, np.newaxis, np.newaxis] * bottom_col_biases[np.newaxis, np.newaxis, :]

# Subtract voltage biases

vvmin = -0.5
vvmax = 1.5

plt.imshow(images[0], 
           vmin=vvmin, 
           vmax=vvmax)
plt.colorbar()
plt.title("Before Bias Subtraction")
plt.show()

top_correction =  top_col_biases[np.newaxis, np.newaxis, :]
bottom_correction =  bottom_col_biases[np.newaxis, np.newaxis, :]
    
images[:, :half_npix, :] -= top_correction
images[:, half_npix:, :] -= bottom_correction

plt.imshow(images[0], 
           vmin=vvmin, 
           vmax=vvmax)
plt.colorbar()
plt.title("After Bias Subtraction")
plt.show()

# Calculate mean FOV radiation and images with non-fov area set to NaN
mcp_rad, mcp_fov = mask_average(images, mask_fov, t_int)

# %%
old_top_biases = np.load('/home/jacob/carruthers/products/column_bias_top.npy')
new_top_biases = np.load('/home/jacob/carruthers/products/column_bias_top_v2.npy')

plt.scatter(range(len(old_top_biases)), old_top_biases, label='Old Top Column Biases', alpha=0.5)
plt.scatter(range(len(new_top_biases)), new_top_biases, label='New Top Column Biases', alpha=0.5)
plt.xlabel('Column')
plt.ylabel('Bias')
plt.title('Comparison of Old and New Top Column Biases')
plt.legend()
plt.show()

plt.plot(old_top_biases - new_top_biases, label='Old Top Column Biases')
plt.show()
# %%
