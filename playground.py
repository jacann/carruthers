import xarray as xr
import glide.science_data_processing.L1A as L1A
import glide.common_components.view_geometry as view_geometry
from glide.common_components.stars import get_beta_angle

# 1. Load the dataset and initialize the L1A object
file_path = 'data/WFI_1A-DRK/CARRUTHERS_GCI-WFI_L1A-DRK_20251007_v1.0.nc'
with xr.open_dataset(file_path, engine='netcdf4') as data:
    l1a_obj = L1A.L1A(data)

# 2. Access a spacecraft instance (e.g., the first one)
ind = 0
scraft = l1a_obj.scrafts[ind]

# 3. Get the roll angle
roll_angle = scraft.moc_roll

# 4. Calculate the beta angle
# Get RA and Dec by projecting the boresight to the Star frame
ra_input, dec_input = scraft.boresight_to_sky('WFI', view_geometry.Star_frame)

# Use those outputs to retrieve the beta angle
beta_angle = get_beta_angle(scraft, ra_input, dec_input)

# You can also access other items via the index
# image = l1a_obj.images[ind]
# filter = l1a_obj.filters[ind]


def main():

    print("Complete")


if __name__ == "__main__":
    main()

    '''
    scraft.moc_roll
    roll angle of spacecraft: scraft.moc_roll
    beta angle: scraft.boresight_to_sky('NFI', view_geometry.Star_frame) and then
    get_beta_angle(spacecraft, ra_input, dec_input) where ra_input and dec_input are the outputs of the boresight_to_sky function

    To load in a dataset and get everything loaded for you:
    import glide.science_data_processing.L1A as L1A
     with xr.open_dataset(file_path, engine='netcdf4') as data:
                    l1a_obj = L1A.L1A(data)
    Then you can do things like l1a_obj.scrafts[ind], or l1a_obj.images[ind], or l1a_obj.filters[ind], or etc.


    Then you can do things like l1a_obj.scrafts[ind], or l1a_obj.images[ind], or l1a_obj.filters[ind], or etc.
    '''
