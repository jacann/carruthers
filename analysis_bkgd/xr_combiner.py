import glob
import xarray as xr



channel = 'WFI'


def combine():
    file_list = sorted(glob.glob("/carrdata/L1A/CARRUTHERS_GCI-" + channel + "_L1A-DRK" + "**" + "v1.0.nc"))

    ds = xr.open_mfdataset(
        file_list,
        engine='netcdf4',
        combine="nested",       
        concat_dim="time",      
        parallel=True,        
        compat="override",    
        coords="minimal",     
        data_vars="minimal",   
        chunks={"time": -1}
    )
    ds['cam_id'] = ds['cam_id'].astype(str)
    ds.to_netcdf(f'{channel}_L1A-DRK_ALL.nc')
    ds.close()

def load():
    with xr.open_dataset(f'{channel}_L1A-DRK_ALL.nc') as ds:
        print(ds)


combine()