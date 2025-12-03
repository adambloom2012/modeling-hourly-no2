import cartopy.crs as ccrs
import earthaccess
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import rcParams
import pandas as pd
import os

auth = earthaccess.login()

if not auth.authenticated:
    # Ask for credentials and persist them in a .netrc file.
    auth.login(strategy="interactive", persist=True)

print(earthaccess.__version__)
results = earthaccess.search_data(
    # TEMPO NO₂ Level-3 product
    short_name="TEMPO_NO2_L3",
    # Version 3 of the data product
    version="V03",
    # first half of may 2024
    temporal=("2024-05-01 12:00", "2024-05-15 23:00"),
)

print(f"Number of granules found: {len(results)}")

open_options = {
    "access": "indirect",  # access to cloud data (faster in AWS with "direct")
    "load": True,  # Load metadata immediately (required for indexing)
    "concat_dim": "time",  # Concatenate files along the time dimension
    "data_vars": "minimal",  # Only load data variables that include the concat_dim
    "coords": "minimal",  # Only load coordinate variables that include the concat_dim
    "compat": "override",  # Avoid coordinate conflicts by picking the first
    "combine_attrs": "override",  # Avoid attribute conflicts by picking the first
}
result_root = earthaccess.open_virtual_mfdataset(
    granules=results, **open_options)
result_product = earthaccess.open_virtual_mfdataset(
    granules=results, group="product", **open_options
)
result_geolocation = earthaccess.open_virtual_mfdataset(
    granules=results, group="geolocation", **open_options
)
# merge
result_merged = xr.merge([result_root, result_product, result_geolocation])
# filter out bad data


df = pd.read_csv("/Volumes/External/data/aqs/hourly_42602_2024.csv")
df['site_id'] = df['State Code'].astype(
    str) + '_' + df['County Code'].astype(str) + '_' + df['Site Num'].astype(str)
df = df[['Latitude', 'Longitude', 'Site Num',
         'State Code', 'County Code', 'site_id']]
df = df.drop_duplicates()
# selected_state = 56
# df = df[df['State Code'] == selected_state]

# df = df.iloc[1:]
# len(df)
# # get row 152 of df
# df = df.iloc[153:]
# df.to_csv('/tmp/df_sites.csv', index=False)
# Error processing site 6_19_242: NetCDF: HDF error
result_merged

for lat, lon, site_id in zip(df['Latitude'], df['Longitude'], df['site_id']):
    try:
        print(f"Processing site {site_id} at lat {lat}, lon {lon}")
        # Subset to a small region around the POI
        lat_margin = 0.1
        lon_margin = 0.1
        ds = result_merged.sel(
            latitude=slice(lat - lat_margin, lat + lat_margin),
            longitude=slice(lon - lon_margin, lon + lon_margin),
        )
        print(f"Data shape before filtering: {ds.dims}")

        quality_mask = ds["main_data_quality_flag"] == 0
        ds = ds.where(quality_mask.compute(), drop=True)

        # only good data
        print(f"Data shape after filtering: {ds.dims}")
        data = ds["vertical_column_troposphere"]

        os.makedirs(f"/Volumes/External/TEMPO/{site_id}", exist_ok=True)
        # save data to netcdf
        data.to_netcdf(
            f"/Volumes/External/TEMPO/{site_id}/site_{site_id}_may1_no2.nc")

        print(f"Successfully processed site {site_id}")

    except Exception as e:
        print(f"Error processing site {site_id}: {str(e)}")
        continue

# testing here
lat = 34.19925
lon = -118.53276
lat_margin = 0.1
lon_margin = 0.1
ds = result_merged.sel(
    latitude=slice(lat - lat_margin, lat + lat_margin),
    longitude=slice(lon - lon_margin, lon + lon_margin),
)

ds = ds['vertical_column_troposphere']
# save ds to file in /tmp
ds.to_netcdf('/tmp/test_tempo_january_no2.nc')

print(f"Data shape: {ds.shape}")
print(f"Data size in memory: {ds.nbytes / 1024**2:.2f} MB")
print(f"Data type: {ds.dtype}")
print(f"Dimensions: {ds.dims}")

# testing loading
test_ds = xr.open_dataset(
    '/Volumes/External/TEMPO/30_31_17/site_30_31_17_january_no2.nc')
test_ds_hourly = test_ds.resample(time='1H').mean()
# drop nas
test_ds_hourly = test_ds_hourly.dropna('time', how='all')
test_ds_hourly
