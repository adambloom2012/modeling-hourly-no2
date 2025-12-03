import xarray as xr
import os
import numpy as np
import pandas as pd

data_dir = '/Volumes/External/TEMPO'
output_dir = '/Volumes/External/projects/hourly-us-no2/data/sentinel-5p'


def resample(ds):
    data_variable = ds['tropospheric_NO2_column_number_density']

    # 2. Define the new, finer 120x120 grid
    new_lat = np.linspace(ds['latitude'].min().values,
                          ds['latitude'].max().values, 120)
    new_lon = np.linspace(ds['longitude'].min().values,
                          ds['longitude'].max().values, 120)
    # 3. Perform the linear interpolation
    # The method='linear' is the default, but it's good practice to be explicit
    interpolated_data = data_variable.interp(
        latitude=new_lat, longitude=new_lon, method='linear')

    # 4. (Optional) Create a new xarray Dataset for the interpolated data
    interpolated_ds = xr.Dataset(
        {'tropospheric_NO2_column_number_density': interpolated_data})
    return interpolated_ds


def hourly_mean(data):
    data_sorted = data.sortby('time')
    hourly_mean = data_sorted.resample(
        time='1h').mean().dropna(dim='time', how='all')
    return hourly_mean


def read_and_concat_netcdfs(dir):
    file_list = []
    for file in os.listdir(dir):
        if file.endswith('.nc'):
            file_path = os.path.join(dir, file)
            try:
                ds = xr.open_dataset(file_path)
                file_list.append(ds)
            except Exception as e:
                print(f"Error opening {file_path}: {str(e)}")
                continue
    if file_list:
        combined_ds = xr.concat(file_list, dim='time')
        return combined_ds
    else:
        return None


for file in os.listdir(data_dir):
    station_name = str(file)
    print(f"{station_name}")
    station_dir = f"{data_dir}/{station_name}"
    try:
        ds = read_and_concat_netcdfs(station_dir)
        if ds is None:
            print(f"No valid netcdf files found in {station_dir}")
            continue
        # rename data variable to tropospheric_NO2_column_number_density
        ds = ds.rename(
            {'vertical_column_troposphere': 'tropospheric_NO2_column_number_density'})
        # if length is 0 skip
        if len(ds['time']) == 0:
            print(
                f"Skipping {station_name} due to zero length time dimension")
            continue
    except Exception as e:
        print(f"Error processing {station_name}: {str(e)}")
        continue
    ds_resampled = hourly_mean(ds)
    ds_hourly_resampled = resample(ds_resampled)
    # remove any nan values
    ds_hourly_resampled = ds_hourly_resampled.dropna(dim='time', how='all')
    os.makedirs(f"{output_dir}/{station_name}", exist_ok=True)
    ds_hourly_resampled.to_netcdf(
        f"{output_dir}/{station_name}/{station_name}.netcdf"
    )


# phase 2


# Create dataframe with all ground truth observations
df = pd.read_csv('/Volumes/External/data/aqs/hourly_42602_2024.csv')

# Create datetime and formatting columns
df['datetime'] = pd.to_datetime(
    df['Date GMT'] + ' ' + df['Time GMT'])
df['date_str'] = (df['datetime'].dt.month.astype(str) + '-' +
                  df['datetime'].dt.day.astype(str) + '-' +
                  df['datetime'].dt.hour.astype(str))
df['id'] = (df['State Code'].astype(str) + '_' +
            df['County Code'].astype(str) + '_' + df['Site Num'].astype(str))

# Initialize list to collect all ground truth data
all_ground_truth = []

# Iterate through all tempo files
for station_dir in os.listdir(output_dir):
    station_path = os.path.join(output_dir, station_dir)
    if not os.path.isdir(station_path):
        continue

    netcdf_file = os.path.join(station_path, f"{station_dir}.netcdf")
    if not os.path.exists(netcdf_file):
        continue

    try:
        # Load tempo data for this station
        ds = xr.open_dataset(netcdf_file)
        # verify exclusion of all NaN values
        ds = ds.dropna(dim='time', how='all')

        # Get available time values for this station
        hourly_date_vals = np.array([str(dt.month) + "-" + str(dt.day) + "-" + str(dt.hour)
                                     for dt in pd.to_datetime(ds.time.values)])

        # Filter ground truth data for this station and available times
        station_data = df[
            df['date_str'].isin(hourly_date_vals) &
            (df['id'] == station_dir)
        ].copy()

        if len(station_data) > 0:
            # Add tempo path
            station_data['s5p_path'] = f'{station_dir}/{station_dir}.netcdf'

            # Add image path (assuming similar structure)
            station_data['img_path'] = f'{station_dir}/{station_dir}.npy'

            station_data['AirQualityStation'] = station_dir

            # Select relevant columns
            station_data = station_data[['date_str', 'Sample Measurement', 'Latitude',
                                         'Longitude', 'AirQualityStation',
                                         's5p_path', 'img_path']]
            # rename Sample Measurement to no2
            station_data = station_data.rename(
                columns={'Sample Measurement': 'no2'})

            all_ground_truth.append(station_data)
            print(
                f"Added {len(station_data)} records for station {station_dir}")

    except Exception as e:
        print(f"Error processing station {station_dir}: {str(e)}")
        continue

# Combine all ground truth data
if all_ground_truth:
    final_ground_truth = pd.concat(all_ground_truth, ignore_index=True)
    final_ground_truth.reset_index(drop=True, inplace=True)
    # rename index to idx
    final_ground_truth.index.name = 'idx'
    # final_ground_truth.to_csv('/opt/projects/Global-NO2-Estimation/data/samples_hourly_data.csv')
    print(f"Saved {len(final_ground_truth)} total ground truth records")
else:
    print("No ground truth data found")


final_ground_truth.to_csv(
    '/opt/projects/Global-NO2-Estimation/data/samples_S2S5P_hourly_data.csv', index=True)

# idx,date_str,no2,Latitude,Longitude,State Code,County Code,AirQualityStation,s5p_path,img_path
# idx	date_str	no2	Latitude	Longitude	State Code	County Code	Site Num	s5p_path	img_path


ds = xr.open_dataset(
    '/Volumes/External/projects/hourly-us-no2/data/sentinel-5p/11_1_41/11_1_41.netcdf'
)

ds
has_nan = ds['tropospheric_NO2_column_number_density'].isnull().any()
print(f"Has NaN values: {has_nan}")
nan_only_ds = ds.where(ds['tropospheric_NO2_column_number_density'].isnull())
only_has_data = ds.where(
    ~ds['tropospheric_NO2_column_number_density'].isnull())
check = only_has_data.where(
    only_has_data['tropospheric_NO2_column_number_density'].isnull())
check

nan_only_ds

# drop all nan values
ds = ds.dropna(dim='time', how='all')

nan_only_ds

# remove timestamps where any tropospheric_NO2_column_number_density is nan
ds_cleaned = ds.dropna(dim='time', how='any')
ds_cleaned


ds
ds_cleaned
# check if ds_cleaned has any nan values
has_nan_cleaned = ds_cleaned['tropospheric_NO2_column_number_density'].isnull(
).any()
print(f"Has NaN values after cleaning: {has_nan_cleaned}")


check = xr.open_dataset(
    '/Volumes/External/projects/hourly-us-no2/data/sentinel-5p/1_73_23/1_73_23.netcdf'
)
check = check.dropna(dim='time', how='all')
check
