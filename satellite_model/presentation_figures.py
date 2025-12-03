from sqlalchemy import create_engine
import sqlalchemy
from shapely.geometry import Point
import matplotlib.colors as mcolors
import geopandas as gpd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.isotonic import IsotonicRegression
from scipy.stats import gaussian_kde
from scipy.stats import linregress
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

X_FONTSIZE = 20
Y_FONTSIZE = 20
TITLE_FONTSIZE = 24
TICK_FONTSIZE = 16

# create a df and add header 'station'
test_stations = pd.read_csv(
    '/Users/adambloom/Downloads/stations_test.txt', header=None)
test_stations.columns = ['station']
train_stations = pd.read_csv(
    '/Users/adambloom/Downloads/stations_train.txt', header=None)
train_stations.columns = ['station']
val_stations = pd.read_csv(
    '/Users/adambloom/Downloads/stations_val.txt', header=None)
val_stations.columns = ['station']

# produce final df
df = pd.read_csv(
    '/Users/adambloom/Downloads/mc_dropout_results_final_results_2.csv')

aqs_df = pd.read_csv(
    '/Volumes/External/data/aqs/hourly_42602_2024.csv')
# site id is state code county code site number
aqs_df['site_id'] = aqs_df['State Code'].astype(str) + '_' + \
    aqs_df['County Code'].astype(str) + '_' + \
    aqs_df['Site Num'].astype(str)

# get side id, county name, state name from aqs_df
site_info = aqs_df[['site_id', 'County Name', 'State Name',
                    'Latitude', 'Longitude']].drop_duplicates()

all_measures = pd.read_csv(
    '../data/samples_S2S5P_hourly_data_with_pop_density.csv')

stations_info = all_measures[['AirQualityStation',
                              'Latitude', 'Longitude', 'PopulationDensity', 'LocationType']].drop_duplicates()
# add group column to each station set
test_stations['group'] = 'test'
train_stations['group'] = 'train'
val_stations['group'] = 'val'

# all_stations = merge site_info and each test, train and val stations
all_stations = pd.concat(
    [test_stations, train_stations, val_stations]).drop_duplicates()
all_stations
all_stations = all_stations.merge(
    site_info[['site_id', 'County Name', 'State Name']], left_on='station', right_on='site_id', how='left')
all_stations.to_csv('/tmp/stations_with_info.csv')
# join population data from station_stats on to df
df = df.merge(
    stations_info, left_on='station', right_on='AirQualityStation', how='left')

df = df.merge(
    site_info[['site_id', 'County Name', 'State Name']], left_on='station', right_on='site_id', how='left')

df['day'] = df['date_str'].apply(lambda x: '-'.join(x.split('-')[:2]))
df['month'] = df['date_str'].apply(lambda x: '-'.join(x.split('-')[:1]))
df_daily = df.groupby(['day', 'station']).agg({
    'measurement': 'mean',
    'prediction': 'mean',
    'prediction_dropout': 'mean',
    'uncertainty_dropout': 'mean'
}).reset_index()

df_monthly = df.groupby(['month', 'station']).agg({
    'measurement': 'mean',
    'prediction': 'mean',
    'prediction_dropout': 'mean',
    'uncertainty_dropout': 'mean'
}).reset_index()
df.to_csv('/tmp/project_output.csv')
df
# FIGURE BREAK OVERALL CORRELATION
plt.figure(figsize=(10, 8))
sns.set_theme(style="whitegrid")
x = df['measurement']
y = df['prediction']
# Calculate point density using gaussian_kde
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
plt.scatter(x, y, c=z, s=20, cmap='Blues')
# Fit line
plt.plot(x, x, color='green', linestyle='--', label='y=x line')
plt.xlim(0, 40)
plt.ylim(0, 40)
plt.xlabel('Measurement (ug/m³)', fontsize=X_FONTSIZE)
plt.ylabel('Prediction (ug/m³)', fontsize=Y_FONTSIZE)
plt.title('Hourly Predictions over CONUS', fontsize=TITLE_FONTSIZE)
plt.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
# add r2, mae and mse
slope, intercept, r_value, p_value, std_err = linregress(x, y)
# hardcoding values I got from test run instead of the recalcualted
r2 = 0.37
mae = 4.20
mse = 38.37
stats_text = f'R² = {r2:.2f}\nMAE = {mae:.2f} ug/m³\nMSE = {mse:.2f} (ug/m³)²'
plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=14,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
plt.legend(fontsize=14)
plt.show()

# FIGURE BREAK Daily
plt.figure(figsize=(10, 8))
sns.set_theme(style="whitegrid")
x = df_daily['measurement']
y = df_daily['prediction']
# Calculate point density using gaussian_kde
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
plt.scatter(x, y, c=z, s=20, cmap='Blues')
# Fit line
# Plot y=x line
plt.plot(x, x, color='green', linestyle='--', label='y=x line')
plt.xlim(0, 40)
plt.ylim(0, 40)
plt.xlabel('Measurement (ug/m³)', fontsize=X_FONTSIZE)
plt.ylabel('Prediction (ug/m³)', fontsize=Y_FONTSIZE)
plt.title('Daily Aggregation over CONUS', fontsize=TITLE_FONTSIZE)
plt.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
# add r2, mae and mse
slope, intercept, r_value, p_value, std_err = linregress(x, y)
r2 = round(r2_score(x, y), 2)
mae = round(mean_absolute_error(x, y), 2)
mse = round(mean_squared_error(x, y), 2)
stats_text = f'R² = {r2:.2f}\nMAE = {mae:.2f} ug/m³\nMSE = {mse:.2f} (ug/m³)²'
plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=14,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
plt.legend(fontsize=14)
plt.show()

# FIGURE BREAK Monthly
plt.figure(figsize=(10, 8))
sns.set_theme(style="whitegrid")
x = df_monthly['measurement']
y = df_monthly['prediction']
# Calculate point density using gaussian_kde
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
plt.scatter(x, y, c=z, s=20, cmap='Blues')
# Fit line
plt.plot(x, x, color='green', linestyle='--', label='y=x line')
plt.xlim(0, 40)
plt.ylim(0, 40)
plt.xlabel('Measurement (ug/m³)', fontsize=X_FONTSIZE)
plt.ylabel('Prediction (ug/m³)', fontsize=Y_FONTSIZE)
plt.title('Monthly Aggregation over CONUS', fontsize=TITLE_FONTSIZE)
plt.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
# add r2, mae and mse
slope, intercept, r_value, p_value, std_err = linregress(x, y)
r2 = round(r2_score(x, y), 2)
mae = round(mean_absolute_error(x, y), 2)
mse = round(mean_squared_error(x, y), 2)
stats_text = f'R² = {r2:.2f}\nMAE = {mae:.2f} ug/m³\nMSE = {mse:.2f} (ug/m³)²'
plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=14,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
plt.legend(fontsize=14)
plt.show()


# FIGURE BREAK mean bias map


df['bias'] = df['prediction'] - df['measurement']

# Group by Station to get the MEAN Bias per location
station_stats = df.groupby('station').agg({
    'bias': 'mean',
    'measurement': 'count'  # Optional: track sample count
}).reset_index()

# Merge with your location data
# Ensure stations_info is unique per station
stations_info = stations_info.drop_duplicates(subset=['AirQualityStation'])
gdf_data = station_stats.merge(
    stations_info[['AirQualityStation', 'Latitude', 'Longitude']],
    left_on='station',
    right_on='AirQualityStation',
    how='left'
)

# Convert to a GeoDataFrame
gdf = gpd.GeoDataFrame(
    gdf_data,
    geometry=gpd.points_from_xy(gdf_data.Longitude, gdf_data.Latitude)
)

# ---------------------------------------------------------
# 2. VISUALIZATION SETUP
# ---------------------------------------------------------

# Set up the plot figure
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
fig.patch.set_facecolor('white')

# Load standard background map (World) and filter for USA
# UPDATED: Loading directly from Natural Earth URL since gpd.datasets is deprecated
natural_earth_url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
world = gpd.read_file(natural_earth_url)

usa = world[world.NAME == "United States of America"]

# Load US states
states_url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_1_states_provinces_lakes.zip"
states = gpd.read_file(states_url)
us_states = states[states.admin == "United States of America"]

# Plot the USA background
usa.plot(ax=ax, color='#f0f0f0', edgecolor='#bdbdbd', linewidth=1)

# Plot US state boundaries
us_states.plot(ax=ax, color='none', edgecolor='#757575', linewidth=0.8)

# ---------------------------------------------------------
# 3. ADVANCED COLOR SCALING (CRITICAL)
# ---------------------------------------------------------

# We want 0.0 to be White.
# We want the extremes to be bounded so one huge outlier doesn't wash out the map.
# Let's use the 5th and 95th percentiles to set the range.
vmin = np.percentile(gdf['bias'], 5)
vmax = np.percentile(gdf['bias'], 95)

# Make the range symmetric for a balanced look (optional, but looks nice)
limit = max(abs(vmin), abs(vmax))
vmin, vmax = -limit, limit

# Create a normalization that centers White at exactly 0
norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)

# ---------------------------------------------------------
# 4. PLOT THE STATIONS
# ---------------------------------------------------------

# Plot the points
plot = gdf.plot(
    ax=ax,
    column='bias',
    cmap='coolwarm',  # Blue (Low) -> White (Zero) -> Red (High)
    norm=norm,
    markersize=60,
    edgecolor='black',  # Adds a crisp border to each dot
    linewidth=0.5,
    alpha=0.9
)

# ---------------------------------------------------------
# 5. FORMATTING & AESTHETICS
# ---------------------------------------------------------

# Zoom in on Continental US (CONUS)
ax.set_xlim([-125, -66])
ax.set_ylim([24, 50])

# Add a custom Colorbar
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
sm._A = []  # Dummy array for the scalar mappable
cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.02)
cbar.set_label('Mean Prediction Bias (NO$_2$ ug/m³)\n(Blue = Underpredict, Red = Overpredict)',
               fontsize=10, weight='bold')

# Titles and Labels
plt.title('Spatial Distribution of Model Bias (Mean Bias)',
          fontsize=24, weight='bold', pad=15)
plt.xlabel('Longitude', fontsize=15)
plt.ylabel('Latitude', fontsize=15)

# Add a faint grid
ax.grid(True, linestyle='--', alpha=0.3)
# Show
plt.tight_layout()
plt.show()


# FIGURE BREAK state_specific
state = 'Arizona'
state_df = df[df['State Name'] == state]
# state in IN ('Colorado', 'California', 'Utah', 'Arizona')
list_states = ['Colorado', 'California',
               'Utah', 'Arizona', 'Nevada', 'Washington']
state_df = df[df['State Name'].isin(list_states)]
len(state_df)

plt.figure(figsize=(10, 8))
sns.set_theme(style="whitegrid")
x = state_df['measurement']
y = state_df['prediction']
# Calculate point density using gaussian_kde
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
plt.scatter(x, y, c=z, s=40, cmap='Blues', alpha=0.8,
            edgecolors='black', linewidth=0.2)
# Fit line
# Plot y=x line
plt.plot(x, x, color='green', linestyle='--', linewidth=2, label='y=x line')
plt.xlim(0, 40)
plt.ylim(0, 40)
plt.xlabel('Measurement (ug/m³)', fontsize=X_FONTSIZE)
plt.ylabel('Prediction (ug/m³)', fontsize=Y_FONTSIZE)
plt.title(
    f'Hourly Predictions over West Coast Region', fontsize=TITLE_FONTSIZE)
plt.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
# add r2, mae and mse
slope, intercept, r_value, p_value, std_err = linregress(x, y)
# hardcoding values I got from test run instead of the recalcualted
r2 = round(r2_score(x, y), 2)
mae = round(mean_absolute_error(x, y), 2)
mse = round(mean_squared_error(x, y), 2)
stats_text = f'R² = {r2:.2f}\nMAE = {mae:.2f} ug/m³\nMSE = {mse:.2f} (ug/m³)²'
plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=14,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
plt.legend(fontsize=14)
plt.show()


# FIGURE BREAK

# local pgres con
con = create_engine('postgresql://adambloom:@localhost:5432/adambloom')
query = '''SELECT DISTINCT 
    split_part(date_str, '-', 1)::INT AS month ,
    split_part(date_str, '-', 2)::INT AS day ,
    split_part(date_str, '-', 3)::INT AS hour ,
    date_str,
    measurement,
    prediction
FROM test.project_output
WHERE station IN ('4_19_1011')
    AND split_part(date_str, '-', 1)::INT = 2
ORDER BY month ASC , day ASC , hour ASC'''

station_df = pd.read_sql_query(query, con)
station_df

# Create a normalized time index for overlapping (day * 24 + hour)
station_df['time_index'] = (station_df['day'] - 1) * 24 + station_df['hour']

# Create a single plot
plt.figure(figsize=(16, 8))

# Plot measurement and prediction for January only
plt.plot(station_df['time_index'], station_df['measurement'],
         color='#1f77b4', linewidth=2, alpha=0.8, label='February Measurement')
plt.plot(station_df['time_index'], station_df['prediction'],
         color='#1f77b4', linewidth=2, alpha=0.8, linestyle='--', label='February Prediction')

plt.xlabel('Hours from Start of Week', fontsize=X_FONTSIZE)
plt.ylabel('NO₂ (ug/m³)', fontsize=Y_FONTSIZE)
plt.title('First Week Pattern - February - Station 4_19_1011',
          fontsize=TITLE_FONTSIZE)
plt.legend(loc='upper right', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
plt.xlim(0, 7*24)  # 7 days * 24 hours
plt.tight_layout()
plt.show()
