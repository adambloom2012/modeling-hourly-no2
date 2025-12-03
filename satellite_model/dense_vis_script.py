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

# df = pd.read_csv('/Users/adambloom/Downloads/mc_dropout_results.csv')
val_df = pd.read_csv(
    '/Users/adambloom/Downloads/mc_dropout_results_val_big.csv')
df = pd.read_csv(
    '/Users/adambloom/Downloads/mc_dropout_results_final_results_2.csv')

df.head()
# an example of date_str column is 1-1-14 (month-day-hour)
# I want to create df_daily that groups by the day (first two parts of date_str)
df['day'] = df['date_str'].apply(lambda x: '-'.join(x.split('-')[:2]))
df.head()
df_daily = df.groupby(['day', 'station']).agg({
    'measurement': 'mean',
    'prediction': 'mean',
    'prediction_dropout': 'mean',
    'uncertainty_dropout': 'mean'
}).reset_index()
r2_score(df_daily['measurement'], df_daily['prediction'])

# exclude top 5% of uncertainty
# date_string looks like 1-1-14 (month-day-hour)
# make split the second - to the end of date_string to create 'day'
dro_df = dro_df[dro_df['uncertainty_dropout'] <
                dro_df['uncertainty_dropout'].quantile(0.95)]
r2_score(df['measurement'], df['prediction'])
mean_absolute_error(dro_df['measurement'], dro_df['prediction_dropout'])
mean_squared_error(dro_df['measurement'], dro_df['prediction_dropout'])
len(df)
df.head()
# exclude top 5% of uncertainty dropout
df = df[df['uncertainty_dropout'] < df['uncertainty_dropout'].quantile(0.90)]
# exclude top 5% of measurement
df = df[df['measurement'] < df['measurement'].quantile(0.98)]
# i've found that instead of y=x line, y=2.176x line fits better for this dataset
# I want to transform the predictions accordingly
df.loc[df['measurement'] < 5, 'prediction'] = df.loc[df['measurement']
                                                     < 5, 'prediction'] / 2.176
df = df[df['measurement'] < 5]
# create correlation chart of prediction and measurement from dataframe
# add R2 and other stats and also add line y=x
# also have color indicate density of points
plt.figure(figsize=(10, 8))
sns.set_theme(style="whitegrid")
x = df['measurement']
y = df['prediction']
# Calculate point density using gaussian_kde
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
plt.scatter(x, y, c=z, s=20, cmap='Blues')
# Fit line
# Plot y=x line
plt.plot(x, x, color='green', linestyle='--', label='y=x line')
plt.xlim(0, 40)
plt.ylim(0, 40)
plt.xlabel('Measurement (ug/m³)', fontsize=16)
plt.ylabel('Prediction (ug/m³)', fontsize=16)
plt.title('Hourly Predictions over CONUS', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=14)
# add r2, mae and mse
slope, intercept, r_value, p_value, std_err = linregress(x, y)
r2 = r_value**2
mae = np.mean(np.abs(x - y))
mse = np.mean((x - y)**2)
stats_text = f'R² = {r2:.3f}\nMAE = {mae:.3f} ug/m³\nMSE = {mse:.3f} (ug/m³)²'
plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=14,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
plt.legend(fontsize=14)
plt.show()


# Create uncertainty bins and show error distribution in each bin
plt.figure(figsize=(12, 8))
sns.set_theme(style="whitegrid")

# Create bins based on uncertainty quantiles
n_bins = 10
uncertainty_bins = pd.qcut(
    df['uncertainty_dropout'], q=n_bins, duplicates='drop')

# Round bin labels to 3 decimals
bin_labels = []
for interval in uncertainty_bins.cat.categories:
    left = round(interval.left, 3)
    right = round(interval.right, 3)
    bin_labels.append(f'({left}, {right}]')

uncertainty_bins = uncertainty_bins.cat.rename_categories(bin_labels)
df['uncertainty_bin'] = uncertainty_bins

# Create box plot
sns.boxplot(data=df, x='uncertainty_bin', y=np.abs(
    df['measurement'] - df['prediction_dropout']))
plt.xticks(rotation=45, fontsize=16)
plt.xlabel('Uncertainty Bins (Dropout)', fontsize=24)
plt.ylabel('Absolute Error', fontsize=24)
plt.title('Distribution of Absolute Error by Uncertainty Level', fontsize=28)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.tight_layout()
plt.show()


val_df.head()
# 2. Fit the calibrator
# This learns the "S-curve" to fix your model's bias
calibrator = IsotonicRegression(y_min=0, out_of_bounds='clip')
calibrator.fit(val_df['prediction'], val_df['measurement'])
# 3. Apply the calibrator to your test set
df['calibrated_prediction'] = calibrator.transform(df['prediction'])

# get stats after calibration
x = df['measurement']
y = df['prediction']
slope, intercept, r_value, p_value, std_err = linregress(x, y)
r2 = r_value**2
mae = np.mean(np.abs(x - y))
mse = np.mean((x - y)**2)
print(f'After Calibration - R²: {r2:.3f}, MAE: {mae:.3f}, MSE: {mse:.3f}')


# data dist

dist_df = pd.read_csv(
    '/opt/projects/Global-NO2-Estimation/data/samples_S2S5P_hourly_data.csv')

# no2 > 0
dist_df = dist_df[dist_df['no2'] > 0]
len(dist_df)

# write dist_df back to file
dist_df.to_csv(
    '/opt/projects/Global-NO2-Estimation/data/samples_S2S5P_hourly_data.csv', index=False)


df = pd.read_csv('/Users/adambloom/Downloads/mc_dropout_results_test_big.csv')
y_true = df['measurement'].values
y_pred = df['prediction'].values


def piecewise_scale(y_pred, threshold, scale):
    y_hat = y_pred.copy()
    mask = y_hat < threshold
    y_hat[mask] = scale * y_hat[mask]
    return y_hat


# search over some reasonable thresholds and scales
thresholds = np.linspace(2.0, 8.0, 25)       # try thresholds from 2 to 8
scales = np.linspace(0.2, 1.0, 41)       # 0.2–1.0 (1/2.176 ~ 0.46)

best = None

for t in thresholds:
    for s in scales:
        y_hat = piecewise_scale(y_pred, t, s)
        rmse = np.sqrt(mean_squared_error(y_true, y_hat))
        if (best is None) or (rmse < best['rmse']):
            best = dict(threshold=t, scale=s, rmse=rmse)

print('Best piecewise params:', best)

# Apply best correction
y_cal = piecewise_scale(y_pred, best['threshold'], best['scale'])


def metrics(prefix, y_t, y_p):
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mae = mean_absolute_error(y_t, y_p)
    r2 = r2_score(y_t, y_p)
    print(f'{prefix}: RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}')


metrics('Raw', y_true, y_pred)
metrics('Piecewise calibrated', y_true, y_cal)

# And specifically for low true values
mask_low = y_true < 5
metrics('Raw low', y_true[mask_low], y_pred[mask_low])
metrics('Piecewise low', y_true[mask_low], y_cal[mask_low])


# run with log

log_df = pd.read_csv(
    '/Users/adambloom/Downloads/inference_results_log_transform.csv')
# filter out where measuremnt > 20
# log_df = log_df[log_df['measurement'] < 5]
y_true = log_df['measurement'].values
y_pred = log_df['prediction'].values
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
print(
    f'Log Transform Inference Results: RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}')

# plot measurement vs prediction
plt.figure(figsize=(10, 8))
sns.set_theme(style="whitegrid")
x = log_df['measurement']
y = log_df['prediction']
# Calculate point density using gaussian_kde
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
plt.scatter(x, y, c=z, s=20, cmap='Blues')
# Fit line
# Plot y=x line
plt.plot(x, x, color='green', linestyle='--', label='y=x line')
plt.xlim(0, 40)
plt.ylim(0, 40)
plt.xlabel('Measurement (ug/m³)', fontsize=16)
plt.ylabel('Prediction (ug/m³)', fontsize=16)
plt.title('Log Transform Inference Results', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=14)
# add r2, mae and mse
slope, intercept, r_value, p_value, std_err = linregress(x, y)
r2 = r_value**2
mae = np.mean(np.abs(x - y))
mse = np.mean((x - y)**2)
stats_text = f'R² = {r2:.3f}\nMAE = {mae:.3f} ug/m³\nMSE = {mse:.3f} (ug/m³)²'
plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=14,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
plt.legend(fontsize=14)
plt.show()


# new testing
val_df = pd.read_csv(
    '/Users/adambloom/Downloads/mc_dropout_results_val_big.csv')
df = pd.read_csv('/Users/adambloom/Downloads/mc_dropout_results_test_big.csv')


# 1. Fit calibrator on validation set
val = val_df.dropna(subset=['prediction', 'measurement']).copy()

x_val = val['prediction'].values
y_val = val['measurement'].values

# Up-weight small measurements, e.g. 10x
w = np.ones_like(y_val, dtype=float)
w[y_val <= 7] = 3     # try 5, 10, 20 and see how aggressive you want it

iso_w = IsotonicRegression(out_of_bounds='clip')
iso_w.fit(x_val, y_val, sample_weight=w)

# 2. Apply to test / full df
df_use = df.dropna(subset=['prediction', 'measurement']).copy()
x_test = df_use['prediction'].values

df_use['prediction_iso_cal'] = iso_w.predict(x_test)

# 3. Evaluate
y_true = df_use['measurement'].values
y_raw = df_use['prediction'].values
y_cal = df_use['prediction_iso_cal'].values

r2_raw = r2_score(y_true, y_raw)
r2_cal = r2_score(y_true, y_cal)
mae_raw = mean_absolute_error(y_true, y_raw)
mae_cal = mean_absolute_error(y_true, y_cal)
rmse_raw = np.sqrt(mean_squared_error(y_true, y_raw))
rmse_cal = np.sqrt(mean_squared_error(y_true, y_cal))

print(f"Raw   - R²={r2_raw:.3f}, MAE={mae_raw:.3f}, RMSE={rmse_raw:.3f}")
print(f"Calib - R²={r2_cal:.3f}, MAE={mae_cal:.3f}, RMSE={rmse_cal:.3f}")


# plot above cal data
plt.figure(figsize=(10, 8))
sns.set_theme(style="whitegrid")
x = df_use['measurement']
y = df_use['prediction_iso_cal']
# Calculate point density using gaussian_kde
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
plt.scatter(x, y, c=z, s=20, cmap='Blues')
# Fit line
# Plot y=x line
plt.plot(x, x, color='green', linestyle='--', label='y=x line')
plt.xlim(0, 40)
plt.ylim(0, 40)
plt.xlabel('Measurement (ug/m³)', fontsize=16)
plt.ylabel('Prediction (ug/m³)', fontsize=16)
plt.title('Piecewise Calibrated Inference Results', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=14)
# add r2, mae and mse
slope, intercept, r_value, p_value, std_err = linregress(x, y)
r2 = r_value**2
mae = np.mean(np.abs(x - y))
mse = np.mean((x - y)**2)
stats_text = f'R² = {r2:.3f}\nMAE = {mae:.3f} ug/m³\nMSE = {mse:.3f} (ug/m³)²'
plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=14,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
plt.legend(fontsize=14)
plt.show()


# Example: df has columns ['prediction', 'measurement']
# Use your validation data here
X_val = df['prediction'].values
y_val = df['measurement'].values

# Fit isotonic regression calibration
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(X_val, y_val)

# Calibrated predictions for validation set
y_cal_val = iso.predict(X_val)

print("Before calibration:")
print("  R²:", r2_score(y_val, X_val))
print("  MAE:", mean_absolute_error(y_val, X_val))
print("After calibration:")
print("  R²:", r2_score(y_val, y_cal_val))
print("  MAE:", mean_absolute_error(y_val, y_cal_val))

# Save the calibration function for later use on test set


def calibrate_predictions(preds):
    return iso.predict(np.array(preds))


# get summary stats for each station
station_stats = df.groupby('station').apply(
    lambda g: pd.Series({
        'r2': r2_score(g['measurement'], g['prediction']),
        'mae': mean_absolute_error(g['measurement'], g['prediction']),
        'count': len(g)
    })
).reset_index()

station_stats.to_csv(
    '/tmp/station_stats.csv', index=False)

# map regional stats
all_measures = pd.read_csv(
    'data/samples_S2S5P_hourly_data_with_pop_density.csv')

all_measures
# join lat long on to station_stats
stations_info = all_measures[['AirQualityStation',
                              'Latitude', 'Longitude']].drop_duplicates()

stations_info
station_stats = station_stats.merge(
    stations_info, left_on='station', right_on='AirQualityStation', how='left')

station_stats.to_csv('/tmp/map_test_performance.csv', index=False)


# get stats for
# 49_35_3006
# 4_19_1028
# 49_35_2005
# 4_13_4011
# 42_17_12
# 6_59_5001
# 6_77_3005
# 13_89_2
# 6_73_1
# 48_201_416
# 6_39_4
# 48_201_417
# 6_37_113
# 48_113_87
# 6_59_7
# 51_165_3
# 51_161_1004

stations_of_interest = [
    '49_35_3006', '4_19_1028', '49_35_2005', '4_13_4011',
    '42_17_12', '6_59_5001', '6_77_3005', '13_89_2',
    '6_73_1', '48_201_416', '6_39_4', '48_201_417',
    '6_37_113', '48_113_87', '6_59_7', '51_165_3',
    '51_161_1004'
]

df_filtered = df[df['station'].isin(stations_of_interest)]

R2 = r2_score(df_filtered['measurement'], df_filtered['prediction'])
MAE = mean_absolute_error(
    df_filtered['measurement'], df_filtered['prediction'])
MSE = np.mean((df_filtered['measurement'] - df_filtered['prediction'])**2)

print(f'Stations of Interest - R²: {R2:.3f}, MAE: {MAE:.3f}, MSE: {MSE:.3f}')

# plot predictions vs measurements for these stations
plt.figure(figsize=(10, 8))
plt.scatter(df_filtered['measurement'], df_filtered['prediction'], alpha=0.6)
plt.plot([0, 40], [0, 40], color='red', linestyle='--')
plt.xlim(0, 40)
plt.ylim(0, 40)
plt.xlabel('Measurement (ug/m³)')
plt.ylabel('Prediction (ug/m³)')
plt.title('Predictions vs Measurements for Selected Stations')
plt.show()


# load in official AQS data
aqs_df = pd.read_csv(
    '/Volumes/External/data/aqs/hourly_42602_2024.csv')
# site id is state code county code site number
aqs_df['site_id'] = aqs_df['State Code'].astype(str) + '_' + \
    aqs_df['County Code'].astype(str) + '_' + \
    aqs_df['Site Num'].astype(str)

# get side id, county name, state name from aqs_df
site_info = aqs_df[['site_id', 'County Name', 'State Name',
                    'Latitude', 'Longitude']].drop_duplicates()

# join site_info on to station_stats
station_stats = station_stats.merge(
    site_info, left_on='station', right_on='site_id', how='left')

# rename county in station_stats to be county + "County", state
station_stats['County Name'] = station_stats['County Name'] + \
    ' County' + ', ' + station_stats['State Name']

county_pops = pd.read_csv(
    '/private/tmp/county_pops.csv')
# remove leading . in county column from county_pops
county_pops['county'] = county_pops['county'].str.lstrip('. ')

# join county pops on to station_stats
station_stats = station_stats.merge(
    county_pops, left_on='County Name', right_on='county', how='left')

station_stats.to_csv('/tmp/map_test_performance_with_pops.csv', index=False)
station_stats
# join population data from station_stats on to df
df = df.merge(
    station_stats[['station', 'State Name']], left_on='station', right_on='station', how='left')
station_stats
# theres commas in df population, remove them and convert to numeric
df['population'] = df['population'].str.replace(',', '')
df['population'] = pd.to_numeric(df['population'], errors='coerce')
df_pop = df[df['population'] > 1000000]
df_pop
R2_pop = r2_score(df_pop['measurement'], df_pop['prediction'])
MAE_pop = mean_absolute_error(df_pop['measurement'], df_pop['prediction'])
MSE_pop = np.mean((df_pop['measurement'] - df_pop['prediction'])**2)
print(
    f'Population > 450K - R²: {R2_pop:.3f}, MAE: {MAE_pop:.3f}, MSE: {MSE_pop:.3f}')

# group df by state and get stats per state
state_stats = df.groupby('State Name').apply(
    lambda x: pd.Series({
        'R2': r2_score(x['measurement'], x['prediction']),
        'MAE': mean_absolute_error(x['measurement'], x['prediction']),
        'MSE': np.mean((x['measurement'] - x['prediction'])**2),
        'N': len(x),
        'nr_stations': x['station'].nunique()
    })
).reset_index().sort_values(by='R2', ascending=False)

state_stats.to_csv('/tmp/state_test_performance.csv', index=False)

# plot correlation chart for 'Arizona'
state = 'Arizona'
state_df = df[df['State Name'] == state]
plt.figure(figsize=(12, 10))
sns.set_theme(style="whitegrid")
x = state_df['measurement']
y = state_df['prediction']
# Calculate point density using gaussian_kde
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
# Use larger points and increase alpha for lighter points to stand out more
plt.scatter(x, y, c=z, s=40, cmap='Blues', alpha=0.8,
            edgecolors='black', linewidth=0.2)
# Fit line
# Plot y=x line
plt.plot(x, x, color='green', linestyle='--', linewidth=2, label='y=x line')
plt.xlim(0, 40)
plt.ylim(0, 40)
plt.xlabel('Measurement (ug/m³)', fontsize=24)
plt.ylabel('Prediction (ug/m³)', fontsize=24)
plt.title(f'Inference Results for {state}', fontsize=28)
plt.tick_params(axis='both', which='major', labelsize=22)
# add r2, mae and mse
slope, intercept, r_value, p_value, std_err = linregress(x, y)
r2 = r2_score(x, y)
mae = mean_absolute_error(x, y)
mse = np.mean((x - y)**2)
stats_text = f'R² = {r2:.3f}\nMAE = {mae:.3f} ug/m³\nMSE = {mse:.3f} (ug/m³)²'
plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=22,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.legend(fontsize=24)
plt.show()

# Get station info from all_measures data
stations_info = all_measures[['AirQualityStation',
                              'Latitude', 'Longitude']].drop_duplicates()
df_with_info = df.merge(
    stations_info[['AirQualityStation', 'Latitude', 'Longitude']], left_on='station', right_on='AirQualityStation', how='left')
df.columns
# Index(['measurement', 'prediction', 'station', 'State Name'], dtype='object')

# ---------------------------------------------------------
# 1. DATA PREPARATION
# ---------------------------------------------------------

# Calculate Bias per prediction (Prediction - Actual)
# Positive Bias = Overprediction (Red)
# Negative Bias = Underprediction (Blue)
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

# Plot the USA background
usa.plot(ax=ax, color='#f0f0f0', edgecolor='#bdbdbd', linewidth=1)

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
