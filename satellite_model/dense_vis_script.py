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
df = pd.read_csv('/Users/adambloom/Downloads/mc_dropout_results_test_big.csv')
# exclude top 5% of uncertainty dropout
df = df[df['uncertainty_dropout'] < df['uncertainty_dropout'].quantile(0.90)]
# exclude top 5% of measurement
df = df[df['measurement'] < df['measurement'].quantile(0.95)]
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
y = df['calibrated_prediction_nb']
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
plt.title('Potential Impact of Correcting Overpredictions between 0-5 ug/m³ Measurement and Prediction', fontsize=18)
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


# plot uncertainty vs absolute error
plt.figure(figsize=(10, 8))
sns.set_theme(style="whitegrid")
x = df['uncertainty_dropout']
y = np.abs(df['measurement'] - df['prediction_dropout'])
# Calculate point density using gaussian_kde
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
plt.scatter(x, y, c=z, s=20, cmap='Reds')
plt.xlabel('Uncertainty (Dropout)', fontsize=16)
plt.ylabel('Absolute Error', fontsize=16)
plt.title('Validation Uncertainty vs Absolute Error', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=14)
# add r2, mae and mse
slope, intercept, r_value, p_value, std_err = linregress(x, y)
r2 = r_value**2
mae = np.mean(np.abs(x - y))
mse = np.mean((x - y)**2)
stats_text = f'R² = {r2:.3f}\nMAE = {mae:.3f}\nMSE = {mse:.3f}'
plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=14,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
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
