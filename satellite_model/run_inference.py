from matplotlib.patches import ConnectionPatch
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import shapely
import xarray as xr
from importlib import import_module
import matplotlib.pyplot as plt
from rasterio.errors import RasterioIOError
from rasterio.enums import Resampling
from rasterio.plot import reshape_as_image
import rasterio as rio
from shapely import wkt
from train_utils import eval_metrics, split_samples, train, test
from utils import load_data, set_seed, step, read_param_file
from model import get_model
from transforms import ChangeBandOrder, ToTensor, DatasetStatistics, Normalize, Randomize, LogTransformNO2
from dataset import NO2PredictionDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn, optim
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy
import sys
import os

os.environ["OMP_NUM_THREADS"] = "6"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "6"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "6"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
run2 = "mlruns/772459609649213407/83ee548de3314746952f36e2ce57c5b1/"

samples_file2 = read_param_file(run2 + "params/samples_file")
datadir2 = read_param_file(run2 + "params/datadir")
sources2 = read_param_file(run2 + "params/sources")
frequency2 = read_param_file(run2 + "params/frequency")
log_transform2 = read_param_file(
    run2 + "params/log_transform").strip().lower() == "true"
heteroscedastic2 = read_param_file(
    run2 + "params/heteroscedastic").strip().lower() == "true"

dropout_config2 = {"p_second_to_last_layer": float(read_param_file(run2 + "params/dropout_p_second_to_last_layer")),
                   "p_last_layer": float(read_param_file(run2 + "params/dropout_p_last_layer")),
                   }

model_package2 = import_module(run2.replace("/", ".") + "artifacts.model")
checkpoint2 = None  # read_param_file(run2 + "params/pretrained_checkpoint")

frequency = "hourly"
print("="*50)
print("INFERENCE WITH LOG TRANSFORMATION")
print("="*50)
print(f"Model run: {run2}")
print(f"Samples file: {samples_file2}")
print(f"Data directory: {datadir2}")
print(f"Log transform: {log_transform2}")
print(f"Heteroscedastic: {heteroscedastic2}")

samples, stations = load_data(datadir2, samples_file2, frequency, sources2)


test_stations_str = read_param_file(run2 + "artifacts/stations_test.txt")

# Parse the string to get actual station IDs
# Assuming it's a comma-separated string or similar format
test_stations = set(test_stations_str.strip().split('\n')
                    )  # Adjust delimiter as needed
print(test_stations)
# Or if it's a different format, adjust accordingly

# Filter stations to only include test stations
stations = {station: data for station,
            data in stations.items() if station in test_stations}
save_stations = stations  # saving because I keep losing it

# Filter samples to only include those from test stations
samples = [
    sample for sample in samples if sample["AirQualityStation"] in test_stations]

print(f"Filtered to {len(stations)} stations and {len(samples)} samples")


# get dataset transforms - include log transform if model was trained with it
datastats = DatasetStatistics()
transform_list = [Normalize(datastats)]
if log_transform2:
    transform_list.append(LogTransformNO2())
transform_list.extend([Randomize(), ToTensor()])
tf = transforms.Compose(transform_list)
dataset = NO2PredictionDataset(
    datadir2, samples, frequency, sources2, transforms=tf, station_imgs=stations)
dataloader = DataLoader(dataset, batch_size=1,
                        num_workers=1, shuffle=False, pin_memory=False)


model_weights2 = run2 + "artifacts/model_state.model"
weights = torch.load(model_weights2)

model2 = model_package2.get_model(
    sources2, device, checkpoint=checkpoint2, dropout=dropout_config2, heteroscedastic=heteroscedastic2)
model2.load_state_dict(weights)
model2.to(device)
model2.eval()
print("Model loaded successfully")


# Run inference using the test function which handles log transformation properly
print("Running inference...")
measurements, predictions = test(sources2, model2, dataloader, device, datastats,
                                 dropout=False, heteroscedastic=heteroscedastic2,
                                 log_transform=log_transform2)

print(f"Generated {len(predictions)} predictions")


# Calculate metrics
r2 = r2_score(measurements, predictions)
mae = mean_absolute_error(measurements, predictions)
mse = mean_squared_error(measurements, predictions)
rmse = np.sqrt(mse)

print(f"\nInference Results:")
print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Log Transform Used: {log_transform2}")
print(f"Heteroscedastic Model: {heteroscedastic2}")

station_names = []
for i, sample in enumerate(samples):
    station = sample["AirQualityStation"]
    # Handle if station is a tensor
    if torch.is_tensor(station):
        station = station.item() if station.numel() == 1 else str(
            station.detach().cpu().numpy())
    # Handle if station is a numpy array
    elif isinstance(station, np.ndarray):
        station = station.item() if station.size == 1 else str(station)
    # Convert to string just in case
    station_names.append(str(station))

# save results to dataframe
results_df = pd.DataFrame({
    "measurement": measurements,
    "prediction": predictions,
    "station": station_names,
})

# Save results
output_file = "logs/inference_results_log_transform.csv"
results_df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")

# Save summary
summary_file = output_file.replace('.csv', '_summary.txt')
with open(summary_file, 'w') as f:
    f.write(f"Inference Results Summary\n")
    f.write(f"========================\n\n")
    f.write(f"Model Run: {run2}\n")
    f.write(f"Samples File: {samples_file2}\n")
    f.write(f"Data Directory: {datadir2}\n")
    f.write(f"Sources: {sources2}\n")
    f.write(f"Frequency: {frequency2}\n")
    f.write(f"Log Transform: {log_transform2}\n")
    f.write(f"Heteroscedastic: {heteroscedastic2}\n\n")
    f.write(f"Performance Metrics:\n")
    f.write(f"R² Score: {r2:.4f}\n")
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"MSE: {mse:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n\n")
    f.write(f"Data Statistics:\n")
    f.write(f"Number of predictions: {len(predictions)}\n")
    f.write(f"Test stations: {len(test_stations)}\n")
    f.write(
        f"Prediction range: [{np.min(predictions):.4f}, {np.max(predictions):.4f}]\n")
    f.write(
        f"Measurement range: [{np.min(measurements):.4f}, {np.max(measurements):.4f}]\n")

print(f"Summary saved to: {summary_file}")
