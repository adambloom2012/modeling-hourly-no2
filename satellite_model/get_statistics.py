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
run2 = "mlruns/379837206754776280/e43fd6a90e444a61abf67ffac5a68b32/"

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
frequency = "hourly"

samples, stations = load_data(datadir2, samples_file2, frequency, sources2)

dataset = NO2PredictionDataset(
    datadir2, 
    samples, 
    frequency, 
    sources2, 
    transforms=ToTensor(),  # Only convert to tensor, no normalization
    station_imgs=stations
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

print("Calculating statistics...")

# Initialize accumulators
img_sum = None
img_sum_sq = None
img_count = 0
s5p_values = []
no2_values = []

# Process batches
for sample in tqdm(dataloader):
    img_batch = sample["img"].float()
    batch_size_actual = img_batch.shape[0]
    
    # Sum across spatial dimensions and batch
    img_batch_mean = img_batch.mean(dim=(0, 2, 3))
    img_batch_sq = (img_batch ** 2).mean(dim=(0, 2, 3))
    
    if img_sum is None:
        img_sum = img_batch_mean * batch_size_actual
        img_sum_sq = img_batch_sq * batch_size_actual
    else:
        img_sum += img_batch_mean * batch_size_actual
        img_sum_sq += img_batch_sq * batch_size_actual
        
    img_count += batch_size_actual
    
    # Collect S5P and NO2 values
    if "s5p" in sample:
        s5p_batch = sample["s5p"].float().numpy().flatten()
        s5p_values.extend(s5p_batch[~np.isnan(s5p_batch)])
    
    if "no2" in sample:
        no2_batch = sample["no2"].float().numpy().flatten()
        no2_values.extend(no2_batch[~np.isnan(no2_batch)])

# Calculate final statistics
channel_means = (img_sum / img_count).numpy()
channel_variance = (img_sum_sq / img_count) - (channel_means ** 2)
channel_std = np.sqrt(np.maximum(channel_variance, 0))

s5p_values = s5p_values[np.isfinite(s5p_values)]
if len(s5p_values) > 0:
    # Remove extreme 1% on each end (keep 98% of data)
    p1, p99 = np.percentile(s5p_values, [1, 99])
    s5p_values = s5p_values[(s5p_values >= p1) & (s5p_values <= p99)]
no2_values = np.array(no2_values)

print(f"channel_means = np.array({list(channel_means)})")
print(f"channel_std = np.array({list(channel_std)})")
print(f"s5p_mean = {np.mean(s5p_values)}")
print(f"s5p_std = {np.std(s5p_values)}")
print(f"no2_mean = {np.mean(no2_values)}")
print(f"no2_std = {np.std(no2_values)}")
