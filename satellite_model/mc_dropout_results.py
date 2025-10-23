import os

os.environ["OMP_NUM_THREADS"] = "6" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "6" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

import sys
import copy
from tqdm  import tqdm

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import NO2PredictionDataset
from transforms import ChangeBandOrder, ToTensor, DatasetStatistics, Normalize, Randomize
from model import get_model
from utils import load_data, set_seed, step, read_param_file
from train_utils import eval_metrics, split_samples, train, test
from shapely import wkt

import rasterio as rio
from rasterio.plot import reshape_as_image
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
import matplotlib.pyplot as plt

from importlib import import_module

import xarray as xr
import shapely
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch

from transforms import ChangeBandOrder, ToTensor, DatasetStatistics, Normalize, Randomize

# load trained dropout model from saved mlflow run 
# /share/atmoschem/abloom/projects/Global-NO2-Estimation/satellite_model/mlruns/169512132705312502/244edd1bdf7741519dd1a9deec94cd4f/params
run = "mlruns/169512132705312502/244edd1bdf7741519dd1a9deec94cd4f/" # heteroscedastic whole_timespan 0.05, 0.05 0.55 R2

samples_file = read_param_file(run + "params/samples_file")
datadir = read_param_file(run + "params/datadir")
verbose = True
sources = read_param_file(run + "params/sources")
frequency = read_param_file(run + "params/frequency")
heteroscedastic = bool(read_param_file(run + "params/heteroscedastic"))

dropout_config = {"p_second_to_last_layer" : float(read_param_file(run + "params/dropout_p_second_to_last_layer")),
                 "p_last_layer" : float(read_param_file(run + "params/dropout_p_last_layer")),
                 }

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_package = import_module(run.replace("/", ".") + "artifacts.model")

checkpoint = None #read_param_file(run + "params/pretrained_checkpoint")


# get original model
#run2 = "mlruns/21/4166c84b6ec149998a92459bbe715719/" # no dropout 0.6 r2
# /share/atmoschem/abloom/projects/Global-NO2-Estimation/satellite_model/mlruns/379969873074757557/8010223d5aa94759a8dc572db53df614/params
run2 = "mlruns/379969873074757557/8010223d5aa94759a8dc572db53df614/" # no dropout 0.6 r2 same model structure as above

samples_file2 = read_param_file(run2 + "params/samples_file")
datadir2 = read_param_file(run2 + "params/datadir")
sources2 = read_param_file(run2 + "params/sources")
frequency2 = read_param_file(run2 + "params/frequency")
# heteroscedastic = bool(read_param_file(run + "params/heteroscedastic"))

dropout_config2 = {"p_second_to_last_layer" : float(read_param_file(run2 + "params/dropout_p_second_to_last_layer")),
                 "p_last_layer" : float(read_param_file(run2 + "params/dropout_p_last_layer")),
                 }

model_package2 = import_module(run2.replace("/", ".") + "artifacts.model")
checkpoint2 = None #read_param_file(run2 + "params/pretrained_checkpoint")

frequency = "hourly"
print(samples_file)
print(datadir)

samples, stations = load_data(datadir, samples_file, frequency, sources)


test_stations_str = read_param_file(run + "artifacts/stations_test.txt")

# Parse the string to get actual station IDs
# Assuming it's a comma-separated string or similar format
test_stations = set(test_stations_str.strip().split('\n'))  # Adjust delimiter as needed
print(test_stations)
# Or if it's a different format, adjust accordingly

# Filter stations to only include test stations
stations = {station: data for station, data in stations.items() if station in test_stations}
save_stations = stations # saving because I keep losing it 

# Filter samples to only include those from test stations
samples = [sample for sample in samples if sample["AirQualityStation"] in test_stations]

print(f"Filtered to {len(stations)} stations and {len(samples)} samples")


#tf = transforms.Compose([ChangeBandOrder()])#, Normalize(datastats), Randomize(), ToTensor()])
# get dataset transforms
datastats = DatasetStatistics()
tf = transforms.Compose([Normalize(datastats), Randomize(), ToTensor()])
dataset = NO2PredictionDataset(datadir, samples, frequency, sources, transforms=tf, station_imgs=stations)
dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, pin_memory=False)

model_weights = run + "artifacts/model_state.model"

model = model_package.get_model(sources, device, checkpoint=checkpoint, dropout=dropout_config, heteroscedastic=heteroscedastic)
model.load_state_dict(torch.load(model_weights, map_location=device))
model.to(device)

model.eval()
model.head.turn_dropout_on()


model_weights2 = run2 + "artifacts/model_state.model"
weights = torch.load(model_weights2, map_location=torch.device('cpu'))

weights["head.fc1.weight"] = weights["head.0.weight"]
weights["head.fc1.bias"] = weights["head.0.bias"]
weights["head.fc2.weight"] = weights["head.2.weight"]
weights["head.fc2.bias"] = weights["head.2.bias"]

del weights["head.0.weight"]
del weights["head.0.bias"]
del weights["head.2.weight"]
del weights["head.2.bias"]

model2 = model_package2.get_model(sources2, device, checkpoint=checkpoint, dropout=dropout_config2)
model2.load_state_dict(weights) #torch.load(model_weights2, map_location=device))
model2.to(device)

model2.eval()
"loaded"
#model.head.turn_dropout_on()


measurements = []
predictions = []
predictions_dropout = []
variances = []
stations = []
T = 100
for idx, sample in tqdm(enumerate(dataloader)):
    model_input = {"img" : sample["img"].float().to(device),
                    "s5p" : sample["s5p"].float().unsqueeze(dim=1).to(device),
                   "hour" : sample["hour"].float().to(device)
                   
                      }
    y = sample["no2"].float().to(device)
    
    y_hat2 = model2(model_input).squeeze()
    measurements.append(y.item())
    predictions.append(y_hat2.item())
    stations.append(sample["AirQualityStation"])
    
    # copy the sample T times along the batch dimension
    model_input["img"] = torch.cat(T*[model_input["img"]])
    model_input["s5p"] = torch.cat(T*[model_input["s5p"]])
    model_input["hour"] = torch.cat(T*[model_input["hour"]])
            
    y_hat = model(model_input).detach().cpu()
    ym = y_hat[:, 0]
    ym_sq = ym**2
    sigma = torch.exp(y_hat[:, 1])
    
    # take mean across T MC-estimates
    mean = ym.mean()
    predictions_dropout.append(mean.item())
    variances.append(torch.sqrt(ym_sq.mean() - mean * mean + sigma.mean()).item())

measurements = np.array(measurements)
predictions = np.array(predictions)
predictions_dropout = np.array(predictions_dropout)
variances = np.array(variances)
stations = np.array(stations)


# save results to dataframe
results_df = pd.DataFrame({
    "station" : stations,
    "measurement" : measurements,
    "prediction" : predictions,
    "prediction_dropout" : predictions_dropout,
    "uncertainty_dropout" : variances
})
results_df.to_csv("logs/mc_dropout_results.csv", index=False)