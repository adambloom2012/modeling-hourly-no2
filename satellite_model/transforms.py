from rasterio.plot import reshape_as_image
import torch
import numpy as np
import copy
import random
import os

os.environ["OMP_NUM_THREADS"] = "6"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "6"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "6"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


# define image transforms

class ChangeBandOrder(object):
    def __call__(self, sample):
        """necessary if model was pre-trained on .npy files of BigEarthNet and should be used on other Sentinel-2 images

        move the channels of a sentinel2 image such that the bands are ordered as in the BigEarthNet dataset
        input image is expected to be of shape (200,200,12) with band order:
        ['B04', 'B03', 'B02', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'B01', 'B09'] (i.e. like my script on compute01 produces)

        output is of shape (12,120,120) with band order:
        ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"] (order in BigEarthNet .npy files)
        """
        img = copy.copy(sample["img"])
        img = np.moveaxis(img, -1, 0)
        reordered_img = np.zeros(img.shape)
        reordered_img[0, :, :] = img[10, :, :]
        reordered_img[1, :, :] = img[2, :, :]
        reordered_img[2, :, :] = img[1, :, :]
        reordered_img[3, :, :] = img[0, :, :]
        reordered_img[4, :, :] = img[4, :, :]
        reordered_img[5, :, :] = img[5, :, :]
        reordered_img[6, :, :] = img[6, :, :]
        reordered_img[7, :, :] = img[3, :, :]
        reordered_img[8, :, :] = img[7, :, :]
        reordered_img[9, :, :] = img[11, :, :]
        reordered_img[10, :, :] = img[8, :, :]
        reordered_img[11, :, :] = img[9, :, :]

        if img.shape[1] != 120 or img.shape[2] != 120:
            reordered_img = reordered_img[:, 40:160, 40:160]

        out = {}
        for k, v in sample.items():
            if k == "img":
                out[k] = reordered_img
            else:
                out[k] = v

        return out


class ToTensor(object):
    def __call__(self, sample):
        img_array = sample["img"]
        if not img_array.flags.c_contiguous:
            img_array = np.ascontiguousarray(img_array)
        img = torch.from_numpy(img_array)

        if sample.get("no2") is not None:
            no2_val = sample["no2"]
            if isinstance(no2_val, (int, float, np.number)):
                no2 = torch.tensor(no2_val, dtype=torch.float32)
            else:
                # Ensure contiguous array for numpy arrays
                if hasattr(no2_val, 'flags') and not no2_val.flags.c_contiguous:
                    no2_val = np.ascontiguousarray(no2_val)
                no2 = torch.from_numpy(no2_val)

        if sample.get("s5p") is not None:
            s5p_val = sample["s5p"]
            if isinstance(s5p_val, (int, float, np.number)):
                s5p = torch.tensor(s5p_val, dtype=torch.float32)
            else:
                # Ensure contiguous array for numpy arrays
                if hasattr(s5p_val, 'flags') and not s5p_val.flags.c_contiguous:
                    s5p_val = np.ascontiguousarray(s5p_val)
                s5p = torch.from_numpy(s5p_val)

        out = {}
        for k, v in sample.items():
            if k == "img":
                out[k] = img
            elif k == "no2":
                out[k] = no2
            elif k == "s5p":
                out[k] = s5p
            else:
                out[k] = v

        return out


class DatasetStatistics(object):
    def __init__(self):
        self.channel_means = np.array([1648.2152, 1800.7183, 2007.9602, 2141.0327, 2411.6934, 2834.076, 3006.8687, 3142.7512, 3166.1133, 3183.226, 3073.6143, 2582.8882])

        self.channel_std = np.array([662.9306, 837.3406, 865.3376, 953.4862, 898.1061, 875.4351, 902.4029, 1003.1600, 922.4142, 815.4863, 859.4661, 769.3292])

        # statistics over the whole of Europe from Sentinel-5P products in 2018-2020:
        # l3_mean_europe_2018_2020_005dg.netcdf mean 1.51449095e+15 std 6.93302798e+14
        # l3_mean_europe_large_2018_2020_005dg.netcdf mean 1.23185273e+15 std 7.51052046e+14
        self.s5p_mean = 5263615565234176
        self.s5p_std = 7.51052046e+14

        # values for averages from 2018-2020 per EEA station, across stations
        self.no2_mean = 6.817475318908691
        self.no2_std = 7.417215824127197


class Normalize(object):
    """normalize a sample, i.e. the image and NO2 value, by subtracting mean and dividing by std"""

    def __init__(self, statistics):
        self.statistics = statistics

    def __call__(self, sample):
        img = copy.copy(reshape_as_image(sample.get("img")))

        if img.shape[0] == 12:  # channels first
            img = np.moveaxis(img, 0, -1)
        img = np.moveaxis((img - self.statistics.channel_means) /
                          self.statistics.channel_std, -1, 0)

        if sample.get("no2") is not None:
            no2 = copy.copy(sample.get("no2"))
#            no2 = np.array((no2 - self.statistics.no2_mean) / self.statistics.no2_std)
            no2 = np.array((no2 - 0) / 1)

        if sample.get("s5p") is not None:
            s5p = copy.copy(sample.get("s5p"))
            s5p = np.array((s5p - self.statistics.s5p_mean) /
                           self.statistics.s5p_std)

        out = {}
        for k, v in sample.items():
            if k == "img":
                out[k] = img
            elif k == "no2":
                out[k] = no2
            elif k == "s5p":
                out[k] = s5p
            else:
                out[k] = v

        return out

    @staticmethod
    def undo_no2_standardization(statistics, no2):
        return (no2 * statistics.no2_std) + statistics.no2_mean


class LogTransformNO2(object):
    """Apply log(NO2 + 1) transformation for training"""

    def __call__(self, sample):
        out = {}
        for k, v in sample.items():
            if k == "no2":
                # Apply log(x + 1) transformation
                out[k] = np.log(v + 1)
            else:
                out[k] = v
        return out

    @staticmethod
    def inverse_transform(log_no2):
        """Convert back from log(NO2 + 1) to NO2"""
        return np.exp(log_no2) - 1


class Randomize():
    def __call__(self, sample):
        img = copy.copy(sample.get("img"))

        s5p_available = False
        if sample.get("s5p") is not None:
            s5p_available = True
            s5p = copy.copy(sample["s5p"])

        if random.random() > 0.5:
            img = np.flip(img, 1)
            if s5p_available:
                s5p = np.flip(s5p, 0)
        if random.random() > 0.5:
            img = np.flip(img, 2)
            if s5p_available:
                s5p = np.flip(s5p, 1)
        if random.random() > 0.5:
            img = np.rot90(img, np.random.randint(0, 4), axes=(1, 2))
            if s5p_available:
                s5p = np.rot90(s5p, np.random.randint(0, 4), axes=(0, 1))

        out = {}
        for k, v in sample.items():
            if k == "img":
                out[k] = img
            elif k == "s5p":
                out[k] = s5p
            else:
                out[k] = v

        return out
