import torch
from dgmr import DGMR

def model_fn(model_dir):
    model = DGMR.from_pretrained(model_dir)
    model.eval()
    model.cuda()

    return model

def predict_fn(data, model):
    # destruct model and tokenizer
    forecast_steps = data.get("forecast_steps", 18)
    
    model.config['forecast_steps'] = forecast_steps
    model.sampler.forecast_steps = forecast_steps
    
    input_frames = data.get("input_frames")

    with torch.no_grad():
        pred_frames = model(input_frames.cuda())
        pred_frames[pred_frames<0] = 0
    
    results = {"pred_frames": pred_frames.cpu(), "forecast_steps": forecast_steps}
    
    return results

import zipfile
import os
import re
from functools import cmp_to_key
from datetime import datetime, timedelta
from PIL import Image
import json
import random
# import pickle
import webdataset as wds
import numpy as np
from tqdm import tqdm
import pandas as pd
import time
import uuid
import torch


def get_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.zip'):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
                
                # try:
                #     with zipfile.ZipFile(file_path, 'r') as zip_file:
                #         # Check if the ZIP file is complete
                #         bad_file = zip_file.testzip()
                #         if bad_file is None:
                #             # The ZIP file is complete
                #             file_paths.append(file_path)
                #         else:
                #             print(f"Warning: '{file_path}' is a corrupted ZIP file.")
                # except zipfile.BadZipFile:
                #     print(f"Warning: '{file_path}' is not a valid ZIP file.")
                    
    return file_paths


def sort_key(file_path):
    match = re.search(r'BABJ_(\d+)_P', file_path)
    if match:
        return int(match.group(1))
    else:
        return float('inf')


# get consecutive time periods
def get_time_from_path(file_path, return_type='datetime'):
    match = re.search(r'BABJ_(\d+)_P', file_path)
    if match:
        time_str = match.group(1)
        if return_type == 'datetime':
            return datetime.strptime(time_str, '%Y%m%d%H%M%S')
        else:
            return time_str
    else:
        return None


def read_data(file_path):
    # Open the ZIP file
    with zipfile.ZipFile(file_path, 'r') as zip_file:
        # Get the name of the file inside the ZIP archive
        file_name = zip_file.namelist()[0]

        # Open the file inside the ZIP archive
        with zip_file.open(file_name) as file:
            # Load the NumPy array from the file
            data = np.loadtxt(file)
    
    return data

        


def read_data(file_path):
    # Open the ZIP file
    with zipfile.ZipFile(file_path, 'r') as zip_file:
        # Get the name of the file inside the ZIP archive
        file_name = zip_file.namelist()[0]

        # Open the file inside the ZIP archive
        with zip_file.open(file_name) as file:
            # Load the NumPy array from the file
            data = np.loadtxt(file)
    
    return data


def read_frames(file_paths, vmin=0, vmax=75):
    
    frames = []
    for file_path in file_paths:
        try:
            data = read_data(file_path)
            frames.append(data)
        except Exception as e:
            print(f"{e}: {file_path}")
    
    frames = np.stack(frames, axis=0)
    frames[np.isnan(frames)] = 0
    frames[frames>vmax] = vmax
    frames[frames<vmin] = vmin
    frames = frames.astype(np.float32)
    
    return frames

def pad_frames(arr, base_num=32):
    # print("Original array shape:", arr.shape)

    # Calculate the padding needed for each dimension
    pad_dim1 = (base_num - arr.shape[1] % base_num) % base_num
    pad_dim2 = (base_num - arr.shape[2] % base_num) % base_num

    # Calculate the padding amount for each side
    pad_width = ((0, 0), (0, pad_dim1), (0, pad_dim2))

    # Pad the array with zeros
    padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
    padded_arr = padded_arr.astype(np.float32)
    
    return padded_arr

if __name__ == '__main__':
    forecast_steps = 20
    model_dir = "/skillful_nowcasting/sagemaker-deploy/dgmr"
    
    model = model_fn(model_dir)
    
    data_dir = "/skillful_nowcasting/data/zuimei-radar"
    
        # Example usage
    file_paths = get_file_paths(data_dir)
    print(f"number of files: {len(file_paths)}")
    print("before sort: \n", file_paths[:5])

    file_paths.sort(key=cmp_to_key(lambda x, y: (sort_key(x) > sort_key(y)) - (sort_key(x) < sort_key(y))))

    print("after sort: \n", file_paths[:5])
    
    times = [get_time_from_path(path) for path in file_paths]
    times.sort()

    periods = []
    current_period = []

    for i in range(len(times)):
        current_time = times[i]
        if i == 0 or (times[i] - times[i - 1]).total_seconds() >= 355 and (times[i] - times[i - 1]).total_seconds() <= 365:
            current_period.append(file_paths[i])
        else:
            periods.append(current_period)
            current_period = [file_paths[i]]

    if current_period:
        periods.append(current_period)

    print(f"Periods of consecutive file paths: {len(periods)}")
    # for period in periods[:5]:
    #     print(period)
    
    
    # Print the loaded NumPy array
    data = read_data(file_paths[0])
    print(type(data), data.shape)

    count = sum([1 for period in periods if len(period)>=24])

    lens = [len(period) for period in periods if len(period)>=0]
    print(count)
    print(lens)
    
    num_total_frames = 24
    period_frames_raw = read_frames(periods[5][:num_total_frames])
    _, width_raw, height_raw = period_frames_raw.shape
    period_frames_raw.shape
    
    period_frames_pad = pad_frames(period_frames_raw)
    period_frames_pad.shape

    period_frames = period_frames_pad
    
    num_input_frames = 4
    num_forecast_frames = 20

    input_frames = period_frames[:num_input_frames]
    target_frames = period_frames[num_input_frames:num_input_frames+num_forecast_frames]

    print(input_frames.shape, target_frames.shape)
    
    width, height = input_frames.shape[1], input_frames.shape[2]
    print(f"width: {width}, height: {height}")
    
    num_subframe = 4
    model.config['forecast_steps'] = forecast_steps
    model.sampler.forecast_steps = forecast_steps
    model.latent_stack.shape = (8, width//num_subframe//32, height//32)

    pred_frames = []

    start_time = time.time()
    with torch.no_grad():
        for i in range(num_subframe):
            pred_subframes = model(torch.tensor(input_frames[:,i*width//num_subframe:(i+1)*width//num_subframe,:]).unsqueeze(0).unsqueeze(2).cuda())
            pred_subframes[pred_subframes<0] = 0

            pred_frames.append(pred_subframes)

    time_cost = time.time() - start_time

    pred_frames = torch.cat(pred_frames, dim=3)

    print(pred_frames.shape)
    print(f"time cost for 1 inference: {time_cost/2}s.")
    
    