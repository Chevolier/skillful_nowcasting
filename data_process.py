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


data_dir = "data/zuimei-radar-test"


def get_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.zip'):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
                    
    return file_paths

def sort_key(file_path):
    match = re.search(r'BABJ_(\d+)_P', file_path)
    if match:
        return int(match.group(1))
    else:
        return float('inf')
    
    
# Example usage
file_paths = get_file_paths(data_dir)
print(f"number of files: {len(file_paths)}")
print("before sort: \n", file_paths[:5])

file_paths.sort(key=cmp_to_key(lambda x, y: (sort_key(x) > sort_key(y)) - (sort_key(x) < sort_key(y))))

print("after sort: \n", file_paths[:5])


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

times = [get_time_from_path(path) for path in file_paths]
times.sort()

periods = []
current_period = []
time_interval = 360
delta = 5

for i in range(len(times)):
    current_time = times[i]
    if i == 0 or (times[i] - times[i - 1]).total_seconds() >= time_interval - delta and (times[i] - times[i - 1]).total_seconds() <= time_interval + delta:
        current_period.append(file_paths[i])
    else:
        periods.append(current_period)
        current_period = [file_paths[i]]

if current_period:
    periods.append(current_period)

print(f"Length of periods of consecutive file paths: {len(periods)}")
# for period in periods[:5]:
#     print(period)
count = sum([1 for period in periods if len(period)>=24])
lens = [len(period) for period in periods if len(period)>=0]
print(f"Number of periods with length>=24: {count}")
print(f"Lengths of all periods: {lens}")


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

# Print the loaded NumPy array
data = read_data(file_paths[0])
print(f"data type: {type(data)}, data shape: {data.shape}")


import numpy as np
from tqdm import tqdm
import time

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


def pad_image(arr, base_num=32):

    # Calculate the padding needed for each dimension
    pad_dim1 = (base_num - arr.shape[0] % base_num) % base_num
    pad_dim2 = (base_num - arr.shape[1] % base_num) % base_num

    # Calculate the padding amount for each side
    pad_width = ((0, pad_dim1), (0, pad_dim2))

    # Pad the array with zeros
    padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
    padded_arr = padded_arr.astype(np.float32)

    # print("Padded array shape:", padded_arr.shape)
    
    return padded_arr

def crop_image_grid(image, crop_size=256, num_thr=1000, threshold=5):
    # cropping the image by dividing the frame into grids
    
    
    print(f"original image shape: {image.shape}")
    padded_image = pad_image(image, crop_size)
    
    width, height = padded_image.shape
    print(f"padded image shape: {padded_image.shape}")
    
    counts = []
    valid_positions = []
    for row in range(0, width, crop_size):
        for col in range(0, height, crop_size):
            cropped_image = padded_image[row:row+crop_size, col:col+crop_size]
            count = np.sum(cropped_image>=threshold)
            if count >= num_thr:
                valid_positions.append((row, col, count))
                counts.append(count)
    
    return {"valid_positions": valid_positions}


crop_size = 256
num_thr = 1000 # for random cropping, ensure that the cropped area has at least this number of elements are nonzero
threshold = 10

num_periods = len(periods)
num_total_frames = 24

count = 0
tar_id = 0
num_per_tar = 20

save_dir = f"data/zuimei-radar-cropped-num_thr{num_thr}-threshold{threshold}-test"
os.makedirs(save_dir, exist_ok=True)

for period in tqdm(periods[:num_periods]):
    if len(period) < num_total_frames:
        continue
            
    sub_period_size = 60  # 24 * 60 / 6 = 240 one day, 6 hours = 
    
    for sub_pid in range(0, len(period), sub_period_size):
        start_reading = time.time()
        sub_period = period[sub_pid:sub_pid+sub_period_size]
        print(f"Start reading {len(sub_period)} frames ...")
        period_frames = read_frames(sub_period)
        end_reading = time.time()
        time_cost_reading = end_reading - start_reading
        print(f"End reading {len(sub_period)} frames, time cost: {time_cost_reading:.2f}s.")

        print("Start croppping and saving")
        try:
            frame_result = crop_image_grid(period_frames[0], crop_size, num_thr, threshold)
        except Exception as e:
            print(f"Error: {e}")
            continue
        
        num_examples = period_frames.shape[0] - num_total_frames + 1
        if num_examples <= 0:
            continue
        
        valid_positions = frame_result['valid_positions']
        for idx in tqdm(range(num_examples)):
            frames = period_frames[idx:idx+num_total_frames]
            
            for position in valid_positions:
                cropped_frames= frames[:,position[0]:position[0]+crop_size, position[1]:position[1]+crop_size]

                frame_start_time = get_time_from_path(period[idx], 'str')

                # Define the output dataset directory
                if count == 0:
                    output_dir = os.path.join(save_dir, f"{tar_id:06d}.tar")
                    writer = wds.TarWriter(output_dir)

                sample = {
                    "__key__": str(uuid.uuid4()),
                    "cropped_frames": cropped_frames.tobytes(),
                    "position": np.array(position, dtype=np.float32).tobytes(),
                    "start_time": frame_start_time.encode(),
                }

                # Write the sample to the dataset
                writer.write(sample)
                count += 1

                if count == num_per_tar:
                    # Close the writer
                    writer.close()
                    count = 0
                    tar_id += 1
    
        end_crop = time.time()
        time_cost_crop = end_crop - end_reading
        print(f"End cropping and saving {num_examples} frames, time cost: {time_cost_crop}s, average time cost: {time_cost_crop/num_examples:.2f}s")

if count < num_per_tar:
    writer.close()
    
    