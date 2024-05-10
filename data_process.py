import numpy as np
import zipfile
import os
import re
from functools import cmp_to_key
from datetime import datetime, timedelta
import tqdm

data_dir = "data/zuimei-radar"

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

# Example usage
file_paths = get_file_paths(data_dir)
print(f"number of files: {len(file_paths)}")
print("before sort: \n", file_paths[:5])

file_paths.sort(key=cmp_to_key(lambda x, y: (sort_key(x) > sort_key(y)) - (sort_key(x) < sort_key(y))))

print("after sort: \n", file_paths[:5])

# keep only items with 6 min intervals
file_paths_new = []

for file_path in file_paths:
    match = re.search(r'BABJ_(\d+)_P', file_path)
    if not match:
        print(file_path)
        continue
    
    minutes = int(match.group(1)[-4:-2])
    if minutes % 6 == 0:
        file_paths_new.append(file_path)
    else:
        print(file_path)
    

file_paths = file_paths_new

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
print(type(data), data.shape)


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


def compute_integral_image(image):
    """Compute the integral image of a given image."""
    integral_image = np.zeros_like(image)
    rows, cols = image.shape
    
    # Compute the integral image
    for i in range(rows):
        for j in range(cols):
            integral_image[i, j] = 1 if image[i, j] > 0 else 0
            if i > 0:
                integral_image[i, j] += integral_image[i-1, j]
            if j > 0:
                integral_image[i, j] += integral_image[i, j-1]
            if i > 0 and j > 0:
                integral_image[i, j] -= integral_image[i-1, j-1]
                
    return integral_image


def get_window_sum(integral_image, top_left, bottom_right):
    """Compute the sum of pixel values in a window using the integral image."""
    top, left = top_left
    bottom, right = bottom_right
    
    window_sum = integral_image[bottom, right]
    if top > 0:
        window_sum -= integral_image[top-1, right]
    if left > 0:
        window_sum -= integral_image[bottom, left-1]
    if top > 0 and left > 0:
        window_sum += integral_image[top-1, left-1]
        
    return window_sum


def crop_image(image, crop_size=256, num_thr=50):
    # Compute the integral image
    integral_image = compute_integral_image(image)
    
    # Get the dimensions of the image
    rows, cols = image.shape
    
    # Initialize variables to keep track of the maximum non-zero count and its position
    max_count = -1
    max_pos = (0, 0)
    
    valid_positions = []
    # Slide the window across the integral image
    for i in range(0, rows - crop_size + 1):
        for j in range(0, cols - crop_size + 1):
            # Compute the sum of pixel values in the current window
            count = get_window_sum(integral_image, (i, j), (i+crop_size-1, j+crop_size-1))
            
            # Update the maximum count and its position if the current count is greater
            if count > max_count:
                max_count = count
                max_pos = (i, j)
            
            if count > num_thr:
                valid_positions.append((i, j))
    
    # Crop the image using the position with the maximum count
    cropped_image_max_nonzero = image[max_pos[0]:max_pos[0]+crop_size, max_pos[1]:max_pos[1]+crop_size]
    
    # If no valid position found, raise an exception
    if not valid_positions:
        raise ValueError("No valid crop position found with the given threshold.")

    # Select a random valid position
    random_idx = np.random.choice(len(valid_positions))
    random_pos = valid_positions[random_idx]

    # Crop the array
    cropped_image_random = image[random_pos[0]:random_pos[0]+crop_size,
                         random_pos[1]:random_pos[1]+crop_size]

    result = {
        "cropped_image_max_nonzero": cropped_image_max_nonzero,
        "max_pos": max_pos,
        "cropped_image_random": cropped_image_random,
        "random_pos": random_pos,
        "valid_positions": valid_positions
    }
    
    return result


def crop_frames(frames, crop_size=256, num_thr=100):
    # used to crop the large frame into small frames
    result = crop_image(frames[0], crop_size=crop_size, num_thr=num_thr)
    
    max_pos = result['max_pos']
    cropped_frames_max_nonzero = frames[:,max_pos[0]:max_pos[0]+crop_size, max_pos[1]+max_pos[1]+crop_size]
    
    random_pos = result['random_pos']
    cropped_frames_random= frames[:,random_pos[0]:random_pos[0]+crop_size, random_pos[1]+random_pos[1]+crop_size]
    
    cropped_result = {
        "cropped_frames_max_nonzero": cropped_frames_max_nonzero,
        "max_pos": max_pos,
        "cropped_frames_random": cropped_frames_random,
        "random_pos": random_pos
    }
    
    return cropped_result


examples = []

crop_size = 256
num_thr = 2000 # for random cropping, ensure that the cropped area has at least this number of elements are nonzero

num_periods = 100
num_total_frames = 24

for period in tqdm(periods[:num_periods]):
    if len(period) < num_total_frames:
        continue
    
    start_reading = time.time()
    print(f"Start reading {len(period)} frames ...")
    period_frames = read_frames(period)
    end_reading = time.time()
    time_cost_reading = end_reading - start_reading
    print(f"End reading {len(period)} frames, time cost: {time_cost_reading:.2f}s.")
    
    freq_update_crop_region = 240 # change crop regions every 24 hours, 240 * 6 min
    
    frame_result = {"max_pos": (0, 0), "valid_positions": [(0, 0)]}
    
    print("Start croppping")
    num_examples = len(period) - num_total_frames + 1
    for idx in tqdm(range(num_examples)):
        frames = period_frames[idx:idx+num_total_frames]
        
        if idx % freq_update_crop_region == 0:
            frame_result = crop_image(frames[0], crop_size, num_thr)
        
        max_pos = frame_result['max_pos']
        cropped_frames_max_nonzero = frames[:,max_pos[0]:max_pos[0]+crop_size, max_pos[1]:max_pos[1]+crop_size]

        random_idx = np.random.choice(len(frame_result["valid_positions"]))
        random_pos = frame_result["valid_positions"][random_idx]
        cropped_frames_random= frames[:,random_pos[0]:random_pos[0]+crop_size, random_pos[1]:random_pos[1]+crop_size]
        
        frame_start_time = get_time_from_path(period[idx], 'str')
        
        cropped_result = {
            "cropped_frames_max_nonzero": cropped_frames_max_nonzero,
            "max_pos": max_pos,
            "cropped_frames_random": cropped_frames_random,
            "random_pos": random_pos,
            "start_time": frame_start_time
        }
        
        examples.append(cropped_result)
    
    end_crop = time.time()
    time_cost_crop = end_crop - end_reading
    print(f"End cropping {num_examples} frames, time cost: {time_cost_crop}s, average time cost: {time_cost_crop/num_examples:.2f}s")
    
import os
from PIL import Image
import json
import random
# import pickle
import webdataset as wds
import numpy as np

from tqdm import tqdm

import pandas as pd
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

count = 0
tar_id = 0
num_per_tar = 10

num_examples = len(examples) # 10 * num_per_tar
# Create a WebDataset writer
for i, example in tqdm(enumerate(examples[:num_examples]), total=num_examples):
    
    # Define the output dataset directory
    if count == 0:
        output_dir = f"data/zuimei-radar-cropped/{tar_id:06d}.tar"
        writer = wds.TarWriter(output_dir)

    sample = {
        "__key__": f"{i:08d}",
        "cropped_frames_max_nonzero": example['cropped_frames_max_nonzero'].tobytes(),
        "max_pos": np.array(example['max_pos'], dtype=np.float32).tobytes(),
        "cropped_frames_random": example['cropped_frames_random'].tobytes(),
        "random_pos": np.array(example['random_pos'], dtype=np.float32).tobytes(),
        "start_time": example['start_time'].encode(),
    }

    # Write the sample to the dataset
    writer.write(sample)
    count += 1
    
    if count == num_per_tar:
        # Close the writer
        writer.close()
        count = 0
        tar_id += 1

if count < num_per_tar:
    writer.close()