import argparse
import zipfile
import os
import re
from functools import cmp_to_key
from datetime import datetime, timedelta
from PIL import Image
import json
import random
import webdataset as wds
import numpy as np
from tqdm import tqdm
import pandas as pd
import time
import uuid


def remove_corrupted_zips(directory):
    # Collect all ZIP files
    zip_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.zip'):
                file_path = os.path.join(root, file)
                zip_files.append(file_path)
    
    # Process each ZIP file with a progress bar
    for file_path in tqdm(zip_files, desc="Checking ZIP files"):
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_file:
                # Check if the ZIP file is complete
                bad_file = zip_file.testzip()
                if bad_file:
                    os.remove(file_path)
                    print(f"Warning: '{file_path}' is a corrupted ZIP file, removed.")
                    
        except zipfile.BadZipFile:
            os.remove(file_path)
            print(f"Warning: '{file_path}' is not a valid ZIP file, removed.")
            
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
    with zipfile.ZipFile(file_path, 'r') as zip_file:
        file_name = zip_file.namelist()[0]
        with zip_file.open(file_name) as file:
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
    frames[frames > vmax] = vmax
    frames[frames < vmin] = vmin
    frames = frames.astype(np.float32)
    return frames

def pad_image(arr, base_num=32):
    pad_dim1 = (base_num - arr.shape[0] % base_num) % base_num
    pad_dim2 = (base_num - arr.shape[1] % base_num) % base_num
    pad_width = ((0, pad_dim1), (0, pad_dim2))
    padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
    padded_arr = padded_arr.astype(np.float32)
    return padded_arr

def crop_image_grid(image, crop_size=256, num_thr=1000, threshold=5):
    padded_image = pad_image(image, crop_size)
    width, height = padded_image.shape
    counts = []
    valid_positions = []
    for row in range(0, width, crop_size):
        for col in range(0, height, crop_size):
            cropped_image = padded_image[row:row+crop_size, col:col+crop_size]
            count = np.sum(cropped_image >= threshold)
            if count >= num_thr:
                valid_positions.append((row, col, count))
                counts.append(count)
    return {"valid_positions": valid_positions}

def main(args):
    data_dir = args.data_dir
    crop_size = args.crop_size
    num_thr = args.num_thr
    threshold = args.threshold
    num_per_tar = args.num_per_tar
    save_dir = args.save_dir
    num_total_frames = args.num_total_frames

    os.makedirs(save_dir, exist_ok=True)
    file_paths = get_file_paths(data_dir)
    file_paths.sort(key=cmp_to_key(lambda x, y: (sort_key(x) > sort_key(y)) - (sort_key(x) < sort_key(y))))
    
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

    count = 0
    tar_id = 0

    for period in tqdm(periods):
        if len(period) < num_total_frames:
            continue
        sub_period_size = 60
        for sub_pid in range(0, len(period), sub_period_size):
            start_reading = time.time()
            sub_period = period[sub_pid:sub_pid+sub_period_size]
            period_frames = read_frames(sub_period)
            end_reading = time.time()
            time_cost_reading = end_reading - start_reading

            try:
                frame_result = crop_image_grid(period_frames[0], crop_size, num_thr, threshold)
            except Exception as e:
                print(f"Error: {e}")
                continue

            num_examples = period_frames.shape[0] - num_total_frames + 1
            if num_examples <= 0:
                continue

            valid_positions = frame_result['valid_positions']
            for idx in range(num_examples):
                frames = period_frames[idx:idx+num_total_frames]
                for position in valid_positions:
                    cropped_frames = frames[:, position[0]:position[0]+crop_size, position[1]:position[1]+crop_size]
                    frame_start_time = get_time_from_path(period[idx], 'str')
                    if count == 0:
                        output_path = os.path.join(save_dir, f"{tar_id:06d}.tar")
                        writer = wds.TarWriter(output_path)

                    sample = {
                        "__key__": str(uuid.uuid4()),
                        "cropped_frames": cropped_frames.tobytes(),
                        "position": np.array(position, dtype=np.float32).tobytes(),
                        "start_time": frame_start_time.encode(),
                    }
                    writer.write(sample)
                    count += 1
                    if count == num_per_tar:
                        writer.close()
                        count = 0
                        tar_id += 1
            end_crop = time.time()
            time_cost_crop = end_crop - end_reading

    if count < num_per_tar:
        writer.close()
        # the last one has less files, remove it to keep all have the same
        os.remove(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process radar data and save as webdataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the radar data zip files")
    parser.add_argument("--crop_size", type=int, default=256, help="Size of the cropped image")
    parser.add_argument("--num_thr", type=int, default=1000, help="Minimum number of elements above threshold in cropped image")
    parser.add_argument("--threshold", type=float, default=10, help="Threshold value for cropping")
    parser.add_argument("--num_per_tar", type=int, default=20, help="Number of samples per tar file")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the processed tar files")
    parser.add_argument("--num_total_frames", type=int, default=24, help="Total number of frames in each period")
    
    args = parser.parse_args()
    
    print("Removing corrupted zip files, need only run once.")
    remove_corrupted_zips(args.data_dir)
    print("Finished removing corrupted zip files.")
    
    main(args)
