#!/bin/bash

git clone https://github.com/openclimatefix/skillful_nowcasting.git

# conda create -n dgmr python=3.10
pip install -r requirements.txt
pip install -e skillful_nowcasting

# chmod +x ./s5cmd

python -u run.py --pretrained_model_path models/dgmr \
                    --train_data_dir data/zuimei-radar-cropped/train \
                    --valid_data_dir data/zuimei-radar-cropped/valid \
                    --output_dir checkpoint/dgmr_forecast20_ep2 \
                    --mixed_precision 16-mixed \
                    --accelerator_device gpu \
                    --num_devices 4 \
                    --strategy ddp \
                    --num_train_epochs 1 \
                    --dataloader_num_workers 8 \
                    # --max_train_samples 100
        