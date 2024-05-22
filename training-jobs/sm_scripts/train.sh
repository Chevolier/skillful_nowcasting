#!/bin/bash

git clone https://github.com/openclimatefix/skillful_nowcasting.git

pip install -r requirements.txt
pip install -e skillful_nowcasting

#                     --pretrained_model_path models/dgmr \
python -u run.py --num_input_frames 4 --num_forecast_frames 18 \
                    --train_data_dir /tmp/data/zuimei-radar-cropped/train \
                    --valid_data_dir /tmp/data/zuimei-radar-cropped/valid \
                    --output_dir checkpoint/dgmr_forecast18_train50k_ep10_init\
                    --num_train_epochs 10 --train_batch_size 1 --valid_batch_size 1\
                    --mixed_precision bf16-mixed \
                    --accelerator_device gpu \
                    --num_devices 8 \
                    --strategy ddp \
                    --dataloader_num_workers 8 \
                    --validation_steps 1000 \
                    --checkpointing_steps 1000
        