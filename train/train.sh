#!/bin/bash

python -u train/run.py --num_input_frames 4 --num_forecast_frames 18 \
                    --pretrained_model_path models/dgmr \
                    --train_data_dir data/zuimei-radar-cropped/train \
                    --valid_data_dir data/zuimei-radar-cropped/valid \
                    --output_dir checkpoint/dgmr_forecast20_ep2 \
                    --num_train_epochs 1 --train_batch_size 1 --valid_batch_size 1\
                    --mixed_precision bf16-mixed \
                    --accelerator_device gpu \
                    --num_devices 1 \
                    --strategy ddp \
                    --dataloader_num_workers 4 \
                    --validation_steps 10 \
                    --checkpointing_steps 20 \
                    --checkpoints_total_limit 5
        