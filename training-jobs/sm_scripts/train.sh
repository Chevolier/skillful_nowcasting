#!/bin/bash

git clone https://github.com/openclimatefix/skillful_nowcasting.git

pip install -r requirements.txt
pip install -e skillful_nowcasting


python -u run.py --num_input_frames 4 --num_forecast_frames 18 \
                    --pretrained_model_path models/dgmr \
                    --train_data_dir data/zuimei-radar-cropped/train \
                    --valid_data_dir data/zuimei-radar-cropped/valid \
                    --output_dir checkpoint/dgmr_forecast18_ep50_max_nonzero0.9\
                    --num_train_epochs 5 --train_batch_size 1 --valid_batch_size 1\
                    --mixed_precision bf16-mixed \
                    --accelerator_device gpu \
                    --num_devices 8 \
                    --strategy ddp \
                    --dataloader_num_workers 8 \
                    --validation_steps 5 \
                    --checkpointing_steps 10 \
                    --max_nonzero_ratio 0.9
        