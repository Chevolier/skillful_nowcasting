#!/bin/bash

git clone https://github.com/openclimatefix/skillful_nowcasting.git

pip install -r requirements.txt
pip install -e skillful_nowcasting

# #                    --pretrained_model_path models/dgmr \

python -u run.py --num_input_frames 4 --num_forecast_frames 18 \
                    --train_data_dir data/zuimei-radar-cropped/train \
                    --valid_data_dir data/zuimei-radar-cropped/valid \
                    --output_dir checkpoint/dgmr_forecast18_ep50 \
                    --num_train_epochs 20 --train_batch_size 1 --valid_batch_size 1\
                    --mixed_precision 16-mixed \
                    --accelerator_device gpu \
                    --num_devices 8 \
                    --strategy ddp \
                    --dataloader_num_workers 8 \
                    --validation_steps 50 \
                    --checkpointing_steps 100 \
                    --checkpoints_total_limit 10 \
                    --max_nonzero_ratio 0.8
        
        