python -u train/run.py --pretrained_model_path models/dgmr \
                    --train_data_dir data/zuimei-radar-cropped/train \
                    --valid_data_dir data/zuimei-radar-cropped/valid \
                    --output_dir checkpoint/dgmr_forecast20_ep2 \
                    --mixed_precision 16-mixed \
                    --accelerator_device gpu \
                    --num_devices 4 \
                    --strategy ddp \
                    --num_train_epochs 2 \
                    --dataloader_num_workers 4 \
                    --checkpointing_steps 500
                    # --max_train_samples 100
        