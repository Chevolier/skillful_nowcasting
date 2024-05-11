import wandb
import torch.utils.data.dataset
from datasets import load_dataset
from pytorch_lightning import (
    LightningDataModule,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from dgmr import DGMR

wandb.init(project="dgmr")
# wandb.init(mode="disabled")
from pathlib import Path

import numpy as np
from numpy.random import default_rng
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

import random
import argparse
import tarfile
from collections import defaultdict

import os

# os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

NUM_INPUT_FRAMES = 4
NUM_FORECAST_FRAMES = 20

def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class WatchModel(Callback):
    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq, log_graph=True)


class UploadCheckpointsAsArtifact(Callback):
    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    @rank_zero_only
    def on_keyboard_interrupt(self, trainer, pl_module):
        self.on_train_end(trainer, pl_module)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in Path(self.ckpt_dir).rglob("*.ckpt"):
                ckpts.add_file(str(path))

        experiment.log_artifact(ckpts)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in Path(self.ckpt_dir).rglob("*.ckpt"):
                ckpts.add_file(str(path))

        experiment.log_artifact(ckpts)

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in Path(self.ckpt_dir).rglob("*.ckpt"):
                ckpts.add_file(str(path))

        experiment.log_artifact(ckpts)


def revert_back_numpy_array(byte_array, size=(24, 256, 256), dtype=np.float32, source_dtype = np.float32):
    # Load the flattened data from disk
    flattened_data = bytearray(byte_array) 

    # Convert the bytearray to a numpy array
    flattened_array = np.frombuffer(flattened_data, dtype=source_dtype)
    # Reshape the flattened array to the original shape
    original_array = flattened_array.reshape(size).astype(dtype)
    
    return original_array


def collate_fn(examples):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    
    inputs, targets = [], []
    for i, example in enumerate(examples):
        cropped_frames_max_nonzero = revert_back_numpy_array(example["cropped_frames_max_nonzero"], size=(24, 256, 256), dtype=np.float32)
        max_pos = revert_back_numpy_array(example["max_pos"], size=(2), dtype=np.uint8, source_dtype=np.float32)
        
        cropped_frames_random = revert_back_numpy_array(example["cropped_frames_random"], size=(24, 256, 256), dtype=np.float32)
        random_pos = revert_back_numpy_array(example["random_pos"], size=(2), dtype=np.uint8, source_dtype=np.float32)
        
        if random.random() < 0.5:
            input_frames = cropped_frames_max_nonzero[:NUM_INPUT_FRAMES, ...]
            target_frames = cropped_frames_max_nonzero[NUM_INPUT_FRAMES:NUM_INPUT_FRAMES+NUM_FORECAST_FRAMES, ...]
        else:
            input_frames = cropped_frames_random[:NUM_INPUT_FRAMES, ...]
            target_frames = cropped_frames_random[NUM_INPUT_FRAMES:NUM_INPUT_FRAMES+NUM_FORECAST_FRAMES, ...]
                        
        inputs.append(input_frames)
        targets.append(target_frames)
        
    inputs_tensor = torch.Tensor(np.stack(inputs)).unsqueeze(2)
    targets_tensor = torch.Tensor(np.stack(targets)).unsqueeze(2)
    
    return inputs_tensor, targets_tensor


def count_distinct_files(folder_path):
    distinct_files = defaultdict(set)
    for filename in os.listdir(folder_path):
        if filename.endswith('.tar'):
            tar_path = os.path.join(folder_path, filename)
            with tarfile.open(tar_path, 'r') as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        file_name = os.path.basename(member.name)
                        file_prefix, _ = os.path.splitext(file_name)
                        distinct_files[file_prefix].add(file_name)
    return len(distinct_files)
    

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a DGMR training script.")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--valid_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the validation data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoint",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    # parser.add_argument(
    #     "--args.num_input_frames", type=int, default=4, help="Number of input frames."
    # )
    # parser.add_argument(
    #     "--num_forecast_frames", type=int, default=18, help="Number of forecasted frames."
    # )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["32", "16-mixed", "bf16-mixed"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--accelerator_device",
        type=str,
        default="auto",
        choices=["auto", "gpu", "cpu", "tpu"],
        help=(
            "accelerator device"
        ),
    )
    parser.add_argument(
        "--num_devices", type=int, default=1, help="Number of GPU devices."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="ddp",
        choices=["ddp", "ddp_find_unused_parameters_true"],
        help=(
            "strategy"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args
    
if __name__ == "__main__":
    
    args = parse_args()
    
    print("*****************start cp data and pretrained models*****************************")
    os.system("chmod +x ./s5cmd")
    os.system("./s5cmd sync {0}* {1}".format(os.environ['TRAIN_DATA_PATH'], args.train_data_dir))
    os.system("./s5cmd sync {0}* {1}".format(os.environ['VALID_DATA_PATH'], args.valid_data_dir))
    os.system("./s5cmd sync {0}* {1}".format(os.environ['PRETRAINED_MODEL_S3_PATH'], args.pretrained_model_path))
    
    ## download the latest checkpoint
    # os.system("./s5cmd sync {0}* {1}".format(os.environ['LATEST_CHECKPOINT_S3_PATH'], args.output_dir+'/checkpoint-60000')) 

    wandb_logger = WandbLogger(logger="dgmr")
    model_checkpoint = ModelCheckpoint(
        monitor="train/g_loss",
        dirpath=args.output_dir,
        filename="best",
    )
    
    train_dataset = load_dataset("webdataset", 
                    data_files={"train": os.path.join(args.train_data_dir,"*.tar")}, 
                    split="train", 
                    streaming=True)
    
    if args.max_train_samples:
        train_dataset_len = args.max_train_samples
    else:
        train_dataset_len = count_distinct_files(args.train_data_dir)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    valid_dataset = load_dataset("webdataset", 
                    data_files={"valid": os.path.join(args.valid_data_dir,"*.tar")}, 
                    split="valid", 
                    streaming=True)
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    trainer = Trainer(
        max_epochs=args.num_train_epochs,
        logger= wandb_logger,
        callbacks=[model_checkpoint],
        accelerator=args.accelerator_device,
        devices=args.num_devices,
        precision=args.mixed_precision,  # "16-mixed"
        strategy= args.strategy,  # "ddp_find_unused_parameters_true" # , "ddp", DDPStrategy(find_unused_parameters=True)
        limit_train_batches=train_dataset_len//(args.train_batch_size*args.num_devices)
    )
    
    if args.pretrained_model_path:
        model = DGMR.from_pretrained(args.pretrained_model_path)
    else:
        model = DGMR(forecast_steps=NUM_FORECAST_FRAMES, generation_steps=6)

    trainer.fit(model, train_loader, valid_loader)
    
     # upload checkpoint to s3
    ############################
    persistant_path = os.environ['OUTPUT_MODEL_S3_PATH'] + str(datetime.now().strftime("%m-%d-%Y-%H-%M-%S")) + '/'
    os.system("./s5cmd sync {0} {1}".format(args.output_dir, persistant_path)) # +'/best_model'
