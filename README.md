# Skillful Nowcasting with Deep Generative Model of Radar (DGMR)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-10-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
Implementation of DeepMind's Skillful Nowcasting GAN Deep Generative Model of Radar (DGMR) (https://arxiv.org/abs/2104.00954) in PyTorch Lightning.

This implementation matches as much as possible the pseudocode released by DeepMind. Each of the components (Sampler, Context conditioning stack, Latent conditioning stack, Discriminator, and Generator) are normal PyTorch modules. As the model training is a bit complicated, the overall architecture is wrapped in PyTorch Lightning.

The default parameters match what is written in the paper.

## Prepare Environment


Clone the repository, then run

```bash
conda create -n dgmr python=3.10

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
pip install -e .
````

## Prepare Training Data

1. Put radar data into folder data/zuimei-radar, in the following format.

```
zuimei-radar
├── 20240301
├── 20240302
├── 20240303
├── 20240304
└── 20240305
```

2. Run data processing, e.g., cropping and group into consecutive frames. Note that removing corrupted zip files only need to run once, after that could be commented out. If you have a lot of day, say >= 3 months, suggest use ml.g5.16xlarge, otherwise, the data processing process may be killed by the system.

```bash
python process_data.py --data_dir data/zuimei-radar --save_dir data/zuimei-radar-cropped  --crop_size 256 --num_thr 1000 --threshold 10 --num_per_tar 20 --num_total_frames 24
```

3. Then can split the data into the following folders.

```
data/zuimei-radar-cropped
├── train
│   ├── 000000.tar
│   ├── 000001.tar
├── valid
│   ├── 000000.tar
│   ├── 000001.tar
│   ├── 000002.tar
└── test
    ├── 000020.tar
    ├── 000021.tar
```


## Download Pretrained Weights

Pretrained weights are be available through [HuggingFace Hub](https://huggingface.co/openclimatefix), currently weights trained on the sample dataset. The whole DGMR model or different components can be loaded as the following:

Download the following pretrained weights into the corresponding folder models.

```
models
├── dgmr
│   ├── config.json
│   ├── pytorch_model.bin
│   └── README.md
├── dgmr-context-conditioning-stack
│   ├── config.json
│   └── pytorch_model.bin
├── dgmr-discriminator
│   ├── config.json
│   └── pytorch_model.bin
├── dgmr-latent-conditioning-stack
│   ├── config.json
│   └── pytorch_model.bin
└── dgmr-sampler
    ├── config.json
    └── pytorch_model.bin
```
    
## Training

we provide two training methods, either way is OK.

1. Direct training in SageMaker, suggest use ml.g5.48xlarge with 8 A10G GPUs.

Change the parameters in train.sh based on your data, the machine you used for training. 
For instance, if you have 8 GPUs, then set --num_devices 8, change 
--validation_steps  --checkpointing_steps larger if you have a lot of data, say 1000.

```bash
bash train/train.sh
```


2. Training using SageMaker training jobs.

Go to training-jobs folder and run train.ipynb step by step.


## Inference 

Follow the steps in inference.ipynb to do inference for both cropped radar images and raw data images.


## For more analysis, e.g., comparisons with pretrained models, use analyze.ipynb.

