"""
A script for training a diffusion model on the Smithsonian Butterflies dataset following the simpleDiffusion paradigm.

This script uses the UNet2D model from the diffusion library and the simpleDiffusion model from the simple_diffusion library. 
"""

from nets.unet import UNetCondition2D
from nets.uvit import UViT
from utils.wavelet import wavelet_dec_2
from utils.plotter import plot_rgb
from diffusion.simple_diffusion import simpleDiffusion

from datasets import load_dataset
from torchvision import transforms
import torch
from diffusers.optimization import get_cosine_schedule_with_warmup

class TrainingConfig:
    # Optimization parameters
    learning_rate = 5e-5
    lr_warmup_steps = 10000
    train_batch_size = 4
    gradient_accumulation_steps = 1
    ema_beta = 0.9999
    ema_warmup = 500
    ema_update_freq = 10

    # Experiment parameters
    resume = False # whether to resume training from a checkpoint
    num_epochs = 200 # the number of training epochs
    save_image_epochs = 50 # how often to save generated images
    evaluation_batches = 1 # the number of batches to use for evaluation
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    experiment_path = "/home/mila/g/gian.favero/simpleDiffusion/ddpm-butterflies-128" # codebase root directory
    
    # Model parameters
    image_size = 128  # the generated image resolution
    backbone = "unet"  # the backbone model to use, either 'unet' or 'uvit'

    # Diffusion parameters
    pred_param = "v" # 'v', 'eps'
    schedule = "shifted_cosine" # shifted_cosine, cosine, shifted_cosine_interpolated
    noise_d = 64 # base noise dimension to shift to
    sampling_steps = 128 # number of steps to sample with

    seed = 0

def main():
    
    config = TrainingConfig

    dataset_name = "huggan/smithsonian_butterflies_subset"

    dataset = load_dataset(dataset_name, split="train")

    preprocess = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]

        # Do a wavelet decomposition on the images
        images = [wavelet_dec_2(image) / 2 for image in images]

        return {"images": images}

    dataset.set_transform(transform)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
    )

    # Images are 128x128x3 -> 64x64x12 after wavelet decomposition
    if config.backbone == "unet":
        # ADM channel maps for 64x64 images with more ResBlocks at 16x16
        backbone = UNetCondition2D(
            sample_size=config.image_size,  # the target image resolution
            in_channels=12,  # the number of input channels, 3 for RGB images
            out_channels=12,  # the number of output channels
            layers_per_block=(1,2,8,2),  # how many ResNet layers to use per UNet block
            block_out_channels=(128,256,512,768),  # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            mid_block_type="UNetMidBlock2D",
        )
    elif config.backbone == "uvit":
        raise ValueError(f"Invalid backbone: {config.backbone}")
    else:
        raise ValueError(f"Invalid backbone: {config.backbone}")

    optimizer = torch.optim.Adam(backbone.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=len(train_loader) * config.num_epochs,
    )

    diffusion_model = simpleDiffusion(
        backbone=backbone,
        config=config,
    )

    diffusion_model.train_loop(
        optimizer=optimizer,
        train_dataloader=train_loader,
        val_dataloader=train_loader,
        lr_scheduler=lr_scheduler,
        plot_function=plot_rgb,
    )

if __name__ == '__main__':
    main()