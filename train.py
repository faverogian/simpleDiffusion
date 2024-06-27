"""
A script for training a diffusion model on the Smithsonian Butterflies dataset following the simpleDiffusion paradigm.

This script uses the UNet2D model from the diffusion library and the simpleDiffusion model from the simple_diffusion library. 
"""

from diffusion.unet import UNet2D
from diffusion.simple_diffusion import simpleDiffusion

from datasets import load_dataset
from torchvision import transforms
import torch
from diffusers.optimization import get_cosine_schedule_with_warmup


class TrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 4
    num_epochs = 100
    gradient_accumulation_steps = 1
    learning_rate = 5e-5
    lr_warmup_steps = 10000
    save_image_epochs = 100
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-butterflies-128"  # the model name locally and on the HF Hub
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
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
        return {"images": images}

    dataset.set_transform(transform)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
    )

    unet = UNet2D(
        sample_size=config.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    optimizer = torch.optim.Adam(unet.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=len(train_loader) * config.num_epochs,
    )

    diffusion_model = simpleDiffusion(
        unet=unet,
        image_size=config.image_size
    )

    diffusion_model.train_loop(
        config=config,
        optimizer=optimizer,
        train_dataloader=train_loader,
        lr_scheduler=lr_scheduler
    )


if __name__ == '__main__':
    main()