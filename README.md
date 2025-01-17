## simple diffusion: End-to-end diffusion for high resolution images
### Unofficial PyTorch Implementation 

**Simple diffusion: End-to-end diffusion for high resolution images**
[Emiel Hoogeboom](https://arxiv.org/search/cs?searchtype=author&query=Hoogeboom,+E), [Jonathan Heek](https://arxiv.org/search/cs?searchtype=author&query=Heek,+J), [Tim Salimans](https://arxiv.org/search/cs?searchtype=author&query=Salimans,+T)
https://arxiv.org/abs/2301.11093

![alt text](https://github.com/faverogian/simpleDiffusion/blob/main/assets/simpleDiffusion.png?raw=true)

### Requirements
* All testing and development was conducted on 4x 16GB NVIDIA V100 GPUs
* 64-bit Python 3.8 and PyTorch 2.1 (or later). See  [https://pytorch.org](https://pytorch.org/)  for PyTorch install instructions.

For convenience, a `requirements.txt` file is included to install the required dependencies in an environment of your choice.

### Usage

The code for training a diffusion model is self-contained in the `simpleDiffusion` class. Set-up and preparation is included in the `train.py` file:

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
Multiple versions of the U-Net architecture are available (UNet2DModel, ADM), with U-ViT and others planning to be included in the future.

### Multi-GPU Training
The `simpleDiffusion` class is equipped with HuggingFace's [Accelerator](https://huggingface.co/docs/accelerate/en/index) wrapper for distributed training. Multi-GPU training is easily done via:
`accelerate launch --multi-gpu train.py`

### Sample Results
A pre-trained model and results can be seen at the HuggingFace [model card](https://huggingface.co/faverogian/Smithsonian128UNet) for this project.

### Citations

    @inproceedings{Hoogeboom2023simpleDE,
	    title   = {simple diffusion: End-to-end diffusion for high resolution images},
	    author  = {Emiel Hoogeboom and Jonathan Heek and Tim Salimans},
	    year    = {2023}
	    }
    
    @InProceedings{pmlr-v139-nichol21a,
	    title       = {Improved Denoising Diffusion Probabilistic Models},
	    author      = {Nichol, Alexander Quinn and Dhariwal, Prafulla},
	    booktitle   = {Proceedings of the 38th International Conference on Machine Learning},
	    pages       = {8162--8171},
	    year        = {2021},
	    editor      = {Meila, Marina and Zhang, Tong},
	    volume      = {139},
	    series      = {Proceedings of Machine Learning Research},
	    month       = {18--24 Jul},
	    publisher   = {PMLR},
	    pdf         = {http://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf},
	    url         = {https://proceedings.mlr.press/v139/nichol21a.html}
	    }

    @inproceedings{Hang2023EfficientDT,
	    title   = {Efficient Diffusion Training via Min-SNR Weighting Strategy},
	    author  = {Tiankai Hang and Shuyang Gu and Chen Li and Jianmin Bao and Dong Chen and Han Hu and Xin Geng and Baining Guo},
	    year    = {2023}
	    }
