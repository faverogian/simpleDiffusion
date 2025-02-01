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

The code for training a diffusion model is self-contained in the `simpleDiffusion` class. Set-up and preparation is included in the `train.py` file, all that is needed is to modify the `TrainingConfig` class .

Multiple diffusion backbones are available (UNet, U-ViT) and others planning to be included in the future (DiT).

### Multi-GPU Training
The `simpleDiffusion` class is equipped with HuggingFace's [Accelerator](https://huggingface.co/docs/accelerate/en/index) wrapper for distributed training. Multi-GPU training is easily done via:
`accelerate launch --multi-gpu train.py`

### Standard Diffusion Implementations:
1. DDPM (ancestral) sampling
2. Continuous time sampling (t $ \in $ [0, 1])
3. Variational diffusion model formulation (logSNR = f(t))
4. Min-SNR loss weighting

### Diffusion Improvements from Paper:
1. Shifted cosine noise schedule with specified base noise dimension 
2. Interpolated (shifted) cosine noise schedule
3. Scaling the architecture (UNet2DCondition model)
4. Single-stage wavelet decomposition to reduce image size

### Architectures from Paper:
1. UNet (with variable ResBlocks, attention)

### TODO:
1. UNet (variable dropout resolutions)
2. UViT architecture

Note: Multi-scale training loss was not implemented as this did not significantly improve performance as reported in the paper.

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
