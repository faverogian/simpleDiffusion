"""
This file contains an implementation of the pseudocode from the paper 
"Simple Diffusion: End-to-End Diffusion for High Resolution Images" 
by Emiel Hoogeboom, Tim Salimans, and Jonathan Ho.

Reference:
Hoogeboom, E., Salimans, T., & Ho, J. (2023). 
Simple Diffusion: End-to-End Diffusion for High Resolution Images. 
Retrieved from https://arxiv.org/abs/2301.11093
"""

import os
import math
import torch
import torch.nn as nn
from torch.special import expm1
from accelerate import Accelerator
from tqdm import tqdm
from ema_pytorch import EMA
import time

# helper
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

class simpleDiffusion(nn.Module):
    def __init__(
        self, 
        backbone: nn.Module,
        config: dict,
    ):
        super().__init__()

        # Training configuration
        self.config = config

        # Training objective
        pred_param = self.config.pred_param
        assert pred_param in ['v', 'eps'], "Invalid prediction parameterization. Must be 'v' or 'eps'"
        self.pred_param = pred_param

        # Sampling schedule
        schedule = self.config.schedule
        assert schedule in ['cosine', 'shifted_cosine', 'shifted_cosine_interpolated'], "Invalid schedule. Must be 'cosine', 'shifted_cosine', or 'shifted_cosine_interpolated'"
        if schedule == 'cosine':
            self.schedule = self.logsnr_schedule_cosine
        elif schedule == 'shifted_cosine':
            self.schedule = self.logsnr_schedule_cosine_shifted
            self.noise_d = self.config.noise_d
            self.image_d = self.config.image_size
        elif schedule == 'shifted_cosine_interpolated':
            self.schedule = self.logsnr_schedule_cosine_interpolated
            self.noise_d1 = self.config.image_size // 2
            self.noise_d2 = self.config.image_size // 8
            self.image_d = self.config.image_size

        # Model
        assert isinstance(backbone, nn.Module), "Model must be an instance of torch.nn.Module."
        self.model = backbone

        # EMA version of the model
        self.ema = EMA(
            self.model,
            beta=config.ema_beta,
            update_after_step=config.ema_warmup,
            update_every=config.ema_update_freq,
        )

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Number of parameters: {num_params}")

    def diffuse(self, x, alpha_t, sigma_t):
        """
        Function to diffuse the input tensor x to a timepoint t with the given alpha_t and sigma_t.

        Args:
        x (torch.Tensor): The input tensor to diffuse.
        alpha_t (torch.Tensor): The alpha value at timepoint t.
        sigma_t (torch.Tensor): The sigma value at timepoint t.

        Returns:
        z_t (torch.Tensor): The diffused tensor at timepoint t.
        eps_t (torch.Tensor): The noise tensor at timepoint t.
        """
        eps_t = torch.randn_like(x)

        z_t = alpha_t * x + sigma_t * eps_t

        return z_t, eps_t

    def logsnr_schedule_cosine(self, t, logsnr_min=-15, logsnr_max=15):
        """
        Function to compute the logSNR schedule at timepoint t with cosine:

        logSNR(t) = -2 * log (tan (pi * t / 2))

        Taking into account boundary effects, the logSNR value at timepoint t is computed as:

        logsnr_t = -2 * log(tan(t_min + t * (t_max - t_min)))

        Args:
        t (int): The timepoint t.
        logsnr_min (int): The minimum logSNR value.
        logsnr_max (int): The maximum logSNR value.

        Returns:
        logsnr_t (float): The logSNR value at timepoint t.
        """
        logsnr_max = logsnr_max + math.log(self.noise_d / self.image_d)
        logsnr_min = logsnr_min + math.log(self.noise_d / self.image_d)
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))

        logsnr_t = -2 * log(torch.tan((t_min + t * (t_max - t_min)).clone().detach()))

        return logsnr_t

    def logsnr_schedule_cosine_shifted(self, t):
        """
        Function to compute the logSNR schedule at timepoint t with shifted cosine:

        logSNR_shifted(t) = logSNR(t) + 2 * log(noise_d / image_d)

        Args:
        t (int): The timepoint t.

        Returns:
        logsnr_t_shifted (float): The logSNR value at timepoint t.
        """
        logsnr_t = self.logsnr_schedule_cosine(t)
        logsnr_t_shifted = logsnr_t + 2 * math.log(self.noise_d / self.image_d)

        return logsnr_t_shifted
    
    def logsnr_schedule_cosine_interpolated(self, t):
        """
        Function to compute the logSNR schedule at timepoint t with interpolation:

        logSNR(t) = tlogSNR(t)_shift1 + (1 - t)logSNR(t)_shift2

        Args:
        t (int): The timepoint t.

        Returns:
        logsnr_t_interpolated (float): The logSNR value at timepoint t.

        Assumes shift_1 is 1/2 the full resolution and shift_2 is 1/8 the full resolution.
        """

        logsnr_t = self.logsnr_schedule_cosine(t)
        logsnr_t_shift1 = logsnr_t + 2 * math.log(self.noise_d1 / self.image_d)
        logsnr_t_shift2 = logsnr_t + 2 * math.log(self.noise_d2 / self.image_d)

        logsnr_t_interpolated = t * logsnr_t_shift1 + (1 - t) * logsnr_t_shift2

        return logsnr_t_interpolated
        
    def clip(self, x):
        """
        Function to clip the input tensor x to the range [-1, 1].

        Args:
        x (torch.Tensor): The input tensor to clip.

        Returns:
        x (torch.Tensor): The clipped tensor.
        """
        return torch.clamp(x, -1, 1)

    @torch.no_grad()
    def ddpm_sampler_step(self, z_t, pred, logsnr_t, logsnr_s):
        """
        Function to perform a single step of the DDPM sampler.

        Args:
        z_t (torch.Tensor): The diffused tensor at timepoint t.
        pred (torch.Tensor): The predicted value from the model (v or eps).
        logsnr_t (float): The logSNR value at timepoint t.
        logsnr_s (float): The logSNR value at the sampling timepoint s.

        Returns:
        z_s (torch.Tensor): The diffused tensor at sampling timepoint s.
        """
        c = -expm1(logsnr_t - logsnr_s)
        alpha_t = torch.sqrt(torch.sigmoid(logsnr_t))
        alpha_s = torch.sqrt(torch.sigmoid(logsnr_s))
        sigma_t = torch.sqrt(torch.sigmoid(-logsnr_t))
        sigma_s = torch.sqrt(torch.sigmoid(-logsnr_s))

        if self.pred_param == 'v':
            x_pred = alpha_t * z_t - sigma_t * pred
        elif self.pred_param == 'eps':
            x_pred = (z_t - sigma_t * pred) / alpha_t

        x_pred = self.clip(x_pred)

        mu = alpha_s * (z_t * (1 - c) / alpha_t + c * x_pred)
        variance = (sigma_s ** 2) * c

        return mu, variance

    @torch.no_grad()
    def sample(self, x):
        """
        Standard DDPM sampling procedure. Begun by sampling z_T ~ N(0, 1)
        and then repeatedly sampling z_s ~ p(z_s | z_t)

        Args:
        x_shape (tuple): The shape of the input tensor.

        Returns:
        x_pred (torch.Tensor): The predicted tensor.
        """
        z_t = torch.randn(x.shape).to(x.device)

        # Create evenly spaced steps, e.g., for sampling_steps=5 -> [1.0, 0.8, 0.6, 0.4, 0.2]
        steps = torch.linspace(1.0, 0.0, self.config.sampling_steps + 1)

        for i in range(len(steps) - 1):  # Loop through steps, but stop before the last element
            
            u_t = steps[i]  # Current step
            u_s = steps[i + 1]  # Next step

            logsnr_t = self.schedule(u_t).to(x.device).unsqueeze(0)
            logsnr_s = self.schedule(u_s).to(x.device).unsqueeze(0)

            pred = self.ema(z_t, logsnr_t)

            mu, variance = self.ddpm_sampler_step(z_t, pred, logsnr_t.clone().detach(), logsnr_s.clone().detach())
            z_t = mu + torch.randn_like(mu) * torch.sqrt(variance)

        # Final step
        logsnr_1 = self.schedule(steps[-2]).to(x.device).unsqueeze(0)
        logsnr_0 = self.schedule(steps[-1]).to(x.device).unsqueeze(0)

        pred = self.model(z_t, logsnr_1)
        x_pred, _ = self.ddpm_sampler_step(z_t, pred, logsnr_1.clone().detach(), logsnr_0.clone().detach())
        
        x_pred = self.clip(x_pred)

        return x_pred
    
    def loss(self, x):
        """
        A function to compute the loss of the model. The loss is computed as the mean squared error
        between the predicted noise tensor and the true noise tensor. Various prediction parameterizations
        imply various weighting schemes as outlined in Kingma et al. (2023)

        Args:
        x (torch.Tensor): The input tensor.

        Returns:
        loss (torch.Tensor): The loss value.
        """
        t = torch.rand(x.shape[0])

        logsnr_t = self.schedule(t).to(x.device)
        alpha_t = torch.sqrt(torch.sigmoid(logsnr_t)).view(-1, 1, 1, 1).to(x.device)
        sigma_t = torch.sqrt(torch.sigmoid(-logsnr_t)).view(-1, 1, 1, 1).to(x.device)
        z_t, eps_t = self.diffuse(x, alpha_t, sigma_t)
        pred = self.model(z_t, logsnr_t)

        if self.pred_param == 'v':
            eps_pred = sigma_t * z_t + alpha_t * pred
        else: 
            eps_pred = pred

        # Apply min-SNR weighting (https://arxiv.org/pdf/2303.09556)
        snr = torch.exp(logsnr_t).clamp_(max = 5)
        if self.pred_param == 'v':
            weight = 1 / (1 + snr)
        else:
            weight = 1 / snr

        weight = weight.view(-1, 1, 1, 1)

        loss = torch.mean(weight * (eps_pred - eps_t) ** 2)

        return loss

    def train_loop(
        self, 
        optimizer, 
        train_dataloader, 
        val_dataloader,
        lr_scheduler, 
        plot_function,
    ):
        """
        A function to train the model.

        Args:
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        train_dataloader (torch.utils.data.DataLoader): The training dataloader.
        val_dataloader (torch.utils.data.DataLoader): The validation dataloader.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        metrics (list): A list of metrics to evaluate.
        plot_function (function): The function to use for plotting the samples.

        Returns:
        None
        """

        # Initialize accelerator
        accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            project_dir=self.config.experiment_path,
        )

        # Prepare the model, optimizer, dataloaders, and learning rate scheduler
        model, ema, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare( 
            self.model, self.ema, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )

        # Check if resume training is enabled
        if self.config.resume:
            checkpoint_path = os.path.join(self.config.experiment_path, "checkpoints")
            start_epoch = self.load_checkpoint(checkpoint_path, accelerator)
        else: # Set up fresh experiment
            start_epoch = 0

        # Train!
        if accelerator.is_main_process:
            print(self.config.__dict__)

        for epoch in range(start_epoch, self.config.num_epochs):
            epoch_start_time = time.time()

            model.train()

            for _, batch in enumerate(train_dataloader):
                x = batch["images"]

                loss = self.loss(x)
                loss = loss.to(next(model.parameters()).dtype)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    model_params = dict(model.named_parameters())
                    accelerator.clip_grad_norm_(model_params.values(), max_norm=1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                ema.update()

            epoch_elapsed = time.time() - epoch_start_time
            if accelerator.is_main_process:
                print(f"Epoch {epoch}/{self.config.num_epochs}: {epoch_elapsed:.2f} s.")

            # Run an evaluation
            if epoch % self.config.save_image_epochs == 0 or epoch == self.config.num_epochs - 1:
                val_evaluation_start_time = time.time()

                model.eval()

                # Make directory for saving images
                training_images_path = os.path.join(self.config.experiment_path, "training_images/")
        
                val_samples = self.evaluate(
                    val_dataloader,
                    stop_idx=self.config.evaluation_batches,
                )

                # Use a provided plot function to plot the samples
                if plot_function is not None:
                    plot_function(
                        output_dir=training_images_path,
                        samples=val_samples,
                        epoch=epoch,
                        process_idx=accelerator.state.process_index,
                    )

                val_evaluation_elapsed = time.time() - val_evaluation_start_time
                if accelerator.is_main_process:
                    print(f"Validation evaluation time: {val_evaluation_elapsed:.2f} s.")
                    self.save_checkpoint(accelerator, epoch)

    @torch.no_grad()
    def evaluate(
        self, 
        val_dataloader,
        stop_idx=None
    ):
        """
        A function to evaluate the model.

        Args:
        val_dataloader (torch.utils.data.DataLoader): The dataloader to use for evaluation.
        stop_idx (int): The number of batches to evaluate.

        Returns:
        samples (list): A list of samples.
        batches (int): The number of batches evaluated.
        """

        val_samples = []

        # Make a progress bar
        progress_bar = tqdm(val_dataloader, desc="Evaluating")
        for idx, batch in enumerate(val_dataloader):
            progress_bar.update(1)

            batch = {k: v for k, v in batch.items()}

            x = batch["images"]

            sample = self.sample(x)
            val_samples.append(sample)

            if stop_idx is not None and idx >= stop_idx:
                break

        return val_samples
    
    def save_checkpoint(self, accelerator, epoch):
        """
        A function to save the model checkpoint.

        Args:
        accelerator (accelerate.Accelerator): The accelerator object.
        epoch (int): The current epoch.

        Returns:
        None
        """
        checkpoint_dir = os.path.join(self.config.experiment_path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save accelerator state
        accelerator.save_state(output_dir=checkpoint_dir)

        # Save experiment state
        experiment_state = {
            "epoch": epoch,
        }

        # Save new checkpoint
        latest_exp_state_path = os.path.join(checkpoint_dir, "experiment_state.pth")
        torch.save(experiment_state, latest_exp_state_path)

        print(f"Checkpoint saved to {latest_exp_state_path}")

    def load_checkpoint(self, checkpoint_path, accelerator):
        """
        A function to load the model checkpoint.

        Args:
        checkpoint_path (str): The path to the checkpoint.
        accelerator (accelerate.Accelerator): The accelerator object.

        Returns:
        start_epoch (int): The starting epoch.
        """
        # Load accelerator state
        accelerator.load_state(checkpoint_path)

        # Load experiment state
        experiment_state_path = os.path.join(checkpoint_path, "experiment_state.pth")
        
        # Load the checkpoint file on CPU
        checkpoint = torch.load(experiment_state_path, map_location="cpu", weights_only=False)

        # Restore the epoch to resume training from
        epoch = checkpoint["epoch"]

        print(f"Loaded checkpoint from {checkpoint_path}")

        return epoch