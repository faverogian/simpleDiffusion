import os
from utils.wavelet import wavelet_enc_2
import matplotlib.pyplot as plt

# Simple RGB plotter
def plot_rgb(
        output_dir: str, 
        samples: list, 
        epoch: int, 
        process_idx: int
    ):
    """
    Plot RGB images.

    Args:
        output_dir (str): The output directory.
        val_samples (list): The validation samples.
        epoch (int): The epoch.
        process_idx (int): The process index.

    Returns:
        None
    """

    for i, sample in enumerate(samples):
        
        for j in range(1):

            sample_item = sample[j] * 2
            sample_item = wavelet_enc_2(sample_item) * 0.5 + 0.5

            pred = sample_item.cpu().detach().numpy().transpose(1, 2, 0)

            plt.imshow(pred)
            plt.axis('off')

            # Make sure the output directory exists
            os.makedirs(output_dir, exist_ok=True)

            plt.savefig(
                f"{output_dir}/process_{process_idx}_epoch_{epoch}_sample_{i}.png"
            )
            plt.close()

    return None
