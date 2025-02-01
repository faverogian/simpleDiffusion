import pywt
import torch

def wavelet_dec_2(images):
    """
    Perform a one-level wavelet decomposition on a tensor [C, H, W].
    """
    # On every channel in a batch, perform a DWT
    wavelet = 'haar' # Daubechies wavelet

    # Get the number of channels
    num_channels = images.shape[0]

    # Initialize the output tensor, which will have 4 times the number of channels
    wavelet_images = torch.zeros(
        4*num_channels, 
        images.shape[1]//2, 
        images.shape[2]//2
    ).to(images.device)

    # Loop through each channel
    for i in range(num_channels):
        # Get the channel
        channel = images[i, :, :]

        # Perform the DWT
        cA, (cH, cV, cD) = pywt.dwt2(channel, wavelet)

        # Store the DWT coefficients (as tensors)
        wavelet_images[4*i, :, :] = torch.tensor(cA)
        wavelet_images[4*i + 1, :, :] = torch.tensor(cH)
        wavelet_images[4*i + 2, :, :] = torch.tensor(cV)
        wavelet_images[4*i + 3, :, :] = torch.tensor(cD)

    return wavelet_images

def wavelet_enc_2(wavelet_images):
    """
    Perform a one-level wavelet reconstruction on a tensor [4*C, H, W].
    """
    # On every channel in a batch, perform a DWT
    wavelet = 'haar' # Daubechies wavelet

    # Get the number of channels
    num_channels = wavelet_images.shape[0] // 4

    # Initialize the output tensor, which will have 4 times less the number of channels
    images = torch.zeros(
        num_channels, 
        wavelet_images.shape[1]*2, 
        wavelet_images.shape[2]*2
    ).to(wavelet_images.device)

    # Loop through each channel
    for i in range(num_channels):
        # Get the DWT coefficients
        cA = wavelet_images[4*i, :, :].cpu().numpy()
        cH = wavelet_images[4*i + 1, :, :].cpu().numpy()
        cV = wavelet_images[4*i + 2, :, :].cpu().numpy()
        cD = wavelet_images[4*i + 3, :, :].cpu().numpy()

        # Perform the inverse DWT
        channel = pywt.idwt2((cA, (cH, cV, cD)), wavelet)

        # Store the channel
        images[i, :, :] = torch.tensor(channel)

    return images