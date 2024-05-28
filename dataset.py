import os
import numpy as np
from PIL import Image
import torch

class N2NJPGDataset(torch.utils.data.Dataset):
    def __init__(self, input_image_path, target_image_path, transform=None):
        """
        Initializes the dataset by loading a single pair of noisy images.

        Parameters:
        - input_image_path: Path to the input noisy image file.
        - target_image_path: Path to the target noisy image file.
        - transform: Optional transform to be applied to each tuple of images.
        """
        self.transform = transform

        # Load images as grayscale and add a channel dimension preemptively
        self.input_image = np.array(Image.open(input_image_path).convert('L'))
        self.target_image = np.array(Image.open(target_image_path).convert('L'))

        # Ensure data has a channel dimension
        if self.input_image.ndim == 2:
            self.input_image = self.input_image[np.newaxis, ...]  # Add channel dimension (C, H, W)
        if self.target_image.ndim == 2:
            self.target_image = self.target_image[np.newaxis, ...]  # Add channel dimension (C, H, W)

    def __len__(self):
        # As there's only one pair of images, the length is 1
        return 1

    def __getitem__(self, index):
        # Check index since there's only one item
        if index != 0:
            raise IndexError("This dataset contains only one pair of images.")

        input_image, target_image = self.input_image, self.target_image
        
        # Apply the transform to the tuple if specified
        if self.transform:
            input_image, target_image = self.transform((input_image, target_image))

        return input_image, target_image


