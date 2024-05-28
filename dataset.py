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

        # Load images based on their file extension
        self.input_image = self.load_image(input_image_path)
        self.target_image = self.load_image(target_image_path)

        # Ensure data has a channel dimension
        if self.input_image.ndim == 2:
            self.input_image = self.input_image[np.newaxis, ...]  # Add channel dimension (C, H, W)
        if self.target_image.ndim == 2:
            self.target_image = self.target_image[np.newaxis, ...]  # Add channel dimension (C, H, W)

    def load_image(self, image_path):
        """
        Load an image from the given path and convert it to grayscale.

        Parameters:
        - image_path: Path to the image file.

        Returns:
        - image: Numpy array of the grayscale image.
        """
        with Image.open(image_path) as img:
            if img.mode == 'I;16':  # Check if the image is 16-bit grayscale
                image = np.array(img, dtype=np.uint16)
            else:
                image = np.array(img.convert('L'), dtype=np.uint8)  # Convert to 8-bit grayscale if not 16-bit
        return image

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
