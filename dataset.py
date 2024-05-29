import numpy as np
from PIL import Image
import torch

class N2NJPG8bitDataset(torch.utils.data.Dataset):
    def __init__(self, input_image_path, target_image_path, transform=None):
        """
        Initializes the dataset by loading a single pair of noisy images.

        Parameters:
        - input_image_path: Path to the input noisy image file.
        - target_image_path: Path to the target noisy image file.
        - transform: Optional transform to be applied to each tuple of images.
        """
        self.transform = transform

        # Load images and ensure they are 8-bit grayscale
        self.input_image = self.load_image(input_image_path)
        self.target_image = self.load_image(target_image_path)

        # Ensure data has a channel dimension
        if self.input_image.ndim == 2:
            self.input_image = self.input_image[..., np.newaxis]
        if self.target_image.ndim == 2:
            self.target_image = self.target_image[..., np.newaxis]

    def load_image(self, image_path):
        """
        Load an image from the given path and ensure it is 8-bit grayscale.

        Parameters:
        - image_path: Path to the image file.

        Returns:
        - image: Numpy array of the 8-bit grayscale image.
        
        Raises:
        - ValueError: If the image is not in 8-bit grayscale format.
        """
        with Image.open(image_path) as img:
            if img.mode != 'L':  # 'L' mode indicates 8-bit grayscale in PIL
                raise ValueError(f"Image at {image_path} is not in 8-bit grayscale format.")
            image = np.array(img, dtype=np.uint8)
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
