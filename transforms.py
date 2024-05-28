import numpy as np
import os
import sys
from skimage import filters, exposure
import matplotlib.pyplot as plt
from PIL import Image
import torch
import copy

from utils import *

class NormalizePair(object):
    """
    Normalize a pair of single-channel images using mean and standard deviation.

    Args:
        mean (float): Mean for the channel.
        std (float): Standard deviation for the channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, imgs):
        """
        Normalize each image in a tuple with dimensions (1, H, W).

        Args:
            imgs (tuple of numpy.ndarray): Tuple of images to be normalized, each expected to be in the format (1, H, W).

        Returns:
            tuple of numpy.ndarray: Tuple of normalized images.
        """
        input_image, target_image = imgs

        # Normalize the input image
        normalized_input_image = (input_image - self.mean) / self.std

        # Normalize the target image
        normalized_target_image = (target_image - self.mean) / self.std

        return (normalized_input_image, normalized_target_image)





class AddGaussianNoise(object):
    """
    Add Gaussian noise to a single-channel image.

    Args:
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.
    """

    def __init__(self, mean=0.0, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        Add Gaussian noise to a single-channel image with dimensions (1, H, W).

        Args:
            img (numpy.ndarray): Image to which noise will be added, expected to be in the format (1, H, W).

        Returns:
            numpy.ndarray: Image with added noise.
        """
        # Ensure the image array is float for accurate noise addition
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        
        # Generate Gaussian noise
        noise = np.random.normal(self.mean, self.std, img.shape)
        
        # Add the noise to the image
        noisy_img = img + noise

        #plot_as_image(noisy_img)

        return noisy_img



class RandomCrop(object):
    """
    Randomly crop a single-channel image to a specified size.
    
    Args:
        output_size (tuple): The target output size (height, width).
    """

    def __init__(self, output_size=(64, 64)):
        """
        Initializes the RandomCrop transformer with the desired output size.

        Parameters:
        - output_size (tuple): The target output size (height, width).
        """
        self.output_size = output_size

    def __call__(self, img):
        """
        Apply random cropping to a single-channel image with dimensions (1, H, W).

        Parameters:
        - img (numpy.ndarray): The image to be cropped, expected to be in the format (1, H, W).

        Returns:
        - numpy.ndarray: Randomly cropped image.
        """
        # Ensure that we're working with a single-channel image
        if img.ndim != 3 or img.shape[0] != 1:
            raise ValueError("Input image must have dimensions (1, H, W).")

        _, h, w = img.shape
        new_h, new_w = self.output_size

        if h > new_h and w > new_w:
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            cropped_img = img[:, top:top+new_h, left:left+new_w]
        else:
            # If the image is smaller than the crop size, padding is required
            padding_top = (new_h - h) // 2 if new_h > h else 0
            padding_left = (new_w - w) // 2 if new_w > w else 0
            padding_bottom = new_h - h - padding_top if new_h > h else 0
            padding_right = new_w - w - padding_left if new_w > w else 0

            cropped_img = np.pad(img, ((0, 0), (padding_top, padding_bottom), (padding_left, padding_right)),
                                 mode='constant', constant_values=0)  # Can modify padding mode and value if needed

        return cropped_img



class RandomCropPair(object):
    """
    Randomly crop a pair of single-channel images to a specified size.
    
    Args:
        output_size (tuple): The target output size (height, width).
    """

    def __init__(self, output_size=(64, 64)):
        """
        Initializes the RandomCrop transformer with the desired output size for pairs of images.

        Parameters:
        - output_size (tuple): The target output size (height, width).
        """
        self.output_size = output_size

    def __call__(self, imgs):
        """
        Apply random cropping to a tuple of single-channel images each with dimensions (1, H, W).

        Parameters:
        - imgs (tuple of numpy.ndarray): Tuple of images to be cropped, each expected to be in the format (1, H, W).

        Returns:
        - tuple of numpy.ndarray: Tuple of randomly cropped images.
        """
        input_image, target_image = imgs

        # Ensure both images have correct dimensions
        assert input_image.ndim == 3 and input_image.shape[0] == 1, "Each image must have dimensions (1, H, W)."
        assert target_image.ndim == 3 and target_image.shape[0] == 1, "Each image must have dimensions (1, H, W)."

        _, h, w = input_image.shape
        new_h, new_w = self.output_size

        if h > new_h and w > new_w:
            # Apply the same random crop to both images
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            cropped_input_image = input_image[:, top:top + new_h, left:left + new_w]
            cropped_target_image = target_image[:, top:top + new_h, left:left + new_w]
        else:
            # If either image is smaller than the crop size, apply padding
            padding_top = (new_h - h) // 2 if new_h > h else 0
            padding_left = (new_w - w) // 2 if new_w > w else 0
            padding_bottom = new_h - h - padding_top if new_h > h else 0
            padding_right = new_w - w - padding_left if new_w > w else 0

            cropped_input_image = np.pad(input_image, ((0, 0), (padding_top, padding_bottom), (padding_left, padding_right)),
                                         mode='constant', constant_values=0)  # Can modify padding mode and value if needed
            cropped_target_image = np.pad(target_image, ((0, 0), (padding_top, padding_bottom), (padding_left, padding_right)),
                                          mode='constant', constant_values=0)  # Can modify padding mode and value if needed

        return (cropped_input_image, cropped_target_image)






class RandomHorizontalFlip:
    """
    Apply random horizontal flipping to a single-channel image.
    
    Args:
        None needed for initialization.
    """

    def __call__(self, img):
        """
        Apply random horizontal flipping to a single-channel image with dimensions (1, H, W).
        
        Args:
            img (numpy.ndarray): The image to potentially flip, expected to be in the format (1, H, W).
        
        Returns:
            numpy.ndarray: Horizontally flipped image, if applied.
        """
        # Ensure that we're working with a single-channel image
        if img.ndim != 3 or img.shape[0] != 1:
            raise ValueError("Input image must have dimensions (1, H, W).")

        # Apply horizontal flipping with a 50% chance
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=2)  # Flip along the width axis, which is axis 2 for (1, H, W)
        return img


class RandomHorizontalFlipPair:
    """
    Apply random horizontal flipping to a pair of single-channel images.
    
    Args:
        None needed for initialization.
    """

    def __call__(self, imgs):
        """
        Apply random horizontal flipping to a tuple of single-channel images each with dimensions (1, H, W).
        
        Args:
            imgs (tuple of numpy.ndarray): Tuple of images to potentially flip, each expected to be in the format (1, H, W).
        
        Returns:
            tuple of numpy.ndarray: Tuple of horizontally flipped images, if applied.
        """
        input_image, target_image = imgs

        # Ensure both images have correct dimensions
        assert input_image.ndim == 3 and input_image.shape[0] == 1, "Each image must have dimensions (1, H, W)."
        assert target_image.ndim == 3 and target_image.shape[0] == 1, "Each image must have dimensions (1, H, W)."

        # Apply horizontal flipping with a 50% chance
        if np.random.rand() > 0.5:
            input_image = np.flip(input_image, axis=2)  # Flip along the width axis
            target_image = np.flip(target_image, axis=2)  # Flip along the width axis

        return (input_image, target_image)





class ToTensor(object):
    """
    Convert dictionaries containing single-channel images to PyTorch tensors. This class is specifically
    designed to handle dictionaries where each value is a single-channel image formatted as (1, H, W).
    """

    def __call__(self, data):
        """
        Convert a dictionary of single-channel images to PyTorch tensors, maintaining the channel position.

        Args:
            data (dict): The input must be a dictionary where each value is a single-channel image
            in the format (1, H, W).

        Returns:
            dict: Each converted image as a PyTorch tensor in the format (1, H, W).
        """
        def convert_image(img):
            # Check image dimensions and convert to tensor
            if img.ndim != 3 or img.shape[0] != 1:
                raise ValueError("Unsupported image format: each image must be 2D with a single channel (1, H, W).")
            return torch.from_numpy(img.astype(np.float32))

        # Ensure data is a dictionary of images
        if isinstance(data, dict):
            return {key: convert_image(value) for key, value in data.items()}
        else:
            raise TypeError("Input must be a dictionary of single-channel images.")

        return converted_tensors


class ToTensorPair(object):
    """
    Convert a tuple of single-channel images to PyTorch tensors. This class is specifically
    designed to handle tuples where each element is a single-channel image formatted as (1, H, W).
    """

    def __call__(self, imgs):
        """
        Convert a tuple of single-channel images to PyTorch tensors, maintaining the channel position.

        Args:
            imgs (tuple): The input must be a tuple where each element is a single-channel image
            in the format (1, H, W).

        Returns:
            tuple: Each converted image as a PyTorch tensor in the format (1, H, W).
        """
        def convert_image(img):
            # Check image dimensions and convert to tensor
            if img.ndim != 3 or img.shape[0] != 1:
                raise ValueError("Unsupported image format: each image must be 2D with a single channel (1, H, W).")
            return torch.from_numpy(img.astype(np.float32))

        if isinstance(imgs, tuple):
            return tuple(convert_image(img) for img in imgs)
        else:
            raise TypeError("Input must be a tuple of single-channel images.")



class ToTensorInference(object):
    """
    Convert images or batches of images to PyTorch tensors, handling both single images
    and tuples of images (input_img, target_img). The input is expected to be in the format
    (b, h, w, c) for batches or (h, w, c) for single images, and it converts them to
    PyTorch's (b, c, h, w) format or (c, h, w) for single images.
    """

    def __call__(self, data):
        """
        Convert input images or a tuple of images to PyTorch tensors, adjusting the channel position.

        Args:
            data (numpy.ndarray or tuple of numpy.ndarray): The input can be a single image (h, w, c),
            a batch of images (b, h, w, c), or a tuple of (input_img, target_img) in similar formats.

        Returns:
            torch.Tensor or tuple of torch.Tensor: The converted image(s) as PyTorch tensor(s) in the
            format (c, h, w) for single images or (b, c, h, w) for batches. If input is a tuple, returns
            a tuple of tensors.
        """
        def convert_image(img):
            if img.ndim == 3:
                return torch.from_numpy(img.astype(np.float32))
            else:
                raise ValueError("Unsupported image format: must be (h, w, c) or (b, h, w, c).")

        # Check if the input is a tuple of images
        if isinstance(data, tuple):
            return tuple(convert_image(img) for img in data)
        else:
            return convert_image(data)


class CropToMultipleOf16Inference(object):
    """
    Crop an image to ensure its height and width are multiples of 16.
    This is particularly useful for models that require input dimensions to be divisible by certain values.
    """

    def __call__(self, data):
        """
        Args:
            img (numpy.ndarray): Image to be cropped.

        Returns:
            numpy.ndarray: Cropped image.
        """
        input_img = data[0]
        _, h, w = input_img.shape  # Assuming img is a numpy array with shape (H, W, C) or (H, W)

        # Compute new dimensions to be multiples of 16
        new_h = h - (h % 16)
        new_w = w - (w % 16)

        # Calculate cropping margins
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        # Generate indices for cropping
        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis].astype(np.int32)
        id_x = np.arange(left, left + new_w, 1).astype(np.int32)

        # Crop the image
        cropped_image = input_img[:, id_y, id_x]

        return cropped_image, cropped_image


class CropToMultipleOf16Video(object):
    """
    Crop a stack of images and a single image to ensure their height and width are multiples of 16.
    This is particularly useful for models that require input dimensions to be divisible by certain values.
    """

    def __call__(self, data):
        """
        Args:
            data (tuple): A tuple where the first element is an input stack with shape (4, H, W, 1)
                          and the second element is a target image with shape (H, W, 1).

        Returns:
            tuple: A tuple containing the cropped input stack and target image.
        """
        input_stack, target_img = data

        # Crop the input stack
        cropped_input_stack = [self.crop_image(frame) for frame in input_stack]

        # Crop the target image
        cropped_target_img = self.crop_image(target_img)

        return np.array(cropped_input_stack), cropped_target_img

    def crop_image(self, img):
        """
        Crop a single image to make its dimensions multiples of 16.

        Args:
            img (numpy.ndarray): Single image to be cropped.

        Returns:
            numpy.ndarray: Cropped image.
        """
        h, w = img.shape[:2]  # Assuming img is a numpy array with shape (H, W, C) or (H, W)

        # Compute new dimensions to be multiples of 16
        new_h = h - (h % 16)
        new_w = w - (w % 16)

        # Calculate cropping margins
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        # Crop the image
        cropped_image = img[top:top+new_h, left:left+new_w]

        return cropped_image



class CropToMultipleOf32Inference(object):
    """
    Crop each slice in a stack of images to ensure their height and width are multiples of 32.
    This is particularly useful for models that require input dimensions to be divisible by certain values.
    """

    def __call__(self, stack):
        """
        Args:
            stack (numpy.ndarray): Stack of images to be cropped, with shape (Num_Slices, H, W).

        Returns:
            numpy.ndarray: Stack of cropped images.
        """

        num_slices, h, w = stack.shape

        # Compute new dimensions to be multiples of 32
        new_h = h - (h % 32)
        new_w = w - (w % 32)

        # Calculate cropping margins
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        # Generate indices for cropping
        id_y = np.arange(top, top + new_h).astype(np.int32)
        id_x = np.arange(left, left + new_w).astype(np.int32)

        # Crop each slice in the stack
        cropped_stack = np.zeros((num_slices, new_h, new_w), dtype=stack.dtype)
        for i in range(num_slices):
            cropped_stack[i] = stack[i, id_y, :][:, id_x]

        return cropped_stack




class ToNumpy(object):

    def __call__(self, data):

        return data.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    




class Denormalize(object):
    """
    Denormalize an image using mean and standard deviation, then convert it to 16-bit format.
    
    Args:
        mean (float or tuple): Mean for each channel.
        std (float or tuple): Standard deviation for each channel.
    """

    def __init__(self, mean, std):
        """
        Initialize with mean and standard deviation.
        
        Args:
            mean (float or tuple): Mean for each channel.
            std (float or tuple): Standard deviation for each channel.
        """
        self.mean = mean
        self.std = std
    
    def __call__(self, img):
        """
        Denormalize the image and convert it to 16-bit format.
        
        Args:
            img (numpy array): Normalized image.
        
        Returns:
            numpy array: Denormalized 16-bit image.
        """
        # Denormalize the image by reversing the normalization process
        img_denormalized = (img * self.std) + self.mean

        # Scale the image to the range [0, 65535] and convert to 16-bit unsigned integer
        img_denormalized = img_denormalized.astype(np.uint16)
        
        return img_denormalized