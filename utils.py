import os
import warnings
import glob
import random
import torch
import numpy as np
import tifffile
import pickle
import matplotlib.pyplot as plt
from PIL import Image

def create_result_dir(project_dir, name='new_results'):

    os.makedirs(project_dir, exist_ok=True)
    results_dir = os.path.join(project_dir, name, 'results')
    os.makedirs(results_dir, exist_ok=True)
    checkpoints_dir = os.path.join(project_dir, name, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    return results_dir, checkpoints_dir


def create_train_val_dir(results_dir):

    os.makedirs(results_dir, exist_ok=True)
    train_dir = os.path.join(results_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)
    val_dir = os.path.join(results_dir, 'val')
    os.makedirs(val_dir, exist_ok=True)

    return train_dir, val_dir


def normalize_dataset(dataset):
    all_means = []
    all_stds = []
    all_sizes = []

    # Compute mean, std, and size for each stack
    for stack in dataset:
        all_means.append(np.mean(stack))
        all_stds.append(np.std(stack))
        all_sizes.append(stack.size)

    # Convert lists to numpy arrays for easier computation
    array_means = np.array(all_means)
    array_stds = np.array(all_stds)
    array_sizes = np.array(all_sizes)

    # Compute weighted average of mean and std based on array sizes
    total_size = np.sum(array_sizes)
    weighted_mean = np.sum(array_means * array_sizes) / total_size
    weighted_std = np.sqrt(np.sum(array_stds**2 * array_sizes) / total_size)

    # Set global mean and std
    mean = weighted_mean
    std = weighted_std

    # Compute global minimum and maximum over the entire dataset
    global_min = np.min([np.min(stack) for stack in dataset])
    global_max = np.max([np.max(stack) for stack in dataset])

    # Apply global normalization to the entire dataset using the global min and max
    normalized_dataset = []
    for stack in dataset:
        # Normalize each slice in the stack using the global mean and std
        stack_normalized = (stack - mean) / std

        # Normalize each slice in the stack using the global min and max
        stack_normalized = (stack - global_min) / (global_max - global_min)

        # Clip and normalize to [0, 1] for each slice in the stack using the global min and max
        stack_normalized = np.clip(stack_normalized, 0, 1)

        normalized_dataset.append(stack_normalized.astype(np.float32))

    return normalized_dataset




def denormalize_image(normalized_img, mean, std):
    """
    Denormalizes an image back to its original range using the provided mean and standard deviation.

    Parameters:
    - normalized_img: The image to be denormalized.
    - mean: The mean used for the initial normalization.
    - std: The standard deviation used for the initial normalization.

    Returns:
    - The denormalized image.
    """
    original_img = (normalized_img * std) + mean
    return original_img.astype(np.float32)


def load_normalization_params(data_dir):
    """
    Loads the mean and standard deviation values from a pickle file located in the specified data directory.

    Parameters:
    - data_dir: Path to the directory containing the 'normalization_params.pkl' file.

    Returns:
    - A tuple containing the mean and standard deviation values.
    """
    # Construct the path to the pickle file
    load_path = os.path.join(data_dir, 'normalization_params.pkl')
    
    # Load the parameters from the pickle file
    with open(load_path, 'rb') as f:
        params = pickle.load(f)
    
    mean = params['mean']
    std = params['std']
    
    return mean, std



def load_min_max_params(data_dir):
    """
    Loads the global minimum and maximum values from a pickle file located in the specified data directory.

    Parameters:
    - data_dir: Path to the directory containing the 'min_max_params.pkl' file.

    Returns:
    - A tuple containing the global minimum and maximum values.
    """
    # Construct the path to the pickle file
    load_path = os.path.join(data_dir, 'min_max_params.pkl')
    
    # Load the parameters from the pickle file
    with open(load_path, 'rb') as f:
        params = pickle.load(f)
    
    global_min = params['global_min']
    global_max = params['global_max']
    
    return global_min, global_max




def plot_intensity_distribution(data_list, bins=256, save_path=None, legend_fontsize=10, plot_type='histogram'):
    plt.figure(figsize=(10, 5))
    
    line_labels = {}
    
    for idx, data in enumerate(data_list):
        label = input(f"Enter the label for dataset {idx + 1} (or press Enter to use the default label): ").strip()
        line_labels[idx] = label if label else f"Dataset {idx + 1}"
    
    for idx, data in enumerate(data_list):
        label = line_labels[idx]
        
        # Convert tensor to numpy array if needed
        if torch.is_tensor(data):
            data = data.numpy()
            if data.ndim == 3:  # Assuming the tensor is in the format (channels, height, width)
                data = np.transpose(data, (1, 2, 0))  # Convert to (height, width, channels)
            data = data.squeeze()  # Remove single-dimensional entries

        # Ensure data is 2D
        if data.ndim == 3 and data.shape[2] == 1:
            data = data[:, :, 0]
        elif data.ndim != 2:
            raise ValueError("Input data must be a 2D array or a tensor that can be converted to a 2D array.")
        
        # Use numpy.histogram to bin the pixel intensity data
        intensity_values, bin_edges = np.histogram(data, bins=bins, range=(0, 255))
        # Calculate bin centers from edges
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        if plot_type == 'line':
            plt.plot(bin_centers, intensity_values, label=label)
        elif plot_type == 'histogram':
            plt.hist(data.ravel(), bins=bins, range=(0, 255), alpha=0.5, label=label)

    plt.title('Pixel Intensity Distribution for 8-bit Grayscale Images')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend(fontsize=legend_fontsize)
    plt.grid(True)

    # Save the figure
    if save_path:
        plt.savefig(os.path.join(save_path, 'intensity_distribution-poisson-8bit.png'), format='png', dpi=300)
        plt.close()  # Close the figure to free up memory
    else:
        plt.show()


def get_file_path(local_path, remote_path):

    path = ''

    # Detect the operating system
    if os.name == 'nt':  # Windows
        path = local_path
    else:  # Linux and others
        path = remote_path
    
    if not os.path.exists(path):
        warnings.warn(f"Project directory '{path}' not found. Please verify the path.")
        return
    print(f"Using file path: {path}")

    return path


def clip_extremes(data, lower_percentile=0, upper_percentile=100):
    """
    Clip pixel values to the specified lower and upper percentiles.
    """
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    return np.clip(data, lower_bound, upper_bound)



def compute_global_min_max_and_save(dataset_path):
    """
    Computes and saves the global minimum and maximum values across all TIFF stacks
    in the given directory and its subdirectories, saving the results in the same directory.

    Parameters:
    - dataset_path: Path to the directory containing the TIFF files.
    """
    global_min = float('inf')
    global_max = float('-inf')
    for subdir, _, files in os.walk(dataset_path):
        for filename in files:
            if filename.lower().endswith(('.tif', '.tiff')):
                filepath = os.path.join(subdir, filename)
                stack = tifffile.imread(filepath)
                stack_min = np.min(stack)
                stack_max = np.max(stack)
                global_min = min(global_min, stack_min)
                global_max = max(global_max, stack_max)
    
    # Define the save_path in the same directory as the dataset
    save_path = os.path.join(dataset_path, 'min_max_params.pkl')

    # Save the computed global minimum and maximum to a file
    with open(save_path, 'wb') as f:
        pickle.dump({'global_min': global_min, 'global_max': global_max}, f)
    
    print(f"Global min and max parameters saved to {save_path}")
    return global_min, global_max



def plot_intensity_line_distribution(image, title='1', bins=200):
    plt.figure(figsize=(10, 5))

    if isinstance(image, torch.Tensor):
        # Ensure it's on the CPU and convert to NumPy
        image = image.detach().numpy()

    # Use numpy.histogram to bin the pixel intensity data, using the global min and max
    intensity_values, bin_edges = np.histogram(image, bins=bins, range=(np.min(image), np.max(image)))
    # Calculate bin centers from edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.plot(bin_centers, intensity_values, label='Pixel Intensity Distribution')
    
    plt.title('Pixel Intensity Distribution ' + title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()



def compute_global_mean_and_std(dataset_path, checkpoints_path):
    """
    Computes and saves the global mean and standard deviation across all JPEG and TIFF images
    in the given directory and its subdirectories, saving the results in the specified path.

    Parameters:
    - dataset_path: Path to the directory containing the image files.
    - checkpoints_path: Path where the normalization parameters will be saved.
    """
    all_means = []
    all_stds = []
    
    for subdir, _, files in os.walk(dataset_path):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.tif', '.tiff')):  # Check for both JPEG and TIFF files
                filepath = os.path.join(subdir, filename)
                image = Image.open(filepath)
                
                if image.mode == 'I;16':  # Check if the image is 16-bit grayscale
                    img_array = np.array(image, dtype=np.float32)  # Convert to float for more accurate mean/std computation
                else:
                    img_array = np.array(image.convert('L'), dtype=np.float32)  # Convert to 8-bit grayscale if not 16-bit

                all_means.append(np.mean(img_array))
                all_stds.append(np.std(img_array))

    global_mean = np.mean(all_means)
    global_std = np.mean(all_stds)
    
    # Define the save_path in the same directory as the dataset
    save_path = os.path.join(checkpoints_path, 'normalization_params.pkl')

    # Save the computed global mean and standard deviation to a file
    with open(save_path, 'wb') as f:
        pickle.dump({'mean': global_mean, 'std': global_std}, f)
    
    print(f"Global mean and std parameters saved to {save_path}")
    return global_mean, global_std



def crop_tiff_depth_to_divisible(path, divisor):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(('.tif', '.tiff')):
                file_path = os.path.join(root, file)
                with tifffile.TiffFile(file_path) as tif:
                    images = tif.asarray()
                    depth = images.shape[0]
                    
                    # Check if depth is divisible by divisor
                    if depth % divisor != 0:
                        # Calculate new depth that is divisible
                        new_depth = depth - (depth % divisor)
                        cropped_images = images[:new_depth]
                        
                        # Save the cropped TIFF stack
                        tifffile.imwrite(file_path, cropped_images, photometric='minisblack')
                        print(f'Cropped and saved: {file_path}')


def get_device():
    if torch.cuda.is_available():
        print("GPU is available")
        device = torch.device("cuda:0")
    else:
        print("GPU is not available")
        device = torch.device("cpu")
    
    return device


def print_tiff_filenames(root_folder_path):
    """
    Prints the filenames of TIFF files in the specified folder and its subdirectories.
    
    Parameters:
    - root_folder_path: Path to the root folder containing TIFF stack files.
    """
    for subdir, _, files in os.walk(root_folder_path):
        sorted_files = sorted([f for f in files if f.lower().endswith(('.tif', '.tiff'))])
        for filename in sorted_files:
            print(filename)


def plot_as_image(data, title='Image Display', cmap='gray', colorbar=True):
    plt.figure(figsize=(6, 6))

    if isinstance(data, torch.Tensor):
        # Ensure it's on the CPU and convert to NumPy
        data = data.detach().cpu().numpy()

    # Check if the data has multiple channels and select the first one if so
    if data.ndim == 3 and (data.shape[0] == 3 or data.shape[0] == 1):
        data = data[0]  # Assume the first channel for visualization if it's a 3-channel image

    data = data.squeeze()

    img = plt.imshow(data, cmap=cmap)
    plt.title(title)
    plt.axis('off')  # Turn off axis numbers and ticks

    if colorbar:
        plt.colorbar(img)

    plt.show()

    print(data.min())
    print(data.max())