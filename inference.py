import os
import sys
sys.path.append(os.path.join(".."))

import torch
import numpy as np
import tifffile
import argparse
import time
import glob
from torchvision import transforms
import matplotlib.pyplot as plt

from model import *
from transforms import *
from utils import *
from dataset import *


import logging

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


logging.basicConfig(filename='logging.log',  # Log filename
                    filemode='a',  # Append mode, so logs are not overwritten
                    format='%(asctime)s - %(levelname)s - %(message)s',  # Include timestamp
                    level=logging.INFO,  # Logging level
                    datefmt='%Y-%m-%d %H:%M:%S')  # Timestamp formatlog_file = open('logfile.log', 'w', buffering=1)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set logging level for console
logging.getLogger('').addHandler(console_handler)

# Redirect stdout and stderr to logging
sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)


def load(checkpoints_dir, model, epoch=1, optimizer=None, device='cpu'):


    # Ensure optimG is not None; it's better to explicitly check rather than using a mutable default argument like []
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())  

    checkpoint_path = os.path.join(checkpoints_dir, f'best_model.pth')
    dict_net = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(dict_net['model'])
    optimizer.load_state_dict(dict_net['optimizer'])
    epoch = dict_net['epoch']

    # If the model and optimizer are expected to be used on a GPU, explicitly move them after loading.
    model.to(device)
    # Note: Optimizers will automatically move their tensors to the device of the parameters they optimize.
    # So, as long as the model parameters are correctly placed, the optimizer's tensors will be as well.

    return model, epoch


def main():


        # Check if the script is running on the server by looking for the environment variable
    if os.getenv('RUNNING_ON_SERVER') == 'true':

        parser = argparse.ArgumentParser(description='Process data directory.')

        parser.add_argument('--data_dir', type=str, help='Path to the data directory')
        parser.add_argument('--project_name', type=str, help='Name of the project')
        parser.add_argument('--inference_name', type=str, help='Name of the inference')
    

        # Parse arguments
        args = parser.parse_args()

        # Now you can use args.data_dir as the path to your data
        data_dir = args.data_dir
        project_name = args.project_name 
        inference_name = args.inference_name
        project_dir = os.path.join('/g', 'prevedel', 'members', 'Rauscher', 'projects', 'FastDVDNet')

        print(f"Using data directory: {data_dir}")
        print(f"Project name: {project_name}")
    else:
        # If not running on the server, perhaps use a default data_dir or handle differently
        train_file_path = r"C:\Users\rausc\Documents\EMBL\data\BSD300_one\double_noisy\noised_12003_16bit_gray.tif"
        target_file_path = r"C:\Users\rausc\Documents\EMBL\data\BSD300_one\double_noisy\noised_12003_16bit_gray.tif"
        project_dir = r"Z:\members\Rauscher\projects\N2N-BSD300\BSD-test_2"
        inference_name = 'BSD1'

    #********************************************************#


    #********************************************************#

    results_dir = os.path.join(project_dir, 'results')
    checkpoints_dir = os.path.join(project_dir, 'checkpoints')

    # Make a folder to store the inference
    inference_folder = os.path.join(results_dir, inference_name)
    os.makedirs(inference_folder, exist_ok=True)



    # check if GPU is accessible
    if torch.cuda.is_available():
        print("\nGPU will be used.")
        device = torch.device("cuda:0")
    else:
        print("\nCPU will be used.")
        device = torch.device("cpu")

    mean, std = load_normalization_params(checkpoints_dir)
    
    inf_transform = transforms.Compose([
        NormalizePair(mean, std),
        CropToMultipleOf16Inference(),
        ToTensorPair(),
    ])

    inv_inf_transform = transforms.Compose([
        ToNumpy(),
        Denormalize(mean, std)
    ])

    inf_dataset = N2NJPGDataset(input_image_path=train_file_path,
                                      target_image_path=target_file_path,
                                      transform=inf_transform)

    batch_size = 8
    print("Dataset size:", len(inf_dataset))
    inf_loader = torch.utils.data.DataLoader(
        inf_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )


    model = NewUNet().to(device)
    model, epoch = load(checkpoints_dir, model)

    model = model.to(device)


    print("starting inference")

    with torch.no_grad():
        model.eval()

        for batch, data in enumerate(inf_loader):
            input_img, _ = data

            # Generate the output images
            output_img = model(input_img)
            output_img_np = inv_inf_transform(output_img)

            # plot_intensity_line_distribution(input_img, 'in#put')
            # plot_intensity_line_distribution(output_img_np, 'out#put')

            # Remove channel dimension if single channel
            output_img_clipped = output_img_np.squeeze(0)
            output_img_clipped = output_img_clipped.squeeze(-1)

            # Convert the numpy array to a PIL Image
            output_img_pil = Image.fromarray(output_img_clipped, mode='I;16')

            # Define the filename and the path
            filename = f'output_image-{inference_name}-epoch{epoch}.tif'
            filepath = os.path.join(inference_folder, filename)

            # Save the image as a TIFF file
            output_img_pil.save(filepath, 'TIFF')

            plt.imsave('output.png', output_img_clipped, cmap='gray')

            print("Output TIFF image created successfully.")



if __name__ == '__main__':
    start_time = time.time()  # Record start time
    main()
    end_time = time.time()    # Record end time
    print(f"Execution time: {end_time - start_time} seconds")