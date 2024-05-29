import os
import sys
import argparse
sys.path.append(os.path.join(".."))

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





from utils import *
from train import *


def main():

    ## parser

    # Check if the script is running on the server by looking for the environment variable
    if os.getenv('RUNNING_ON_SERVER') == 'true':

        parser = argparse.ArgumentParser(description='Process data directory.')

        parser.add_argument('--data_dir', type=str, help='Path to the train data directory')
        parser.add_argument('--train_file_path', type=str, help='Path to the train file')
        parser.add_argument('--target_file_path', type=str, help='Path to the target file')
        parser.add_argument('--project_name', type=str, help='Name of the project')
        parser.add_argument('--train_continue', type=str, default='off', choices=['on', 'off'],
                            help='Flag to continue training: "on" or "off" (default: "off")')
    

        # Parse arguments
        args = parser.parse_args()

        # Now you can use args.data_dir as the path to your data
        data_dir = args.data_dir
        train_file_path = args.train_file_path
        target_file_path = args.target_file_path
        project_name = args.project_name 
        train_continue = args.train_continue
        project_dir = os.path.join('/g', 'prevedel', 'members', 'Rauscher', 'projects', 'N2N-BSD300')

        print(f"Using train data directory: {train_file_path}")
        print(f"Using val data directory: {target_file_path}")
        print(f"Project name: {project_name}")
        print(f"Train continue: {train_continue}")
    else:
        # If not running on the server, perhaps use a default data_dir or handle differently
        data_dir = r"C:\Users\rausc\Documents\EMBL\data\BSD300_one\gaussian-8bit\double_noisy"
        train_file_path = r"C:\Users\rausc\Documents\EMBL\data\BSD300_one\gaussian-8bit\double_noisy\noised_12003_8bit_gray.jpg"
        target_file_path = r"C:\Users\rausc\Documents\EMBL\data\BSD300_one\gaussian-8bit\double_noisy\noised2_12003_8bit_gray.jpg"
        project_dir = os.path.join('C:\\', 'Users', 'rausc', 'Documents', 'EMBL', 'projects', 'N2N-BSD300')
        project_name = 'BSD-test_1'
        train_continue = 'off'


    data_dict = {}

    data_dict['data_dir'] = data_dir
    data_dict['train_file_path'] = train_file_path
    data_dict['target_file_path'] = target_file_path
    data_dict['project_dir'] = project_dir
    data_dict['project_name'] = project_name

    data_dict['num_epoch'] = 5000
    data_dict['batch_size'] = 8
    data_dict['lr'] = 1e-5

    data_dict['num_freq_disp'] = 100

    data_dict['train_continue'] = train_continue


    TRAINER = Trainer(data_dict)
    TRAINER.train()


if __name__ == '__main__':
    main()


