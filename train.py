import os
import torch
import matplotlib.pyplot as plt
import pickle
import time
import logging

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


from utils import *
from transforms import *
from dataset import *
from model import *


class Trainer:

    def __init__(self, data_dict):

        self.data_dir = data_dict['data_dir']
        self.train_file_path = data_dict['train_file_path']
        self.target_file_path = data_dict['target_file_path']

        self.project_dir = data_dict['project_dir']
        self.project_name = data_dict['project_name']

        self.results_dir, self.checkpoints_dir = create_result_dir(self.project_dir, self.project_name)
        self.train_results_dir, self.val_results_dir = create_train_val_dir(self.results_dir)

        self.num_epoch = data_dict['num_epoch']
        self.batch_size = data_dict['batch_size']
        self.lr = data_dict['lr']

        self.num_freq_disp = data_dict['num_freq_disp']

        self.train_continue = data_dict['train_continue']

        self.device = get_device()

        self.writer = SummaryWriter(self.results_dir + '/tensorboard_logs')



    def save(self, checkpoints_dir, model, optimizer, epoch):
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch}, 
                os.path.join(checkpoints_dir, 'best_model.pth'))

        

    def load(self, checkpoints_dir, model, epoch=[], optimizer=[]):

        dict_net = torch.load('%s/best_model.pth' % (checkpoints_dir), map_location=torch.device('cpu'))

        model.load_state_dict(dict_net['model'])
        optimizer.load_state_dict(dict_net['optimizer'])
        epoch = dict_net['epoch']

        print('Loaded %dth network' % epoch)

        return model, optimizer, epoch
    

    def train(self):

        ### transforms ###

        start_time = time.time()
        mean, std = compute_global_mean_and_std(self.data_dir, self.checkpoints_dir)
        print(mean)
        print(std)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")

        transform_train = transforms.Compose([
            NormalizePair(mean, std),
            RandomCropPair(output_size=(64,64)),
            RandomHorizontalFlipPair(),
            ToTensorPair()
        ])

        transform_inv_train = transforms.Compose([
            ToNumpy(),
            Denormalize(mean, std)
        ])


        train_dataset = N2NJPGDataset(input_image_path=self.train_file_path,
                                      target_image_path=self.target_file_path,
                                      transform=transform_train)
    

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2)


        ### initialize network ###

        model = NewUNet().to(self.device)

        criterion = nn.MSELoss().to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), self.lr)

        st_epoch = 0
        best_train_loss = float('inf')
        cum_train_loss = [1]
        old_cum_train_loss = np.average(cum_train_loss)

        if self.train_continue == 'on':
            print(self.checkpoints_dir)
            model, optimizer, st_epoch = self.load(self.checkpoints_dir, model, st_epoch, optimizer)
            model = model.to(self.device)

        for epoch in range(st_epoch + 1, self.num_epoch + 1):
            model.train()  # Ensure model is in training mode
            train_loss = 0.0

            for batch, data in enumerate(train_loader, 0):
                optimizer.zero_grad()
                input_img, target_img = data
                input_img = input_img.to(self.device)
                target_img = target_img.to(self.device)

                output_img = model(input_img)
                loss = criterion(output_img, target_img)
                train_loss += loss.item() 
                cum_train_loss.append(loss.item())
                loss.backward()
                optimizer.step()
                
            if epoch % self.num_freq_disp == 0:
                input_img_np = transform_inv_train(input_img)
                target_img_np = transform_inv_train(target_img)
                output_img_np = transform_inv_train(output_img)

                for j in range(target_img_np.shape[0]):
                    base_filename = f"sample{j:03d}"

                    input_filename = os.path.join(self.train_results_dir, f"{base_filename}_input.png")
                    target_filename = os.path.join(self.train_results_dir, f"{base_filename}_target.png")
                    output_filename = os.path.join(self.train_results_dir, f"{base_filename}_output.png")

                    plt.imsave(input_filename, input_img_np[j, :, :, 0], cmap='gray')
                    plt.imsave(target_filename, target_img_np[j, :, :, 0], cmap='gray')
                    plt.imsave(output_filename, output_img_np[j, :, :, 0], cmap='gray')

                new_cum_train_loss = np.average(cum_train_loss)
                if new_cum_train_loss < old_cum_train_loss:
                    old_cum_train_loss = new_cum_train_loss
                    self.save(self.checkpoints_dir, model, optimizer, epoch)
                    print(f"Saved best model at epoch {epoch} with train loss {new_cum_train_loss:.4f}.")


            print(f'Epoch [{epoch}/{self.num_epoch}], Train Loss: {np.average(cum_train_loss):.4f}')

            

            
 
        self.writer.close()