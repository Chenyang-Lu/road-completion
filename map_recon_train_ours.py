import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tensorboardX import SummaryWriter
import random

from data_loader import *
from map_recon_nets_unet import *


mode = 'full'
seed = 5
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)



with_msk_channel = False
num_epochs = 60
batch_size = 16
restore = True
num_labels = 5
checkpoint_path = 'checkpoints/map_recon_checkpoint_dilate_trainval_' + mode + '_seed_' + str(seed) +'.pth.tar'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()


                
# Define dataloaders
val_set = OccMapDataset('dataset/val_50K_top32.csv', transform=transforms.Compose([ToTensor()]))
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
train_set = OccMapDataset('dataset/trainval_50K_top32.csv', transform=transforms.Compose([ToTensor()]))
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
road_set = RoadDataset('dataset/road_layout_train.csv', transform=transforms.Compose([ToTensor_road()]))
road_loader = DataLoader(road_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
road_iter = iter(road_loader)
dataloaders = {'train': train_loader, 'val': val_loader}

G = encdec_road_layout(with_msk_channel=with_msk_channel).to(device)
bce_loss = nn.BCELoss()
optimizerG = optim.Adam(G.parameters(), lr=0.0001, weight_decay=0.0001)
schedulerG = lr_scheduler.StepLR(optimizerG, step_size=100, gamma=0.1)


if restore:
    if os.path.isfile(checkpoint_path):
        state = torch.load(checkpoint_path)
        epoch = state['epoch']
        G.load_state_dict(state['state_dict_G'])
        optimizerG.load_state_dict(state['optimizer_G'])
        schedulerG.load_state_dict(state['scheduler_G'])
    else:
        epoch = 0
else:
    epoch = 0


while epoch < num_epochs:
    print(' ')
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            schedulerG.step()
            G.train()  # Set model to training mode
        else:
            G.eval()  # Set model to evaluate mode

        running_loss_0 = 0.0
        running_loss_1 = 0.0
        running_loss_2 = 0.0

        # Iterate over data.

        road_placeholder = torch.LongTensor(batch_size*64*64, 1).to(device) % num_labels
        road_onehot = torch.FloatTensor(batch_size*64*64, num_labels).to(device)

        for i, temp_batch in enumerate(dataloaders[phase]):
            temp_map = temp_batch['map'].long().to(device)
            temp_style_tgt = temp_batch['style'].float().to(device)

            road_onehot.zero_()
            temp_map = road_onehot.scatter_(1, temp_map.view(-1, 1), 1).view(batch_size, 64, 64, 5).permute(0, 3, 1, 2)
            temp_map_input = temp_map[:, 0, :, :] + 0.5 * temp_map[:, 4, :, :]
            temp_map_input = temp_map_input.unsqueeze(1)
            dropout = torch.rand(batch_size, 1, 64, 64) > 0.85
            temp_map_input[dropout] = 1 - temp_map_input[dropout]

            try:
                temp_road = next(road_iter)['road'].float().to(device)
            except StopIteration:
                road_iter = iter(road_loader)
                temp_road = next(road_iter)['road'].float().to(device)

            temp_road_input_nomsk = temp_road[:, 0, :, :].unsqueeze(1)
            temp_road_input = (1 - temp_map[:, 4, :, :]) * temp_road[:, 0, :, :] + 0.5 * temp_map[:, 4, :, :]
            temp_road_input = temp_road_input.unsqueeze(1)
            temp_road_input[dropout] = 1 - temp_road_input[dropout]
            msk_channel = temp_map[:, 4, :, :].unsqueeze(1)

            with torch.set_grad_enabled(phase == 'train'):
                optimizerG.zero_grad()
                if with_msk_channel:
                    pred_road = G(torch.cat([temp_map_input, msk_channel], dim=1), phase == 'train')
                    pred_road_1 = G(torch.cat([temp_road_input, msk_channel], dim=1), phase == 'train')
                else:
                    pred_road = G(temp_map_input.clone().detach(), phase == 'train')
                    pred_road_1 = G(temp_road_input, phase == 'train')

                loss_road_1 = loss_function_road_pred(pred_road, temp_map)
                loss_road_2 = loss_function_pre_selection(pred_road, temp_style_tgt)
                loss_road_3 = loss_function_road_layout(pred_road_1, temp_road)

                if mode == 'full':
                    loss_all = 0.5 * loss_road_1 + 0.25 * loss_road_2 + 0.25 * loss_road_3
                elif mode == 'wo_1':
                    loss_all = 0.5 * loss_road_2 + 0.5 * loss_road_3
                elif mode == 'wo_2':
                    loss_all = 2/3 * loss_road_1 + 1/3 * loss_road_3
                elif mode == 'wo_3':
                    loss_all = 2/3 * loss_road_1 + 1/3 * loss_road_2
                elif mode == 'wo_12':
                    loss_all = loss_road_3
                elif mode == 'wo_13':
                    loss_all = loss_road_2
                elif mode == 'wo_23':
                    loss_all = loss_road_1


                if phase == 'train':
                    loss_all.backward()
                    optimizerG.step()

            running_loss_0 += loss_road_1.item()
            running_loss_1 += loss_road_2.item()
            running_loss_2 += loss_road_3.item()

            # tensorboardX logging
            if phase == 'train':
                writer.add_scalar(phase+'_loss_road_0', loss_road_1.item(), epoch * len(train_set) / batch_size + i)
                writer.add_scalar(phase+'_loss_road_1', loss_road_2.item(), epoch * len(train_set) / batch_size + i)
                writer.add_scalar(phase+'_loss_road_2', loss_road_3.item(), epoch * len(train_set) / batch_size + i)

            # statistics
        if phase == 'train':
            running_loss_0 = running_loss_0 / len(train_set)
            running_loss_1 = running_loss_1 / len(train_set)
            running_loss_2 = running_loss_2 / len(train_set)
        else:
            running_loss_0 = running_loss_0 / len(val_set)
            running_loss_1 = running_loss_1 / len(val_set)
            running_loss_2 = running_loss_2 / len(val_set)

        print(phase, running_loss_0, running_loss_1, running_loss_2)
        if phase == 'val':
            writer.add_scalar(phase+'_loss_road_0', loss_road_1.item(), (epoch+1) * len(train_set) / batch_size)
            writer.add_scalar(phase+'_loss_road_1', loss_road_2.item(), (epoch+1) * len(train_set) / batch_size)
            writer.add_scalar(phase+'_loss_road_2', loss_road_3.item(), (epoch+1) * len(train_set) / batch_size)

    # save model per epoch
    torch.save({
        'epoch': epoch + 1,
        'state_dict_G': G.state_dict(),
        'optimizer_G': optimizerG.state_dict(),
        'scheduler_G': schedulerG.state_dict(),
        }, checkpoint_path)
    print('model after %d epoch saved...' % (epoch+1))
    epoch += 1

writer.close()

