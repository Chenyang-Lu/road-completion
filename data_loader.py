import pandas as pd
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils



class RoadDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        self.examples = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        road_layout = io.imread(self.examples.iloc[item, 0])

        example = {
                   'road': road_layout
                  }
        if self.transform:
            example = self.transform(example)

        return example


class ToTensor_road(object):

    def __call__(self, sample):
        road = np.expand_dims(sample['road'], 0)

        # road = road.transpose((2, 0, 1))
        road = (road.astype(float) / 65535. > 0.5).astype(float)

        # road = road + np.random.normal(0, 0.01, (1, 64, 64))
        road = torch.from_numpy(road)
        return {'road': road
                }


class OccMapDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        self.examples = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # rgb = io.imread(self.examples.iloc[item, 0])
        map = io.imread(self.examples.iloc[item, 3])
        style = io.imread(self.examples.iloc[item, 4])
        # seg = io.imread(self.examples.iloc[item, 2])

        example = {#'rgb': rgb,
                   'map': map,
                   'style': style
                   #'seg': seg
                  }
        if self.transform:
            example = self.transform(example)

        return example


class ToTensor(object):

    def __call__(self, sample):
        # rgb = sample['rgb']
        map = np.expand_dims(sample['map'], 0)
        style = np.expand_dims(sample['style'], 0)

        style = style / 65535.
        # seg = np.expand_dims(sample['seg'], 0)
        # print(map.shape)
        # print(rgb.shape)

        # rgb = rgb.transpose((2, 0, 1))
        # rgb = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                              std=[0.229, 0.224, 0.225])(torch.from_numpy(rgb))
        # rgb = torch.from_numpy(rgb)
        map = torch.from_numpy(map)
        style = torch.from_numpy(style)
        # seg = torch.from_numpy(seg)
        return {#'rgb': rgb,
                'map': map,
                'style': style}
                #'seg': seg}


class Rescale(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        # rgb = sample['rgb']
        map = sample['map']
        # seg = sample['seg']

        # rgb = transform.resize(rgb, self.output_size, mode='constant', preserve_range=False)
        # seg = transform.resize(seg, self.output_size, order=0, mode='constant', preserve_range=True)

        return {#'rgb': rgb,
                'map': map}
                #'seg': seg}


class Normalize(object):

    def __call__(self, sample):
        rgb = sample['rgb']
        map = sample['map']
        seg = sample['seg']
        rgb = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])(rgb)
        return {'rgb': rgb,
                'map': map,
                'seg': seg}


if __name__ == '__main__':
    val_set = OccMapDataset('dataset/val.csv', transform=transforms.Compose([Rescale((256, 512)), ToTensor()]))
    print('number of val examples:', len(val_set))
    print(val_set[0]['rgb'].shape)
    print(val_set[0]['map'].shape)
    print(val_set[0]['seg'].shape)


    val_loader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=8)
    print('show 3 examples')
    for i, temp_batch in enumerate(val_loader):
        if i == 0:
            print(temp_batch['rgb'])
            print(temp_batch['map'])
            print(temp_batch['seg'])
        break


    road_set = RoadDataset('dataset/road_layout_64_train_subset.csv', transform=transforms.Compose([ToTensor_road()]))
    print('number of val examples:', len(road_set))
    print(road_set[0]['road'].shape)
    print(road_set[0])
    for i, temp_batch in enumerate(road_set):
        print(temp_batch['road'])
        if i == 3:
            break


