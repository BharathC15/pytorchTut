# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

from __future__ import print_function, division

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

if __name__ == '__main__':

    data_dir = 'hymenoptera_data/' 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Check if cuda is available or not
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])}

    # Image Folder (PATH, Transform)
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x]) 
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                                  batch_size=4,
                                                  shuffle=True, 
                                                  num_workers=4) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    """
    image_datasets = {
        'train' : datasets.ImageFolder(os.path.join(data_dir,'train'),data_transforms['train']),
        'test' : datasets.ImageFolder(os.path.join(data_dir,'test'),data_transforms['test']),
    }
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'],batch_size = 4, shuffle = True, num_workers = 4),
        'test': torch.utils.data.DataLoader(image_datasets['test'],batch_size = 4, shuffle = True, num_workers = 4),
    }
    dataset_sizes = {
        'train': len(image_datasets['train']),
        'test': len(image_datasets['test'])
    }    
    """
    
    
    class_names = image_datasets['train'].classes

    print(image_datasets['train'])
    print(dir(image_datasets['train']))
    print(class_names)
    
    print(image_datasets['train'].__getitem__(index=200)) # returns image and class
