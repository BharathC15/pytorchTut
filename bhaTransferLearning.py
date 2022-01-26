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
import cv2

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()

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
    
    # dataset.ImageFolder
    A generic data loader where the images are arranged in this way by default:

    root/dog/xxx.png
    root/dog/xxy.png
    root/dog/[...]/xxz.png

    root/cat/123.png
    root/cat/nsdf3.png
    root/cat/[...]/asd932_.png

    """
    
    
    class_names = image_datasets['train'].classes

    print(image_datasets['train'])
    print(dir(image_datasets['train']))
    print(class_names)
    
    print(image_datasets['train'].__getitem__(index=200)) # returns image and class
    
    img,img_name = image_datasets['train'].__getitem__(index=200)
    
    '''
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])
    '''
    
    '''
    cv2.imshow(str(img_name), np.array(img))
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
    '''
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img_float32 = np.float32(img.numpy().transpose((1, 2, 0)))
    lab_image = cv2.cvtColor(img_float32, cv2.COLOR_BGR2RGB)
    
    newImage = std * lab_image + mean
    cv2.imshow(str(img_name),newImage)
    cv2.waitKey(3000)  # wait for 3 seconds
    cv2.destroyAllWindows() 
    #print(newImage)
    
    