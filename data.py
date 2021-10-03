# define data generator for feeding data into model

import pandas as pd
from PIL import Image
import os

from torch.utils import data
from torchvision import transforms as T


class Cactus(data.Dataset):
    
    def __init__(self, mode, transforms=None):
        # mode='train', 'val', or 'test'
        
        self.mode=mode
        
        if mode=='test':
            
            df=pd.read_csv('../data/sample_submission.csv')
            root='../data/test'
            
        elif mode=='train' or mode=='val':
            
            df=pd.read_csv('../data/train.csv')
            root='../data/train'
            
        else:
            raise Exception("mode has to be 'train', 'val', or 'test'")
    
        imgs=(df['id']).values
        imgs=[os.path.join(root, img) for img in imgs]
        imgs_num=len(imgs)
        
        labels=(df['has_cactus']).values
        
        if mode=='test':
            
            self.imgs=imgs
            self.labels=(df['id']).values
            
        elif mode=='train':
            
            self.imgs=imgs[:int(0.7*imgs_num)]
            self.labels=labels[:int(0.7*imgs_num)]
            # store some weights info for Sampler
            # for dealing with the unbalanced dataset
            count_0=sum(self.labels==0)
            count_1=sum(self.labels==1)
            # inverse the number of sample in each class
            # to be the weight of sampling
            self.weight_of_labels=[
                1/count_1 if cac==1 else 1/count_0 for cac in self.labels]
            
        elif mode=='val':
            
            self.imgs=imgs[int(0.7*imgs_num):]
            self.labels=labels[int(0.7*imgs_num):]
            
            
        if transforms is None:
            
            normalize=T.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5])
            
            if mode=='test' or mode=='val':
                
                self.transforms=T.Compose([T.ToTensor(),normalize])
                
            else: # mode=='train'
                
                # some data augmentations
                
                 self.transforms=T.Compose([
                    T.RandomVerticalFlip(),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(),
                    T.ToTensor(),normalize])
            
    def __getitem__(self, index):
        
        img_path=self.imgs[index]
        
        label=self.labels[index]
        
        data=Image.open(img_path)
        data=self.transforms(data)
        
        return data, label
    
    def __len__(self):
        return len(self.imgs)
            
            