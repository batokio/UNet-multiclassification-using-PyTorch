from PIL import Image 
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import numpy as np
import os
import glob

class SegmentationDataset(Dataset):
    def __init__(self, images_path, mask_path, transform=None, eval=False):
        self.transform = transform

        self.XImg_list = sorted(glob.glob(images_path+'*.jpg'))
        self.yLabel_list = sorted(glob.glob(mask_path+'*.png'))
        self.eval = eval
                
    def __len__(self):
        return len(self.XImg_list) # Return the length of the data set
    

    def __getitem__(self, index):
        image = Image.open(self.XImg_list[index])
        y = Image.open(self.yLabel_list[index])

        if self.transform is not None:
            image = self.transform(image)
            y = self.transform(y)

        image = transforms.ToTensor()(image) # Rescale the image with value between 0 and 1
        y = np.array(y) # convert the mask to numpy array
        y = torch.from_numpy(y)  # convert the mask to Tensor
        
        y = y.type(torch.LongTensor) # Change the type to LongTensor
        return image, y