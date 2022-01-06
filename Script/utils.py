import torch
from torch.utils.data import DataLoader
from torchvision import datasets, utils, transforms
from datasets import SegmentationDataset
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def get_dataloader(
    images_path,
    mask_path,
    transforms=None,
    batch_size=1,
    eval=False,
    shuffle=True,
    pin_memory=True,
    ):
  
  dataset = SegmentationDataset(images_path=images_path, mask_path=mask_path, transform=transforms, eval=eval)

  dataloaded = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)

  return dataloaded

## Useful functions

# Function to save result on a .txt file
def write_results(ff, save_folder, epoch, train_acc, val_acc, train_loss, val_loss):
    ff=open('{}/progress.txt'.format(save_folder),'a')
    ff.write(' E: ')
    ff.write(str(epoch))
    ff.write('         ')
    ff.write(' TRAIN_OA: ')
    ff.write(str('%.3f' % train_acc))
    ff.write(' VAL_OA: ')
    ff.write(str('%.3f' % val_acc))
    ff.write('         ')
    ff.write(' TRAIN_LOSS: ')
    ff.write(str('%.3f' % train_loss))
    ff.write(' VAL_LOSS: ')
    ff.write(str('%.3f' % val_loss))
    ff.write('\n')


# Function to compute the accuracy of the predictions
def model_accuracy(output, target):

  # Transform the output to get the right format
  output_softmax = F.softmax(output)
  output_argmax = torch.argmax(output_softmax, dim=1)

  # Get the correct predictions as a boolean mask
  corrects = (output_argmax  == y)

  # Compute accuracy
  accuracy = corrects.sum().float() / float( y.size(0) * y.size(1) * y.size(2) )

  return accuracy


# Function to create train-val graph and save the figure
def save_graph(train_loss, val_loss, nb_epochs, save_folder):
    plt.plot(list(range(nb_epochs+1))[1:], train_loss)
    plt.plot(list(range(nb_epochs+1))[1:], val_loss)
    plt.legend(['train', 'val'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('{}/chart.png'.format(save_folder))