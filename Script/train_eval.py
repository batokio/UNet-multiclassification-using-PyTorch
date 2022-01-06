import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from utils import *
from model_unet import UNET
from datasets import *


if torch.cuda.is_available():
    device = 'cuda:0'
    print('Running on the GPU')
else:
    device = "cpu"
    print('Running on the CPU')


# create a directory for saving the models and the training progress
save_folder = 'SAVE_FOLDER_PATH'

load_model = False

train_images_path = 'TRAIN_IMAGES_PATH'
train_masks_path = 'TRAIN_MASKS_PATH'

val_images_path = 'VAL_IMAGES_PATH'
val_masks_path = 'VAL_MASKS_PATH'

img_height = 200 # To increase if we want to increase the accuracy of our model 
img_width = 200 # To increase if we want to increase the accuracy of our model 

batch_size = 8
learning_rate = 0.0005
epochs = 25

def main():
  
  # Open the .txt file
  ff=open('{}/progress.txt'.format(save_folder),'w')

  # Defining the model, optimizer and loss function
  model = UNET(in_channels=3, classes=25).to(device)
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  criterion = nn.CrossEntropyLoss(ignore_index=255)

  # Defining the transform
  transform = transforms.Compose([
          transforms.Resize((img_height, img_width), interpolation=Image.NEAREST),
      ])

  train_set = get_dataloader(
      train_images_path,
      train_masks_path,
      transforms=transform,
      batch_size=batch_size,
      shuffle=True,
      )

  val_set = get_dataloader(
      val_images_path,
      val_masks_path,
      transforms=transform,
      batch_size=batch_size,
      shuffle=True,
      eval=True
      )

  total_train_losses = []
  total_val_losses = []


  for epoch in range(1, epochs+1):
  
    # TRAINING
    model.train()
    train_losses = []
    train_accuracy = []

    for index, batch in enumerate(tqdm(train_set)): 
          X, y = batch
          X, y = X.to(device), y.to(device)
          output = model(X)
      
          loss = criterion(output, y)
          train_acc = model_accuracy(output, y)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          train_losses.append(loss.item())
          train_accuracy.append(train_acc)
          

    train_acc_mean = torch.mean(torch.stack(train_accuracy))
    train_loss_mean = np.mean(train_losses)
    total_train_losses.append(train_loss_mean)


    ##VALIDATION##
    model.eval()
    val_losses = []
    val_accuracy = []

    for index, batch in enumerate(tqdm(val_set)): 
          X, y = batch
          X, y = X.to(device), y.to(device)
          outputs = model(X)
          val_acc = model_accuracy(outputs, y)

          loss = criterion(outputs, y)

          val_losses.append(loss.item())
          val_accuracy.append(val_acc)

    val_acc_mean = torch.mean(torch.stack(val_accuracy))
    val_loss_mean = np.mean(val_losses)
    total_val_losses.append(val_loss_mean)

    print('EPOCH: ', epoch)
    print('TRAIN_LOSS: ', '%.3f' % train_loss_mean, 'TRAIN_ACC: ', '%.3f' % train_acc_mean)
    print('VAL_LOSS: ', '%.3f' % val_loss_mean, 'VAL_ACC: ', '%.3f' % val_acc_mean)

    write_results(ff, save_folder, epoch, train_acc_mean, val_acc_mean, train_loss_mean, val_loss_mean)

    torch.save(model.state_dict(), save_folder + '/model_{}.pt'.format(epoch))
  
  save_graph(total_train_losses, total_val_losses, epochs, save_folder)