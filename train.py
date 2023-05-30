import torch
import argparse
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from discriminator import Discriminator
from generator import Generator
from dataset import dataset
from torch.utils.data import DataLoader
from utils import save_images, save_checkpoint
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Train the model.')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
parser.add_argument('--save_interval', type=int, default=10, help='Interval for saving model')

args = parser.parse_args()
epochs = args.epochs
save_interval = args.save_interval

#setting up the hyper parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train/train_A"
learning_rate = 2e-4
BATCH_SIZE = 1
L1_LAMBDA = 100
log_dir = "logs"
writer = SummaryWriter(log_dir)

disc = Discriminator(in_channels=3).to(DEVICE)
gen = Generator(num_classes=3).to(DEVICE)
opt_disc = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.5, 0.999),)
opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))
BCE = nn.BCEWithLogitsLoss()
L1_LOSS = nn.L1Loss()

trainset = dataset(root_dir=TRAIN_DIR)
train_loader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True)


random_save_index = random.randint(0, len(train_loader))

def train_model(disc, gen, dataloader, opt_disc, opt_gen, 
                l1_loss, bce, epochs,
                save_interval=10):
  
    total_g_loss, total_d_loss = [], []
    save_sample = False
    for epoch in range(epochs):
      print(f"Epoch: {epoch+1}/{epochs}")
      save_sample = True
      count = 0
      for image, label in tqdm(dataloader, leave=False):
          image = image.to(DEVICE)
          label = label.to(DEVICE)

          Z = gen(image)    # Z distribution created by the Generator
          disc_real = disc(image=image, label=label)
          disc_real_loss = bce(disc_real, torch.ones_like(disc_real))
          disc_fake = disc(image=image, label=Z.detach())
          disc_fake_loss = bce(disc_fake, torch.zeros_like(disc_fake))
          disc_loss = (disc_real_loss + disc_fake_loss) / 2
          total_d_loss.append(disc_loss.detach().item())

          opt_disc.zero_grad()
          disc_loss.backward()
          opt_disc.step()

          disc_fake = disc(image, Z)   # because the previous disc_fake tensor became a part of the computational graph, had to do it again
          gen_fake_loss = bce(disc_fake, torch.ones_like(disc_fake))
          L1 = l1_loss(Z, label) * L1_LAMBDA
          gen_loss = gen_fake_loss + L1
          total_g_loss.append(gen_loss.detach().item())

          opt_gen.zero_grad()
          gen_loss.backward()
          opt_gen.step()

          if save_sample:
            if count % random_save_index == 0:
              save_images(gen, image, epoch)       #saving random image for the visualisation
              save_sample = False
          
          count += 1

      writer.add_scalar('Generator_Loss', np.mean(total_g_loss), epoch)      #writing the tensorboard graphs for the generator model
      writer.add_scalar('Discriminator_Loss', np.mean(total_d_loss), epoch)  #writing the tensorboard graphs for the discriminator model
          
      print(f" G_Loss: {np.mean(total_g_loss)}   D_Loss: {np.mean(total_d_loss)}")

      if epoch % save_interval == 0:
        #saving the generator
        save_checkpoint(gen, opt_gen, epoch, gen_loss, "Generator")
        #saving the discriminator
        save_checkpoint(disc, opt_disc, epoch, disc_loss, "Discriminator")
if __name__ == "__main__": 
  train_model(disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, epochs, save_interval)
