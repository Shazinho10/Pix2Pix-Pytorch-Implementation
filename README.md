
# Pix2Pix From scratch with Pytorch

In this implementation, the famous Pix2Pix paper from Berkley AI Reseach (BAIR) has been implemented using Pytorch

# Generator
The generator here is UNet architechture

# Discriminator
This is a PatchGAN model.


### Training
Use the following script to train this model
python train.py --epochs <int> --save_interval <int>

### Training Details
The model has been trained on the gray images, which produce, colored images in return. But this model should be tested on other images as well.

1. During the training random samples are saved in the "output" directory.
2. The weights will be saved in the checkpoints directory depending upon the save_interval
3. The input images will be in the "data/train_A" directory while the target images will be in the "data/train_B" directory. Here the goal is to translate the images from A ---> B (train_A --> train_B)