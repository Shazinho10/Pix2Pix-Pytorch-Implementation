import torch.nn as nn
import torch

# 70x70 convolution
class Discriminator(nn.Module):
  def __init__(self, in_channels=3):
    super().__init__()
    self.first_conv = nn.Conv2d(in_channels=in_channels*2, out_channels=64, kernel_size=4, 
                                 stride=2, padding=1, padding_mode = "reflect", bias=False)   # Here we will be giving the concatenated image and the label thus we haave 6 channels
    self.lrelu =  nn.LeakyReLU(0.2)
    
    self.second_conv = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding_mode="reflect")
    self.second_bn = nn.BatchNorm2d(128)

    self.third_conv = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding_mode="reflect")
    self.third_bn = nn.BatchNorm2d(256)

    self.fourth_conv = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding_mode="reflect")
    self.fourth_bn = nn.BatchNorm2d(512)

    self.final_layer = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")  # in the final layer padding was 1 n the paper

  def forward(self, image, label):
    img = torch.cat([image, label], axis=1)   # giving the context of the target distribution into the GAN
    x = self.first_conv(img)
    x = self.lrelu(x)

    x = self.second_conv(x)
    x = self.second_bn(x)
    x = self.lrelu(x)
    
    x = self.third_conv(x)
    x = self.third_bn(x)
    x = self.lrelu(x)

    x = self.fourth_conv(x)
    x = self.fourth_bn(x)
    x = self.lrelu(x)

    x = self.final_layer(x)
    return x