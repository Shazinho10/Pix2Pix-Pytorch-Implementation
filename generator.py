import torch.nn as nn
import torch

def conv(in_channels, out_channels):
  x = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect"),
      nn.BatchNorm2d(out_channels),
      nn.LeakyReLU(0.2)
  )
  return x


def trans_conv(in_channels, out_channels, use_dropout=False):
  if use_dropout:
    conv = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Dropout(0.5))
  else:
    conv = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU())
  return conv

def up_conv(in_channels, out_channels):
  x = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=2),
      nn.ReLU(),
      nn.Conv2d(out_channels, out_channels, kernel_size=2),
      nn.ReLU()
  )
  return x


def skip_connection(src, trg):
  src_w = src.shape[2]
  src_h = src.shape[3]

  trg_w = trg.shape[2]
  trg_h = trg.shape[3]

  src = src[:, :, : trg_w, :trg_h]
  return torch.cat((src, trg), 1)


class Encoder(nn.Module):
  def __init__(self):
    super().__init__()
    
    #block1:
    self.conv_1 = conv(in_channels=3, out_channels=64)

    #block2:
    self.conv_2 = conv(64, 128)

    #block3:
    self.conv_3 = conv(128, 256)

    #block4:
    self.conv_4 = conv(256, 512)

    #block5:
    self.conv_5 = conv(512, 512)

    #block6:
    self.conv_6 = conv(512, 512)

    #block7:
    self.conv_7 = conv(512, 512)

    #block7:
    self.final_down = nn.Sequential(
        nn.Conv2d(512, 512, 4, 2, 1), 
        nn.ReLU()   # in the last layer we do not apply the batch normalization according to the research paper
        )
    

  def forward(self, img):
    #block1
    d1 = self.conv_1(img)
    # print('First Convolution Block', d1.shape)

    #block2
    d2 = self.conv_2(d1)
    # print('Second Convolution Block', d2.shape)

    #block3
    d3 = self.conv_3(d2)
    # print('Third Convolution Block', d3.shape)

    #block4
    d4 = self.conv_4(d3)
    # print('Fourth Convolution Block', d4.shape)

    #block5
    d5 = self.conv_5(d4)
    # print('Fifth Convolution Block', d5.shape)

    #block6:
    d6 = self.conv_6(d4)
    # print('sixth Convolution Block', d6.shape)
    
    #block7:
    d7 = self.conv_7(d4)
    # print('last Convolution Block', d7.shape)

    #bottom
    final_down = self.final_down(d7)
    # print('bottom layer', final_down.shape)
    return d1, d2, d3, d4, d5, d6, d7, final_down


class Generator(nn.Module):
  def __init__(self, num_classes=3):
    super().__init__()

    # defining the architechture of the decoder
    #block1:
    self.trans_1 = trans_conv(in_channels=512, out_channels=512, use_dropout=True)   #transpose convolutions
    
    #block2:
    self.trans_2 = trans_conv(in_channels=512*2, out_channels=512, use_dropout=True)  # because of skip connctioons we will have 512 * 2 dimensions.

    #block3:
    self.trans_3 = trans_conv(in_channels=512*2, out_channels=512, use_dropout=False)

    #block4:
    self.trans_4 = trans_conv(in_channels=512*2, out_channels=512, use_dropout=False)
    
    #block5:
    self.trans_5 = trans_conv(in_channels=512*2, out_channels=256, use_dropout=False)

    #block6:
    self.trans_6 = trans_conv(in_channels=256*2, out_channels=128, use_dropout=False)
    
    #block6:
    self.trans_7 = trans_conv(in_channels=128*2, out_channels=64, use_dropout=False)

    #final layer:
    self.final_up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64 * 2, out_channels=num_classes, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    #encoder:
    self.encoder = Encoder()


  def forward(self, x):
    # Here the encoder and decoder will be combined to form a full UNet architechture
    #down sampling from the encoder
    out = self.encoder(x)
    d1 = out[0]
    d2 = out[1]
    d3 = out[2]
    d4 = out[3]
    d5 = out[4]
    d6 = out[5]
    d7 = out[6]
    final_down = out[7]
    #Now we begin the up sampling

    #block1

    # print("*"*30, "Decoder", "*"*30)
    trans1 = self.trans_1(final_down)   #transpose convolution
    skip_1 = skip_connection(trans1, d7)
    # print('1st Upsampled Convolution')

    #block2:
    trans2 = self.trans_2(skip_1)
    skip_2 = skip_connection(trans2, d6)
    # print('2nd Upsampled Convolution')

    #block3:
    trans3 = self.trans_3(skip_2)
    skip_3 = skip_connection(trans3, d5)
    # print('3rd Upsampled Convolution')

    #block4:
    trans4 = self.trans_4(skip_3)
    skip_4 = skip_connection(trans4, d4)
    # print('4th Upsampled Convolution')

    #block5
    # import pdb; pdb.set_trace()
    trans5 = self.trans_5(skip_4)
    skip_5 = skip_connection(trans5, d3)
    # print('5th Upsampled Convolution')

    #block6:
    trans6 = self.trans_6(skip_5)
    skip_6 = skip_connection(trans6, d2)
    # print('6th Upsampled Convolution')

    #block7:
    trans7 = self.trans_7(skip_6)
    skip_7 = skip_connection(trans7, d1)
    # print('7th Upsampled Convolution')


    out = self.final_up(skip_7)
    
    return out