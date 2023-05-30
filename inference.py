import torch
import cv2
import argparse
import numpy as np
from generator import Generator
from torchvision.io import read_image
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser(description='Train the model.')
parser.add_argument('--model_path', type=str, help='path to checkpoints')
parser.add_argument('--image_path', type=str, help='path to image for testing')


class Predict:
  def __init__(self, PATH):
    self.PATH = PATH

  def preprocessing(self, image):
    image = image.unsqueeze(0)
    transform = transforms.Resize((256, 256))
    image = transform(image)
    image = image.float()
    return image

  def add_channels(self, image):
    rgb_image = torch.stack([image] * 3, dim=1)
    rgb_image = rgb_image.squeeze(0)
    return rgb_image


  def inference(self, image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = read_image(image_path)
    channels = image.shape[0]
    if channels == 1:
      image = self.add_channels(image)
      
    image = self.preprocessing(image)
    gen = Generator()
    checkpoint = torch.load(self.PATH, map_location=device)
    gen.load_state_dict(checkpoint["model_state_dict"])
    fake_image = gen(image)
    fake_image = fake_image * 0.5 + 0.5
    fake_image = fake_image.squeeze(0).detach().cpu().numpy()
    fake_image = np.transpose(fake_image, (1, 2, 0))
    fake_image = cv2.cvtColor(fake_image, cv2.COLOR_RGB2BGR)
    return fake_image


if __name__ == "__main__":
  args = parser.parse_args()
  model_path = args.model_path
  image_path = args.image_path

  predict = Predict(model_path)
  image = predict.inference(image_path)
  cv2.imshow("Fake Image", image)
  cv2.waitKey(0)