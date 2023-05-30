import numpy as np
import os
from PIL import Image
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

both_transform = A.Compose(
    [A.Resize(width=256, height=256),], additional_targets={"label": "image"},
)

transform_only_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)


class dataset(Dataset):
  def __init__(self, root_dir):
    self.root_dir = root_dir
    self.list_files = os.listdir(root_dir)

  def __len__(self):
    return len(self.list_files)

  def __getitem__(self, idx):
    file = self.list_files[idx]
    full_image_path = os.path.join(self.root_dir, file)
    full_label_path = full_image_path.replace("train_A", "train_B")  #train_A will have the x distribusion, while train_B will have the y distribution
    input_image = np.array(Image.open(full_image_path))
    target_image = np.array(Image.open(full_label_path))

    augmentation = both_transform(image=input_image, label=target_image)
    input_image = augmentation["image"]
    target_image = augmentation["label"]

    input_image = transform_only_input(image=input_image)["image"]
    target_image = transform_only_mask(image=target_image)["image"]

    return input_image, target_image
