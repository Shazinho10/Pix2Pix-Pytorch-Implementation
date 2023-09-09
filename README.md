
## Pix-to-Pix Model Implementation (From Scratch)

This instantiation involves the utilization of the renowned Pix2Pix paper from Berkeley AI Research (BAIR), implemented through the PyTorch framework.
## Training Details
This model has undergone training using grayscale input images, generating corresponding colorized output images. It is essential to evaluate the model's performance on a broader spectrum of image data.

* Throughout the training process, random samples have been preserved and stored in the "output" directory for further analysis and assessment.

* The model's trained weights are systematically saved in the "checkpoints" directory, based on a specified save interval, ensuring that the model's progress is effectively recorded.

* Input images for this model are sourced from the "data/train/train_A" directory, while the corresponding target images are located in the "data/train/train_B" directory. The primary objective here is to execute image translation, converting images from the "train_A" set to their corresponding representations in the "train_B" set.

## Training

After putting the data as described above, do the following

```bash
  python train.py --epochs --save_interval
```

## Inference

```bash
  python inference.py --model_path --image_path
```
