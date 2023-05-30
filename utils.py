import torch
from torchvision.utils import save_image

def save_checkpoint(model, optimizer, epoch, loss, model_type : str):

  #model_type indicates if this weight is opf discriminator or the generator
  checkpoint = {
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'epoch': epoch,
      'loss': loss}
  checkpoint_path = f"checkpoints/{model_type}_epoch{epoch}.pt"
  torch.save(checkpoint, checkpoint_path)


def save_images(gen, image, epoch, folder="output"):
  gen.eval()
  with torch.no_grad():
    y_fake = gen(image)
    y_fake = y_fake * 0.5 + 0.5  # rescaling the image
    save_image(y_fake, folder + f"/image_gen_epoch_{epoch}.png")
    save_image(image * 0.5 + 0.5, folder + f"/image_input_epoch_{epoch}.png")
    # print("saved image: ", folder + f"/gen_{epoch}.png")