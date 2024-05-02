import torch
import os
import config
import numpy as np
from PIL import Image
from torchvision.utils import save_image, make_grid
import glob

def plot_examples(root_dir, gen, epoch):
    try:
        # Use glob to search for files with specified extensions recursively
        files = glob.glob(os.path.join(root_dir, '**', '*.*'), recursive=True)
        print("Number of files found:", len(files))  # Print number of files found
        print("Found files:", files)

        gen.eval()
        images = []
        for idx, file in enumerate(files):
            image_path = file
            print("Processing image:", image_path)
            image = Image.open(image_path)
            with torch.no_grad():
                upscaled_img = gen(
                    config.test_transform(image=np.asarray(image))["image"]
                    .unsqueeze(0)
                    .to(config.DEVICE)
                )
            print("Generated image:", upscaled_img.shape)
            images.append(upscaled_img)
            save_path = f"/Users/shubh/Desktop/GANS_PROJECT_PYCHARM/SR_GAN_JUPYTER/EPOCHS/epoch_{epoch+1}_image{idx+1}.jpeg"
            save_image(upscaled_img * 0.5 + 0.5, save_path, normalize=True)
            print("Saved image:", save_path)
        grid_image = make_grid(torch.cat(images), nrow=4, padding=2, normalize=True)
        grid_save_path = f"/Users/shubh/Desktop/GANS_PROJECT_PYCHARM/SR_GAN_JUPYTER/EPOCHS/epoch_{epoch+1}_grid.jpeg"
        save_image(grid_image, grid_save_path)
        print("Saved grid image:", grid_save_path)
        gen.train()
    except Exception as e:
        print("Error:", e)
