import torch
import config
import plotting
from torch import nn
from torch import optim
from utils import plot_examples
from loss import VGGLoss
from torch.utils.data import DataLoader
from model import Generator, Discriminator
from tqdm import tqdm
from dataset import MyImageFolder

torch.backends.cudnn.benchmark = True

import matplotlib.pyplot as plt

def plot_loss_curve(loss_disc, gen_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_disc, label='Discriminator Loss', color='blue')
    plt.plot(gen_loss, label='Generator Loss', color='orange')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Discriminator and Generator Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")  # Save the loss curve plot
    plt.show()


def train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss):
    for epoch in range(config.NUM_EPOCHS):
        epoch_losses = []
        loop = tqdm(loader, leave=True, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")

        for idx, (low_res, high_res) in enumerate(loop):
            high_res = high_res.to(config.DEVICE)
            low_res = low_res.to(config.DEVICE)

            fake = gen(low_res)
            disc_real = disc(high_res)
            disc_fake = disc(fake.detach())
            disc_loss_real = bce(
                disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)
            )
            disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = disc_loss_fake + disc_loss_real

            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            disc_fake = disc(fake)
            adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
            loss_for_vgg = 0.006 * vgg_loss(fake, high_res)
            gen_loss = loss_for_vgg + adversarial_loss

            opt_gen.zero_grad()
            gen_loss.backward()
            opt_gen.step()

            epoch_losses.append((loss_disc.item(), gen_loss.item()))

            # Save generated images every iteration
            if idx % 20 == 0:
                # Call plot_examples with correct parameters
                plot_examples("/Users/shubh/Pictures/GAN_PROJECT_DATA/RESIZED_IMAGES", gen, epoch)

        # Calculate average losses for the epoch
        avg_disc_loss = sum(l[0] for l in epoch_losses) / len(epoch_losses)
        avg_gen_loss = sum(l[1] for l in epoch_losses) / len(epoch_losses)

        # Display epoch losses
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Avg Discriminator Loss: {avg_disc_loss:.4f} | Avg Generator Loss: {avg_gen_loss:.4f}")



def main():
    dataset = MyImageFolder(root_dir="/Users/shubh/Pictures/GAN_PROJECT_DATA/RESIZED_IMAGES")
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=config.NUM_WORKERS,
    )
    gen = Generator(in_channels=3).to(config.DEVICE)
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = VGGLoss()

    train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss)

if __name__ == "__main__":
    main()