import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from model import Discriminator, Generator, initialize_weights
from config import *

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define data transformations
# We resize to IMAGE_SIZE and normalize to [-1, 1] for the Tanh activation
transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)

# Load MNIST dataset
dataset = datasets.MNIST(root=DATA_DIR, train=True, transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize models and move to device
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(DEVICE)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(DEVICE)
initialize_weights(gen)
initialize_weights(disc)

# Setup optimizers for both models
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# A fixed noise vector to see the progress of the generator over time
fixed_noise = torch.randn(64, Z_DIM, 1, 1).to(DEVICE)

# Start training
print("🚀 Starting Training...")
gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    loop = tqdm(loader, leave=True)
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(DEVICE)
        noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(DEVICE)
        fake = gen(noise)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        # Train with real images
        disc.zero_grad()
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        
        # Train with fake images
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        # Combine losses and update
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        loss_disc.backward()
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        gen.zero_grad()
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        loss_gen.backward()
        opt_gen.step()
        
        # Update progress bar
        loop.set_postfix(
            epoch=epoch+1,
            loss_D=f"{loss_disc.item():.4f}",
            loss_G=f"{loss_gen.item():.4f}",
        )

    # At the end of each epoch, save generated images and model checkpoints
    with torch.no_grad():
        fake_images = gen(fixed_noise)
        torchvision.utils.save_image(fake_images, f"{OUTPUT_DIR}/fake_epoch_{epoch+1}.png", normalize=True)
        torch.save(gen.state_dict(), f"{OUTPUT_DIR}/generator_epoch_{epoch+1}.pth")
        torch.save(disc.state_dict(), f"{OUTPUT_DIR}/discriminator_epoch_{epoch+1}.pth")

print("✅ Training complete.")