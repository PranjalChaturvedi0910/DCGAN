import torch

# Configuration settings and hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4  # From the DCGAN paper
BATCH_SIZE = 128
IMAGE_SIZE = 64 # MNIST images are 28x28, we will resize them to 64x64
CHANNELS_IMG = 1 # MNIST is grayscale
Z_DIM = 100 # Latent vector dimension
NUM_EPOCHS = 10 # Train for more epochs for better results
FEATURES_DISC = 64 # Base size for discriminator feature maps
FEATURES_GEN = 64 # Base size for generator feature maps

# Paths
OUTPUT_DIR = "outputs"
DATA_DIR = "data"