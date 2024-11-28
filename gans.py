import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg

# Hyperparameters
z_dim = 20  # Latent vector dimension
image_size = 28  # Size of the generated images (MNIST images are 28x28)
channels = 1  # Number of channels in the generated images (e.g. grayscale)
batch_size = 64
num_epochs = 5
lr = 0.0002
beta1 = 0.5  # Beta 1 for Adam optimizer

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator class
class Generator(nn.Module):
    def __init__(self, z_dim, img_channels):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, img_channels * image_size * image_size),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.model(z)
        return out.view(-1, channels, image_size, image_size)

# Discriminator class
class Discriminator(nn.Module):
    def __init__(self, img_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_channels * image_size * image_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img = img.view(img.size(0), -1)  # Flatten image
        return self.model(img)

# Initialize generator and discriminator
generator = Generator(z_dim, channels).to(device)
discriminator = Discriminator(channels).to(device)

# Loss function
criterion = nn.BCELoss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Data loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)  # Normalize to range [-1, 1]
])
dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Directory to save generated images
os.makedirs('generated_images', exist_ok=True)

# Training loop
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        # Train Discriminator
        discriminator.zero_grad()
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # Real images
        outputs = discriminator(real_images).view(-1, 1)  # Reshape output to match the labels
        loss_real = criterion(outputs, real_labels)
        loss_real.backward()
        
        # Fake images
        z = torch.randn(batch_size, z_dim).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images.detach()).view(-1, 1)  # Reshape output to match the labels
        loss_fake = criterion(outputs, fake_labels)
        loss_fake.backward()
        
        optimizer_D.step()

        # Train Generator
        generator.zero_grad()
        outputs = discriminator(fake_images).view(-1, 1)  # Reshape output to match the labels
        loss_G = criterion(outputs, real_labels)
        loss_G.backward()
        optimizer_G.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] | Step [{i}/{len(dataloader)}] | Loss D: {loss_real.item() + loss_fake.item()} | Loss G: {loss_G.item()}")

    # Save generated images for each epoch
    with torch.no_grad():
        z = torch.randn(64, z_dim).to(device)
        fake_images = generator(z).cpu()
        grid = torchvision.utils.make_grid(fake_images, nrow=8, normalize=True)
        plt.imshow(grid.permute(1, 2, 0))
        plt.title(f"Generated Images at Epoch {epoch+1}")
        plt.axis('off')
        plt.savefig(f'generated_images/epoch_{epoch+1}.png')
        plt.close()

# Dynamically load and display the latest generated image
import os

# Check for the latest image file in the 'generated_images' folder
epoch_files = [f for f in os.listdir('generated_images') if f.endswith('.png')]
latest_epoch_file = sorted(epoch_files)[-1]  # Get the latest file (highest epoch)

# Read and display the latest generated image
img = mpimg.imread(f'generated_images/{latest_epoch_file}')
plt.imshow(img)
plt.axis('off')
plt.show()

print("Training complete.")
