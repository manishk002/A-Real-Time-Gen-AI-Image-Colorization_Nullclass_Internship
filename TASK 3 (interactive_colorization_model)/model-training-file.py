import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class UserGuidedColorizationNet(nn.Module):
    def __init__(self):
        super(UserGuidedColorizationNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)  # 4 input channels: 1 for grayscale, 3 for user hints
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x, user_hints):
        x = torch.cat([x, user_hints], dim=1)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = nn.functional.relu(self.conv5(x))
        x = torch.tanh(self.conv6(x))
        return x

class ColorizationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Convert to LAB color space
        lab = rgb_to_lab(image)
        L = lab[[0], ...] / 50.0 - 1.0  # between -1 and 1
        ab = lab[[1, 2], ...] / 110.0  # between -1 and 1
        
        # Create random user hints
        user_hints = torch.zeros_like(image)
        num_hints = np.random.randint(1, 10)
        for _ in range(num_hints):
            x, y = np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[2])
            color = torch.rand(3)
            user_hints[:, x, y] = color
        
        return L, ab, user_hints

# Helper functions
def rgb_to_lab(rgb):
    # Convert RGB to LAB color space
    # This is a placeholder. You should implement a proper RGB to LAB conversion
    return rgb

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 32
num_epochs = 50
learning_rate = 0.001

# Data loading
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_dataset = ColorizationDataset(root_dir='path/to/your/training/images', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Initialize the model, loss, and optimizer
model = UserGuidedColorizationNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i, (L, ab, user_hints) in enumerate(train_loader):
        L, ab, user_hints = L.to(device), ab.to(device), user_hints.to(device)

        # Forward pass
        outputs = model(L, user_hints)
        loss = criterion(outputs, ab)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

print("Training completed!")

# Save the model
torch.save(model.state_dict(), 'interactive_colorization_model.pth')

# Visualization function
def visualize_colorization(model, L, user_hints):
    model.eval()
    with torch.no_grad():
        output = model(L.unsqueeze(0).to(device), user_hints.unsqueeze(0).to(device))
    output = output.squeeze().cpu()
    L = L.squeeze().cpu()
    
    # Convert LAB to RGB (placeholder function, implement proper conversion)
    rgb_output = lab_to_rgb(torch.cat([L.unsqueeze(0), output], dim=0))
    
    plt.imshow(rgb_output.permute(1, 2, 0))
    plt.axis('off')
    plt.show()

# Test the model on a sample image
sample_L, _, sample_hints = next(iter(train_loader))
visualize_colorization(model, sample_L[0], sample_hints[0])
