import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

# Step 2: Data Loading and Preprocessing
# Modify your dataset to work with grayscale images
class CustomShapeDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.data = ImageFolder(root, transform)  # Use ImageFolder to load grayscale images
        self.classes = self.data.classes

    def __getitem__(self, index):
        img, _ = self.data[index]
        labels = self.extract_labels(index)
        return img, labels

    def extract_labels(self, index):
        image_filename, _ = self.data.samples[index]
        labels = image_filename.split('/')[-1].split('.')[0].split('_')
        labels = [label for label in labels if label in self.classes]

        # Create an empty tensor filled with zeros
        label_tensor = torch.zeros(len(self.classes), dtype=torch.float32)

        # Set the corresponding positions to 1 for detected labels
        for label in labels:
            label_index = self.classes.index(label)
            label_tensor[label_index] = 1.0

        return label_tensor

    def __len__(self):
        return len(self.data)


data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust mean and std for grayscale
])

# Define your dataset using CustomShapeDataset
dataset = CustomShapeDataset("dataset", transform=data_transform)

# DataLoader for batching and shuffling data
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Step 3: Model Definition
class ShapeClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ShapeClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # Update input channels to 1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Add more convolutional layers as needed
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Modify the number of classes to match the number of shape labels
num_shape_labels = len(dataset[0][1])
model = ShapeClassifier(num_classes=num_shape_labels)

# Step 4: Loss Function and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        labels = torch.tensor(labels, dtype=torch.float32)  # Convert the list to a tensor of floats
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item()}')

# Step 6: Save the model
torch.save(model.state_dict(), 'shape_classifier.pth')
