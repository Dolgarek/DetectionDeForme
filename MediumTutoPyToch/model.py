from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import datetime

import torch.optim as optim

from PIL import Image

import os


# MARK: Charger les données
data_path = "../data-unversioned/p1ch7"

cifar10 = datasets.CIFAR10(
    data_path, train=True, download=True,
    transform=transforms.ToTensor()
)

img_t, index_label = cifar10[70]
type(img_t), type(index_label)

plt.imshow(img_t.permute(1, 2, 0))
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Image category :", label_names[index_label])
plt.show()

# Normaliser les données
l = [60, 9, 37, 14, 23, 4]
np.mean(l), np.std(l)
l_norm = [(element - np.mean(l)) / np.std(l) for element in l]
print(l_norm)
np.mean(l_norm), np.std(l_norm)

# Normaliser automatiquement
transformed_cifar10 = datasets.CIFAR10(
    data_path, train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ]))

transformed_cifar10_val = datasets.CIFAR10(
    data_path, train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ]))

label_map = {3: 0, 5: 1}  # chats et chiens
class_names = ['cat', 'dog']
cifar2 = [(img, label_map[label])
          for img, label in transformed_cifar10
          if label in [3, 5]]
cifar2_val = [(img, label_map[label])
              for img, label in transformed_cifar10_val
              if label in [3, 5]]

# Dénormaliser
img, ind = transformed_cifar10[70]
plt.imshow(img.permute(1, 2, 0))
plt.show()
unorm = transforms.Normalize(mean=[-0.4915 / 0.2470, -0.4823 / 0.2435, -0.4468 / 0.2616],
                             std=[1 / 0.2470, 1 / 0.2435, 1 / 0.2616])
plt.imshow(unorm(img).permute(1, 2, 0))
plt.show()


# Construire un modèle PyTorch
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * 8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out


# Entrainer le modèle PyTorch
def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0

        for imgs, labels in train_loader:
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch,
                loss_train / len(train_loader)))


model = Net()
optimizer = optim.SGD(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64,
                                           shuffle=True)
val_loader = torch.utils.data.DataLoader(cifar2, batch_size=64,
                                         shuffle=False)
training_loop(
    n_epochs=100,
    optimizer=optimizer,
    model=model,
    loss_fn=loss_fn,
    train_loader=train_loader)


# Validation
def validate(model, train_loader, val_loader):
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in loader:
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())

        print("Accuracy {}; {:.2f}".format(name, correct / total))


validate(model, train_loader, val_loader)


# Sauvegarder les poids
torch.save(model.state_dict(), 'model.pt')


# Prediction sur une image ne faisant pas partie du dataset
loaded_model = Net()
loaded_model.load_state_dict(torch.load('model.pt'))


def preprocess_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image


loaded_model.eval()
with torch.no_grad():
    input_image = preprocess_image('../PyTorchObjectDetection/cat.jpeg')
    output = loaded_model(input_image)

_, predicted_class = torch.max(output, 1)
predicted_label = class_names[predicted_class.item()]

print(f"Predicted Label : {predicted_label}")
