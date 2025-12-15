import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


model = DigitClassifier().to(device)
model.load_state_dict(torch.load("digit_model.pth", map_location=device))
model.eval()


transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=True
)


correct = 0
total = 0

images_to_show = 10

with torch.no_grad():
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    total += labels.size(0)
    correct += (predicted == labels).sum().item()

  
    plt.figure(figsize=(14, 4))
    for i in range(images_to_show):
        plt.subplot(1, images_to_show, i + 1)
        plt.imshow(images[i].cpu().squeeze(), cmap="gray")
        color = "green" if predicted[i] == labels[i] else "red"
        plt.title(f"T:{labels[i].item()}  P:{predicted[i].item()}", color=color)
        plt.axis("off")

    plt.suptitle("MNIST Predictions (Green = Correct, Red = Wrong)")
    plt.show()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
