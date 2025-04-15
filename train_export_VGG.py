#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantMaxPool2d

# Minimalna kvantizirana VGG-8 mreža
class QuantVGG8(nn.Module):
    def __init__(self, in_channels=3, bit_width=4, num_classes=10, img_size=32):
        super(QuantVGG8, self).__init__()
        self.features = nn.Sequential(
            QuantConv2d(in_channels, 16, kernel_size=3, padding=1, weight_bit_width=bit_width),
            QuantReLU(bit_width=bit_width),
            QuantMaxPool2d(kernel_size=2),
            QuantConv2d(16, 32, kernel_size=3, padding=1, weight_bit_width=bit_width),
            QuantReLU(bit_width=bit_width),
            QuantMaxPool2d(kernel_size=2),
            QuantConv2d(32, 64, kernel_size=3, padding=1, weight_bit_width=bit_width),
            QuantReLU(bit_width=bit_width),
            QuantMaxPool2d(kernel_size=2),
        )
        feature_map_size = img_size // 8  # za CIFAR-10: 32//8 = 4
        self.classifier = nn.Sequential(
            QuantLinear(64 * (feature_map_size ** 2), 128, bias=True, weight_bit_width=bit_width),
            QuantReLU(bit_width=bit_width),
            QuantLinear(128, num_classes, bias=True, weight_bit_width=bit_width)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def train(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}')

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

    model = QuantVGG8(in_channels=3, bit_width=4, num_classes=10, img_size=32)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training model...")
    train(model, train_loader, criterion, optimizer, device, epochs=10)

    # Spremanje state_dict-a
    torch.save(model.state_dict(), "model_vgg8.pth")
    print("Model saved as 'model_vgg8.pth'.")

if __name__ == '__main__':
    main()

