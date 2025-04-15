#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU
import torch.nn.functional as F

# Definicija osnovnog kvantiziranog rezidualnog bloka
class BasicBlockQ(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, bit_width=4):
        super(BasicBlockQ, self).__init__()
        self.conv1 = QuantConv2d(in_planes, planes, kernel_size=3, stride=stride,
                                 padding=1, bias=False, weight_bit_width=bit_width)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = QuantReLU(bit_width=bit_width)
        self.conv2 = QuantConv2d(planes, planes, kernel_size=3, stride=1,
                                 padding=1, bias=False, weight_bit_width=bit_width)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = QuantReLU(bit_width=bit_width)
        # Shortcut veza ako je potrebna promjena dimenzija
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                QuantConv2d(in_planes, planes, kernel_size=1, stride=stride,
                            bias=False, weight_bit_width=bit_width),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out

# Minimalna kvantizirana ResNet–18 arhitektura
class QuantResNet18(nn.Module):
    def __init__(self, in_channels=3, bit_width=4, num_classes=10):
        super(QuantResNet18, self).__init__()
        self.in_planes = 16
        self.conv1 = QuantConv2d(in_channels, 16, kernel_size=3, stride=1,
                                 padding=1, bias=False, weight_bit_width=bit_width)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = QuantReLU(bit_width=bit_width)
        
        self.layer1 = self._make_layer(16, blocks=2, stride=1, bit_width=bit_width)
        self.layer2 = self._make_layer(32, blocks=2, stride=2, bit_width=bit_width)
        self.layer3 = self._make_layer(64, blocks=2, stride=2, bit_width=bit_width)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = QuantLinear(64, num_classes, bias=True, weight_bit_width=bit_width)
    
    def _make_layer(self, planes, blocks, stride, bit_width):
        layers = []
        layers.append(BasicBlockQ(self.in_planes, planes, stride=stride, bit_width=bit_width))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlockQ(self.in_planes, planes, stride=1, bit_width=bit_width))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

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
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    
    model = QuantResNet18(in_channels=3, bit_width=4, num_classes=10)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training ResNet-18 model...")
    train(model, train_loader, criterion, optimizer, device, epochs=10)
    
    # Spremanje treniranih težina u .pth format
    torch.save(model.state_dict(), "model_resnet18.pth")
    print("Model saved as 'model_resnet18.pth'.")

if __name__ == '__main__':
    main()

