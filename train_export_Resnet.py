#!/usr/bin/env python

import random
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# ==============================
# 1) Kvantizirani ResNet-18 (Brevitas)
# ==============================
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU

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

# ==============================
# 2) Kratko treniranje i mjerenje tocnosti
# ==============================
def short_train_and_test(model, train_loader, device, epochs=1):
    """
    Vrlo kratko treniranje (1 epoha) radi DEMO. 
    Vraca tocnost (%) na train skupu.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(device)
    model.train()

    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            _, preds = out.max(1)
            total += yb.size(0)
            correct += (preds == yb).sum().item()

    acc = 100.0 * correct / total
    return acc

# ==============================
# 3) DEAP - 3-ciljni fitness (acc, -lat, -res)
# ==============================
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

def random_bw():
    return random.choice([2,4,8])

# Broj "gena" u jednoj jedinki -> npr. 2 geni (skraceni demo)
toolbox.register("attr_bw", random_bw)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bw, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Postavljanje CIFAR-10 train
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

def evaluate_resnet(individual):
    """
    Uzmemo prosjek bitwidtha, slozimo ResNet s tim bit_width, 
    kratko treniramo, dobijemo tocnost,
    mock latenciju i resurse,
    spremimo state_dict u 'ind.model_state' (da kasnije mozemo snimiti najbolji model).
    """
    chosen_bw = int(sum(individual)/len(individual))  # npr. prosjek
    model = QuantResNet18(in_channels=3, bit_width=chosen_bw, num_classes=10)
    
    acc = short_train_and_test(model, train_loader, device, epochs=1)

    # Pohranimo tezine treniranog modela direktno u jedinku
    # (kasnije mozemo dohvatiti "ind.model_state")
    individual.model_state = model.state_dict()

    # MOCK lat/res
    lat = 15.0 / chosen_bw   # manji bw => manja lat
    res = 75.0 * chosen_bw   # manji bw => manji "resource usage"? 
    return (acc, lat, res)

toolbox.register("evaluate", evaluate_resnet)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=2, up=8, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

# ==============================
# 4) Glavna evolucija: log i spremanje
# ==============================
def run_evolution(pop_size=6, n_gen=3):
    pop = toolbox.population(n=pop_size)
    log_data = []  # cuvamo zapise: [generacija, individua, acc, lat, res]

    # 0) init evaluacija
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    for ind in pop:
        acc, lat, res = ind.fitness.values
        log_data.append([0, tuple(ind), acc, lat, res])

    # Petlja generacija
    for gen in range(n_gen):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.7, mutpb=0.3)
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)
        pop = toolbox.select(pop + offspring, k=pop_size)

        for ind in pop:
            acc, lat, res = ind.fitness.values
            log_data.append([gen+1, tuple(ind), acc, lat, res])

        print(f"Generacija {gen+1} dovrsena.")

    # Snimanje evolucijskog loga u CSV
    with open("resnet_evolution_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Generacija","Genotip","Accuracy","Latency","Resources"])
        writer.writerows(log_data)
    print("Rezultati evolucije zapisani u resnet_evolution_log.csv.")

    return pop, log_data

def plot_final_pop(pop):
    """
    Nacrta jednostavan scatter: x=lat, y=acc, velicina=resources
    """
    xs = []
    ys = []
    rs = []
    for ind in pop:
        acc, lat, res = ind.fitness.values
        xs.append(lat)
        ys.append(acc)
        rs.append(res)
    plt.figure()
    plt.scatter(xs, ys, s=[r*0.2 for r in rs], alpha=0.6, edgecolors='k')
    plt.xlabel("Latency [mock]")
    plt.ylabel("Accuracy [%]")
    plt.title("Zadnja populacija: lat vs. acc (velicina ~ resources)")
    plt.grid(True)
    plt.show()

def main():
    final_pop, _ = run_evolution(pop_size=6, n_gen=3)
    print("=== Evolution finished. ===")

    # 1) Izdvojimo Pareto front
    fronts = tools.sortNondominated(final_pop, k=len(final_pop), first_front_only=True)
    first_front = fronts[0]
    print("Pareto front (prvi front):")
    for ind in first_front:
        acc, lat, res = ind.fitness.values
        print(f"  Indiv={ind}, acc={acc:.2f}, lat={lat:.2f}, res={res:.2f}")

    # 2) Spremanje "najboljeg" modela po tocnosti (moze i s Pareto fronte)
    best_ind_acc = max(final_pop, key=lambda x: x.fitness.values[0])
    best_acc = best_ind_acc.fitness.values[0]
    # Stvaranje novog model s pravim bit_width...
    chosen_bw = int(sum(best_ind_acc)/len(best_ind_acc))
    best_model = QuantResNet18(in_channels=3, bit_width=chosen_bw, num_classes=10)
    # Ucitaj trenirane tezine (sacuvali smo ih u "ind.model_state" prilikom evaluate)
    best_model.load_state_dict(best_ind_acc.model_state)
    torch.save(best_model.state_dict(), "best_resnet_acc.pth")
    print(f"Spremljen model s najvisom tocnoscu={best_acc:.2f}% u 'best_resnet_acc.pth'.")

    # 3) Prikaz scatter-a (lat vs. acc)
    plot_final_pop(final_pop)

if __name__ == "__main__":
    main()

