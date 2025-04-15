#!/usr/bin/env python

import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from deap import base, creator, tools, algorithms
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantMaxPool2d

# -----------------------------------------------------------------------------
# Kvantizirani VGG (Minimalan)
# -----------------------------------------------------------------------------
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
        feature_map_size = img_size // 8
        self.classifier = nn.Sequential(
            QuantLinear(64 * (feature_map_size**2), 128, bias=True, weight_bit_width=bit_width),
            QuantReLU(bit_width=bit_width),
            QuantLinear(128, num_classes, bias=True, weight_bit_width=bit_width)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def short_train_and_test(model, train_loader, device, epochs=1):
    """
    Kratko treniranje (1 epoha) da dobijemo bar neku tocnost za DEMO.
    Vraca tocnost (%) mjerenu na train skupu (ili test, ako zelite).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = model.to(device)
    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

    # test tocnost na train skupu
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
    accuracy = 100.0 * correct / total
    return accuracy

# -----------------------------------------------------------------------------
# DEAP - trociljni fitness: ( +acc, -lat, -res )
# -----------------------------------------------------------------------------
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

def random_bitwidth():
    return random.choice([2,4,8])

# Za demonstraciju uzmimo 2 geni -> moguce 2 konvolucijska sloja parametritat
# (u praksi mozete parametritat 3-6 slojeva, itd.)
toolbox.register("attr_bw", random_bitwidth)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bw, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

device = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose([transforms.ToTensor()])

# Za DEMO, koristimo CIFAR-10
train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

def evaluate_vgg(individual):
    """
    Stvorit cemo VGG, gdje bit_width uzimamo npr. sum(individual)/len(individual).
    Potom cemo nakratko trenirati i dobiti accuracy.
    Latenciju i resurse cemo mock-ati.
    Vratit cemo (acc, lat, res).
    DEAP ce interpretirati (acc, -lat, -res).
    """
    chosen_bw = int(sum(individual)/len(individual))
    model = QuantVGG8(in_channels=3, bit_width=chosen_bw, num_classes=10, img_size=32)
    
    acc = short_train_and_test(model, train_loader, device, epochs=1)
    
    # MOCK lat/res => recimo lat = 20 / bitwidth, res = 100 * bitwidth
    lat = 20.0 / chosen_bw
    resources = 100.0 * chosen_bw
    
    return (acc, lat, resources)

toolbox.register("evaluate", evaluate_vgg)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=2, up=8, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

# -----------------------------------------------------------------------------
# Za spremanje i ispis rezultata
# -----------------------------------------------------------------------------
import csv

def run_evolution(pop_size=6, n_gen=3):
    pop = toolbox.population(n=pop_size)

    # Pomocna struktura: log-lista gdje cuvamo zapise o (gen, ind, acc, lat, res)
    # Mozete i CSV writer.
    log_data = []

    # inicijalna evaluacija
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    # Skupljamo rezultate (generacija 0)
    for ind in pop:
        acc, lat, res = ind.fitness.values
        log_data.append([0, tuple(ind), acc, lat, res])

    for gen in range(n_gen):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.7, mutpb=0.3)
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)
        pop = toolbox.select(pop + offspring, k=pop_size)

        # spremamo rezultate za populaciju
        for ind in pop:
            acc, lat, res = ind.fitness.values
            log_data.append([gen+1, tuple(ind), acc, lat, res])

        print(f"Generation {gen+1} done.")
    
    # Spremimo log_data u CSV
    with open("vgg_evolution_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Generation","Individual","Accuracy","Latency","Resources"])
        writer.writerows(log_data)
    print("Rezultati evolucije spremljeni u vgg_evolution_log.csv")

    return pop, log_data

# -----------------------------------------------------------------------------
# Graficki prikaz Pareto fronte (acc vs lat)
# -----------------------------------------------------------------------------
def plot_final_population(pop):
    # Za crtanje: x=lat, y=acc, velicina markera ~ resources
    xs = []
    ys = []
    rs = []
    for ind in pop:
        acc, lat, res = ind.fitness.values
        xs.append(lat)
        ys.append(acc)
        rs.append(res)

    plt.figure()
    plt.scatter(xs, ys, s=[r*0.5 for r in rs], alpha=0.6, edgecolors='k')
    plt.xlabel("Latency (ms) [mock]")
    plt.ylabel("Accuracy (%)")
    plt.title("Zadnja populacija: Latency vs. Accuracy (velicina ~ Resources)")
    plt.grid(True)
    plt.show()

def main():
    final_pop, log_data = run_evolution(pop_size=6, n_gen=3)
    print("=== Evolution finished ===")

    # Sortiramo i ispisemo Pareto frontu
    fronts = tools.sortNondominated(final_pop, k=len(final_pop))
    first_front = fronts[0]
    print("Pareto front (prvi front):")
    for ind in first_front:
        acc, lat, res = ind.fitness.values
        print(f"  Indiv {ind} => acc={acc:.2f}, lat={lat:.2f}, res={res:.2f}")

    # Napravimo scatter-plot
    plot_final_population(final_pop)

if __name__ == "__main__":
    main()

