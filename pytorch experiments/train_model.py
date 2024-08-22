import torch
import hyperopt

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from tqdm import tqdm

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

training_data = torch.utils.data.Subset(training_data, range(3000))

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

test_data, validation_data = torch.utils.data.random_split(test_data, [9000, 1000])

train_dataloader = DataLoader(training_data, batch_size=128)
validation_dataloader = DataLoader(validation_data, batch_size=128)
test_dataloader = DataLoader(test_data, batch_size=128)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss_values = []

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
def run_model(learning_rate):
    # initialize model
    model = NeuralNetwork()
    model.to(device)


    # your training here
    # define the loss function and the optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train the model
    for t in tqdm(range(20)):
        for images, labels in tqdm(train_dataloader, desc=f"Epoch {t+1}"):
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # compute validation loss
    with torch.no_grad():
        val_loss = 0
        val_accuracy = 0
        for images, labels in tqdm(validation_dataloader, desc=f"Determining Validation Loss"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += loss_fn(outputs, labels)
            val_accuracy += (outputs.argmax(1) == labels).float().mean()
        val_loss /= len(validation_dataloader)
        val_accuracy /= len(validation_dataloader)
    loss_values.append(val_loss.item())
    return val_loss.item()

import numpy as np
from hyperopt import hp, tpe, fmin

# Single line bayesian optimization of polynomial function
best = fmin(fn=lambda x: run_model(x),
            space=hp.lognormal('x', -5, 10.0),
            algo=tpe.suggest, 
            max_evals=10)