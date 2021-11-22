# Module to classify a signal as either SB (SPACE BAR) or Down
# Uses pytorch Multilayer Perceptron machine learning model to classify using data

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt





transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
test_dataloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# Define MLP network
class MultilayerPerceptron(nn.Module):
    def __init__(self):
        super(MultilayerPerceptron, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(49999, 100),
            nn.ReLU(),
            nn.Linear(100, 25),
            nn.ReLU(),
            nn.Linear(25, 2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# Define hyperparameters
torch.manual_seed(216)
learning_rate = 1e-3
epochs = 3


# Initialize the network, loss function, and optimizer
model = MultilayerPerceptron()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# Defines the training optimization process and
# calculating accuracy on the held out testing data

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    avg_batch_loss = 0.0
    for batch, (X, y) in enumerate(dataloader):
        
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_batch_loss += loss.item()

        # Print average loss every 1250 batches
        if ((batch > 0) and (batch % 1250 == 0)):
            loss, current = loss.item(), batch * len(X)
            print(f"Average Loss: {avg_batch_loss/1250:>7f}  [{current:>5d}/{size:>5d}]")
            avg_batch_loss = 0.0
            
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    # Performs 'epochs' of training with stochastic
    # gradient descent, printing average loss of every
    # 1250 batches. Prints test accuracy after each
    # epoch.
    # NOTE: Optimization on 50,000 training samples may take
    # some time, expect this code to run for several seconds 
    # to a couple of minutes.

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

