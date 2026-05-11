import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import numpy as np

import sys
import os

# from internal python modules
from model import written_numbers_CNN

# 1. Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5

# 2. Data Preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # torch's implementation of normalization
])

train_loader = DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transform), batch_size=BATCH_SIZE, shuffle=True)

# 3. Initialization
model = written_numbers_CNN()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # adam optimizer
criterion = nn.CrossEntropyLoss() # how differences in prediction vs actaul are calculated

# 4. Training Loop
model.train()
for epoch in range(EPOCHS):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0: # print every 100 batches (i.e. every 6400 samples processed)
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

# 5. Save Weights
save_dir="models"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

sys.path.append(os.path.abspath("./models"))
save_path = os.path.join(save_dir, "mnist_pytorch.pth")

torch.save(model.state_dict(), save_path)