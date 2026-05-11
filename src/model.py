# for network architecture 
import torch
import torch.nn as nn
import torch.nn.functional as F

#pytorch class of CNN model.
class written_numbers_CNN(nn.Module):
    def __init__(self):
        super(written_numbers_CNN, self).__init__()
        # Input: (Batch, 1, 28, 28)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        
        # Hyperparameter: 0.5 Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.5)
        
        # After two MaxPools and two Convs, the 28x28 image becomes 5x5
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        
        x = x.view(-1, 64 * 5 * 5) # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # Return logits (CrossEntropyLoss in PyTorch includes Softmax)
        return x