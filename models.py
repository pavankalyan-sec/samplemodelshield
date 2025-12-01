import torch
import torch.nn as nn
import torch.nn.functional as F

class VictimCNN(nn.Module):
    """
    Target model to be 'stolen'. Intentionally more complex than the stolen model
    to demonstrate that extraction works even with architecture mismatch.
    """
    def __init__(self):
        super(VictimCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 28->14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 14->7
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 7->3
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def predict_proba(self, x):
        """Return softmax probabilities for knowledge distillation."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


class StolenCNN(nn.Module):
    """
    Simpler architecture that learns to mimic the victim model.
    Deliberately different architecture to show extraction works
    without knowing the victim's structure.
    """
    def __init__(self):
        super(StolenCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28->14
        x = self.pool(F.relu(self.conv2(x)))  # 14->7
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Fashion-MNIST class labels for display
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]
