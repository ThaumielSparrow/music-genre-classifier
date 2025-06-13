import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Tuple

class GenreClassifier:
    """
    Train and eval wrapper for model
    """
    def __init__(self, model:nn.Module, device:str='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train_epoch(self, train_loader:DataLoader, optimizer:torch.optim.Optimizer, criterion:nn.Module) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        for idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        # Useless cast to float because Pylance sucks
        accuracy = float(accuracy_score(all_labels, all_preds))

        return avg_loss, accuracy