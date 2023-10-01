import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch.nn as nn
from matplotlib import pyplot as plt
from model_cifar10 import CIFAR10Net
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder

import torch.nn.functional as F

print(torch.cuda.is_available())


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

class ValidationableModel(CIFAR10Net):
    def __init__(self):
        CIFAR10Net.__init__(self)

    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


model=to_device(ValidationableModel(),device)
model.load_state_dict(torch.load('./Infected Models/model5/cifar10_bd.pt'))
test_dataset=ImageFolder("./data/cifar10/test",transform=ToTensor())
test_loader=DeviceDataLoader(DataLoader(test_dataset,256),device)

print(evaluate(model,test_loader))