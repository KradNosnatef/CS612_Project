import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch.nn as nn
from matplotlib import pyplot as plt

print(torch.cuda.is_available())

