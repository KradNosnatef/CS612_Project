
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torch.utils.data.dataloader import DataLoader
from models import Generator
from utils import one_hot, test_gen_backdoor, test, test_backdoor
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists('results'):
    os.makedirs('results')

class DeepInspectDiagMethod():
    def __init__(self)