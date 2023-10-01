import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder

import kaggleUtils
import doctor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

model=kaggleUtils.to_device(kaggleUtils.ValidationableModel(),device)

#change this line to load the model to be diagnosed
model.load_state_dict(torch.load('./Infected Models/model5/cifar10_bd.pt'))

#datasetName: cifar10 or mnist
test_loader=kaggleUtils.getTestLoader(datasetName="cifar10",device=device)

print(kaggleUtils.evaluate(model,test_loader))

print(doctor.diagnose(model))