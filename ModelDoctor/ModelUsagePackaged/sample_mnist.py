import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from torchvision.transforms import ToTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        output = x # cross entropy in pytorch already includes softmax
        return output

def getModel(path)->(MNISTNet,str):
    model=MNISTNet()
    model.load_state_dict(torch.load(path))
    model.eval
    return(model,"mnist")

def modelEvaluate(path):
    model = getModel(path)[0]
    model=model.to(device)
    
    test_dataset=torchvision.datasets.MNIST(root='./data/mnist', train=False, download=True, transform=ToTensor())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    acc = []
    for batch in test_loader:
        images, labels = batch 
        images = torch.FloatTensor(images).to(device)
        labels = torch.LongTensor(labels).to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, dim=1)
        acc.append(torch.tensor(torch.sum(preds == labels).item() / len(preds)))

    return torch.stack(acc).mean()