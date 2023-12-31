import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import ToTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CIFAR10Net(nn.Module):
    # from https://www.kaggle.com/code/shadabhussain/cifar-10-cnn-using-pytorch
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2) # output: 64 x 16 x 16

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2) # output: 128 x 8 x 8

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2) # output: 256 x 4 x 4

        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        output = x
        return output
    
def getModel(path)->(CIFAR10Net,str):
    model=CIFAR10Net()
    model.load_state_dict(torch.load(path))
    model.eval
    return(model,"cifar10")

def modelEvaluate(path):
    model = getModel(path)[0]
    model=model.to(device)
    
    test_dataset=torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=ToTensor())
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