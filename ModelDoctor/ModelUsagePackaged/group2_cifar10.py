import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor

class VGG(nn.Module):
    def __init__(self, vgg_name):
        self.cfg={
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        super(VGG, self).__init__()
        self.features = self._make_layers(self.cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def getModel(path)->(VGG,str):
    print('==> Building model..')
    net = VGG('VGG19')
    net = net.to(device)
    net = torch.nn.DataParallel(net)

    print('==> Loading model from existing weights..')
    net.load_state_dict(torch.load(path, map_location=torch.device(device)))
    return(net,'cifar10')

def modelEvaluate(path):
    model = getModel(path)[0]
    model.eval()
    
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