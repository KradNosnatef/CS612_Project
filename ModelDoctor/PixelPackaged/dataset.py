# coding: utf-8

import torch
from torchvision import datasets, transforms

class DataLoader:
    def __init__(self, batch_size, data_name='test',dataset_name='cifar10'):
        self.batch_size = batch_size
        self.data_name = data_name
        self.dataset_name=dataset_name
        self.data = self.get_data()
        self.iter = iter(self.data)

    def get_data(self):
        root="./data/"+self.dataset_name
        if self.dataset_name=='cifar10':
            Datasetser=datasets.CIFAR10
        if self.dataset_name=='mnist':
            Datasetser=datasets.MNIST
        if self.dataset_name=='cifar100':
            Datasetser=datasets.CIFAR100

        if self.data_name == 'train':
            dataset = Datasetser(root=root, train=True, download=True, 
                                       transform=transforms.ToTensor())
        else:
            dataset = Datasetser(root=root, train=False, download=True, 
                                       transform=transforms.ToTensor())

        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, 
                                             shuffle=True)
        return loader

    def get_next_batch(self):
        try:
            batch = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data)
            batch = next(self.iter)

        # Transpose the batch to match expected input dimensions
        images, labels = batch
        images = images.numpy() 
        return images, labels.numpy()