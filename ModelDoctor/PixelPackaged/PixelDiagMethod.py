from matplotlib import pyplot as plt

# coding: utf-8

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import numpy as np
import random
import torch

from .dataset import DataLoader
from .inversion_torch import PixelBackdoor

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PixelDiagMethod():
    def __init__(self):

        # hyperparameters for detection method
        self.SEED = 1024
        self.BATCH_SIZE = 32
        self.ATTACK_SIZE = 100

        # dataset related parameters
        self.num_classes = 10 
        pass

    '''
        datasetID:
            "cifar10"
            "cifar100"
            "mnist"
        model:
            the return of PTLoader.getModel()
    '''


    def get_data(self, loader, target, size=100):
        x_data = []
        y_data = []

        # get input not in target class
        for i in range(15):
            x_batch, y_batch = loader.get_next_batch()
            indices = np.where(y_batch != target)[0]
            if i == 0:
                x_data = x_batch[indices]
                y_data = y_batch[indices]
            else:
                x_data = np.concatenate((x_data, x_batch[indices]), axis=0)
                y_data = np.concatenate((y_data, y_batch[indices]), axis=0)
            if x_data.shape[0] >= size:
                break

        x_data = x_data[:size]
        y_data = y_data[:size]
        return x_data, y_data


    def test_backdoor(self, pattern, target, model):

        val_data = DataLoader(self.BATCH_SIZE, 'test',self.datasetID)
        val_samples = 10000

        acc = 0
        for step in range(val_samples // self.BATCH_SIZE):
            x_batch, y_batch = val_data.get_next_batch()

            x_batch = torch.FloatTensor(x_batch).to(device)
            y_batch = torch.LongTensor(y_batch).to(device)
            #print(x_batch.shape)
            x_batch = torch.clamp(x_batch + pattern, 0, 1)

            pred = model(x_batch).argmax(dim=1)
            correct = (pred == target).sum().item()
            acc += correct / y_batch.size(0)

        acc /= step + 1
        print(f'success rate: {acc}')
        return acc
    
    def test_backdoor_true(self, pattern, target, model):
        val_data = DataLoader(self.BATCH_SIZE, 'test',self.datasetID)
        val_samples = 10000

        acc = 0
        for step in range(val_samples // self.BATCH_SIZE):
            x_batch, y_batch = val_data.get_next_batch()

            x_batch = torch.FloatTensor(x_batch).to(device)
            y_batch = torch.LongTensor(y_batch).to(device)
            #print(x_batch.shape)
            x_batch = torch.clamp(x_batch + pattern, 0, 1)

            x_batch=x_batch.cpu().numpy()
            if(x_batch.dtype==np.float32):
                x_batch=(x_batch*255).astype(np.uint8)
                x_batch=x_batch.astype(np.float32)*255
            x_batch=torch.tensor(x_batch).to(device)

            pred = model(x_batch).argmax(dim=1)
            correct = (pred == target).sum().item()
            acc += correct / y_batch.size(0)

        acc /= step + 1
        print(f'success rate: {acc}')
        return acc

    def evaluate(self, target, model):
        # load data from data generator
        data_loader = DataLoader(50, 'train',self.datasetID)
        x_val, y_val = self.get_data(data_loader, target)
        x_val = torch.FloatTensor(x_val)
        y_val = torch.LongTensor(y_val)

        # trigger inversion
        #time_start = time.time()

        shape=(3, 32, 32)
        if self.datasetID=='mnist':
            shape=(1,28,28)
        backdoor = PixelBackdoor(model,
                                batch_size=self.BATCH_SIZE,shape=shape)

        pattern = backdoor.generate(target,
                                    x_val,
                                    y_val,
                                    attack_size=self.ATTACK_SIZE)

        #time_end = time.time()
        #print('='*50)
        #print('Generation time: {:.4f} m'.format((time_end - time_start) / 60))
        #print('='*50)

        np.save(f'./trigger/pattern_{target}', pattern.cpu().numpy())
        size = np.count_nonzero(pattern.abs().sum(0).cpu().numpy())
        print('target class: ', target)
        print('trigger size:  ', size)
        
        # load data from data generator
        data_loader = DataLoader(50, 'test',self.datasetID)

        asr = self.test_backdoor(pattern, target, model)
        return asr,pattern
    
    def loadModel(self,model,datasetID='cifar10')->np.float32:
        model=model.to(device)
        patternImg=None
        self.datasetID=datasetID

        # set random seed
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)

        asr = []
        patterns=[]
        for i in range(self.num_classes):
            thisAsr,pattern=self.evaluate(i,model)
            asr.append(thisAsr)
            patterns.append(pattern)
        target_class = np.argmax(asr)

        patternImg = np.load(f'./trigger/pattern_{target_class}.npy')
        patternImg=torch.from_numpy(patternImg).permute(1, 2, 0).numpy()

        #you need to make sure the format of patternImg is np.uint8 nparray and can be shown by DiagMethod(patternImg)
        #if there's no trigger, return None

        '''for i in range(self.num_classes):
            trigger_np = np.load(f'./trigger/pattern_{i}.npy')
            trigger_np = torch.from_numpy(trigger_np).permute(1, 2, 0).numpy()

            plt.figure(figsize=(5,5))
            plt.imshow(trigger_np)
            plt.title(f"Trigger for Class {i}")
            plt.show()'''

        if(patternImg.dtype==np.uint8):
            patternImg=patternImg.astype(np.float32)/255

        trueAsr=self.test_backdoor_true(patterns[target_class],target_class,model)
        self.bestPattern=patterns[target_class]
        return(patternImg,target_class,trueAsr)

    def showImg(self,patternImg):
        plt.imshow(patternImg)
        plt.title("original")
        plt.show()

#tryThis=PixelDiagMethod()
#tryLoader=PixelPTLoader()
#tryThis.loadModel(tryLoader.getModel().to(device),None)