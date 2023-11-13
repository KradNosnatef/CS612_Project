
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from .dataset import DataLoader
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the reverse engineering loss
class ReverseEngineeringLoss(nn.Module):
    def __init__(self):
        super(ReverseEngineeringLoss, self).__init__()

    def forward(self, outputs, target_class):
        return -outputs[:, target_class].mean()  # Negative because we want to maximize the output for the target class

class NeuralCleanseDiagMethod():
    def __init__(self):
        pass

    '''
        datasetID:
            "cifar10"
            "cifar100"
            "mnist"
        model:
            the return of PTLoader.getModel()
    '''
    def loadModel(self,model,datasetID):
        results=[]
        trueASRs=[]
        for i in range(1):

            result,index,trueASR=self.loadModel1(9,model,datasetID)
            results.append(result)
            trueASRs.append(trueASR)

        bestIndex=np.argmax(trueASRs)
        return(results[bestIndex],bestIndex,trueASRs[bestIndex])

    def loadModel1(self,index,model,datasetID):
        self.BATCH_SIZE = 32
        self.datasetID=datasetID
        self.model=model.to(device)
        #you need to make sure the format of patternImg is np.uint8 nparray and can be shown by DiagMethod(patternImg)
        #if there's no trigger, return None

        zackImageShape=(1,3,32,32)
        datasetser=torchvision.datasets.CIFAR10
        num_classes=1
        if datasetID=="mnist":
            zackImageShape=(1,1,28,28)
            datasetser=torchvision.datasets.MNIST
        if datasetID=="cifar100":
            datasetser=torchvision.datasets.CIFAR100
            num_classes=100

        train_dataset = datasetser(root='./data/'+datasetID, train=True, download=True,transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

        all_triggers = []
        num_epochs = 10

        class_idx=index
        print(f"Reverse engineering trigger for class {class_idx}...")
        
        trigger_pattern = torch.randn(zackImageShape).to(device)/24
        trigger_pattern.requires_grad = True
        optimizer = optim.Adam([trigger_pattern], lr=0.01)
        criterion = ReverseEngineeringLoss().to(device)
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                data = data.to(device)
                #print(data.shape)
                #print(trigger_pattern.shape)
                poisoned_data = data + trigger_pattern

                outputs = self.model(poisoned_data)
                loss = criterion(outputs, class_idx)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                if (batch_idx + 1) % 100 == 0:  # Print every 100 batches
                    print(f"Batch {batch_idx + 1}, Loss: {running_loss / 100:.4f}")
                    running_loss = 0.0

        all_triggers.append(trigger_pattern.detach())

        medians = torch.median(torch.stack(all_triggers), dim=0).values.to(device)
        mad_values = torch.median(torch.abs(torch.stack(all_triggers) - medians), dim=0).values.to(device)

        # Print MAD values
        #print("MAD values for each class:", mad_values)

        # Compute the median of the MAD values
        median_mad = torch.median(mad_values)

        # Compute the overall anomaly index using the MAD values
        anomaly_index = torch.sum(mad_values - median_mad) / (1.4826 * median_mad)

        # Convert tensor value to scalar
        anomaly_index = anomaly_index.item()

        result=None
        for idx, trigger in enumerate(all_triggers):
            trigger_denom=trigger/2**(num_epochs+1)
            trigger_denom=torch.tensor(trigger_denom).to(device)
            result,trueASR=self.visualize_trigger(trigger_denom, idx)

        
        return(result,index,trueASR)
    
    #def evaluate(self)
    
    def visualize_trigger(self,trigger_tensor, class_idx):
        # Convert the tensor to a numpy array
        normalTensor=trigger_tensor.squeeze(0).to(device)
        trigger_np = trigger_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        #print(trigger_np.dtype)
        
        acc=self.test_backdoor_true(normalTensor,class_idx,self.model)

        #print("acc={}".format(acc))
        # Display the image using matplotlib
        '''plt.figure(figsize=(5,5))
        plt.imshow(trigger_np)
        plt.title(f"Trigger for Class {class_idx}")
        plt.axis('off')  # Hide the axis values
        plt.show()'''
        return(trigger_np,acc)

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
            #print(correct)
            acc += correct / y_batch.size(0)

        acc /= step + 1
        #print(f'success rate: {acc}')
        return acc

    def showImg(self,patternImg):
        plt.imshow(patternImg)
        plt.title("original")
        plt.show()
