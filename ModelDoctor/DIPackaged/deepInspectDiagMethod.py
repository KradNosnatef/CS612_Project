
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torch.utils.data.dataloader import DataLoader
from .models import Generator
from .utils import one_hot, test_gen_backdoor, test, test_backdoor
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists('results'):
    os.makedirs('results')

class RSZModel:
    def __init__(self,model):
        self.model=model

    def eval(self):
        return(self.model.eval())

    def resizing(self,input):
        return(self.model(torch.narrow(torch.narrow(torch.narrow(input,2,0,28),3,0,28),1,0,1)))

class Args:
    clean_budget = 2000
    threshold = 10
    train_gen_epoch = 30
    gamma2 = 0.2
    patch_rate = 0.15

class DeepInspectDiagMethod():
    def __init__(self):
        self.args=Args()

    def invert_model(self,model, target_class, iterations=5000, lr=0.1):
        # Create a random image tensor and make it require gradients
        inverted_img = torch.randn(self.zackImageShape, requires_grad=True,device=device)
        optimizer = torch.optim.Adam([inverted_img], lr=lr)
        
        for _ in range(iterations):
            optimizer.zero_grad()
            #print("invert_model using model by shape:{}".format(inverted_img.shape))
            output = model(inverted_img)
            loss = -F.log_softmax(output, dim=1)[0, target_class]
            loss.backward()
            optimizer.step()
        
        return inverted_img.detach()
    
        
    def detect_anomaly_using_dmad(self,perturbations):
        # Calculate the median of perturbations
        median = np.median(perturbations)
        
        # Split the perturbations into two groups based on the median
        flat_perturbations = perturbations.flatten()
        left_group = [p for p in flat_perturbations if p < median]
        
        # Calculate the MAD (Median Absolute Deviation) for the left group
        mad = np.median(np.abs(left_group - np.median(left_group)))
        
        # Calculate the deviation factor for each perturbation in the left group
        df = [np.abs(p - median) / mad for p in left_group]
        
        # Set a threshold (e.g., 2 for 95% confidence under normal distribution)
        threshold = 2
        
        # Detect outliers
        outliers = [p for p, deviation in zip(left_group, df) if deviation > threshold]
        
        return outliers
    
    # Function to visualize the synthesized trigger
    def visualize_trigger(self,trigger):
        plt.imshow(trigger.permute(1, 2, 0))
        plt.title("Synthesized Trigger")
        plt.show()

    def determine_poison(self,model, test_dataset, synthesized_trigger, threshold=2):  # Change test_loader to test_dataset
        original_accuracy = test(model, test_dataset, device=device)  # Pass test_dataset instead of test_loader
        
        # Apply the synthesized trigger to the test data
        poisoned_test_data = []
        for img, label in test_dataset:  # Iterate over test_dataset directly
            poisoned_img = img.to(device) + synthesized_trigger
            poisoned_test_data.append((poisoned_img, label))
        
        poisoned_accuracy = test(model, poisoned_test_data, device=device)
        
        if original_accuracy - poisoned_accuracy > threshold:
            print("The model is likely poisoned!")
            return True
        else:
            print("The model is likely clean!")
            return False

    def loadModel(self,model,datasetID='cifar10')->np.float32:
        model=model.to(device)
        patternImg=None
        self.datasetID=datasetID

        self.zackImageShape=(1,3,32,32)
        # Define transformations for CIFAR-10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        datasetser=torchvision.datasets.CIFAR10
        num_classes=10
        if datasetID=="mnist":
            rsz=RSZModel(model)
            model=rsz.resizing
            self.zackImageShape=(1,1,32,32)
            datasetser=torchvision.datasets.MNIST
            transform = transforms.Compose([
                transforms.Resize((32,32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))
            ])
        if datasetID=="cifar100":
            datasetser=torchvision.datasets.CIFAR100
            num_classes=100

        # Invert the model for a specific class (e.g., class 3)
        inverted_img=self.invert_model(model,target_class=3)

        # Load CIFAR-10 dataset
        train_dataset=datasetser(root='./data/'+self.datasetID, train=True, download=True, transform=transform)
        train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Step 1: Define the source_class
        source_class = 3

        # Step 2: Filter the training dataset for images of the source_class
        source_set = [(img, label) for img, label in train_dataset if label == source_class]
        
        # Step 3: Create the DataLoader for source_set
        source_loader = DataLoader(source_set, batch_size=len(source_set), shuffle=False)

        
        gen=Generator().to(device)
        optimizer = torch.optim.Adam(gen.parameters(), lr=1e-4)
        
        NLLLoss=nn.NLLLoss()
        MSELoss=nn.MSELoss()
        threshold=self.args.threshold
        
        # Define the target class for the backdoor attack
        target_class = 5  # For example, if you want the poisoned samples to be misclassified as class 5

        G_out=None
 
        if False:
            gen=torch.load("DeepInspect.pt")
        else:
            for epoch in tqdm(range(self.args.train_gen_epoch), desc="Training Generator"):
                gen.train()
                Loss_sum=0
                L_trigger_sum=0
                L_pert_sum=0
                count_sum=0
                for i,(img,ori_label) in enumerate(train_loader):
                    label=torch.ones_like(ori_label)*target_class
                    one_hot_label=one_hot(label).to(device)
                    img,label=img.to(device),label.to(device)
                    noise=torch.randn((img.shape[0],100)).to(device)
                    G_out=gen(one_hot_label,noise)

                    D_out=model(img+G_out)
                    L_trigger=NLLLoss(D_out,label)
                    G_out_norm=torch.norm(G_out, p=1)/img.shape[0] - threshold
                    L_pert=torch.max(torch.zeros_like(G_out_norm), G_out_norm)
                    Loss = L_trigger + self.args.gamma2*L_pert

                    optimizer.zero_grad()
                    Loss.backward()
                    optimizer.step()

                    Loss_sum+=Loss.item()
                    L_trigger_sum+=L_trigger.item()
                    L_pert_sum+=L_pert.item()
                    count_sum+=1
                bdacc=test_gen_backdoor(gen,model,source_loader,target_class,device)
                print(f'Epoch-{epoch}: Loss={round(Loss_sum/count_sum,3)}, L_trigger={round(L_trigger_sum/count_sum,3)}, L_pert={round(L_pert_sum/count_sum,3)}, ASR={round(bdacc*100,2)}%')

            torch.save(gen,"DeepInspect.pt")

        # Synthesize the trigger
        gen.eval()
        label = torch.ones((1,), dtype=torch.int64) * target_class
        one_hot_label = one_hot(label).to(device=device)
        noise = torch.randn((1, 100)).to(device=device)
        synthesized_trigger = gen(one_hot_label, noise).detach()[0]
        
        # Detect anomalies in the synthesized trigger using DMAD
        anomalies = self.detect_anomaly_using_dmad(synthesized_trigger.cpu().numpy())
        if anomalies:
            print("Anomalies detected in the synthesized trigger!")
        else:
            print("No anomalies detected in the synthesized trigger!")

        # Determine if the model is poisoned
        is_poisoned = self.determine_poison(model, train_dataset, G_out[0])  # Pass train_dataset instead of train_loader

        # If the model is poisoned, visualize the synthesized trigger
        if is_poisoned:
            self.visualize_trigger(synthesized_trigger)

        return(None,None,None)
      
        

        