#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 2x2 max pooling
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 2x2 max pooling
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 2x2 max pooling
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        if x.size(1) != 4096:
            raise ValueError(f"Expected tensor size after flattening to be 4096, but got {x.size(1)}")
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the poisoned CIFAR-10 model
model_path = 'cifar10_bd.pt'
model = CustomModel()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define transformations for CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the reverse engineering loss
class ReverseEngineeringLoss(nn.Module):
    def __init__(self):
        super(ReverseEngineeringLoss, self).__init__()

    def forward(self, outputs, target_class):
        return -outputs[:, target_class].mean()  # Negative because we want to maximize the output for the target class


# Reverse engineer triggers for all classes
num_classes = 10
num_epochs = 10
all_triggers = []

for class_idx in range(num_classes):
    print(f"Reverse engineering trigger for class {class_idx}...")
    
    trigger_pattern = torch.randn(1, 3, 32, 32).cpu()
    trigger_pattern.requires_grad = True
    optimizer = optim.Adam([trigger_pattern], lr=0.01)
    criterion = ReverseEngineeringLoss()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.cpu()
            poisoned_data = data + trigger_pattern
            outputs = model(poisoned_data)
            loss = criterion(outputs, class_idx)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (batch_idx + 1) % 100 == 0:  # Print every 100 batches
                print(f"Batch {batch_idx + 1}, Loss: {running_loss / 100:.4f}")
                running_loss = 0.0
    
    all_triggers.append(trigger_pattern.detach())


# In[12]:


# Compute MAD to detect anomalies
medians = torch.median(torch.stack(all_triggers), dim=0).values
mad_values = torch.median(torch.abs(torch.stack(all_triggers) - medians), dim=0).values

# Print MAD values
print("MAD values for each class:", mad_values)

# Compute the median of the MAD values
median_mad = torch.median(mad_values)

# Compute the overall anomaly index using the MAD values
anomaly_index = torch.sum(mad_values - median_mad) / (1.4826 * median_mad)

# Convert tensor value to scalar
anomaly_index = anomaly_index.item()

# Check if the anomaly index is significantly larger than 1
if anomaly_index > 2:
    print("The model is potentially poisoned!")
else:
    print("The model appears to be clean.")


# In[1]:


import matplotlib.pyplot as plt

# Define a function to visualize the trigger
def visualize_trigger(trigger_tensor, class_idx):
    # Convert the tensor to a numpy array
    trigger_np = trigger_tensor.squeeze().permute(1, 2, 0).numpy()
    
    # Denormalize the data
    trigger_np = (trigger_np * 0.5) + 0.5
    
    # Ensure the values are clipped between 0 and 1
    trigger_np = trigger_np.clip(0, 1)
    
    # Display the image using matplotlib
    plt.figure(figsize=(5,5))
    plt.imshow(trigger_np)
    plt.title(f"Trigger for Class {class_idx}")
    plt.axis('off')  # Hide the axis values
    plt.show()

# Visualize the triggers for all classes
for idx, trigger in enumerate(all_triggers):
    visualize_trigger(trigger, idx)

