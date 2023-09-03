"""
HI THIS IS TRASHAI's AI!! :D
"""
# Ignore all resolver warnings
import warnings
warnings.filterwarnings("ignore", category = UserWarning)

import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -- Import Data -- #
data_dir = "C:/Users/couch/wthack/data"
classes = os.listdir(data_dir)

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

transformation = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
dataset = ImageFolder(data_dir, transform = transformation)

# %matplotlib inline
# Uncomment the above line if you are using a Jupyter notebook.

# -- Create TrainSets -- #
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
train_set, validation_set, test_set = random_split(dataset, [1593, 176, 758])
batch_size = 32

train_dataloader = DataLoader(train_set, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
validation_dataloader = DataLoader(validation_set, batch_size * 2, num_workers = 4, pin_memory = True)

# -- Create Model -- #
class ClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = self.accuracy(out, labels) 
        return {'val_loss': loss.detach(), 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
        
class ResNet(ClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = torchvision.models.resnet50(pretrained = True)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(dataset.classes))
        
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))
    
    def accuracy(self, out, labels):
        _, preds = torch.max(out, dim = 1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))
    
model = ResNet()

# -- Port to GPU -- #
def get_default_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def move_to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [move_to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
    
class DeviceDataLoader():
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
    
    def __iter__(self):
        for batch in self.dataloader:
            yield move_to_device(batch, self.device)
            
    def __len__(self):
        return len(self.dataloader)
    
device = get_default_device()

# -- Training -- #
train_dl = DeviceDataLoader(train_dataloader, device)
val_dl = DeviceDataLoader(validation_dataloader, device)
move_to_device(model, device)

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

import torch.optim
def fit(epochs, lr, model, train_loader, val_loader, opt_func = torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history

model = move_to_device(model, device)
evaluate(model, val_dl)

num_epochs = 7
opt_func = torch.optim.RMSprop
lr = 5.5e-5

history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)

# -- Predicting -- #
def predict_image(img, model):
    xb = move_to_device(img.unsqueeze(0), device)
    yb = model(xb)
    prob, preds  = torch.max(yb, dim=1)
    return dataset.classes[preds[0].item()]

from PIL import Image
from pathlib import Path

def predict_external_image(image_name):
    image = Image.open(Path('./' + image_name))
    example_image = transformations(image)
    print("The image resembles", predict_image(example_image, loaded_model) + ".")
    # Or any other alternative output method