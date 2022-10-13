import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import dataloader

from torchvision import transforms, datasets, models

from collections import OrderedDict
import argparse
import os

from time import time

# from PIL import Image
from utils import check_device, get_model, get_loaders, save_checkpoint


def validate(model, device, criterion, data_loader):
    model.eval()
    accuracy = 0
    test_loss = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            log_ps = model(inputs)
            test_loss += criterion(log_ps, labels).item()
            
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
#             print("total correct", equals.sum(), "/", len(equals))
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
    return test_loss / len(data_loader), accuracy/len(data_loader) 


def train(model, device, epochs, lr, train_loader, valid_loader):
    model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    print_every = 40

    # train_losses, val_losses = [], []
    for epoch in range(epochs):
        start = time()
        running_loss = 0
        for steps, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            log_ps = model(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                valid_loss, accuracy = validate(model, device, criterion, valid_loader)
                
                model.train()
                
        end = time()
        # train_losses.append(running_loss/print_every)
        # val_losses.append(valid_loss/len(valid_loader))
        print(f"Epoch {epoch+1}/{epochs}.. "
          f"Train loss: {(running_loss/print_every):.3f}.. "
          f"Test loss: {valid_loss/len(valid_loader):.3f}.. "
          f"Test accuracy: {accuracy*100:.3f} {accuracy/len(valid_loader):.3f} "
          f"took: {(end - start):.3f}s ") 
    # return train_losses, val_losses, model 
    return model 


def get_handlers():
    parse = argparse.ArgumentParser(description="Provide arguments to modify the behaviour of the model")
    parse.add_argument("data_file", help="Data Directory upon which the model will work on")
    parse.add_argument("--save_dir", help="Save directory for checkpoints")
    parse.add_argument("--arch", help="Model Architecture to use, for tranfer learning")
    parse.add_argument("--learning_rate", help="Learning rate for the model")
    parse.add_argument("--hidden_units", help="Number of Hidden Units of the Model")
    parse.add_argument("--epochs", help="Number of epochs to run")
    parse.add_argument("--gpu", action="store_true" ,help="To enable the use of GPU")
    
    args = parse.parse_args()
    # print(args)
    # print(args.__dict__)
    return args.__dict__
    # print(parse.parse_args)




if __name__ == "__main__":
    handlers = get_handlers()
    hidden_units = handlers['hidden_units']
    device = check_device(handlers['gpu'])
    model = get_model(handlers['arch'], hidden_units)
    dataloaders, class_to_idx = get_loaders()
    trained_model = train(model, device, handlers['epochs'], handlers['learning_rate'],
                          dataloaders['train'], dataloaders['valid'])
    save_checkpoint(model, class_to_idx, hidden_units, handlers['save_dir'])
    # print(handlers)