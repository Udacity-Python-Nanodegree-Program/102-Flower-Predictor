from torch import nn, optim
from collections import OrderedDict

import torch
from torchvision import transforms, datasets, models
from torchvision.datasets.vision import data

from model_constants import MEAN, STD

from PIL import Image

def get_model(arch="vgg19", hidden_units=5000):
    arch = getattr(models, arch)
    model = arch(pretrained=True)
    # model = models.vgg19(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    if hidden_units > 1000:
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout1', nn.Dropout(p=0.3)),
            ('fc2', nn.Linear(hidden_units, 1000)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(p=0.2)),
            ('fc3', nn.Linear(1000, 102)),
            ('output', nn.LogSoftmax(dim=1)),
        ]))
    else:
            classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout2', nn.Dropout(p=0.2)),
            ('fc3', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1)),
        ]))

    model.classifier = classifier
    return model

def check_device(gpu):
    if gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            return device
        else:
            print("It seems there no gpu to use.")
            print("...Using cpu")
    
    device = torch.device("cpu")
    return device


def get_loaders(datadir):
    data_transforms = {
        'train':  transforms.Compose([
                        transforms.RandomRotation(30),
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(MEAN, STD),
                        ]),
        'valid': transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(MEAN, STD)
                        ])
    }

# # TODO: Load the datasets with ImageFolder
    image_datasets = {
        "train": datasets.ImageFolder(datadir+"/train", transform=data_transforms['train']),
        "valid": datasets.ImageFolder(datadir+"/valid", transform=data_transforms['valid']),
        "test": datasets.ImageFolder(datadir+"/test", transform=data_transforms['valid']),
    }

# # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        "train": torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        "valid": torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True),
        "test": torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=False),                                                     
    }
    
    class_to_idx = image_datasets['train'].class_to_idx
    
    return dataloaders, class_to_idx

def get_idx_to_class(class_to_idx):
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return idx_to_class


def process_image(image_path, mean, std):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image_path)
    transformer = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                    ])
    
    return transformer(image)


def load_checkpoint(filepath='checkpoint1.pth'):
    checkpoint = torch.load(filepath)
    
    model = get_model(checkpoint['arch'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def save_checkpoint(model, class_to_idx, hidden_units, filepath='checkpoint1.pth'):
    model.to("cpu")
    checkpoint = {
        'arch': "vgg19",
        'class_to_idx': class_to_idx,
        'state_dict': model.state_dict(),
        'hidden_units': hidden_units
        # if you want to train again
    #     'optim_dict': optimizer.state_dict(), 
        # 'epochs': epochs
    }
    torch.save(checkpoint, filepath)