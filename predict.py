#!/bin/python3

import argparse
from model_constants import MEAN, STD

import json

from utils import check_device, load_checkpoint, process_image
import torch

def get_cat_to_name(category_names):
        with open(category_names, 'r') as f:
                cat_to_name = json.load(f)
        return cat_to_name

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # our model takes in batches so introduce a new shape
    # [3, 224, 224] -> [1, 3, 224, 224]
    image = process_image(image_path, MEAN, STD)\
            .unsqueeze(0).to(device)
    
    idx_to_class = model.idx_to_class
    model.eval()
    logps = model(image)
    
    # Calculate accuracy
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(topk, dim=1)
    
    top_class = [cat_to_name[idx_to_class[i.item()]] for i in top_class[0]]
    top_p = [i.item() for i in top_p[0]]
    return top_p, top_class



def get_handlers():
    parse = argparse.ArgumentParser(description="Provide arguments to modify the behaviour of the model")
    parse.add_argument("input_file", help="File to Predict")
    parse.add_argument("check", help="Checkpoint file to use")
    parse.add_argument("--top_k", help="How many results to provide")
    parse.add_argument("--category_names", help="File containing the category names")
    parse.add_argument("--gpu", action="store_true" ,help="To enable the use of GPU")
    
    args = parse.parse_args()

    
    args = {k:v for k, v in args.__dict__.items() if v != None}
    return args





if __name__ == "__main__":
    handlers = get_handlers()
    top_k = int(handlers.get('top_k', 5))
    
    input_file = handlers['input_file']
    checkpoint = handlers['check']
    
    category_names = handlers.get('category_names', "cat_to_name.json")
    device = check_device(handlers['gpu'])
    
    cat_to_name = get_cat_to_name(category_names)
    model = load_checkpoint(device, checkpoint)
    ps, cls = predict(input_file, model, top_k)
#     for ps, cls in zip(ps, cls):
    for ps, cls in zip(*predict(input_file, model, top_k)):
            print(f"{cls}: {ps:.6f}")
    