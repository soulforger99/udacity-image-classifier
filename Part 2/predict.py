# */image-classifier-part-1-workspace/home/aipnd-project/predict.py
# 
# PROGRAMMER: Lai Chee Fong
# DATE CREATED: 29 June 2024                                
# REVISED DATE: 30 June 2024
# PURPOSE: This code load a pre-trained deep learning model and use it to predict the class 
# (or classes) of an input image. It allows users to specify the path to the image file, 
# the path to the checkpoint file containing the trained model weights, and 
# various options such as the number of top predictions to display, the path to a 
# JSON file mapping class indices to category names, and whether to use a GPU for 
# inference. The script preprocesses the input image, loads the trained model and 
# class-to-index mapping from the checkpoint file, and then uses the model to 
# generate class probabilities for the input image. Finally, it prints out the 
# top K predicted classes and their associated probabilities, optionally using 
# the provided category name mapping.

import argparse
import torch
from torchvision import models
import json
from PIL import Image
import numpy as np
import os
from pathlib import Path
from utils import process_image, load_checkpoint

def predict(image_path, model, class_to_idx, top_k=5, device='cpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Preprocess the image
    img = process_image(image_path)
    img = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(0)
    img = img.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Get the logits from the model
    with torch.no_grad():
        logits = model(img)
    
    # Get the probabilities from the logits
    ps = torch.exp(logits)
    
    # Get the top k probabilities and class indices
    top_probs, top_indices = ps.topk(top_k)
    
    # Convert the class indices to class labels
    top_classes = [list(class_to_idx.keys())[list(class_to_idx.values()).index(idx)] for idx in top_indices[0].tolist()]
    
    # Convert the probabilities to scalars
    top_probs = top_probs[0].tolist()
    
    return top_probs, top_classes

def predict_from_image(image_path, checkpoint_path, top_k=5, category_names_path='cat_to_name.json', use_gpu=False):
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load the class_to_idx mapping from the checkpoint directory
    checkpoint_dir = Path(checkpoint_path).parent
    class_to_idx_path = checkpoint_dir / 'class_to_idx.json'
    if not class_to_idx_path.exists():
        raise FileNotFoundError(f"Class-to-index mapping file not found: {class_to_idx_path}")
    with open(class_to_idx_path, 'r') as f:
        class_to_idx = json.load(f)

    model, arch, _ = load_checkpoint(checkpoint_path, device, class_to_idx)

    # Load the class-to-name mapping
    with open(category_names_path, 'r') as f:
        cat_to_name = json.load(f)

    probs, classes = predict(image_path, model, class_to_idx, top_k, device)

    print("Top {} predictions:".format(top_k))
    for prob, class_idx in zip(probs, classes):
        class_name = cat_to_name[str(class_idx)]
        print("- {}: {:.3f}".format(class_name, prob))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict flower name from an image')
    parser.add_argument('input', type=str, help='Path to the image file')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to the JSON file that maps the class values to category names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()

    predict_from_image(args.input, args.checkpoint, args.top_k, args.category_names, args.gpu)