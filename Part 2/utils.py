import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json
from pathlib import Path

def load_checkpoint(checkpoint_path, device, class_to_idx):
    """
    Load a checkpoint and rebuild the model.
    Args:
        checkpoint_path (str): The path to the checkpoint file.
        device (torch.device): The device (CPU or GPU) to use for loading the model.
        class_to_idx (dict): The class-to-index mapping.
    Returns:
        tuple: The loaded model, the model architecture, and class-to-index mapping.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get the architecture and hidden units from the checkpoint
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']

    # Create a new model with the same architecture as the checkpoint
    model = getattr(models, arch)(pretrained=True)

    # Freeze the parameters of the pre-trained model
    for param in model.parameters():
        param.requires_grad = False

    # Build the classifier with the provided class_to_idx mapping
    classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, len(class_to_idx)),
        nn.LogSoftmax(dim=1)
    )

    # Replace the classifier of the pre-trained model
    model.classifier = classifier
    model.to(device)

    # Load the state dictionary from the checkpoint
    model.load_state_dict(checkpoint['state_dict'])

    return model, arch, class_to_idx

def save_checkpoint(model, optimizer, epochs, class_to_idx, checkpoint_path, arch, hidden_units):
    """
    Save the trained model to a checkpoint.
    Args:
        model (torch.nn.Module): The trained model.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        epochs (int): The number of training epochs.
        class_to_idx (dict): The class-to-index mapping.
        checkpoint_path (str): The path to save the checkpoint.
        arch (str): The architecture of the model.
        hidden_units (int): The number of hidden units in the classifier.
    """
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs
    }
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved to {checkpoint_path}')

    # Save the class_to_idx mapping as a separate file
    class_to_idx_path = Path(checkpoint_path).with_name('class_to_idx.json')
    with open(class_to_idx_path, 'w') as f:
        json.dump(class_to_idx, f)
    print(f'Class-to-index mapping saved to {class_to_idx_path}')

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns a Numpy array
    '''
    # Open the image
    img = Image.open(image_path)
    # Get the original dimensions
    original_width, original_height = img.size
    # Find the shorter side and create settings to crop it to 256
    if original_width < original_height:
        img.thumbnail((256, 256 * original_height // original_width))
    else:
        img.thumbnail((256 * original_width // original_height, 256))
    # Get the center 224x224 portion
    left = (256 - 224) / 2
    top = (256 - 224) / 2
    right = left + 224
    bottom = top + 224
    # Crop the image to 224x224 centered portion
    img = img.crop((left, top, right, bottom))
    # Convert to numpy, normalize, and transpose
    np_image = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    return np_image

def save_class_to_name_mapping(class_to_idx, category_names_path):
    """
    Save the class-to-name mapping to a JSON file.
    Args:
        class_to_idx (dict): The class-to-index mapping.
        category_names_path (str): The path to save the JSON file.
    """
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    with open(category_names_path, 'w') as f:
        json.dump({idx_to_class[i]: idx_to_class[i] for i in range(len(idx_to_class))}, f)