# */image-classifier-part-1-workspace/home/aipnd-project/train.py
# 
# PROGRAMMER: Lai Chee Fong
# DATE CREATED: 29 June 2024                                
# REVISED DATE: 30 June 2024
# PURPOSE: This code train a deep learning model on a dataset of images. It allows users 
# to specify various parameters such as the directory containing the dataset, 
# the model architecture to use (e.g., VGG16, ResNet18, AlexNet), the learning rate, 
# the number of hidden units in the classifier, the number of training epochs, and 
# whether to use a GPU for training. The script loads the dataset, creates 
# data loaders, initializes the chosen pre-trained model, and replaces the classifier 
# with a new feed-forward network. It then trains the model on the training set, 
# evaluates it on the validation set after each epoch, and saves the best-performing 
# model weights to a checkpoint file. The script also supports features like 
# learning rate scheduling and early stopping to improve training performance.

import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm
from utils import load_checkpoint, save_checkpoint, data_transforms

def train_model(data_dir, save_dir=None, arch='vgg16', learning_rate=0.001, hidden_units=2048, epochs=10, gpu=False):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

    # Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms['train']),
        'valid': datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=data_transforms['valid']),
        'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=data_transforms['test'])
    }

    # Create the dataloaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': DataLoader(image_datasets['valid'], batch_size=64),
        'test': DataLoader(image_datasets['test'], batch_size=64)
    }

    # Load the pre-trained model
    model = getattr(models, arch)(pretrained=True)

    # Freeze the parameters of the pre-trained model
    for param in model.parameters():
        param.requires_grad = False

    # Define the classifier
    classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, len(image_datasets['train'].classes)),
        nn.LogSoftmax(dim=1)
    )

    # Replace the classifier of the pre-trained model
    model.classifier = classifier
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Implement learning rate scheduling
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        # Train loop
        for inputs, labels in tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(dataloaders['train'])

        model.eval()
        val_loss = 0
        accuracy = 0

        # Validation loop
        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model(inputs)
                batch_loss = criterion(logps, labels)
                val_loss += batch_loss.item()

                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        val_loss /= len(dataloaders['valid'])
        val_accuracy = accuracy / len(dataloaders['valid'])

        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {train_loss:.3f}.. "
              f"Validation loss: {val_loss:.3f}.. "
              f"Validation accuracy: {val_accuracy:.3f}")

        # Save the model with the best validation accuracy
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            best_model_wts = model.state_dict()

        # Update the learning rate
        scheduler.step(val_loss)

    # Load the best model weights
    model.load_state_dict(best_model_wts)

    if save_dir:
        save_checkpoint(model, optimizer, epochs, image_datasets['train'].class_to_idx, os.path.join(save_dir, 'checkpoint.pth'), arch, hidden_units)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a new network on a dataset')
    parser.add_argument('data_dir', type=str, help='The directory containing the dataset')
    parser.add_argument('--save_dir', type=str, default=None, help='The directory to save the checkpoint and class_to_idx mapping')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'resnet18', 'alexnet'], help='The architecture to use for the model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='The learning rate for the optimizer')
    parser.add_argument('--hidden_units', type=int, default=4096, help='The number of units in the hidden layers')
    parser.add_argument('--epochs', type=int, default=10, help='The number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Whether to use a GPU for training')

    args = parser.parse_args()

    train_model(args.data_dir, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)