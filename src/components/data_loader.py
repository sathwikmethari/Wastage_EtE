import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision as tv
from torchvision import transforms, models
from dataclasses import dataclass, field

from src.utils import *
from src.logger import logging
from src.exceptions import CustomException

from src.components.data_acquiring import DataAcquisition, Data_Ingestion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")


class Model_and_Data_loader:
    def __init__(self):
        pass
    def load_model(self,classes):
        logging.info('Downloading model')
        weights = tv.models.ResNet50_Weights.DEFAULT # .DEFAULT = best available weights
        logging.info('Downloading model weights...')
        preprocessing = weights.transforms()

        model = tv.models.resnet50(weights=weights).to(device)
        logging.info('Using ResNet50 model with downloaded weights.')

        for param in model.parameters():
            param.requires_grad = True

    # Get the length of class_names (one output unit for each class)
        output_shape = len(classes)

        # Recreate the classifier layer and seed it to the target device
        model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), 
        torch.nn.Linear(in_features=1280, 
                        out_features=output_shape, # same number of output units as our number of classes
                        bias=True)).to(device)

        logging.info('Model downloaded and saved, optimizer and loss function set up.')
        return model, preprocessing

    def load_data(classes,train_folder_path, test_folder_path, preprocessing,batch_size=32):
        logging.info('Loading data...')
        train_dataset = ImageDataset(train_folder_path,classes,preprocessing)
        test_dataset = ImageDataset(test_folder_path, classes,preprocessing)
        train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,pin_memory=True)
        test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,pin_memory=True)
        logging.info('Data loaded.')
        return train_dataloader, test_dataloader

if __name__=='__main__':
    data_acquisition=Data_Ingestion()
    data_acquisition.download_zip_file()
    classes=data_acquisition.train_test_split()
    model_and_data_loader=Model_and_Data_loader()
    model, preprocessing=model_and_data_loader.load_model(classes)
    train_dataloader, test_dataloader=model_and_data_loader.load_data(classes,data_acquisition)


