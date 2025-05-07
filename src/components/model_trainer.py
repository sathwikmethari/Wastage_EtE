import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision as tv
from torchvision import transforms, models
from dataclasses import dataclass,field

from src.utils import *
from src.logger import logging
from src.exceptions import CustomException

from src.components.data_acquiring import DataAcquisition, Data_Ingestion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

@dataclass
class ModelTrainerConfig:
    model_save_path: str = field(default_factory=lambda: os.path.join('artifacts', 'pre_trained_model.pth'))

class Model_trainer:
    def __init__(self):
        self.path_config=ModelTrainerConfig()
    def train_model(model):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        logging.info(f"Using Loss Function: {loss_fn}, Optimizer: {optimizer}")

        # save_path=self.path_config.model_save_path
        # torch.save(model.state_dict(), save_path)
