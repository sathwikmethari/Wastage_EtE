import sys
import torch
from torch.utils.data import DataLoader

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) Uncomment while running this file.

from src.utils import *
from src.logger import logging
from src.exceptions import CustomException

from src.components.data_acquiring import DataAcquisition, Data_Ingestion


class Model_and_Data_loader:
    def __init__(self):
        self.dirs_path=DataAcquisition()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        self.preprocessing=preprocessing_for_loaders()
        
    def load_data(self, classes, batch_size=32):
        try:            
            logging.info('Loading data...')
            train_dataset = ImageDataset(self.dirs_path.train_data_path,classes,self.preprocessing)
            test_dataset = ImageDataset(self.dirs_path.test_data_path, classes,self.preprocessing)
            train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,pin_memory=True,num_workers=4)
            test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,pin_memory=True,num_workers=4)
            logging.info('Train, Test dataloaders loaded.')
            return train_dataloader, test_dataloader
        except Exception as e:
            logging.error(CustomException(e,sys))
            raise CustomException(e,sys)

if __name__=='__main__':
    data_acquisition=Data_Ingestion()
    data_acquisition.download_zip_file()
    classes=data_acquisition.train_test_split()
    model_and_data_loader=Model_and_Data_loader()
    model, preprocessing=model_and_data_loader.load_model(classes=classes)
    train_dataloader, test_dataloader=model_and_data_loader.load_data(classes=classes,preprocessing=preprocessing)


