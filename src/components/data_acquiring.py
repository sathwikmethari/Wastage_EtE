import os, sys, zipfile
from dataclasses import dataclass

from src.exceptions import CustomException
from src.logger import logging
from src.utils import sampler_utils

@dataclass
class DataAcquisition:
    train_data_path: str= os.path.join('artifacts','train')
    test_data_path: str= os.path.join('artifacts','test')
    zip_path: str= os.path.join('artifacts','zip')

class Data_Ingestion:
    def __init__(self):
        self.create_paths = DataAcquisition()
    def Initiate_data_ingestion(self):
        logging.info("Data Ingestion method starts")
        try:
            pass

        except Exception as e:
            raise CustomException(e, sys) from e
        