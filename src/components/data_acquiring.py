import os, sys, zipfile, requests, shutil, random,tqdm
from PIL import Image
from dataclasses import dataclass

from io import BytesIO
from timeit import default_timer as timer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.exceptions import CustomException
from src.logger import logging
#from src.utils import sampler_utils

@dataclass
class DataAcquisition:
    parent_dir_path=os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))
    artifacts_path=os.path.join(parent_dir_path,'artifacts')
    train_data_path: str= os.path.join(artifacts_path,'train')
    test_data_path: str= os.path.join(artifacts_path,'test')
    zip_path: str= os.path.join(artifacts_path,'zip')

class Data_Ingestion:
    def __init__(self,url='https://archive.ics.uci.edu/static/public/908/realwaste.zip'):
        self.create_dirs = DataAcquisition()
        self.url=url
    def download_zip_file(self):
        try:
            if os.path.exists(self.create_dirs.artifacts_path) and os.path.exists(self.create_dirs.zip_path):
                logging.info("You may have already downloaded the zip file!")
            else:
                os.mkdir(self.create_dirs.artifacts_path)
                os.mkdir(self.create_dirs.zip_path)
                logging.info('Created artifacts and zip directories.')
                start_time = timer()
                logging.info('Downloading the file...')
                file = requests.get(self.url, params={'format': 'zip'})
                end_time = timer()
                time_taken= end_time - start_time
                logging.info(f'Downloading finished. Time taken: {time_taken:.2f} seconds')
                logging.info('Loading zip file from memory...')
                zip_file=zipfile.ZipFile(BytesIO(file.content))
                logging.info('Unzipping file...')

                zip_file.extractall(self.create_dirs.zip_path)
                
                logging.info('Unzipping completed.')
                zip_file.close()
                logging.info('File downloaded and extracted successfully.')
        except Exception as e:
            raise CustomException(e, sys)
        
    def train_test_split(self):
        try:
            if os.path.exists(self.create_dirs.artifacts_path) and  os.path.exists(self.create_dirs.train_data_path) and os.path.exists(self.create_dirs.test_data_path):
                logging.info('Train and test directories already exist.')
            else:
                start_time=timer()
                path=self.create_dirs.zip_path
                files=os.listdir(path)
                while len(files)<3:
                    for file in files:
                        if os.path.isdir(os.path.join(path, file)):
                            path=os.path.join(path,file)
                            files=os.listdir(path)
                            break
                os.mkdir(self.create_dirs.train_data_path)
                logging.info('Successfully created train directory.')
                for file in files:
                    shutil.move(os.path.join(path,file),os.path.join(self.create_dirs.train_data_path,file))

                classes=(os.listdir(self.create_dirs.train_data_path))
                class_paths=[os.path.join(self.create_dirs.train_data_path,cls) for cls in classes]
                split=0.8
                os.mkdir(self.create_dirs.test_data_path)
                logging.info('Successfully created test directory.')
                for cls,class_path in zip(classes,class_paths):
                    files=os.listdir(class_path)
                    num_of_files=len(files)
                
                    for i in range(0,num_of_files,2):  #for loop for flipping alternate images. Used cv,could also use PIL.
                        cur_file_path=os.path.join(class_path,files[i])
                        image=Image.open(cur_file_path)
                        image= image.transpose(Image.FLIP_LEFT_RIGHT)
                        image.save(cur_file_path)
                    random.shuffle(files)
                    os.mkdir(os.path.join(self.create_dirs.test_data_path,cls))

                    for file in files[int(num_of_files*split):]:
                        shutil.move(os.path.join(class_path,file),os.path.join(self.create_dirs.test_data_path,cls,file))
                end_time=timer()
                time_taken=end_time-start_time
                logging.info(f'Successfully created train and test folders. Time taken: {time_taken:.2f} seconds')
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=='__main__':
    data_acquisition=Data_Ingestion()
    data_acquisition.download_zip_file()
