import os, sys, tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer as timer
from dataclasses import dataclass,field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils import *
from src.logger import logging
from src.exceptions import CustomException

from src.components.data_acquiring import Data_Ingestion
from src.components.data_loader import Model_and_Data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

@dataclass
class ModelTrainerConfig:
    model_save_path: str = field(default_factory=lambda: os.path.join('artifacts', 'trained_model.pth'))

class Model_trainer_and_tester:
    def __init__(self):
        self.path_config=ModelTrainerConfig()

    def train_model(self,train_dataloader,epochs=10):
        try:
            model=model_loader(classes=classes,device=device)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            logging.info(f"Using Loss Function: {loss_fn}, Optimizer: {optimizer}")
            start_time = timer()
        # Setup training and save the results
            logging.info('Training the model...')
            for epoch in tqdm(range(epochs)):
                # Train the model for one epoch
                train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
                print(f"Epoch: {epoch+1}--->train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f}")
                logging.info(f"Epoch: {epoch+1}--->train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f}")
            # End the timer and print out how long it took
            end_time = timer()
            total_time = end_time - start_time
            logging.info("Training completed!")
            logging.info(f"Total training time: {total_time:.3f} seconds")  
            
            save_path=self.path_config.model_save_path
            torch.save(model.state_dict(), save_path)

        except Exception as e:
            logging.error(CustomException(e,sys))
            raise CustomException(e,sys)
        
    def test_model(self,classes,test_dataloader):
        try:
            model=model_loader(classes,device)
            save_path=self.path_config.model_save_path
            model.load_state_dict(torch.load(save_path))
            logging.info("Model loaded for testing.")
            model.eval()
            loss_fn=nn.CrossEntropyLoss()
            test_loss,test_acc,batch_count=0,0,0
            with torch.inference_mode():
                for batch, (X, y) in enumerate(test_dataloader):
                # Send data to target device
                    X, y = X.to(device), y.to(device)

                    batch_count+=1
                    # 1. Forward pass
                    test_pred_logits = model(X)

                    # 2. Calculate and accumulate loss
                    loss = loss_fn(test_pred_logits, y)
                    test_loss += loss.item()

                    # Calculate and accumulate accuracy
                    test_pred_labels = test_pred_logits.argmax(dim=1)
                    test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

                    
            logging.info("Testing completed!")
            test_loss/=batch_count
            test_acc/=batch_count
            logging.info(f"Test_loss: {test_loss:.4f} | train_acc: {test_acc:.4f}")

        except Exception as e:
            logging.error(CustomException(e,sys))
            raise CustomException(e,sys)
        
if __name__=='__main__':
    data_acquisition=Data_Ingestion()
    data_acquisition.download_zip_file()
    classes=data_acquisition.train_test_split()
    data_loader=Model_and_Data_loader()
    train_dataloader, test_dataloader=data_loader.load_data(classes=classes)
    training_and_tester=Model_trainer_and_tester()
    training_and_tester.train_model(train_dataloader=train_dataloader)
    training_and_tester.test_model(classes=classes,test_dataloader=test_dataloader)
    

