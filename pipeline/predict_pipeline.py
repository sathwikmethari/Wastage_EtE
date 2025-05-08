import os, sys, torch
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components.data_acquiring import DataAcquisition
from src.components.model_trainer_tester import ModelTrainerConfig
from src.utils import model_loader, preprocessing_for_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Prediction:
    def __init__(self):
        self.get_path=DataAcquisition()
        self.get_m_path=ModelTrainerConfig()
    def predict(self,image):
        classes=os.listdir(self.get_path.train_data_path)
        model=model_loader(classes,device)
        model.load_state_dict(torch.load(self.get_m_path.model_save_path))
        preprocessing=preprocessing_for_loaders()

        image = preprocessing(image)
        image=image.to(device)
        image = image.unsqueeze(0)  # Add batch dimension

        model.eval()
        print(classes)
        print("Model in eval mode.")
        with torch.inference_mode():
            y_pred_logits=model(image)
            y_pred_probs = torch.softmax(y_pred_logits , dim=1)
            y_preds = torch.argmax(y_pred_probs, dim=1)
            index=y_preds.to('cpu').numpy()[0]
            output=classes[index]

        return output