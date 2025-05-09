import os, sys, torch
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.components.data_acquiring import DataAcquisition
from src.components.model_trainer_tester import ModelTrainerConfig
from src.utils import model_loader, preprocessing_for_loaders



class Prediction:
    def __init__(self):
        self.get_path=DataAcquisition()
        self.get_m_path=ModelTrainerConfig()
        self.device=device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes=os.listdir(self.get_path.train_data_path)
        self.model=model_loader(self.classes,self.device)
        self.model.load_state_dict(torch.load(self.get_m_path.model_save_path))
        self.preprocessing=preprocessing_for_loaders()

    def predict(self,image):        
        image = self.preprocessing(image)
        image=image.to(self.device)
        image = image.unsqueeze(0)  # Add batch dimension

        self.model.eval()
        #print(self.classes)
        print("Model in eval mode.")
        with torch.inference_mode():
            y_pred_logits=self.model(image)
            # Check model output shape
            #print("Model output shape:", y_pred_logits.shape)  # Should be [1, 10] if 10 classes
            y_pred_probs = torch.softmax(y_pred_logits , dim=1)
            y_preds = torch.argmax(y_pred_probs, dim=1)
            index=y_preds.to('cpu').numpy()[0]
            output=self.classes[index]

        return output
    
if __name__=='__main__':
    predictor=Prediction()
    image='/home/satwik/PYTHON_ENV/Wastage_EtoE_class/artifacts/test/Plastic/Plastic_15.jpg'
    image=Image.open(image)
    out=predictor.predict(image=image)
    print(out)