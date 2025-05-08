import os,sys
from flask import Flask,request,render_template
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.predict_pipeline import Prediction


app=Flask(__name__)
@app.route('/',methods=['GET','POST'])
def predict_final():
    image = request.files.get('image')
    if image:
        predictor=Prediction()
        img = Image.open(image.stream).convert('RGB')
        output=predictor.predict(img)
        #pass          
    return render_template('home.html',output=output)


if __name__=='__main__':
    app.run(debug=True)