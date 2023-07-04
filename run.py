# -*- coding: utf-8 -*-
from flask import Flask,render_template
from flask import *
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model


app=Flask(__name__)


UPLOAD_FOLDER='static/upload/'
DETECT_FOLDER='static/detect/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/',methods=['POST','GET'])
def upload_file():
    if request.method=='POST':
        f=request.files['file']
        full_filename=os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
        f.save(full_filename)
        return render_template("predict.html",user_image=full_filename)
    else:
       return render_template('index.html') 
   
    
   
@app.route('/Detect/<path:user_image>',methods=['GET'])
def detect(user_image):
    path=user_image
    IMG = cv2.imread(path)  
    IMG=cv2.resize(IMG,(224,224))
    image=IMG /255 
    img=image.astype(np.float32)
    img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=img.reshape(1,img.shape[0],img.shape[1],1)
    model=load_model('model.h5')
    predicted=model.predict(img,verbose=0)
    predicted=predicted.reshape(predicted.shape[1],predicted.shape[2])
    masked=np.zeros(IMG.shape)
    predicted[predicted<0.5]=0
    predicted[predicted>0.5]=1
    masked[:,:,-1]=predicted
    output = cv2.add(image,masked)
    output=output*255
    detected_path=user_image.split("/")[-1]
    detected_image_path = os.path.join(DETECT_FOLDER,f"{detected_path}")
    path=os.path.join("http://127.0.0.1:5000/static/upload/",f"{detected_path}")
    cv2.imwrite(detected_image_path, output)
    local_image_path = os.path.join("http://127.0.0.1:5000/static/detect/", f"{detected_path}")
    return render_template('result.html', image=path, detected=local_image_path)
    
      

if __name__=="__main__":
    app.run(debug=False)
    


