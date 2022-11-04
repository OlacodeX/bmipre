import pandas as pd
import os
import json
#from keras.preprocessing import image
#from keras.utils.layer_utils import get_source_inputs
#from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from scripts.models import FacePrediction
from mtcnn.mtcnn import MTCNN
import cv2
from flask import Flask, request, app, jsonify, url_for, render_template


## Create a new instance of the flask app
app = Flask(__name__)
def crop_img(im,x,y,w,h):
    return im[y:(y+h),x:(x+w),:]

## Load the model
model_type = 'vgg16'
model_dir = './saved_model/model_vgg16_base.h5'
freeze_backbone = True # True => transfer learning; False => train from scratch
model = FacePrediction(img_dir = './data/face_aligned/', model_type = model_type)
model.define_model(freeze_backbone = freeze_backbone)
model.load_weights(model_dir)

## Define the landing page route
@app.route('/')
## Define the landing page resource
def home():
    return render_template('home.html')

## Define route for prediction API for communicating with your app
@app.route('/predict_api', methods=['POST'])
## Define the API resource
def predict_api():
    data=request.json['data']
    print(data)
    #convert image to an array
    detector = MTCNN()
    img = cv2.cvtColor(cv2.imread(str(data['img_dir'])), cv2.COLOR_BGR2RGB)
    #detect face from image
    box = detector.detect_faces(img)[0]
    im = plt.imread(str(data['img_dir']))
    #crop image to save only the face area
    cropped = crop_img(im, *box['box'])
    #save the cropped area in a new folder
    plt.imsave('data/processed/face.jpg', cropped)
    output = model.predict('data/processed/face.jpg', show_img=True)
    ## this output is returned as a list of nested arrays. 
    ## Now I just pick the first element in the list which is an array containing the BMI value(in another array) and its data type.
    # I then pick the first element in the array of the BMI
    return jsonify(float(output[0][0][0]))

@app.route('/predict', methods=['POST'])
def predict():
    # initialize the camera
    # If you have multiple camera connected with
    # current device, assign a value in cam_port
    # variable according to that
    cam_port = 0
    cam = cv2.VideoCapture(cam_port)

    # reading the input using the camera
    result, image = cam.read()

    # If image will detected without any error,
    # show result
    if result:
        cv2.namedWindow("cam-test")
        cv2.imshow("cam-test",image)
        cv2.waitKey(1)
        cv2.destroyWindow("cam-test")
        cv2.imwrite("data/face/passport.jpg",image)
        cam.release() 
        # showing result, it take frame name and image
        # output
        #cv.imshow("passport", image)
        
        #convert image to an array
        detector = MTCNN()
        img = cv2.cvtColor(cv2.imread("data/face/passport.jpg"), cv2.COLOR_BGR2RGB)
        #detect face from image
        box = detector.detect_faces(img)[0]
        im = plt.imread(str("data/face/passport.jpg"))
        #crop image to save only the face area
        cropped = crop_img(im, *box['box'])
        #save the cropped area in a new folder
        plt.imsave('data/processed/face.jpg', cropped)
        # saving image in local storage
        #cv.imwrite("passport.jpg", image)

        # If keyboard interrupt occurs, destroy image
        # window
        #cv.waitKey(0)
        #destroy image window after capture
        #cv.destroyWindow("passport")

        # If captured image is corrupted, move to else part
    else:
        print("No image detected. Please! try again")

    output = model.predict('data/processed/face.jpg', show_img=True)
    ## this output is returned as a list of nested arrays. 
    ## Now I just pick the first element in the list which is an array containing the BMI value(in another array) and its data type.
    # I then pick the first element in the array of the BMI
    
    return render_template("result.html", prediction_text="Your predicted BMI is {}".format(output[0][0][0]))


## Run the app now
if __name__=="__main__":
    app.run(debug=True)