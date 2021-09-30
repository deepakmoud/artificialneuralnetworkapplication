from flask.helpers import send_file
from jinja2 import Template
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from os import path
import re
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
# Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib
import numpy as np
from flask import Flask, request, jsonify, render_template

import pandas as pd

# coding=utf-8
import sys
import os
import glob
import re
import cv2
from  PIL import Image, ImageOps
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import io


matplotlib.use('Agg')

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

#------------------------------ Saving dataset---------------------------------
# this is the path to save dataset for preprocessing
pathfordataset = "static/data-preprocess/"
pathfordatasetNew = "data-preprocess/new/"  
 
app.config['DFPr'] = pathfordataset
app.config['DFPrNew'] = pathfordatasetNew
#------------------------------ Saving dataset for Linear regression-------------------------------------------
# this is the path to save dataset for single variable LR
pathforonevarLR = "static/Regression/onevarLR"
pathforonevarLRplot = "Regression/onevarLR/plot"
app.config['LR1VAR'] = pathforonevarLR
app.config['LR1VARplot'] = pathforonevarLRplot

#------------------------------ Saving image for K means-------------------------------------------
# this is the path to save figure of K menas
pathforelbowplot = "kmeans/plot"
#pathforonevarLRplot = "Regression/onevarLR/plot"
#app.config['LR1VAR'] = pathforonevarLR
app.config['elbowplot'] = pathforelbowplot
#print(app.config['elbowplot'])

# for index page
#------------------------------ Launcing undex page-------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

#------------------------------Data Preprocessing-------------------------------------------
# for data preprocessing
def model_predict(file_path, model):
    img = image.load_img(file_path, target_size=(128, 128))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')
    preds = model.predict(x)
    return preds

@app.route('/downloadNewDataset')
def download_file():
    path1 = "static/data-preprocess/new/trained_dataset.csv"
    return send_file(path1,as_attachment=True)

#------------------------------Download Model-------------------------------------------
@app.route('/downloadmodel')
def download_model():
    path1 = "static/data-preprocess/model/model.pkl"
    return send_file(path1,as_attachment=True)

#------------------------------About us-------------------------------------------
@app.route('/aboutusnew')
def aboutusnew():
    return render_template('aboutusnew.html')
#------------------------------Artificial Neural network-------------------------------------------


@app.route('/ann')
def ann():
    return render_template('/ann/ann.html')

#------------------------------Signature Verificationn-------------------------------------------

def model_predict(file_path, model):
    img = image.load_img(file_path, target_size=(128, 128))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')
    preds = model.predict(x)
    return preds

@app.route('/ann/signatureverification/signatureverification')
def signatureverification():
    return render_template('/ann/signatureverification/signatureverification.html')


@app.route('/ann/signatureverification/signatureverification',  methods=['GET', 'POST'])
def signatureverification1():
   
    if request.method == 'POST':
        my_dataset = request.files['my_dataset']
        my_model_name = request.form['name_of_model']
        
        dataset_path = os.path.join(pathfordataset, secure_filename(my_dataset.filename))
        my_dataset.save(dataset_path)
        print(my_dataset)
        get_dastaset = os.path.join(app.config['DFPr'],secure_filename(my_dataset.filename))
        print(get_dastaset)
        input=secure_filename(my_dataset.filename)
        extension= input.split(".")
        extension=extension[1]
        print(extension)
        model = load_model("static/data-preprocess/model/model_vgg19.h5")
        # Make prediction
        preds = model_predict(get_dastaset, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        if(preds> 0.5):
            result= 'Genuine'
        elif(preds< 0.5):
            result='Forged'
        plt.plot(get_dastaset)

        
        fig = plt.gcf()
        img_name1 = 'vgg19'
        fig.savefig('static/kmeans/plot/vgg19.png', dpi=1500)
        #elbow_plot = os.path.join(app.config['elbowplot'], '%s.png' % img_name1)
        vgg19_plot = os.path.join('kmeans\plot', '%s.png' % img_name1)
        plt.clf()

        return render_template('/ann/signatureverification/signatureverificationoutput.html', model_name=my_model_name,my_dataset=my_dataset, pred=result, visualize=input )

#-----------------------Digit Recognition---------------------------------------------
model_digit = load_model("static/data-preprocess/model/MNISTANN.h5")

def import_and_predict(image_data):
  
  image_resized = cv2.resize(image_data, (28, 28)) 
   
  prediction = model_digit.predict(image_resized.reshape(1,784))
  print('Prediction Score:\n',prediction[0])
  thresholded = (prediction>0.5)*1
  print('\nThresholded Score:\n',thresholded[0])
  print('\nPredicted Digit:',np.where(thresholded == 1)[1][0])
  digit = np.where(thresholded == 1)[1][0]
  #st.image(image_data, use_column_width=True)
  return digit



@app.route('/ann/digit/digit')
def digit():
    return render_template('/ann/digit/digit.html')


@app.route('/ann/digit/digit',  methods=['GET', 'POST'])
def digit1():
   
    if request.method == 'POST':
        input_image = request.files['input_image']
        print(input_image)
        my_model_name = request.form['name_of_model']
        
        dataset_path = os.path.join(pathfordataset, secure_filename(input_image.filename))
        input_image.save(dataset_path)
        
        get_dastaset = os.path.join(app.config['DFPr'],secure_filename( input_image .filename))
        input=secure_filename(input_image.filename)
        
        image=Image.open(input_image)
        image=np.array(image)
        
        #image=np.array(input_image)
        preds = import_and_predict(image)

        

        return render_template('/ann/digit/digitoutput.html', model_name=my_model_name,my_dataset=input_image, pred=preds, visualize=input )

#----------------------Image Classification cat/ Dog------------------------------
model_cat = load_model("static/data-preprocess/model/FDPCNN1.h5")

def import_and_predict_cat(image_data):
  #x = cv2.resize(image_data, (48, 48)) 
  #img = image.load_img(image_data, target_size=(48, 48))
  #x = image.img_to_array(img)
  size=(64, 64)
  image=ImageOps.fit(image_data, size, Image.ANTIALIAS)
  img=np.asarray(image)
  img_reshape=np.expand_dims(img, axis=1)
  img_reshape=img[np.newaxis,...]
  result = model_cat .predict(img_reshape)
  print(result)
  #training_set.class_indices
  if result[0][0] == 1:
    prediction = "Dog" 
    
  else:
    prediction = 'Cat'
    #x = np.expand_dims(x, axis=1)
  
  
  return prediction


@app.route('/ann/cat/cat')
def cat():
    return render_template('/ann/cat/cat.html')


@app.route('/ann/cat/cat',  methods=['GET', 'POST'])
def cat1():
   
    if request.method == 'POST':
        input_image = request.files['input_image']
        print(input_image)
        my_model_name = request.form['name_of_model']
        
        dataset_path = os.path.join(pathfordataset, secure_filename(input_image.filename))
        input_image.save(dataset_path)
        
        get_dastaset = os.path.join(app.config['DFPr'],secure_filename( input_image .filename))
        input=secure_filename(input_image.filename)
        
        image=Image.open(input_image)
        #image=np.array(image)
        
        #image=np.array(input_image)
        preds = import_and_predict_cat(image)

        

        return render_template('/ann/cat/catoutput.html', model_name=my_model_name,my_dataset=input_image, pred=preds, visualize=input )


#-------------Signature recognition-----------------------------------------------
model_signaturerecognition = load_model("static/data-preprocess/model/signatureRecognition_VGG16folder_model.h5")
SIGNATURE_CLASSES = ['001', '002', '003','004','006','009','012','013','014','015','016','017','018','019','020','021','022','023','024','025','026','027','028','029','030','031','032','033','034','035','036','037','038','039','040','041','042','043','044','045','046','047','048','049','050','051','052','053','054','055','056','057','058','059','060','061','062','063','064','065','066','067','068','069']
def import_and_predict_recognition(image_data, model):
  #img = image.load_img(image_data, target_size=(224, 224))
  #image = image.img_to_array(img)
  #img_reshap= np.expand_dims(image, axis=0)
  #img_reshap = preprocess_input(img_reshap)
  size=(224, 224)
  image=ImageOps.fit(image_data, size, Image.ANTIALIAS)
  img=np.asarray(image)
  img_reshape=np.expand_dims(img, axis=1)
  img_reshape=img[np.newaxis,...]
  block4_pool_features = model.predict(img_reshape)
  label_index=block4_pool_features.argmax()
  print(block4_pool_features)
  result=SIGNATURE_CLASSES[label_index]
  return result


@app.route('/ann/signaturerecognition/signaturerecognition')
def signaturerecognition():
    return render_template('/ann/signaturerecognition/signaturerecognition.html')


@app.route('/ann/signaturerecognition/signaturerecognition',  methods=['GET', 'POST'])
def signaturerecognition1():
   
    if request.method == 'POST':
        input_image = request.files['input_image']
        
        my_model_name = request.form['name_of_model']
        
        dataset_path = os.path.join(pathfordataset, secure_filename(input_image.filename))
        input_image.save(dataset_path)
        
        get_dastaset = os.path.join(app.config['DFPr'],secure_filename( input_image .filename))
        input=secure_filename(input_image.filename)
        
        image=Image.open(input_image)
        #image=np.array(image)
        
        #image=np.array(input_image)
        preds = import_and_predict_recognition(image, model_signaturerecognition)

        

        return render_template('/ann/signaturerecognition/signaturerecognitionoutput.html', model_name=my_model_name,my_dataset=input_image, pred=preds, visualize=input )

#--------------------Animal Breed identification---------------------------------

model_breed = load_model("static/data-preprocess/model/resnet_model.h5")
def model_predict_breed(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds

@app.route('/ann/breed/breed')
def breed():
    return render_template('/ann/breed/breed.html')


@app.route('/ann/breed/breed',  methods=['GET', 'POST'])
def breed1():
   
    if request.method == 'POST':
        my_dataset = request.files['input_image']
        my_model_name = request.form['name_of_model']
        
        dataset_path = os.path.join(pathfordataset, secure_filename(my_dataset.filename))
        my_dataset.save(dataset_path)
        
        get_dastaset = os.path.join(app.config['DFPr'],secure_filename(my_dataset.filename))
        
        input=secure_filename(my_dataset.filename)
        
        
        # Make prediction
        preds = model_predict_breed(get_dastaset, model_breed)

        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string

        return render_template('/ann/breed/breedoutput.html', model_name=my_model_name,my_dataset=my_dataset, pred=result, visualize=input )
#-----------------------Character Recognition---------------------------------------------
model_char = load_model("static/data-preprocess/model/alphabet.h5")

def predict_char(image_data):
  
  test_image = image.load_img(image_data, target_size = (32,32))
  test_image = image.img_to_array(test_image)
  test_image = np.expand_dims(test_image, axis = 0)
  result = model_char.predict(test_image)
  result = get_result(result)
  return result
  
def get_result(result):
    if result[0][0] == 1:
        return('a')
    elif result[0][1] == 1:
        return ('b')
    elif result[0][2] == 1:
        return ('c')
    elif result[0][3] == 1:
        return ('d')
    elif result[0][4] == 1:
        return ('e')
    elif result[0][5] == 1:
        return ('f')
    elif result[0][6] == 1:
        return ('g')
    elif result[0][7] == 1:
        return ('h')
    elif result[0][8] == 1:
        return ('i')
    elif result[0][9] == 1:
        return ('j')
    elif result[0][10] == 1:
        return ('k')
    elif result[0][11] == 1:
        return ('l')
    elif result[0][12] == 1:
        return ('m')
    elif result[0][13] == 1:
        return ('n')
    elif result[0][14] == 1:
        return ('o')
    elif result[0][15] == 1:
        return ('p')
    elif result[0][16] == 1:
        return ('q')
    elif result[0][17] == 1:
        return ('r')
    elif result[0][18] == 1:
        return ('s')
    elif result[0][19] == 1:
        return ('t')
    elif result[0][20] == 1:
        return ('u')
    elif result[0][21] == 1:
        return ('v')
    elif result[0][22] == 1:
        return ('w')
    elif result[0][23] == 1:
        return ('x')
    elif result[0][24] == 1:
        return ('y')
    elif result[0][25] == 1:
        return ('z')

@app.route('/ann/character/character')
def character():
    return render_template('/ann/character/character.html')


@app.route('/ann/character/character',  methods=['GET', 'POST'])
def character1():
   
    if request.method == 'POST':
        my_dataset = request.files['input_image']
        my_model_name = request.form['name_of_model']
        
        dataset_path = os.path.join(pathfordataset, secure_filename(my_dataset.filename))
        my_dataset.save(dataset_path)
        
        get_dastaset = os.path.join(app.config['DFPr'],secure_filename(my_dataset.filename))
        
        input=secure_filename(my_dataset.filename)
        
        
        # Make prediction
        preds = predict_char(get_dastaset)

        

        return render_template('/ann/character/characteroutput.html', model_name=my_model_name,my_dataset=my_dataset, pred=preds, visualize=input )
    

#-------------------Flask Application--------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
    
# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response
app.config["CACHE_TYPE"] = "null"




