import numpy as np
#import keyboard
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os


#from keras.utils import *
model=load_model('model2.h5py')

#for image taking 
import cv2

def predict_class(ip_file):
    img=image.load_img(ip_file,target_size=(256,256))
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    img_data=preprocess_input(x)
    return int(model.predict(img_data)[0][1]>0.5)
#%% Predict from the model
classes_names = ['BEANS', 'CAKE', 'CANDY', 'CEREAL', 'CHIPS', 'CHOCOLATE', 'COFFEE',
                 'CORN', 'FISH', 'FLOUR', 'HONEY', 'JAM', 'JUICE', 'MILK', 'NUTS', 'OIL',
                 'PASTA', 'RICE', 'SODA', 'SPICES', 'SUGAR', 'TEA', 'TOMATO_SAUCE', 'VINEGAR', 'WATER']
image=cv2.imread(r'E:\PHD Project\Datasets\freiburg_groceries_dataset\images\FLOUR\FLOUR0001.png')
imgh,imgw = 256,256
image_resize=cv2.resize(image,(imgh,imgw))

image1=np.expand_dims(image_resize,axis=0)
pred=model.predict(image1)
output_class=classes_names[np.argmax(pred)]
print(output_class)
#%%
model1=load_model('my_model_LSTM.h5')
alphabet = ['BEANS', 'CAKE', 'CANDY', 'CEREAL', 'CHIPS', 'CHOCOLATE', 'COFFEE',
                 'CORN', 'FISH', 'FLOUR', 'HONEY', 'JAM', 'JUICE', 'MILK', 'NUTS', 'OIL',
                 'PASTA', 'RICE', 'SODA', 'SPICES', 'SUGAR', 'TEA', 'TOMATO_SAUCE', 'VINEGAR', 'WATER']
img_array = img_to_array(image)
img_array = np.expand_dims(img_array, axis=0)
prediction = model.predict(img_array)
predicted_label = alphabet[np.argmax(prediction)]
print('Predicted label:', predicted_label)
