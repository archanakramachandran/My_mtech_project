# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 22:24:02 2023

@author: hp
"""

#from scipy import misc,ndimage
#from scipy.ndimage.interpolation import zoom
#from scipy.misc import imread


#from keras import backend as K
from keras.utils.np_utils import to_categorical
#from keras import layers
#from keras.preprocessing.image import save_img
#from keras.utils.vis_utils import model_to_dot
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.applications.vgg16 import VGG16
from keras.models import Sequential,Model
from keras.layers import Dense,Flatten,Dropout,Concatenate,GlobalAveragePooling2D,Lambda,ZeroPadding2D
from keras.layers import SeparableConv2D,BatchNormalization,MaxPooling2D,Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,CSVLogger,ReduceLROnPlateau,LearningRateScheduler
from tensorflow.keras.preprocessing.image import save_img
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from imutils import paths
import os
import cv2   
import numpy
#%%


train_path=r'I:\End_sem_major_project\data\train'

num_classes = 25
input_shape = (256, 256,3)
batch_size=32
pathtoimages=list(paths.list_images(train_path))

data=[] #x_train
lables=[] #y_train
lable=[] 
for images in pathtoimages: #loop in image folder
  lable.clear()
  lable.append(images.split(os.path.sep)[-2])
  #if label not in grocerieslabels:
  #  continue 
  image=cv2.imread(images)
  image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
  data.append(image)
  lables.append(lable)
print(lables)

#%%
val_path= r'I:\End_sem_major_project\data\val' 
test_pathtoimages=list(paths.list_images(val_path))

x_test=[]
y_test=[]
label=[]
for images in test_pathtoimages:
  label.clear()
  label.append(images.split(os.path.sep)[-2])
  image=cv2.imread(images)
  image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
  x_test.append(image)
  y_test.append(label)

x_train=numpy.array(data)
y_train=numpy.array(lables)
x_test=numpy.array(x_test)
y_test=numpy.array(y_test)

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.02),
        layers.RandomWidth(0.2),
        layers.RandomHeight(0.2),
    ]
)
data_augmentation.layers[0].adapt(x_train)

