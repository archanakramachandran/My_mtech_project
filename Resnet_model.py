# -*- coding: utf-8 -*-
"""
Created on Mon May 22 01:09:02 2023

@author: hp
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 15 22:08:01 2023

@author: hp
"""

#%% Import the required packages
import tensorflow as tf
from tensorflow import keras 
from keras import layers
import matplotlib.pyplot as plt
from matplotlib import style
from keras.preprocessing.image import ImageDataGenerator
import random
import numpy as np
import os
from tensorflow.python.keras.layers import Dense,Flatten,Dropout,Conv2D,MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
import cv2
import pathlib
import PIL
from keras.models import load_model
from keras.models import Model

#%%Prepare the dataset
data_dirt = r'I:\End_sem_major_project\output\train'
data_dirt = pathlib.Path(data_dirt)

data_dirv = r'I:\End_sem_major_project\output\val' 
data_dirv = pathlib.Path(data_dirv)

#roses= list(data_dir.glob("BEANS/*"))
imgh,imgw = 256,256
batch_size=32

#%%Split the dataset
train_datagen =ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(data_dirt, 
                                                             seed=123,
                                                             class_mode= 'categorical', 
                                                             target_size= (imgh,imgw),
                                                             batch_size=batch_size)

val_generator=val_datagen.flow_from_directory(data_dirv,
                                                           seed=123,
                                                           class_mode= "categorical", 
                                                           target_size= (imgh,imgw),
                                                           batch_size=batch_size)

#%%
classes_names=train_generator.classes
print(classes_names)

#%%Choose the pretrained model
model=Sequential()

pretrainedmodel=tf.keras.applications.ResNet50(include_top=False,
                                               input_shape=(imgh,imgw,3),
                                               pooling="max",
                                               classes=25,
                                               weights="imagenet")
for layer in pretrainedmodel.layers:
    layer.trainable=False

model.add(pretrainedmodel)
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(25, activation="softmax"))
model.compile(optimizer=Adam(learning_rate=0.0001),loss="categorical_crossentropy",metrics=["accuracy"])
epochs = 100
history=model.fit(train_generator,
                            validation_data=val_generator,
                            steps_per_epoch=97,
                            epochs=epochs,
                            batch_size=batch_size)


#%%model summary
print(model.summary())


#%%graph plotting 

fig1=plt.figure()
plt.subplot(1,1,1)
plt.title("Graphical Representation for\n Accuracies vs Epochs\n ")
plt.plot(history.history["accuracy"],label='Training Accuracy')
plt.plot(history.history["val_accuracy"],label='Validation Acuuracy')
plt.xlabel("EPOCH").set_color('r')                                          
plt.ylabel("ACCURACY").set_color('r')
plt.grid(True)
plt.legend()

fig2=plt.figure()
plt.subplot(1,1,1)
plt.title("Graphical Representation for\n Losses vs Epochs\n ")
plt.plot(history.history["loss"],label='Training Loss')
plt.plot(history.history["val_loss"],label='Validation Loss')
plt.xlabel("EPOCH").set_color('r')         
plt.ylabel("LOSS").set_color('r')
plt.grid(True)
plt.legend()


#%%#%%
model.save('model3.h5py')   

