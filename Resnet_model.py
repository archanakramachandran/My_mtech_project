# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 22:48:51 2023

@author: hp
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 22:42:06 2023

@author: hp
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.python.keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adamax
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Conv2D, MaxPooling2D


# Update your data paths here
train_path=r'I:\End_sem_major_project\data\train'
val_path= r'I:\End_sem_major_project\data\val' 
test_path= r'I:\End_sem_major_project\data\test' 

imgh,imgw = 256,256
batch_size=64

alphabet = ['BEANS', 'CAKE', 'CANDY', 'CEREAL', 'CHIPS', 'CHOCOLATE', 'COFFEE',
                 'CORN', 'FISH', 'FLOUR', 'HONEY', 'JAM', 'JUICE', 'MILK', 'NUTS', 'OIL',
                 'PASTA', 'RICE', 'SODA', 'SPICES', 'SUGAR', 'TEA', 'TOMATO_SAUCE', 'VINEGAR', 'WATER']
#Path where you want to save the model
os.chdir(r'I:\End_sem_major_project\New_testing_results\Resnet2_changes')

train_datagen = ImageDataGenerator(
    #featurewise_center=True,
    #featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2)
    #rescale=1./255
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(r'I:\End_sem_major_project\data\train',
        target_size=(256, 256),
        batch_size=64,
        color_mode='rgb',
        seed=123,
        class_mode='categorical')

val_generator = val_datagen.flow_from_directory(r'I:\End_sem_major_project\data\val' ,
                                              target_size=(256, 256),
                                              batch_size=64,
                                              color_mode='rgb',
                                              seed=123,
                                              class_mode='categorical' )


num_classes = len(alphabet)
print(num_classes)

model=Sequential()

pretrainedmodel=tf.keras.applications.ResNet50(include_top=False,
                                               input_shape=(imgh,imgw,3),
                                               pooling="max",
                                               weights="imagenet")
for layer in pretrainedmodel.layers:
    layer.trainable=False

model.add(pretrainedmodel)
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(25, activation="softmax"))
model.summary()
model.compile(optimizer=Adamax(learning_rate=0.001), 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])
epochs = 100 
#Update this path to save the best model
best_model_path = r'I:\End_sem_major_project\New_testing_results\Resnet2_changes'
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, verbose = 1)
model_checkpoint = ModelCheckpoint(best_model_path, verbose = 1, save_best_only = True)

history=model.fit(train_generator,
                  validation_data=(val_generator),
                  epochs=epochs,
                  workers=8,
                  callbacks = [early_stopping, model_checkpoint]
                 )

fig1=plt.figure()
plt.subplot(1,1,1)
plt.plot(history.history["accuracy"],label='Training Accuracy')
plt.plot(history.history["val_accuracy"],label='Validation Acuuracy')
plt.xlabel("EPOCH").set_color('r')                                          
plt.ylabel("ACCURACY").set_color('r')
plt.grid(True)
plt.legend()

fig2=plt.figure()
plt.subplot(1,1,1)
plt.plot(history.history["loss"],label='Training Loss')
plt.plot(history.history["val_loss"],label='Validation Loss')
plt.xlabel("EPOCH").set_color('r')         
plt.ylabel("LOSS").set_color('r')
plt.grid(True)
plt.legend()






