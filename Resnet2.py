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
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Update your data paths here
train_path=r'I:\End_sem_major_project\data\train'
val_path= r'I:\End_sem_major_project\data\val' 
test_path= r'I:\End_sem_major_project\data\test' 

imgh,imgw = 256,256
batch_size=32

alphabet = ['BEANS', 'CAKE', 'CANDY', 'CEREAL', 'CHIPS', 'CHOCOLATE', 'COFFEE',
                 'CORN', 'FISH', 'FLOUR', 'HONEY', 'JAM', 'JUICE', 'MILK', 'NUTS', 'OIL',
                 'PASTA', 'RICE', 'SODA', 'SPICES', 'SUGAR', 'TEA', 'TOMATO_SAUCE', 'VINEGAR', 'WATER']
#Path where you want to save the model
os.chdir(r'I:\End_sem_major_project\New_testing')

train_datagen = ImageDataGenerator(
    #featurewise_center=True,
    #featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        r'I:\End_sem_major_project\data\train',
        target_size=(256, 256),
        batch_size=32,
        color_mode='rgb',
        seed=123,
        class_mode='categorical')

val_generator = val_datagen.flow_from_directory(r'I:\End_sem_major_project\data\val' ,
                                              target_size=(256, 256),
                                              batch_size=32,
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
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])
epochs = 100

#Update this path to save the best model
best_model_path = r'I:\End_sem_major_project'
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
#plt.xlabel("").set_color('r')                                          
#plt.ylabel("ACCURACY").set_color('r')
plt.grid(True)
plt.legend()

fig2=plt.figure()
plt.subplot(1,1,1)
plt.plot(history.history["loss"],label='Training Loss')
plt.plot(history.history["val_loss"],label='Validation Loss')
#plt.xlabel("EPOCH").set_color('r')         
#plt.ylabel("LOSS").set_color('r')
plt.grid(True)
plt.legend()



# train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_path, 
#                                                                 seed=123, 
#                                                                 label_mode="categorical", 
#                                                                 image_size=(imgh,imgw), 
#                                                                 batch_size=batch_size)

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(val_path, 
#                                                                 seed=123, 
#                                                                 label_mode="categorical", 
#                                                                 image_size=(imgh,imgw), 
#                                                                 batch_size=batch_size)

# test_ds = tf.keras.preprocessing.image_dataset_from_directory(test_path, 
#                                                                 seed=123, 
#                                                                 label_mode="categorical", 
#                                                                 image_size=(imgh,imgw), 
#                                                                 batch_size=batch_size)
# train_imgs = []
# train_labels = []
# for label in os.listdir(train_path):
#     for img_name in os.listdir(os.path.join(train_path, label)):
#         img = load_img(os.path.join(train_path, label, img_name), color_mode='grayscale', target_size=(imgh, imgw))
#         img_array = img_to_array(img)
#         train_imgs.append(img_array)
#         train_labels.append(label)
        
# val_imgs = []
# val_labels = []
# for label in os.listdir(val_path):
#     for img_name in os.listdir(os.path.join(val_path, label)):
#           img = load_img(os.path.join(val_path, label, img_name), color_mode='grayscale', target_size=(imgh, imgw))
#           img_array = img_to_array(img)
#           val_imgs.append(img_array)
#           val_labels.append(label)
         
         
# train_imgs = np.array(train_imgs)
# train_labels = np.array(train_labels)
# val_imgs = np.array(val_imgs)
# val_labels = np.array(val_labels)

#print(num_classes)
# train_labels = tf.keras.utils.to_categorical([alphabet.index(label) for label in train_labels ], num_classes)
# val_labels = tf.keras.utils.to_categorical([alphabet.index(label) for label in val_labels], num_classes)



# augs_gen = ImageDataGenerator(
#         featurewise_center=False,  
#         samplewise_center=False, 
#         featurewise_std_normalization=False,  
#         samplewise_std_normalization=False,  
#         zca_whitening=False,  
#         rotation_range=10,  
#         zoom_range = 0.1, 
#         width_shift_range=0.2,  
#         height_shift_range=0.2, 
#         horizontal_flip=True,  
#         vertical_flip=False) 



#train_generator=augs_gen.flow_from_directory(r'I:\End_sem_major_project\data')
#augs_gen.fit(train_ds)


