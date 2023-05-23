import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.python.keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, EarlyStopping

# Update your data paths here
train_path=r"D:/test/data/data_used/train"
val_path=r"D:/test/data/data_used/val"
test_path=r"D:/test/data/data_used/test"

imgh,imgw = 256,256
batch_size=32

#Path where you want to save the model
os.chdir(r"D:/test/")

train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_path, 
                                                               seed=123, 
                                                               label_mode="categorical", 
                                                               image_size=(imgh,imgw), 
                                                               batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(val_path, 
                                                               seed=123, 
                                                               label_mode="categorical", 
                                                               image_size=(imgh,imgw), 
                                                               batch_size=batch_size)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(test_path, 
                                                               seed=123, 
                                                               label_mode="categorical", 
                                                               image_size=(imgh,imgw), 
                                                               batch_size=batch_size)

#Creating the model	                                                       
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
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])
epochs = 100

#Update this path to save the best model
best_model_path = r'D:\test\resnet50_model_clf.h5'
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, verbose = 1)
model_checkpoint = ModelCheckpoint(best_model_path, verbose = 1, save_best_only = True)

history=model.fit(train_ds,
                  validation_data=val_ds,
                  epochs=epochs,
                  workers=8,
                  #Comment below line if it gives errors while running
                  callbacks = [early_stopping, model_checkpoint]
                 )

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

"""
#uncomment and run this part if line 68 doesn't cause any issues

from keras.models import load_model
best_model = load_model(r'D:\test\resnet50_model_clf.h5')

import numpy as np
predictions = np.array([])
labels =  np.array([])

for x, y in test_ds:
    predictions = np.concatenate([predictions, np.argmax(best_model.predict(x), axis=-1)])
    labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])

test_accuracy = sum([1 for yt, yp in zip(labels, predictions) if yt==yp]) / len(labels)
test_accuracy

# code to be used for computing tpr/tnr/etc...
tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy()

"""