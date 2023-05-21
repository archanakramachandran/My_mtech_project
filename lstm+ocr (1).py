import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Define the paths for training and validation datasets
train_path =  r'I:\End_sem_major_project\output\train'
val_path = r'I:\End_sem_major_project\output\val' 

# Define the input size for the images
img_rows, img_cols = 180,180

# Define the alphabet of characters for OCR
# alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
alphabet = ['BEANS', 'CAKE', 'CANDY', 'CEREAL', 'CHIPS', 'CHOCOLATE', 'COFFEE',
                 'CORN', 'FISH', 'FLOUR', 'HONEY', 'JAM', 'JUICE', 'MILK', 'NUTS', 'OIL',
                 'PASTA', 'RICE', 'SODA', 'SPICES', 'SUGAR', 'TEA', 'TOMATO_SAUCE', 'VINEGAR', 'WATER']

# Load the training dataset, data augmentation
train_imgs = []
train_labels = []
for label in os.listdir(train_path):
    for img_name in os.listdir(os.path.join(train_path, label)):
        img = load_img(os.path.join(train_path, label, img_name), color_mode='grayscale', target_size=(img_rows, img_cols))
        img_array = img_to_array(img)
        train_imgs.append(img_array)
        train_labels.append(label)

# Load the validation dataset
val_imgs = []
val_labels = []
for label in os.listdir(val_path):
    for img_name in os.listdir(os.path.join(val_path, label)):
        img = load_img(os.path.join(val_path, label, img_name), color_mode='grayscale', target_size=(img_rows, img_cols))
        img_array = img_to_array(img)
        val_imgs.append(img_array)
        val_labels.append(label)

# Convert the images and labels to numpy arrays
train_imgs = np.array(train_imgs)
# train_labels = np.array(train_labels)
val_imgs = np.array(val_imgs)
val_labels = np.array(val_labels)  #ocr ends


# Convert the labels to one-hot encoding
# d=[alphabet.index(label) for label in train_labels]
num_classes = len(alphabet)
# print(len(train_labels),len(train_imgs))
train_labels = tf.keras.utils.to_categorical([alphabet.index(label) for label in train_labels ], num_classes)
val_labels = tf.keras.utils.to_categorical([alphabet.index(label) for label in val_labels], num_classes)

# Define the LSTM network architecture
model = Sequential()
model.add(LSTM(128, input_shape=(img_rows, img_cols), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
optimizer = Adamax(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
model.fit(train_imgs, train_labels, validation_data=(val_imgs, val_labels), epochs=50, batch_size=32)


#%%
#model.save_weights('my_model_weights_LSTM.h5')   #Save model weights to a file
model.save('my_model_LSTM.h5')  
