# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 09:36:15 2023

@author: hp
"""

import tensorflow as tf
import numpy
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from tensorflow.keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense, Activation, Flatten
from keras.layers import BatchNormalization
import cv2                 #import packages 
import os
from imutils import paths
cv2.__version__
from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2
import pytesseract
from pytesseract import image_to_string
from matplotlib import pyplot as plt
from imutils import paths
#%%
net = cv2.dnn.readNet(r"C:\Users\hp\Downloads\frozen_east_text_detection.pb")

def text_detector(image):
    orig = image.copy()
    (H, W) = image.shape[:2]
    (newW, newH) = (256, 256)
    rW = W / float(newW)
    rH = H / float(newH)
    
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]
    #print(H,W)
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]
    
    blob = cv2.dnn.blobFromImage(image, 1.0, (W,H), 
                                 (123.68, 116.78, 103.94), swapRB=True, crop= False)
    
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    recognized_text_embeddings = ""
    
    for y in range(0, numRows):
        
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        
        for x in range(0, numCols):
            if scoresData[x] < 0.5:
                continue
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            
            if((endX>0) and (endX<=W)):
                endX=endX
            elif endX>W:
                endX=W
            else:
                endX=0
                
            if((endY>0) and (endY<=H)):
                endY=endY
            elif endY>H:
                endY=H
            else:
                endY=0
                
            if((startX>0) and (endX<=W)): 
                startX=startX
            elif startX>W:
                startX=W
            else:
                startX=0
            if((startY>0) and (startY<=H)):
                startY=startY
            elif startY>H:
                startY=H
            else:
                startY=0
                
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])     
    boxes=non_max_suppression(np.array(rects), probs=confidences)
    
    for (startX,startY,endX,endY) in boxes: 
        startX = int(startX*rW)
        startY = int(startY*rH)
        endX = int(endX*rW)
        endY = int(endY*rH)
        boundary = 2
            
        text=orig[startY-boundary:endY+boundary, startX - boundary:endX + boundary]
        text=cv2.cvtColor(text.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        textRecognized=pytesseract.image_to_string(text)
        recognized_text_embeddings+=textRecognized
        cv2.rectangle(orig, (startX, startY),(endX,endY), (0, 255, 0), 3)
        orig = cv2.putText(orig, textRecognized, (endX,endY+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            
    return orig, recognized_text_embeddings
    
#%%
image0 = cv2.imread(r"I:\End_sem_major_project\data\train\BEANS\BEANS0036.png")
image = cv2.resize(image0, (960,960), interpolation = cv2.INTER_AREA)
textImage, textDetected = text_detector(image)
print(textDetected)
cv2.imshow("Text Detection", textImage)
time.sleep(2)
k= cv2.waitKey(30)



            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    