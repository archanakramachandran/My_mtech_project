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
#net = cv2.dnn.readNet(r"C:\Users\hp\Downloads\frozen_east_text_detection.pb")
#print(pytesseract.get_tesseract_version)//checking version of tessaract
#pytesseract.get_tesseract_version()
# train_path =  r'I:\End_sem_major_project\data\train'
# val_path = r'I:\End_sem_major_project\data\val' 
# pathtoimages=list(paths.list_images(train_path))
# print(pathtoimages)

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

	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]


	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)

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

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability, ignore it
			if scoresData[x] < 0.5:
				continue

			# compute the offset factor as our resulting feature maps will
			# be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and then
			# compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height of
			# the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates for
			# the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score to
			# our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	boxes = non_max_suppression(np.array(rects), probs=confidences)

	for (startX, startY, endX, endY) in boxes:

		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)
		boundary = 2

		if startX > W:
        		startX=W-2
		elif startX < 2:
                	startX=2
		if startY > W:
            		startY=W-2
        	elif startY < 2:
            		startY=2
        
        	if endX > W:
            		endX=W-2
        	elif endX < 2:
            		endX=2
        	if endY > W:
            		endY=W-2
        	elif endY < 2:
            		endY=2

		text = orig[startY-boundary:endY+boundary, startX - boundary:endX + boundary]
		text = cv2.cvtColor(text.astype(np.uint8), cv2.COLOR_BGR2GRAY)
		textRecognized = pytesseract.image_to_string(text)
		recognized_text_embeddings += textRecognized
		cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 3)
		orig = cv2.putText(orig, textRecognized, (endX,endY+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    
	return orig, recognized_text_embeddings


#%%
# data=[] #x_train
# lables=[] #y_train
# lable=[] 
# for img in pathtoimages: #loop in image folder
#   lable.clear()
#   #lable.append(images.split(os.path.sep)[-2])
#   #if label not in grocerieslabels:
#   #  continue 
#   image=cv2.imread(img)
#   orig = cv2.resize(image, (640,320), interpolation = cv2.INTER_AREA)
#   image=cv2.resize(image, (640,320), interpolation = cv2.INTER_AREA)
 
#   textDetected = text_detector(image)
#   #cv2.imshow("Orig Image",orig)
#   cv2.imshow("Text Detection", textDetected)
#   time.sleep(2)
#   k = cv2.waitKey(30)
#   if k == 27:
#      break
 

#%%
image0 = cv2.imread(r"I:\End_sem_major_project\data\train\CAKE\CAKE0038.png")
image = cv2.resize(image0, (256, 256), interpolation = cv2.INTER_AREA)
textImage, textDetected = text_detector(image)
print(textDetected)
cv2.imshow("Text Detection", textImage)
time.sleep(2)
k= cv2.waitKey(30)

# for i in range(0,2):
# for img in array:
#orig = cv2.resize(image0, (640,320), interpolation = cv2.INTER_AREA)
#cv2.imshow("Orig Image",orig)
    # if k == 27:
    #     break
#cv2.destroyAllWindows()



