import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow import keras
# from csv import writer

import pandas as pd
from tensorflow.keras import backend as K
import numpy as np
import sys
from numpy import genfromtxt
#np.set_printoptions(threshold=sys.maxsize)

saliency_numpied = genfromtxt('saliencyMAP.csv', delimiter=',')

saved_model = load_model('/home/ege/SeniorDesign/sddec21-13/Tensorflow/TrainedModel/trainedModel.h5') #Load MODEL
#saved_model.summary()


train_X = pd.read_csv('trainX.csv', header=None)
train_Y = pd.read_csv('trainY.csv', header=None)
trainY_numpy = train_Y.to_numpy()
trainX_numpy = train_X.to_numpy()


#CONSTANTS
mapSize = 6000 #Size of saliency map & inputs
numberOfSamples = len(train_X.index)
numberOfWebsites = 9
saliencyClass = 3
numberOfSamplesPerClass = 130
desyncFactor = 5 #desync when trying to inject noise (NOT IMPLEMENTED) 

#PARAMETERS
Noise = 15 #Maybe 15?
threshold = 0.2 #If above this, add noise here
widthOfNoise = 15 #How wide the noise introduced will be



arrayToCheckFillRatio = np.zeros(6000, dtype=np.int8)
#We did not factor in the fact that we most likely won't be able to fully sync
# for sample in range((numberOfSamplesPerClass*numberOfWebsites)-1):
# 	for col in range(6000-1):
# 		if(saliency_numpied[col] > threshold):
# 			for num in range(widthOfNoise):
# 				if ((col-(widthOfNoise//2)+num) < 6000):
# 					arrayToCheckFillRatio[(col-(widthOfNoise//2))+num] = 1
# 					trainX_numpy[sample][(col-(widthOfNoise//2))+num] = Noise   


times = 200
for sample in range((numberOfSamplesPerClass*numberOfWebsites)-1):
	for col in range(6000-1):
		trainX_numpy[sample][col] += (saliency_numpied[col]*times)
		arrayToCheckFillRatio[col] += (saliency_numpied[col]*times)



#print("Max Noise Induced: " + str(np.argmax(arrayToCheckFillRatio)))
arrayToCheckFillRatio.tofile('InducedNoise.csv', sep = ',')

#print("Sum: "+ str(sum(arrayToCheckFillRatio)))
#print("Percentage: "+ str(sum(arrayToCheckFillRatio)/6000))



#THESE ARE FOR CREATING SUBSET OF SPECIFIC CLASS FROM THE DATASET
artificialX = trainX_numpy[3::numberOfWebsites]
artificialY = trainY_numpy[3::numberOfWebsites]

#print(artificialX)



#AUTOMATIC PREDICTIONS USING TF.EVALUATE
deneme = saved_model.evaluate(artificialX,artificialY, batch_size=16, verbose=1)


#MANUAL CLASSIFICATION USING CONFIDENCES
out = saved_model.predict(artificialX, batch_size=16, verbose=1, workers=1, use_multiprocessing=False)
meanConfidence = 0
classes = []
for x in range(len(out)): 
	classes.append(np.argmax(out[x]))
	meanConfidence = meanConfidence + out[x][0]

meanConfidence = meanConfidence / numberOfSamplesPerClass
print("Mean Confidence: "+ str(meanConfidence))
print(classes)