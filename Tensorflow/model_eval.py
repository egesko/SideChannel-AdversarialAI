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
import matplotlib.pyplot as plt


saved_model = load_model('/home/ege/SeniorDesign/sddec21-13/Tensorflow/TrainedModel/trainedModel.h5') #Load MODEL

testX = pd.read_csv("testX.csv", header=None)
testXNumpied = testX.to_numpy()

testY = pd.read_csv("testY.csv", header=None)
testYNumpied = testY.to_numpy()


saved_model.evaluate(testX,testY,batch_size=16, verbose=1, workers=1, use_multiprocessing=False)

# out = saved_model.predict(NOnoiseNumpy, batch_size=16, verbose=1, workers=1, use_multiprocessing=False) #NO NOISE
# outNoise = saved_model.predict(noiseNumpy, batch_size=16, verbose=1, workers=1, use_multiprocessing=False) #NOISE



# yAdobe = np.zeros(10)
# yApple = np.zeros(10)
# yGithub = np.zeros(10)
# yGoogle = np.zeros(10)
# yMozilla = np.zeros(10)
# yNyTimes = np.zeros(10)
# yOuva = np.zeros(10)
# ySteam = np.zeros(10)
# yYouTube = np.zeros(10)

# for x in range(10):
# 	file = str(x + 1)+"SecondsAdobe.csv"
# 	train_X = pd.read_csv(file, header=None)
# 	file_numpy = train_X.to_numpy()

# 	#MANUAL CLASSIFICATION USING CONFIDENCES
# 	out = saved_model.predict(file_numpy, batch_size=16, verbose=1, workers=1, use_multiprocessing=False)
# 	print(str(x+1)+" confidences:")
# 	print("Github: "+str(out[0][2]))
# 	print("Adobe: "+str(out[0][0]))

# 	yAdobe[x] = out[0][0]
# 	yApple[x] = out[0][1]
# 	yGithub[x] = out[0][2]
# 	yMozilla[x] = out[0][3]
# 	yGoogle[x] = out[0][4]
# 	yNyTimes[x] = out[0][5]
# 	yOuva[x] = out[0][6]
# 	ySteam[x] = out[0][7]
# 	yYouTube[x] = out[0][8]

# 	print(" ")


# fig = plt.figure()

# #plt.plot(results)
# #for row in range(websiteProbeAmnt):

# plt.plot(yAdobe,label = "Adobe")
# plt.plot(yApple, label = "Apple")

# plt.plot(yGithub,label = "Github")
# plt.plot(yMozilla, label = "Mozilla")

# plt.plot(yGoogle,label = "Google")
# plt.plot(yNyTimes, label = "NyTimes")

# plt.plot(yOuva,label = "Ouva")
# plt.plot(ySteam, label = "Steam")

# plt.plot(yYouTube, label = "YouTube")

# plt.legend()
# plt.yticks(np.arange(0, 1,0.1))
# plt.xticks(np.arange(0, 10,1))
# plt.grid()

# #plt.axhline(linewidth=1, color='r')
# plt.xlabel("Github Noise Injection Interval")
# plt.ylabel("Confidence")

# fig.savefig('AdobeGithub.png',dpi=200)

# meanConfidence = 0
# classes = []
# for x in range(len(out)): 
# 	classes.append(np.argmax(out[x]))
# 	meanConfidence = meanConfidence + out[x][0]

# meanConfidence = meanConfidence / numberOfSamplesPerClass
# print("Mean Confidence: "+ str(meanConfidence))
