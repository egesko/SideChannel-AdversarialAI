import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from tensorflow.keras.models import load_model

from tensorflow import keras
#from tensorflow.keras import layers
# from csv import writer
import pandas as pd
from tf_keras_vis.utils import num_of_gpus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tensorflow.keras import backend as K
from tf_keras_vis.saliency import Saliency
import numpy as np
import sys
import matplotlib.pyplot as plt
#np.set_printoptions(threshold=sys.maxsize)

_, gpus = num_of_gpus()
print('Tensorflow recognized {} GPUs'.format(gpus))
# import matplotlib.pyplot as plt

#Print tf and keras versions
print("keras      {}".format(tf.keras.__version__))
print("tensorflow {}".format(tf.__version__))

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0],True)

train_X = pd.read_csv('trainX.csv', header=None)
train_Y = pd.read_csv('trainY.csv', header=None)

saved_model = load_model('/home/ege/SeniorDesign/sddec21-13/Tensorflow/TrainedModel/trainedModel.h5')
#saved_model.summary()

replace2linear = ReplaceToLinear()

# Create Saliency object.
saliency = Saliency(saved_model,
                    model_modifier=replace2linear,
                    clone=True)


#CONSTANTS
numberOfSamples = len(train_X.index)
numberOfWebsites = 9
numberOfSamplesPerClass = 130


def oneSaliency(row,category):
    dataSet_Row=train_X.iloc[row].to_numpy()
    row_reshaped = dataSet_Row.reshape((6000,1))
    score = CategoricalScore([category])
    saliency_map = saliency(score=score, seed_input = row_reshaped,smooth_samples=20,smooth_noise=0.20,gradient_modifier=None,normalize_map=False)
    return saliency_map


def makeSaliency(singleORmultiple,saliencyClass):    #TRUE=SINGLE FALSE=MULTIPLE

    if(singleORmultiple == True): #If we are creating saliency map of only one class

        finalSaliency = np.zeros(6000)
        i = saliencyClass

        for x in range(numberOfSamplesPerClass):
            print(str(x+1)+" out of "+str(numberOfSamplesPerClass))
            singleSaliency = oneSaliency(i,saliencyClass)
            finalSaliency = np.add(singleSaliency,finalSaliency)
            i = i + numberOfWebsites

        return (finalSaliency / numberOfSamplesPerClass)
    else: #If we are creating saliency map of entire dataset

        finalSaliency = np.zeros(6000)
        i = 0
        for x in range((numberOfSamplesPerClass*numberOfWebsites)-1):
        
            print(str(x+1)+" out of "+str(numberOfSamplesPerClass*numberOfWebsites))

            singleSaliency = oneSaliency(x,i)
            finalSaliency = np.add(singleSaliency,finalSaliency)

            i = (i+1)%numberOfWebsites
        return (finalSaliency / (numberOfSamplesPerClass*numberOfWebsites))


def makeChangeSaliency(inputClass,scoreClass): #Saliency for making one class more like another.

    finalSaliency = np.zeros(6000)
    i = inputClass

    for x in range(numberOfSamplesPerClass):
        print(str(x+1)+" out of "+str(numberOfSamplesPerClass))
        singleSaliency = oneSaliency(i,scoreClass)
        finalSaliency = np.add(singleSaliency,finalSaliency)
        i = i + numberOfWebsites
    
    return (finalSaliency / numberOfSamplesPerClass)


#saliencyFINAL = makeSaliency(True, 0) #TRUE=SINGLE FALSE=MULTIPLE

saliencyFINAL = makeChangeSaliency(3,5) #TEST, DELETE LATER

#saliencyFINAL1 = makeSaliency(True, 1) #TRUE=SINGLE FALSE=MULTIPLE
#saliencyFINAL = saliencyFINAL1 - saliencyFINAL0 

#print(saliencyFINAL)
saliencyFINAL.tofile('saliencyMAP.csv', sep = ',')


# history = model.fit(train_X, train_Y,validation_split = 0.3, batch_size=16, epochs=20, verbose=1)
# #model.evaluate(x_test,y_test, batch_size=64, verbose=2)



fig = plt.figure()
plt.plot(saliencyFINAL[0])
plt.yticks(np.arange(min(saliencyFINAL[0])-0.05, max(saliencyFINAL[0])+0.05,0.01))
plt.grid()
plt.axhline(linewidth=1, color='r')
plt.xlabel("Positions (5ms each)")
plt.ylabel("Importance")

fig.savefig('saliency0to1.png',dpi=200)


