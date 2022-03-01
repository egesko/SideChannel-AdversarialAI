import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from csv import writer
import pandas as pd

import numpy as np
from tf_keras_vis.utils import num_of_gpus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tensorflow.keras import backend as K
from tf_keras_vis.saliency import Saliency

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import ticker as ticker


#print(tf.test.is_gpu_available())
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)

saved_model = load_model('/home/ege/Research/SideChannel-AdversarialAI/Tensorflow/TrainedModel/trainedModel.h5')

replace2linear = ReplaceToLinear()
# Create Saliency object.
saliency = Saliency(saved_model,
                    model_modifier=replace2linear,
                    clone=True)

train_X = pd.read_csv('/home/ege/Research/SideChannel-AdversarialAI/Tensorflow/DataSet/trainX.csv', header=None)
train_Y = pd.read_csv('/home/ege/Research/SideChannel-AdversarialAI/Tensorflow/DataSet/trainY.csv', header=None)

trainY = train_Y.to_numpy()
trainX = train_X.to_numpy()

trainX = np.expand_dims(trainX,axis=2)


dropoutLevel = 0.3
#CONSTANTS
numberOfSamples = len(train_X.index)
numberOfWebsites = 15
numberOfSamplesPerClass = 120

def my_model_sddec():

    input_1 = keras.Input(shape = (6000,1))
    conv1d_1 = layers.Conv1D(256,16,strides=3,padding='valid',activation='relu',use_bias=True,kernel_initializer='VarianceScaling',bias_initializer = 'Zeros')(input_1)#possibly update kernel_initializer
    max_pooling1d_1 = layers.MaxPooling1D(pool_size = 4,strides = 4, padding = 'same')(conv1d_1)
    conv1d_2 = layers.Conv1D(32,8,strides=3,padding='same',activation='relu',use_bias=True,kernel_initializer='VarianceScaling',bias_initializer = 'Zeros')(max_pooling1d_1)#possibly update kernel_initializer
    max_pooling1d_2 = layers.MaxPooling1D(pool_size = 4,strides = 4, padding = 'same')(conv1d_2)
    lstm_1 = layers.LSTM(32,activation='tanh',recurrent_activation='hard_sigmoid',use_bias=True,kernel_initializer='VarianceScaling',recurrent_initializer = 'orthogonal',bias_initializer='Zeros', return_sequences = True)(max_pooling1d_2) #Variance Scaling
    flatten_1 = layers.Flatten()(lstm_1)
    #dropout_1 = layers.Dropout(0.3)(flatten_1)
    dense_1 = layers.Dense(300,activation = 'relu')(flatten_1) #newly added
    dropout_2 = layers.Dropout(0.2)(dense_1)
    dense_2= layers.Dense(numberOfWebsites, kernel_regularizer = 'l2',activation = 'softmax', kernel_initializer = 'VarianceScaling', bias_initializer = 'zeros')(dropout_2)
    model = keras.Model(inputs = input_1, outputs = dense_2)
    return model


def oneSaliency(row,category):
    dataSet_Row=train_X.iloc[row].to_numpy()
    row_reshaped = dataSet_Row.reshape((6000,1))
    score = CategoricalScore([category])
    saliency_map = saliency(score=score, seed_input = row_reshaped,smooth_samples=20,smooth_noise=0,gradient_modifier=None,normalize_map=False)
    return saliency_map


def makeChangeSaliency(inputClass,scoreClass): #Saliency for making one class more like another.

    finalSaliency = np.zeros(6000)
    i = inputClass

    for x in range(numberOfSamplesPerClass):
        #print(str(x+1)+" out of "+str(numberOfSamplesPerClass))
        singleSaliency = oneSaliency(i,scoreClass)
        finalSaliency = np.add(singleSaliency,finalSaliency)
        i = i + numberOfWebsites

    #To erase negative gradients
    for i in range(6000):
        if(finalSaliency[0][i] < 0):
            finalSaliency[0][i] = 0
    
    return (finalSaliency / numberOfSamplesPerClass)



#CONSTANTS
numberOfUpdates = 100

#saliencyTarget = [5,5,8,2,1,2,1,1,2,1] #Dictionary that determines what to change a class input to
saliencyTarget = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] #Dictionary that determines what to change a class input to

#saliencyChange = makeChangeSaliency(0,5)


for sourceWebsite in range(numberOfWebsites):

    print("Website #" + str(sourceWebsite))
    
    saliencyChange = makeChangeSaliency(sourceWebsite,saliencyTarget[sourceWebsite])
    saliencyChange = saliencyChange*numberOfUpdates
    

    i = sourceWebsite
    for sample in range(numberOfSamplesPerClass):
        

        #trainX[i] += saliencyChange[0]
        #Updating with gradients
        for col in range(6000):
            trainX[i][col] += saliencyChange[0][col]

        i = i + numberOfWebsites


model = my_model_sddec()

model.compile(
	loss=keras.losses.SparseCategoricalCrossentropy(),
	optimizer=keras.optimizers.Adam(),
	metrics=["accuracy"]
)

history = model.fit(trainX, trainY,validation_split = 0.2, batch_size=16, epochs=20, verbose=1)


model.save('/home/ege/Research/SideChannel-AdversarialAI/Tensorflow/TrainedModel/trainedModelwNoise.h5')