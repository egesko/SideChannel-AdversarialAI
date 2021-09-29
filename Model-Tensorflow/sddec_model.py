import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

from csv import writer

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)

#upload dataset here


train_csv = pd.read_csv('train_normalized.csv')
val_csv = pd.read_csv('val_normalized.csv')

train_label_csv = pd.read_csv('train_label.csv')
val_label_csv = pd.read_csv('val_label.csv')



#print(train_csv.head(2))

#x_train = train_csv.astype("float32")
#y_train = train_label_csv.astype("float32")






dropoutLevel = 0.3



numberOfWebsites = 3
def my_model_sddec():

    input_1 = keras.Input(shape = (6000,1))
    
    conv1d_1 = layers.Conv1D(256,16,strides=3,padding='valid',activation='relu',use_bias=True,kernel_initializer='VarianceScaling',bias_initializer = 'Zeros')(input_1)#possibly update kernel_initializer
    
    max_pooling1d_1 = layers.MaxPooling1D(pool_size = 4,strides = 4, padding = 'same')(conv1d_1)
    
    conv1d_2 = layers.Conv1D(32,8,strides=3,padding='same',activation='relu',use_bias=True,kernel_initializer='VarianceScaling',bias_initializer = 'Zeros')(max_pooling1d_1)#possibly update kernel_initializer

    max_pooling1d_2 = layers.MaxPooling1D(pool_size = 4,strides = 4, padding = 'same')(conv1d_2)

    #lstm_1 = layers.LSTM(32,activation='tanh',recurrent_activation='hard_sigmoid',use_bias=True,kernel_initializer='VarianceScaling',recurrent_initializer = 'orthogonal',bias_initializer='Zeros', return_sequences = True)(max_pooling1d_2) #Variance Scaling

    flatten_1 = layers.Flatten()(max_pooling1d_2)

    dropout_1 = layers.Dropout(0.3)(flatten_1)

    dense_1 = layers.Dense(300,activation = 'relu')(dropout_1)

    dropout_2 = layers.Dropout(0.3)(dense_1)

    dense_2= layers.Dense(numberOfWebsites,kernel_regularizer = 'l2',activation = 'softmax', kernel_initializer = 'VarianceScaling', bias_initializer = 'zeros')(dropout_2)

    model = keras.Model(inputs = input_1, outputs = dense_2)
    return model


#model1 = Sequential()
model = my_model_sddec()

#model2 = tf.keras.models.model_from_json("model_config.json")

model.compile(
	loss=keras.losses.SparseCategoricalCrossentropy(),
	optimizer=keras.optimizers.Adam(),
	metrics=["accuracy"]
)

history = model.fit(train_csv, train_label_csv,validation_data = (val_csv, val_label_csv), batch_size=16, epochs=10,shuffle = True, verbose=1)


#model.evaluate(x_test,y_test, batch_size=64, verbose=2)


fig = plt.figure()

#plotting
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training','testing'], loc='upper left')
fig.savefig('plot.png')

