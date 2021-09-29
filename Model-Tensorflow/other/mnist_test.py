import os
os.environ['TF_CPP_MIN_LOG_LEVEL']= '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

physical_devices = tf.config.list_physical_devices('GPU')

print(physical_devices)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

#Sequential API

model = keras.Sequential(

	[
		layers.Dense(512,activation='relu'),
		layers.Dense(216, activation = 'relu'),
		layers.Dense(10)
	]
)

model.compile(
	loss= keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	optimizer=keras.optimizers.Adam(lr=0.001),
	metrics=["accuracy"],
)

model.fit(x_train,y_train,batch_size=32,epochs=5,verbose=2)

model.evaluate(x_test, y_test, batch_size= 32, verbose=2)

