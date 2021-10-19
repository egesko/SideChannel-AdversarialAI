import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

from matplotlib import pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)

(x_train,y_train), (x_test, y_test)= cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

dropoutLevel = 0.3

def my_model():
	inputs = keras.Input(shape= (32,32,3))

	#Layer1-CNN
	x = layers.Conv2D(32,3, padding = 'same')(inputs)
	x = keras.activations.relu(x)
	x = layers.Dropout(dropoutLevel)(x)
	x = layers.BatchNormalization()(x)

	#Layer2-CNN
	x = layers.Conv2D(64,3,padding = 'same')(x)
	x = keras.activations.relu(x)
	x = layers.MaxPooling2D()(x)
	x = layers.Dropout(dropoutLevel)(x) #Newly added
	x = layers.BatchNormalization()(x)

	
	x = layers.Conv2D(128,3, padding = 'valid')(x)
	x = layers.BatchNormalization()(x)
	x = keras.activations.relu(x)
	x= layers.Dropout(dropoutLevel)(x) #Newly added
	x=layers.Flatten()(x)
	x = layers.Dense(128,activation = 'relu')(x) #Changed from 64 to 128
	x = layers.Dense(64, activation = 'relu')(x) #Newly added
	outputs = layers.Dense(10)(x)
	model = keras.Model(inputs=inputs, outputs=outputs)
	return model

model =my_model()

model.compile(
	loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	optimizer=keras.optimizers.Adam(lr=3e-4),
	metrics=["accuracy"]
)

history = model.fit(x_train, y_train,validation_split=0.33, batch_size=64, epochs=10, verbose=1)
model.evaluate(x_test,y_test, batch_size=64, verbose=2)


fig = plt.figure()

#plotting
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training','testing'], loc='upper left')
fig.savefig('plot.png')

