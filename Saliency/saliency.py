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
np.set_printoptions(threshold=sys.maxsize)

_, gpus = num_of_gpus()
print('Tensorflow recognized {} GPUs'.format(gpus))
# import matplotlib.pyplot as plt

#Print tf and keras versions
print("keras      {}".format(tf.keras.__version__))
print("tensorflow {}".format(tf.__version__))

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0],True)

train_X = pd.read_csv('trainX.csv')
train_Y = pd.read_csv('trainY.csv')


row_1=train_X.iloc[0].to_numpy()
print(row_1)
exit()
row_reshaped = row_1.reshape((6000,1))


saved_model = load_model('/home/ege/SeniorDesign/sddec21-13/Tensorflow/TrainedModel/trainedModel.h5')
saved_model.summary()


replace2linear = ReplaceToLinear()

# 1 is the imagenet index corresponding to Goldfish, 294 to Bear and 413 to Assault Rifle.
score = CategoricalScore([1])


# Create Saliency object.
saliency = Saliency(saved_model,
                    model_modifier=replace2linear,
                    clone=True)

saliency_map = saliency(score, row_reshaped,smooth_samples=20,smooth_noise=0.20)

saliency_map.tofile('data1.csv', sep = ',')


exit()





# model.compile(
# 	loss=keras.losses.SparseCategoricalCrossentropy(),
# 	optimizer=keras.optimizers.Adam(),
# 	metrics=["accuracy"]
# )

# history = model.fit(train_X, train_Y,validation_split = 0.3, batch_size=16, epochs=20, verbose=1)
# #model.evaluate(x_test,y_test, batch_size=64, verbose=2)



# #plotting
# fig = plt.figure()
# plt.locator_params(axis="x", nbins=20)
# plt.locator_params(axis="y", nbins=10)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['training','validation'], loc='upper left')
# fig.savefig('plot.png')



