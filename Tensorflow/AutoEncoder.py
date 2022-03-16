import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K
import matplotlib.pyplot as plt
import pandas as pd
import datetime

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon






latent_dim = 10

encoder_inputs = keras.Input(shape=(6000,1))

# conv1d_1 = layers.Conv1D(32,8,strides=1,padding='same',activation='relu',use_bias=True,kernel_initializer='VarianceScaling',bias_initializer = 'Zeros')(encoder_inputs)#possibly update kernel_initializer
# #max_pooling1d_1 = layers.MaxPooling1D(pool_size = 2,strides = 4, padding = 'same')(conv1d_1)
# conv1d_2 = layers.Conv1D(32,8,strides=2,padding='same',activation='relu',use_bias=True,kernel_initializer='VarianceScaling',bias_initializer = 'Zeros')(conv1d_1)#possibly update kernel_initializer


x = layers.Conv1D(256,16,strides=2,padding='same',activation='relu',use_bias=True,kernel_initializer='VarianceScaling',bias_initializer = 'Zeros')(encoder_inputs)#possibly update kernel_initializer
#x = layers.MaxPooling1D(pool_size = 4,strides = 4, padding = 'same')(x)

x = layers.Conv1D(128,8,strides=2,padding='same',activation='relu',use_bias=True,kernel_initializer='VarianceScaling',bias_initializer = 'Zeros')(x)#possibly update kernel_initializer
#x = layers.MaxPooling1D(pool_size = 4,strides = 4, padding = 'same')(x)

x = layers.Conv1D(32,8,strides=2,padding='same',activation='relu',use_bias=True,kernel_initializer='VarianceScaling',bias_initializer = 'Zeros')(x)#possibly update kernel_initializer
#x = layers.MaxPooling1D(pool_size = 4,strides = 4, padding = 'same')(x)

shape_before_flattening = K.int_shape(x)

flatten_1 = layers.Flatten()(x)
x = layers.Dense(16 , activation="relu")(flatten_1)

z_mean = layers.Dense(latent_dim, name="z_mean",kernel_initializer='Zeros',bias_initializer = 'Zeros')(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var",kernel_initializer='Zeros',bias_initializer = 'Zeros')(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()



#DECODER
latent_inputs = keras.Input(shape=(latent_dim,))

# x = layers.Dense(1500, activation="relu")(latent_inputs)

#x = layers.Dense(1500, activation="relu")(latent_inputs)

x = layers.Dense(np.prod(shape_before_flattening[1:]), activation="relu")(latent_inputs)
x = layers.Reshape(shape_before_flattening[1:])(x)

x = layers.Conv1DTranspose(32, 8, activation="relu", strides=2, padding="same")(x)
x = layers.Conv1DTranspose(128, 8, activation="relu", strides=2, padding="same")(x)
x = layers.Conv1DTranspose(256, 16, activation="relu", strides=2, padding="same")(x)


decoder_outputs = layers.Conv1DTranspose(1, 16, padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()



# latent_inputs = keras.Input(shape=(latent_dim,))
# x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
# x = layers.Reshape((7, 7, 64))(x)
# x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
# x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
# decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
# decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
# decoder.summary()





class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=-1))
            total_loss = reconstruction_loss + kl_loss
            #total_loss = reconstruction_loss #ABSOLUTELY CHANGE!
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }





# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


train_X = pd.read_csv('/home/ege/Repo/SideChannel-AdversarialAI/Tensorflow/DataSet/trainX13.csv', header=None)
train_Y = pd.read_csv('/home/ege/Repo/SideChannel-AdversarialAI/Tensorflow/DataSet/trainY13.csv', header=None)

trainY = train_Y.to_numpy()
trainX = train_X.to_numpy()
trainX = np.expand_dims(trainX,axis=2)
#train_Y = pd.read_csv('/home/ege/Repo/SideChannel-AdversarialAI/Tensorflow/DataSet/trainY.csv', header=None)

#print(trainX[0])



# print(trainX.shape)
# print(trainXCUT.shape)
# exit()





minimum = np.amin(trainX)
maximum = np.amax(trainX)

trainX = (trainX-minimum)/(maximum-minimum)


# classToCheck = 7
# trainXCUT = trainX[classToCheck::14]
# trainYCUT = trainY[classToCheck::14]



vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
history = vae.fit(trainX, epochs=50, batch_size=32)


sampleToPredict = 0

# z_mean, z_log_var, z = vae.encoder.predict(trainXCUT)
# reconstructed_x = vae.decoder.predict(z)*(maximum-minimum)+minimum


# (reconstructed_x[sampleToPredict]).tofile('prediction.csv', sep = ',')
# ((trainXCUT[sampleToPredict].T)*(maximum-minimum)+minimum).tofile('predictionTARGET.csv', sep = ',')


# z_mean, z_log_var, z = vae.encoder.predict(trainX[5].T)
# print(vae.decoder.predict(z)*(maximum-minimum)+minimum)





classification_model = tf.keras.models.load_model('TrainedModel/trainedModel.h5')



for i in range(14):
    

    trainXCUT = trainX[i::14]
    trainYCUT = trainY[i::14]



    z_mean, z_log_var, z = vae.encoder.predict(trainXCUT)
    reconstructed_x = vae.decoder.predict(z)*(maximum-minimum)+minimum

    # if(i == 5):
    #     sampleToPredict = 0
    #     (reconstructed_x[sampleToPredict]).tofile('prediction.csv', sep = ',')
    #     ((trainXCUT[sampleToPredict].T)*(maximum-minimum)+minimum).tofile('predictionTARGET.csv', sep = ',')

    result = classification_model.evaluate(((reconstructed_x)*(maximum-minimum)+minimum),trainYCUT)





# reconstructed_x = np.expand_dims(reconstructed_x,axis=2)

#reconstructed_x[0].tofile('reconstructed0.csv', sep = ',')

# print(reconstructed_x[0].shape)
# print(trainX[0].shape)

# print(reconstructed_x)
# print("-------------------------------")
# print(trainX)


# classification_model.evaluate(reconstructed_x,trainY,verbose=1)




result = classification_model.evaluate(((reconstructed_x)*(maximum-minimum)+minimum),trainYCUT)
#print(str(np.argmax(result[sampleToPredict])))
    #print(result.shape)




# print(z)

# exit()


# for i, x_i in enumerate(trainX):
    
#     z_mean, z_log_var, z = vae.encoder.predict(x_i.T)
#     reconstructed_x = vae.decoder.predict(z)*(maximum-minimum)+minimum













# print(z_mean)
# print(z_log_var)
# print(z)

# exit()

#x1 = vae.decoder.predict(z)

#x1 = vae.decoder.predict(z)*(maximum-minimum)+minimum
#x2 = (maximum-vae.decoder.predict([[0,1]]))/(maximum-minimum)
#x3 = (maximum-vae.decoder.predict([[-1,0]]))/(maximum-minimum)
#x4 = (maximum-vae.decoder.predict([[0,-1]]))/(maximum-minimum)

#sample1 = x1[0].T
#sample2 = trainX[5].T*(maximum-minimum)+minimum
#sample3 = x3[0].T
#sample4 = x4[0].T

#final_sample1 = sample1[0]



#sample1[0].tofile('prediction1.csv', sep = ',')
#sample2.tofile('predictionTARGET.csv', sep = ',')
#sample3[0].tofile('prediction3.csv', sep = ',')
#sample4[0].tofile('prediction4.csv', sep = ',')


plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig('loss.png')

# def plot_latent_space(vae, n=30, figsize=15):
#     # display a n*n 2D manifold of digits
#     digit_size = 28
#     scale = 1.0
#     figure = np.zeros((digit_size * n, digit_size * n))
#     # linearly spaced coordinates corresponding to the 2D plot
#     # of digit classes in the latent space
#     grid_x = np.linspace(-scale, scale, n)
#     grid_y = np.linspace(-scale, scale, n)[::-1]

#     for i, yi in enumerate(grid_y):
#         for j, xi in enumerate(grid_x):
#             z_sample = np.array([[xi, yi]])
#             x_decoded = vae.decoder.predict(z_sample)
#             digit = x_decoded[0].reshape(digit_size, digit_size)
#             figure[
#                 i * digit_size : (i + 1) * digit_size,
#                 j * digit_size : (j + 1) * digit_size,
#             ] = digit

#     plt.figure(figsize=(figsize, figsize))
#     start_range = digit_size // 2
#     end_range = n * digit_size + start_range
#     pixel_range = np.arange(start_range, end_range, digit_size)
#     sample_range_x = np.round(grid_x, 1)
#     sample_range_y = np.round(grid_y, 1)
#     plt.xticks(pixel_range, sample_range_x)
#     plt.yticks(pixel_range, sample_range_y)
#     plt.xlabel("z[0]")
#     plt.ylabel("z[1]")
#     plt.imsave(fname = "figure.png",arr=figure, cmap="Greys_r")


#plot_latent_space(vae)
