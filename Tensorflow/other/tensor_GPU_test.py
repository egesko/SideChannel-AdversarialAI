import tensorflow as tf
if tf.test.gpu_device_name():
	print('GPU:{}'.format(tf.test.gpu_device_name()))
else:
	print("INSTALL GPU")
