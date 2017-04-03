# Import this file to prevent Tensorflow from destroying your GPU memory
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
