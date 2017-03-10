import tensorflow as tf
from keras import layers, models
import numpy as np
from keras.layers import TimeDistributed as TD

BATCH_SIZE = 1
IMG_WIDTH = 11
IMG_HEIGHT = IMG_WIDTH
GRU_SIZE = 20

print("Keras won't let you change the batch size, so set it to 1")
img = layers.Input(batch_shape=(1, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 3))

x = TD(layers.Convolution2D(1,1,1))(img)


print("Transforming space into time...")
rows = layers.Lambda(lambda x: tf.reshape(x, [1, -1, IMG_WIDTH, 1]))(x)

t = layers.Lambda(lambda x: tf.transpose(x, [0, 1, 3, 2, 4]))(x)
cols = layers.Lambda(lambda x: tf.reshape(x, [1, -1, IMG_HEIGHT, 1]))(t)

groo = TD(layers.GRU(GRU_SIZE, return_sequences=True))
backwards = layers.Lambda(lambda x: tf.reverse(x, axis=[1]))
left_rnn = groo(rows)
down_rnn = groo(cols)
right_rnn = groo(backwards(rows))
up_rnn = groo(backwards(cols))

reshape_to_mask = layers.Lambda(lambda x: tf.reshape(x, [1, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, GRU_SIZE]))

left_out = reshape_to_mask(left_rnn)
right_out = reshape_to_mask(backwards(right_rnn))

down_out = layers.Lambda(lambda x: tf.transpose(x, [0, 2, 1, 3]))(down_rnn)
down_out = reshape_to_mask(down_out)

up_out = layers.Lambda(lambda x: tf.transpose(x, [0, 2, 1, 3]))(up_rnn)
up_out = reshape_to_mask(backwards(up_out))

concat_out = layers.merge([left_out, right_out, up_out, down_out], mode='concat', concat_axis=-1)

print("... now transforming time back into space")
output_mask = TD(layers.Convolution2D(1,1,1))(concat_out)

moo = models.Model(input=img, output=output_mask)
rando = np.random.rand(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH,3)
rando = np.expand_dims(rando, axis=0)
preds = moo.predict(rando)
print("Got output preds shape {}".format(preds.shape))
