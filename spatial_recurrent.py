import numpy as np
from keras import layers, models
from PIL import Image
import tensorflow as tf
from keras import backend as K

import imutil
from cgru import SpatialCGRU

print("Setting arrays to pretty-print")
np.set_printoptions(formatter={'float_kind':lambda x: "% .1f" % x})


cat = np.array(Image.open('kitten.jpg').resize((32,32)))
dog = np.array(Image.open('puppy.jpg').resize((32,32)))
def example():
    pixels = np.zeros((224, 224, 3))
    rand = lambda: np.random.randint(1, 224-32-1)
    cx, cy = rand(), rand()
    pixels[cy:cy+32, cx:cx+32] = cat
    dx, dy = rand(), rand()
    pixels[dy:dy+32, dx:dx+32] = dog

    # Target: Light up for all pixels that are ABOVE the cat AND RIGHT OF the dog
    target = up_cone(cx, cy) + right_cone(dx, dy)
    target = (target > 1).astype(np.float)
    return pixels, target


def up_cone(x, y, input_shape=(224,224), scale=7/224.):
    height, width = int(input_shape[0]*scale), int(input_shape[1]*scale)
    Y = np.zeros((height, width, 1))
    pos_y, pos_x = int(y * scale), int(x * scale)
    for i in range(pos_y, -1, -1):
        left = pos_x - (pos_y - i)
        right = pos_x + (pos_y - i) + 1
        left = max(0, left)
        Y[i, left:right, 0] = 1.0
    return Y
    

def right_cone(x, y, input_shape=(224,224), scale=7/224.):
    height, width = int(input_shape[0]*scale), int(input_shape[1]*scale)
    Y = np.zeros((height, width, 1))
    pos_y, pos_x = int(y * scale), int(x * scale)
    for i in range(pos_x, width):
        bot = pos_y - (i - pos_x)
        top = pos_y + (i - pos_x) + 1
        bot = max(0, bot)
        Y[bot:top, i, 0] = 1.0
    return Y


def map_to_img(Y, scale=224./7):
    output = np.zeros((224,224,3))
    from scipy.misc import imresize
    output[:,:,0] = imresize(Y[:,:,0], scale)
    output *= Y.max()
    return output


def build_model():
    BATCH_SIZE = 1
    IMG_WIDTH = 224
    IMG_HEIGHT = IMG_WIDTH
    IMG_CHANNELS = 3

    CRN_INPUT_SIZE = 8
    CRN_OUTPUT_SIZE = 5

    img = layers.Input(batch_shape=(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # Apply the convolutional layers of VGG16
    from keras.applications.vgg16 import VGG16
    vgg = VGG16(include_top=False)
    for layer in vgg.layers:
        layer.trainable = False

    # Convolve the image
    x = vgg(img)
    x = layers.Conv2D(CRN_INPUT_SIZE, (1,1))(x)

    # Statefully scan the image in each of four directions
    cgru = SpatialCGRU(CRN_OUTPUT_SIZE, return_sequences=True)

    r = layers.Lambda(lambda x: K.reverse(x, 1))
    t = layers.Lambda(lambda x: tf.transpose(x, [0, 2, 1, 3]))

    down_rnn = cgru(x)
    up_rnn = r(cgru(r(x)))
    left_rnn = cgru(t(x))
    right_rnn = r(cgru(r(t(x))))

    concat_out = layers.merge([left_rnn, right_rnn, up_rnn, down_rnn], mode='concat', concat_axis=-1)

    # Convolve the image some more
    output_mask = layers.Conv2D(1, (1,1), activation='sigmoid')(concat_out)

    moo = models.Model(inputs=img, outputs=output_mask)
    moo.compile(optimizer='adam', loss='mse')
    return moo


def train(model):
    X, Y = example()
    print("Input X:")
    imutil.show(X)

    print("Target Y:")
    imutil.show(map_to_img(Y))

    while True:
        for i in range(100):
            X, Y = example()
            X = np.expand_dims(X, axis=0)
            Y = np.expand_dims(Y, axis=0)
            h = model.train_on_batch(X, Y)

        preds = model.predict(X)
        print("Input:")
        imutil.show(X)
        print("Network Output:")
        imutil.show(map_to_img(preds[0]))
        print("Min {:.2f} Max {:.2f}".format(preds.min(), preds.max()))
        print("Ground Truth:")
        imutil.show(map_to_img(Y[0]))


if __name__ == '__main__':
    model = build_model()
    train(model)
