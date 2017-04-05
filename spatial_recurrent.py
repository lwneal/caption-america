import sys
import math
import numpy as np
from keras import layers, models
from PIL import Image
import tensorflow as tf
from keras import backend as K

import imutil
from cgru import SpatialCGRU, transpose, reverse

print("Setting arrays to pretty-print")
np.set_printoptions(formatter={'float_kind':lambda x: "% .1f" % x})

IMG_WIDTH = 320

# Level of downsampling performed by the network
SCALE = 16

cat = np.array(Image.open('kitten.jpg').resize((32,32)))
dog = np.array(Image.open('puppy.jpg').resize((32,32)))
def example():
    pixels = np.zeros((IMG_WIDTH, IMG_WIDTH, 3))
    rand = lambda: np.random.randint(1, IMG_WIDTH-32-1)
    cx, cy = rand(), rand()
    pixels[cy:cy+32, cx:cx+32] = cat
    dx, dy = rand(), rand()
    pixels[dy:dy+32, dx:dx+32] = dog

    # Easy Target: A single layer CGRU gets this right away
    # Light up the row and column centered on the cat
    target = crosshair(cx/SCALE, cy/SCALE)

    # Easy Target:
    # Light up a cone to the right of the cat
    #target = right_cone(cx, cy)

    # Medium Target: Light up for all pixels that are ABOVE the cat AND RIGHT OF the dog
    # Takes a little more training but one layer figures this out
    #target = up_cone(cx, cy) + right_cone(dx, dy)
    #target = (target > 1).astype(np.float)

    # Medium Target: Light up a fixed-radius circle around the cat
    # The only hard part here is learning to ignore the dog
    #target = circle(cx/SCALE, cy/SCALE, 4)

    # Hard Target: Light up the midway point between the cat and the dog
    # This can't be done at distance without two layers
    #target = circle((dx+cx)/2/SCALE, (dy+cy)/2/SCALE, 1)

    # Hard Target: Light up a circle around the cat BUT
    # with radius equal to the distance to the dog
    #rad = math.sqrt((dx-cx)**2 + (dy-cy)**2)
    #target = circle(cx/SCALE, cy/SCALE, rad/SCALE)
    return pixels, target


def up_cone(x, y, input_shape=(IMG_WIDTH,IMG_WIDTH), scale=1.0 / 32):
    height, width = int(input_shape[0]*scale), int(input_shape[1]*scale)
    Y = np.zeros((height, width, 1))
    pos_y, pos_x = int(y * scale), int(x * scale)
    for i in range(pos_y, -1, -1):
        left = pos_x - (pos_y - i)
        right = pos_x + (pos_y - i) + 1
        left = max(0, left)
        Y[i, left:right, 0] = 1.0
    return Y
    

def right_cone(x, y, input_shape=(IMG_WIDTH,IMG_WIDTH), scale=1.0 / 32):
    height, width = int(input_shape[0]*scale), int(input_shape[1]*scale)
    Y = np.zeros((height, width, 1))
    pos_y, pos_x = int(y * scale), int(x * scale)
    for i in range(pos_x, width):
        bot = pos_y - (i - pos_x)
        top = pos_y + (i - pos_x) + 1
        bot = max(0, bot)
        Y[bot:top, i, 0] = 1.0
    return Y


def crosshair(x, y):
    width = IMG_WIDTH / SCALE
    height = width
    Y = np.zeros((height, width, 1))
    Y[y,:] = 1.0
    Y[:,x] = 1.0
    return Y


def circle(x, y, r):
    width = IMG_WIDTH / SCALE
    height = width
    Y = np.zeros((height, width, 1))
    for t in range(628):
        yi = y + r * math.cos(t / 100.)
        xi = x + r * math.sin(t / 100.)
        if 0 <= yi < height and 0 <= xi < width:
            Y[int(yi), int(xi)] = 1.0
    return Y


def map_to_img(Y):
    output = np.zeros((IMG_WIDTH,IMG_WIDTH,3))
    from scipy.misc import imresize
    output[:,:,0] = imresize(Y[:,:,0], (IMG_WIDTH, IMG_WIDTH))
    output *= Y.max()
    return output


def build_model():
    BATCH_SIZE = 1
    IMG_HEIGHT = IMG_WIDTH
    IMG_CHANNELS = 3

    CRN_INPUT_SIZE = 8
    CRN_OUTPUT_SIZE = 8

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
    x = SpatialCGRU(x, CRN_OUTPUT_SIZE)
    x = SpatialCGRU(x, CRN_OUTPUT_SIZE)

    # Upsample and convolve
    x = layers.UpSampling2D((2,2))(x)
    x = layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

    moo = models.Model(inputs=img, outputs=x)
    moo.compile(optimizer='adam', loss='mse')
    moo.summary()
    return moo


def train(model):
    X, Y = example()
    print("Input X:")
    imutil.show(X)

    print("Target Y:")
    imutil.show(map_to_img(Y))


    if 'load' in sys.argv:
        model.load_weights('spatial_recurrent.h5')

    while True:
        for i in range(32):
            X, Y = example()
            X = np.expand_dims(X, axis=0)
            Y = np.expand_dims(Y, axis=0)
            h = model.train_on_batch(X, Y)

        preds = model.predict(X)
        print("Input:")
        imutil.show(X)
        print("Ground Truth vs. Network Output:")
        print("Min {:.2f} Max {:.2f}".format(preds.min(), preds.max()))
        print("Ground Truth:")
        imutil.show(map_to_img(Y[0]))
        print("Network Output:")
        imutil.show(X[0] + map_to_img(preds[0]))

        if 'save' in sys.argv:
            model.save_weights('spatial_recurrent.h5')


if __name__ == '__main__':
    model = build_model()
    train(model)
