import sys
import math
import numpy as np
from keras import layers, models
from PIL import Image
import tensorflow as tf
from keras import backend as K

import imutil
from cgru import SpatialCGRU, transpose, reverse
from visualizer import Visualizer

print("Setting arrays to pretty-print")
np.set_printoptions(formatter={'float_kind':lambda x: "% .1f" % x})

IMG_WIDTH = 320

# Level of downsampling performed by the network
SCALE = 16

# Output RGB
CHANNELS = 3

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
    #target = crosshair(cx/SCALE, cy/SCALE, color=0)

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

    # Hard Target: Line from cat to dog
    # This can't be done at distance without two layers
    target = line(dx/SCALE, dy/SCALE, cx/SCALE, cy/SCALE, color=0)

    # Hard Target: Light up the midway point between the cat and the dog
    #target = circle((dx+cx)/2/SCALE, (dy+cy)/2/SCALE, 1)

    # Hard Target: Light up a circle around the cat BUT
    # with radius equal to the distance to the dog
    #rad = math.sqrt((dx-cx)**2 + (dy-cy)**2)
    #target = circle(cx/SCALE, cy/SCALE, rad/SCALE)

    # For fun, ALSO draw a blue circle around the cat
    target += circle(cx/SCALE, cy/SCALE, 4, color=2)
    target = np.clip(target, 0, 1)

    # Add a little epsilon to stave off dead gradient
    #target += .05

    # Gaussian blur to smooth the gradient
    #from scipy.ndimage.filters import gaussian_filter
    #target = gaussian_filter(target, sigma=.5)

    return pixels, target


def up_cone(x, y, input_shape=(IMG_WIDTH,IMG_WIDTH), scale=1.0 / 32):
    height, width = int(input_shape[0]*scale), int(input_shape[1]*scale)
    Y = np.zeros((height, width, CHANNELS))
    pos_y, pos_x = int(y * scale), int(x * scale)
    for i in range(pos_y, -1, -1):
        left = pos_x - (pos_y - i)
        right = pos_x + (pos_y - i) + 1
        left = max(0, left)
        Y[i, left:right, 0] = 1.0
    return Y
    

def right_cone(x, y, input_shape=(IMG_WIDTH,IMG_WIDTH), scale=1.0 / 32):
    height, width = int(input_shape[0]*scale), int(input_shape[1]*scale)
    Y = np.zeros((height, width, CHANNELS))
    pos_y, pos_x = int(y * scale), int(x * scale)
    for i in range(pos_x, width):
        bot = pos_y - (i - pos_x)
        top = pos_y + (i - pos_x) + 1
        bot = max(0, bot)
        Y[bot:top, i, 0] = 1.0
    return Y


def crosshair(x, y, color=0):
    width = IMG_WIDTH / SCALE
    height = width
    Y = np.zeros((height, width, CHANNELS))
    Y[y,:, color] = 1.0
    Y[:,x, color] = 1.0
    return Y


def circle(x, y, r, color=0):
    width = IMG_WIDTH / SCALE
    height = width
    Y = np.zeros((height, width, CHANNELS))
    for t in range(628):
        yi = y + r * math.cos(t / 100.)
        xi = x + r * math.sin(t / 100.)
        if 0 <= yi < height and 0 <= xi < width:
            Y[int(yi), int(xi), color] = 1.0
    return Y


def line(x0, y0, x1, y1, color=0):
    width = IMG_WIDTH / SCALE
    height = width
    Y = np.zeros((height, width, CHANNELS))
    for t in range(100):
        yi = y0 + (t / 100.) * (y1 - y0)
        xi = x0 + (t / 100.) * (x1 - x0)
        Y[int(yi), int(xi), color] = 1.0
    return Y


def map_to_img(Y):
    output = np.zeros((IMG_WIDTH,IMG_WIDTH,3))
    from scipy.misc import imresize
    output[:,:] = imresize(Y[:,:], (IMG_WIDTH, IMG_WIDTH))
    output *= Y.max()
    return output


BATCH_SIZE = 16

def build_model():
    IMG_HEIGHT = IMG_WIDTH
    IMG_CHANNELS = 3

    CRN_INPUT_SIZE = 8
    CRN_OUTPUT_SIZE = 64

    img = layers.Input(batch_shape=(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # Apply the convolutional layers of VGG16
    from keras.applications.vgg16 import VGG16
    vgg = VGG16(include_top=False)
    for layer in vgg.layers:
        layer.trainable = False

    # Run a pretrained network
    x = vgg(img)

    # Upsample

    # Statefully scan the image in each of four directions
    x = SpatialCGRU(x, CRN_OUTPUT_SIZE)
    # Do spatial CGRU layers work kinda like convolutional layers?
    x = SpatialCGRU(x, CRN_OUTPUT_SIZE)

    # Upsample and convolve
    x = layers.UpSampling2D((2,2))(x)

    # Output an RGB image
    x = layers.Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)

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
            examples = [example() for _ in range(BATCH_SIZE)]
            batch_X, batch_Y = map(np.array, zip(*examples))
            h = model.train_on_batch(np.array(batch_X), np.array(batch_Y))

        preds = model.predict(batch_X)[-1]
        X = batch_X[-1]
        Y = batch_Y[-1]
        print("Input:")
        imutil.show(X)
        print("Ground Truth vs. Network Output:")
        print("Min {:.2f} Max {:.2f}".format(preds.min(), preds.max()))
        print("Ground Truth:")
        imutil.show(map_to_img(Y))
        print("Network Output:")
        imutil.show(X + map_to_img(preds))

        if 'save' in sys.argv:
            model.save_weights('spatial_recurrent.h5')


if __name__ == '__main__':
    model = build_model()
    train(model)
