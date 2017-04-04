# Image Utils
# Fun interactive utilities for image manipulation
import math
import os
import tempfile
from distutils import spawn
import numpy as np
from PIL import Image
from StringIO import StringIO


# Swiss army knife for image decoding
def decode_jpg(jpg, box=None, crop_to_box=None, preprocess=True):
    if jpg.startswith('\xFF\xD8'):
        # jpg is a JPG buffer
        img = Image.open(StringIO(jpg))
    else:
        # jpg is a filename
        img = Image.open(jpg)
    img = img.convert('RGB')
    width = img.width
    height = img.height
    if crop_to_box:
        # Crop to bounding box
        x0, x1, y0, y1 = crop_to_box
        img = img.crop((x0,y0,x1,y1))
    if preprocess:
        img = img.resize((224, 224))
    pixels = np.array(img).astype(float)
    if preprocess:
        pixels = imagenet_process(pixels)
    if box:
        # Transform a bounding box after resizing
        x0, x1, y0, y1 = box
        xs = float(pixels.shape[1]) / width
        ys = float(pixels.shape[0]) / height
        x0 *= xs
        x1 *= xs
        y0 *= ys
        y1 *= ys
        return pixels, (x0, x1, y0, y1)
    return pixels


def encode_jpg(pixels):
    img = Image.fromarray(pixels.astype(np.uint8)).convert('RGB')
    fp = StringIO()
    img.save(fp, format='JPEG')
    return fp.getvalue()


def imagenet_process(x):
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    # 'RGB'->'BGR'
    return x[:, :, ::-1]


# Swiss-army knife for putting an image on the screen
# Accepts numpy arrays, PIL Image objects, or jpgs
# Numpy arrays can consist of multiple images, which will be collated
def show(data, video=None, box=None):
    if type(data) == type(np.array([])):
        pixels = data
    elif type(data) == Image.Image:
        pixels = np.array(data)
    else:
        pixels = decode_jpg(data, preprocess=False)
    if box:
        draw_box(pixels, box)

    if len(pixels.shape) > 3:
        pixels = combine_images(pixels)

    # Display image in the terminal if imgcat is available
    if spawn.find_executable('imgcat'):
        with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp:
            #tmp.write(encode_jpg(pixels))
            #os.system('imgcat {}'.format(tmp.name))
            open('/tmp/foobar.jpg', 'w').write(encode_jpg(pixels))
            os.system('imgcat /tmp/foobar.jpg')

    # Output JPG files can be collected into a video with ffmpeg -i *.jpg
    if video:
        open(video, 'a').write(encode_jpg(pixels))


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:]
    image = np.zeros((height*shape[0], width*shape[1], shape[2]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        a0, a1 = i*shape[0], (i+1)*shape[0]
        b0, b1 = j*shape[1], (j+1)*shape[1]
        image[a0:a1, b0:b1] = img
    return image


def draw_box(img, box, color=1.0):
    x0, x1, y0, y1 = (int(val) for val in box)
    height, width, channels = img.shape
    x0 = np.clip(x0, 0, width-1)
    x1 = np.clip(x1, 0, width-1)
    y0 = np.clip(y0, 0, height-1)
    y1 = np.clip(y1, 0, height-1)
    img[y0:y1,x0] = color
    img[y0:y1,x1] = color
    img[y0,x0:x1] = color
    img[y1,x0:x1] = color
