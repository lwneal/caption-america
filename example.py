import os
import numpy as np
from PIL import Image
import dataset_grefexp
from util import decode_jpg, encode_jpg

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

jpg, box, text = dataset_grefexp.example()
pixels = decode_jpg(jpg, preprocess=False)
draw_box(pixels, box)
with open('/tmp/example.jpg', 'w') as fp:
    fp.write(encode_jpg(pixels))
os.system('imgcat /tmp/example.jpg')
print(text)
