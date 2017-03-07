import sys
from keras import models
from util import predict, decode_jpg


if __name__ == '__main__':
    img_filename = sys.argv[1]
    model_filename = sys.argv[2]

    pixels = decode_jpg(img_filename)
    model = models.load_model(model_filename)
    print(predict(model, pixels))
