import os
import sys
import json
import random

import redis
import numpy as np

DATA_DIR = '/home/nealla/data'

KEY_GREFEXP_TRAIN = 'dataset_grefexp_train'
KEY_GREFEXP_VAL = 'dataset_grefexp_val'

conn = redis.Redis()
categories = {v['id']: v['name'] for v in json.load(open('coco_categories.json'))}


def random_annotation(reference_key=KEY_GREFEXP_TRAIN):
    key = conn.srandmember(reference_key)
    return get_annotation_for_key(key)


def get_all_keys(reference_key=KEY_GREFEXP_VAL):
    keys = conn.smembers(reference_key)
    return keys


def get_annotation_for_key(key):
    grefexp = json.loads(conn.get(key))
    anno_key = 'coco2014_anno_{}'.format(grefexp['annotation_id'])
    anno = json.loads(conn.get(anno_key))
    img_key = 'coco2014_img_{}'.format(anno['image_id'])
    img_meta = json.loads(conn.get(img_key))

    jpg_data = open(os.path.join(DATA_DIR, img_meta['filename'])).read()
    return grefexp, anno, img_meta, jpg_data


def example(reference_key=KEY_GREFEXP_TRAIN):
    grefexp, anno, img_meta, jpg_data = random_annotation(reference_key)
    x0, y0, width, height = anno['bbox']
    box = (x0, x0 + width, y0, y0 + height)
    text = random.choice(grefexp['refexps'])['raw']
    category = categories[anno['category_id']]
    return jpg_data, box, text
