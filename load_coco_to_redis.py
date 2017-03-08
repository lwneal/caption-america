"""
There are a few tasks we need to do in O(1) time:

    1. Pick a random image from COCO2014 training, get its width/height and filename
    2. Pick a random annotation from COCO2014 validation
    3. Given an image ID, get the width/height and filename
    4. Given an annotation ID, get the annotation and its associated image
    5. Given an image ID, get all annotations associated to it

We accomplish this by storing each 'image' and 'annotation' JSON dict as a value in Redis.
A training 'reference key' contains a set of all the keys for training set images
"""
import sys
import json
import os
import redis


KEY_COCO2014_IMAGES_TRAIN = 'dataset_coco2014_images_train'
KEY_COCO2014_IMAGES_VAL = 'dataset_coco2014_images_val'
KEY_COCO2014_ANNOTATIONS_TRAIN = 'dataset_coco2014_annotations_train'
KEY_COCO2014_ANNOTATIONS_VAL = 'dataset_coco2014_annotations_val'


def load_coco_images(conn, json_filename, img_directory, reference_key):
    instances = json.load(open(json_filename))
    images = instances['images']
    for img in images:
        redis_image_key = 'coco2014_img_{}'.format(img['id'])
        value = json.dumps({
            'filename': os.path.join(img_directory, img['file_name']),
            'width': img['width'],
            'height': img['height'],
        })
        conn.set(redis_image_key, value)
        conn.sadd(reference_key, redis_image_key)
    print("Added {} images to redis: {} now contains {} items".format(len(images), reference_key, conn.scard(reference_key)))


def add_annotation_to_image(conn, image_id, annotation_id):
    img_key = 'coco2014_img_{}'.format(image_id)
    img = json.loads(conn.get(img_key))
    img['annotations'] = img.get('annotations', [])
    img['annotations'].append(annotation_id)
    # Remove duplicates to ensure this process is idempotent
    img['annotations'] = list(set(img['annotations']))
    conn.set(img_key, json.dumps(img))


def load_coco_annotations(conn, json_filename, reference_key):
    instances = json.load(open(json_filename))
    annotations = instances['annotations']
    for anno in annotations:
        redis_annotation_key = 'coco2014_anno_{}'.format(anno['id'])
        value = json.dumps({
            'image_id': anno['image_id'],
            'segmentation': anno['segmentation'],
            'area': anno['area'],
            'iscrowd': anno['iscrowd'],
            'bbox': anno['bbox'],
            'category_id': anno['category_id'],
        })
        conn.set(redis_annotation_key, value)
        conn.sadd(reference_key, redis_annotation_key)
        add_annotation_to_image(conn, anno['image_id'], anno['id'])
    print("Added {} annotations to redis: {} now contains {} items".format(len(annotations), reference_key, conn.scard(reference_key)))


def main(data_dir, conn):
    os.chdir(data_dir)
    print("Loading COCO metadata to Redis...")
    load_coco_images(conn, 'coco/annotations/instances_train2014.json', 'coco/train2014', KEY_COCO2014_IMAGES_TRAIN)
    load_coco_annotations(conn, 'coco/annotations/instances_train2014.json', KEY_COCO2014_ANNOTATIONS_TRAIN)
    load_coco_images(conn, 'coco/annotations/instances_val2014.json', 'coco/val2014', KEY_COCO2014_IMAGES_VAL)
    load_coco_annotations(conn, 'coco/annotations/instances_val2014.json', KEY_COCO2014_ANNOTATIONS_VAL)
    print("Finished loading COCO into Redis")


def test_coco_images(data_dir, conn):
    assert conn.scard(KEY_COCO2014_IMAGES_TRAIN) == 82783
    assert conn.scard(KEY_COCO2014_IMAGES_VAL) == 40504
    assert conn.scard(KEY_COCO2014_ANNOTATIONS_TRAIN) == 604907
    assert conn.scard(KEY_COCO2014_ANNOTATIONS_VAL) == 291875
    
    img_key = conn.srandmember(KEY_COCO2014_IMAGES_TRAIN)
    random_img = json.loads(conn.get(img_key))
    assert type(random_img['width']) is int
    assert random_img['height'] > 0
    filename = os.path.join(data_dir, random_img['filename'])
    assert os.path.exists(filename)

    anno_key = conn.srandmember(KEY_COCO2014_ANNOTATIONS_VAL)
    random_anno = json.loads(conn.get(anno_key))
    assert 'bbox' in random_anno
    assert 'category_id' in random_anno
    img_ref_key = 'coco2014_img_{}'.format(random_anno['image_id'])

    ref_img = json.loads(conn.get(img_ref_key))
    anno_id = int(anno_key.split('_')[-1])
    assert 'annotations' in ref_img
    assert anno_id in ref_img['annotations']

    print("Tests complete!")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: {} /path/to/datasets [--test]".format(sys.argv[0]))
        exit()
    data_dir = sys.argv[1]
    conn = redis.StrictRedis()
    if '--test' not in sys.argv:
        main(data_dir, conn)
    test_coco_images(data_dir, conn)

