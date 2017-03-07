import redis
import json
import os
import sys

KEY_GREFEXP_TRAIN = 'dataset_grefexp_train'
KEY_GREFEXP_VAL = 'dataset_grefexp_val'

def load_refexp_to_redis(conn, refexp_file, reference_key):
    grefexp = json.load(open(refexp_file))

    # First create a lookup table to refer to refexps by id
    refexps = {}
    for r in grefexp['refexps']:
        key = r['refexp_id']
        refexps[key] = {
            'tokens': r['tokens'],
            'parse': r['parse'],
            'raw': r['raw'],
        }

    # Now save a json dict in Redis for each annotation
    for a in grefexp['annotations']:
        key = 'grefexp_{}'.format(a['annotation_id'])
        value = json.dumps({
            'annotation_id': a['annotation_id'],
            'region_candidates': a['region_candidates'],
            'refexps': [refexps[i] for i in a['refexp_ids']],
        })
        conn.set(key, value)
        conn.sadd(reference_key, key)
    count = len(grefexp['annotations'])
    print("Uploaded {} annotations: {} now contains {} items".format(count, reference_key, conn.scard(reference_key)))


def main(data_dir, conn):
    os.chdir(data_dir)
    # The gRefExp training dataset consists of 44822 annotations
    # We save each annotation in Redis as a json dict
    load_refexp_to_redis(conn, 'grefexp/google_refexp_train_201511.json', KEY_GREFEXP_TRAIN)
    load_refexp_to_redis(conn, 'grefexp/google_refexp_val_201511.json', KEY_GREFEXP_VAL)

    
def test_grefexp(conn):
    # We should be able to select a COCO2014 train image at random
    assert conn.type(KEY_GREFEXP_TRAIN) == 'set'
    # TODO: Reconcile the 2 annotations with overlapping IDs
    #assert conn.scard(KEY_GREFEXP_TRAIN) == 44822
    assert conn.scard(KEY_GREFEXP_TRAIN) == 44820
    random_key = conn.srandmember(KEY_GREFEXP_TRAIN)
    assert conn.get(random_key)
    grefexp_anno = json.loads(conn.get(random_key))

    # Each grefexp annotation should point to a valid COCO annotation
    assert 'annotation_id' in grefexp_anno
    coco_anno = conn.get('coco2014_anno_{}'.format(grefexp_anno['annotation_id']))
    print("Test successful")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: {} /path/to/datasets".format(sys.argv[0]))
        exit()
    data_dir = sys.argv[1]
    conn = redis.StrictRedis()
    main(data_dir, conn)
    test_grefexp(conn)
