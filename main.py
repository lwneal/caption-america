"""
Usage:
        main.py <module_name> [options]

Arguments:
        module_name: Name of the Python module to run

Options:
      --model_filename=<model>          Name of saved .h5 parameter files after each epoch. [default: default_model]
      --epochs=<epochs>                 Number of epochs to train [default: 200].
      --batches_per_epoch=<b>           Number of batches per epoch [default: 100].
      --batch_size=<size>               Batch size for training [default: 16]
      --foobar=<foobar>                 Enable foobar during training [default: False].
"""
from docopt import docopt
from pprint import pprint


def get_params():
    args = docopt(__doc__)
    return {argname(k): argval(args[k]) for k in args}


def argname(k):
    return k.strip('<').strip('>').strip('--').replace('-', '_')


def argval(val):
    if hasattr(val, 'lower') and val.lower() in ['true', 'false']:
        return val.lower().startswith('t')
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


if __name__ == '__main__':
    params = get_params()
    pprint(params)
    import train
    train.train(**params)
