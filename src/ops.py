import numpy as np
import tensorflow as tf


def tf_pad(tensor, paddings, mode):
    """
    Pads a tensor according to paddings.

    mode can be 'ZERO' or 'EDGE' (Just use tf.pad for other modes).

    'EDGE' padding is equivalent to repeatedly doing symmetric padding with all
    pads at most 1.

    Args:
        tensor (Tensor).
        paddings (list of list of non-negative ints).
        mode (str).

    Returns:
        Padded tensor.
    """
    paddings = np.array(paddings, dtype=int)
    assert np.all(paddings >= 0)
    while not np.all(paddings == 0):
        new_paddings = np.array(paddings > 0, dtype=int)
        paddings -= new_paddings
        new_paddings = tf.constant(new_paddings)
        if mode == 'ZERO':
            tensor = tf.pad(tensor, new_paddings, 'CONSTANT', constant_values=0)
        elif mode == 'EDGE':
            tensor = tf.pad(tensor, new_paddings, 'SYMMETRIC')
        else:
            raise Exception('pad type {} not recognized'.format(mode))

    return tensor