import tensorflow as tf
import os
import numpy as np
from inspect import signature
from functools import wraps
import heapq
import itertools
import time
from concurrent.futures import ThreadPoolExecutor


def activation_function(act,act_input):
        act_func = None
        if act == "sigmoid":
            act_func = tf.nn.sigmoid(act_input)
        elif act == "tanh":
            act_func = tf.nn.tanh(act_input)
            
        elif act == "relu":
            act_func = tf.nn.relu(act_input)
        
        elif act == "elu":
            act_func = tf.nn.elu(act_input)
           
        elif act == "identity":
            act_func = tf.identity(act_input)
            
        elif act == "softmax":
            act_func = tf.nn.softmax(act_input)
         
        elif act == "selu":
            act_func = tf.nn.selu(act_input) 
        
        else:
            raise NotImplementedError("ERROR")
        return act_func  

def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)

def save_dict_to_file(dic, filename):
    f = open(filename,'w')
    f.write(str(dic))
    f.close()

def load_dict_from_file(filename):
    f = open(filename,'r')
    data=f.read()
    f.close()
    return eval(data)

def csr_to_user_dict(train_matrix):
    """convert a scipy.sparse.csr_matrix to a dict,
    where the key is row number, and value is the
    non-empty index in each row.
    """
    train_dict = {}
    for idx, value in enumerate(train_matrix):
        if len(value.indices):
            train_dict[idx] = value.indices.copy().tolist()
    return train_dict


def df_to_positive_dict_per_user(df):
    positive_dict = {}
    unique_context = df['context_id'].unique()
    for context_id in unique_context:
        df_context = df[df['context_id'] == context_id]
        positive_dict[context_id] = df_context['item_id'].tolist()
    return positive_dict


def df_to_positive_dict(df_train):
    positive_dict = {}
    for idx in range(len(df_train)):
        user_id = df_train['user_id'][idx]
        item_id = df_train['item_id'][idx]
        context_id = df_train['context_id'][idx]
        if user_id not in positive_dict:
            positive_dict[user_id] = {}
        if context_id not in positive_dict[user_id]:
            positive_dict[user_id][context_id] = [item_id]
        else:
            positive_dict[user_id][context_id].append(item_id)
    # unique_user = df_train['user_id'].unique()
    # for user in unique_user:
    #     df_user = df_train[df_train['user_id'] == user]
    #     positive_dict[user] = df_to_positive_dict_per_user(df_user)
    return positive_dict


def get_initializer(init_method, stddev):
        if init_method == 'tnormal':
            return tf.truncated_normal_initializer(stddev=stddev)
        elif init_method == 'uniform':
            return tf.random_uniform_initializer(-stddev, stddev)
        elif init_method == 'normal':
            return tf.random_normal_initializer(stddev=stddev)
        elif init_method == 'xavier_normal':
            return tf.contrib.layers.xavier_initializer(uniform=False)
        elif init_method == 'xavier_uniform':
            return tf.contrib.layers.xavier_initializer(uniform=True)
        elif init_method == 'he_normal':
            return tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode='FAN_IN', uniform=False)
        elif init_method == 'he_uniform':
            return tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode='FAN_IN', uniform=True)
        else:
            return tf.truncated_normal_initializer(stddev=stddev)  


def randint_choice(high, size=None, replace=True, p=None, exclusion=None):
    """Return random integers from `0` (inclusive) to `high` (exclusive).
    """
    a = np.arange(high)
    if exclusion is not None:
        if p is None:
            p = np.ones_like(a)
        else:
            p = np.array(p, copy=True)
        p = p.flatten()
        p[exclusion] = 0
        p = p / np.sum(p)
    sample = np.random.choice(a, size=size, replace=replace, p=p)
    return sample


def batch_random_choice(high, size, replace=True, p=None, exclusion=None):
    """Return random integers from `0` (inclusive) to `high` (exclusive).
    :param high: integer
    :param size: 1-D array_like
    :param replace: bool
    :param p: 2-D array_like
    :param exclusion: a list of 1-D array_like
    :return: a list of 1-D array_like sample
    """

    if p is not None and (len(p) != len(size) or len(p[0]) != high):
        raise ValueError("The shape of 'p' is not compatible with the shapes of 'array' and 'size'!")

    if exclusion is not None and len(exclusion) != len(size):
        raise ValueError("The shape of 'exclusion' is not compatible with the shape of 'size'!")

    def choice_one(idx):
        p_tmp = p[idx] if p is not None else None
        exc = exclusion[idx] if exclusion is not None else None
        return randint_choice(high, size[idx], replace=replace, p=p_tmp, exclusion=exc)

    with ThreadPoolExecutor() as executor:
        results = executor.map(choice_one, range(len(size)))

    return [result for result in results]


def typeassert(*type_args, **type_kwargs):
    def decorate(func):
        sig = signature(func)
        bound_types = sig.bind_partial(*type_args, **type_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_values = sig.bind(*args, **kwargs)
            for name, value in bound_values.arguments.items():
                if name in bound_types:
                    if not isinstance(value, bound_types[name]):
                        raise TypeError('Argument {} must be {}'.format(name, bound_types[name]))
            return func(*args, **kwargs)
        return wrapper
    return decorate


def argmax_top_k(a, top_k=50):
    ele_idx = heapq.nlargest(top_k, zip(a, itertools.count()))
    return np.array([idx for ele, idx in ele_idx], dtype=np.intc)


def pad_sequences(array, value=0, max_len=None, padding='post', truncating='post'):
    """padding: String, 'pre' or 'post':
            pad either before or after each sequence.
       truncating: String, 'pre' or 'post':
            remove values from sequences larger than `maxlen`,
            either at the beginning or at the end of the sequences.
    """
    array = tf.keras.preprocessing.sequence.pad_sequences(array, maxlen=max_len, value=value, dtype='int32',
                                                          padding=padding, truncating=truncating)

    return array


def inner_product(a, b, name="inner_product"):
    with tf.name_scope(name=name):
        return tf.reduce_sum(tf.multiply(a, b), axis=-1)


def timer(func):
    """The timer decorator
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print("%s function cost: %fs" % (func.__name__, end_time - start_time))
        return result
    return wrapper


def l2_loss(*params):
    return tf.add_n([tf.nn.l2_loss(w) for w in params])


def get_available_gpus(gpu_id):
    from tensorflow.python.client import device_lib as _device_lib
    local_device_protos = _device_lib.list_local_devices()
    for x in local_device_protos:
        if x.device_type == 'GPU' and gpu_id in x.name:
            return True
        else:
            return False
