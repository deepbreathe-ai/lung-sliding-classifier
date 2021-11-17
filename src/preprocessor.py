import numpy as np
import pandas as pd
import os
import yaml
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.ops import random_ops


cfg = yaml.full_load(open(os.getcwd() + '/../config.yml', 'r'))


def random_flip_left_right_clip(x):
    '''
    Applies a left-right augmentation to a video

    :param x: Tensor of shape (Clip_length, Height, Width, 3)

    :return: A Tensor of shape (Clip_length, Height, Width, 3)
    '''

    r = random_ops.random_uniform([], 0, 1)
    if r < cfg['TRAIN']['PARAMS']['AUGMENTATION_CHANCE']:
        x = tf.image.flip_left_right(x)
    return x


def random_flip_up_down_clip(x):
    '''
    Applies an up-down augmentation to a video

    :param x: Tensor of shape (Clip_length, Height, Width, 3)

    :return: A Tensor of shape (Clip_length, Height, Width, 3)
    '''

    r = random_ops.random_uniform([], 0, 1)
    if r < cfg['TRAIN']['PARAMS']['AUGMENTATION_CHANCE']:
        x = tf.image.flip_up_down(x)
    return x


def random_rotate_clip(x):
    '''
    Randomly rotates a video

    :param x: Tensor of shape (Clip_length, Height, Width, 3)

    :return: A Tensor of shape (Clip_length, Height, Width, 3)
    '''

    r = random_ops.random_uniform([], 0, 1)
    if r < cfg['TRAIN']['PARAMS']['AUGMENTATION_CHANCE']:
        angle = random_ops.random_uniform([], cfg['TRAIN']['PARAMS']['AUGMENTATION']['ROTATE_RANGE'][0],
                                          cfg['TRAIN']['PARAMS']['AUGMENTATION']['ROTATE_RANGE'][1])
        x = tfa.image.rotate(x, angle)
    return x


def random_shift_clip(x):
    '''
    Applies a random horizontal and vertical translation to a video

    :param x: Tensor of shape (Clip_length, Height, Width, 3)

    :return: A Tensor of shape (Clip_length, Height, Width, 3)
    '''

    r = random_ops.random_uniform([], 0, 1)
    if r < cfg['TRAIN']['PARAMS']['AUGMENTATION_CHANCE']:
        h, w = cfg['PREPROCESS']['PARAMS']['IMG_SIZE'][0], cfg['PREPROCESS']['PARAMS']['IMG_SIZE'][1]  # (clip_length, h, w, channels)
        dx = random_ops.random_uniform([], cfg['TRAIN']['PARAMS']['AUGMENTATION']['SHIFT_LEFTRIGHT_BOUNDS'][0],
                                       cfg['TRAIN']['PARAMS']['AUGMENTATION']['SHIFT_LEFTRIGHT_BOUNDS'][1]) * h
        dy = random_ops.random_uniform([], cfg['TRAIN']['PARAMS']['AUGMENTATION']['SHIFT_UPDOWN_BOUNDS'][0],
                                       cfg['TRAIN']['PARAMS']['AUGMENTATION']['SHIFT_UPDOWN_BOUNDS'][1]) * w
        translations = [[dx, dy]] * cfg['PREPROCESS']['PARAMS']['WINDOW']
        x = tfa.image.translate(x, translations)
    return x


def random_shear_clip(x):
    '''
    Applies a random shear to a video

    :param x: Tensor of shape (Clip_length, Height, Width, 3)

    :return: A Tensor of shape (Clip_length, Height, Width, 3)
    '''

    r = random_ops.random_uniform([], 0, 1)
    if r < cfg['TRAIN']['PARAMS']['AUGMENTATION_CHANCE']:
        level = random_ops.random_uniform([], cfg['TRAIN']['PARAMS']['AUGMENTATION']['SHEAR_RANGE'][0],
                                          cfg['TRAIN']['PARAMS']['AUGMENTATION']['SHEAR_RANGE'][1])
        replace = tf.constant([0.0, 0.0, 0.0])
        x = tf.map_fn(lambda x1: tfa.image.shear_x(x1, level, replace), x)
        x = tf.map_fn(lambda x1: tfa.image.shear_y(x1, level, replace), x)
    return x


def random_zoom_clip(x):
    '''
    Applies a random OUTWARDS zoom to a video

    :param x: Tensor of shape (Clip_length, Height, Width, 3)

    :return: A Tensor of shape (Clip_length, Height, Width, 3)
    '''
    
    r = random_ops.random_uniform([], 0, 1)
    if r < cfg['TRAIN']['PARAMS']['AUGMENTATION_CHANCE']:
        prop = random_ops.random_uniform([], cfg['TRAIN']['PARAMS']['AUGMENTATION']['ZOOM_RANGE'][0],
                                         cfg['TRAIN']['PARAMS']['AUGMENTATION']['ZOOM_RANGE'][1])
        h_orig = cfg['PREPROCESS']['PARAMS']['IMG_SIZE'][0]
        w_orig = cfg['PREPROCESS']['PARAMS']['IMG_SIZE'][1]
        h_changed = int(prop * h_orig)
        w_changed = int(prop * w_orig)
        x = tf.map_fn(lambda x1: tf.image.pad_to_bounding_box(x1, int((h_changed-h_orig)/2), int((w_changed-w_orig)/2),
                                                              h_changed, w_changed), x)
        x = tf.map_fn(lambda x1: tf.image.resize(x1, (h_orig, w_orig)), x)
    return x


def augment_clip(x):
    '''
    Applies a series of transformations to a video

    :param x: Tensor of shape (Clip_length, Height, Width, 3)

    :return: A Tensor of shape (Clip_length, Height, Width, 3)
    '''

    x = tf.map_fn(lambda x1: tf.image.random_brightness(x1, max_delta=cfg['TRAIN']['PARAMS']['AUGMENTATION']['BRIGHTNESS_DELTA']), x)
    x = tf.map_fn(lambda x1: tf.image.random_hue(x1, max_delta=cfg['TRAIN']['PARAMS']['AUGMENTATION']['HUE_DELTA']), x)
    x = tf.map_fn(lambda x1: tf.image.random_contrast(x1, cfg['TRAIN']['PARAMS']['AUGMENTATION']['CONTRAST_BOUNDS'][0],
                                                      cfg['TRAIN']['PARAMS']['AUGMENTATION']['CONTRAST_BOUNDS'][1]), x)
    #x = tf.map_fn(lambda x1: random_shift_clip(x1), x)
    x = tf.map_fn(lambda x1: random_flip_left_right_clip(x1), x)
    x = tf.map_fn(lambda x1: random_flip_up_down_clip(x1), x)
    #x = tf.map_fn(lambda x1: random_rotate_clip(x1), x)
    #x = tf.map_fn(lambda x1: random_shear_clip(x1), x)
    #x = tf.map_fn(lambda x1: random_zoom_clip(x1), x)
    return x


def parse_fn(filename, label):
    '''
    Loads a video from its filename and returns its label

    :param filename: Path to an .npz file
    :param label: Binary label for the video

    :return: Tuple of (Loaded Video, One-hot Tensor)
    '''

    with np.load(filename, allow_pickle=True) as loaded_file:
        clip = loaded_file['frames']
    clip = tf.cast(clip, tf.float32)
    return clip, (1 - tf.one_hot(label, 1))


def parse_tf(filename, label):
    '''
    Loads a video from its filename and returns its label as proper tensors

    :param filename: Path to an .npz file
    :param label: Binary label for the video

    :return: Tuple of (Loaded Video, One-hot Tensor)
    '''

    img_size_tuple = cfg['PREPROCESS']['PARAMS']['IMG_SIZE']
    num_frames = cfg['PREPROCESS']['PARAMS']['WINDOW']
    shape = (num_frames, img_size_tuple[0], img_size_tuple[1], 3)
    clip, label = tf.numpy_function(parse_fn, [filename, label], (tf.float32, tf.float32))
    tf.ensure_shape(clip, shape)
    label.set_shape((1,))
    tf.ensure_shape(label, (1,))
    return clip, label


def parse_flow(filename, label):
    '''
    Loads a flow video (2 channels - x and y flow) from its filename and returns its label as proper tensors

    :param filename: Path to an .npz file
    :param label: Binary label for the video

    :return: Tuple of (Loaded Video, One-hot Tensor)
    '''

    img_size_tuple = cfg['PREPROCESS']['PARAMS']['IMG_SIZE']
    num_frames = cfg['PREPROCESS']['PARAMS']['WINDOW']
    shape = (num_frames, img_size_tuple[0], img_size_tuple[1], 2)
    clip, label = tf.numpy_function(parse_fn, [filename, label], (tf.float32, tf.float32))
    tf.ensure_shape(clip, shape)
    label.set_shape((1,))
    tf.ensure_shape(label, (1,))
    return clip, label


class Preprocessor:

    '''
    Preprocessor for datasets consisting of LUS clips (or mini-clips) and their corresponding labels
    '''

    def __init__(self, preprocessing_fn):
        self.batch_size = cfg['TRAIN']['PARAMS']['BATCH_SIZE']
        self.autotune = tf.data.AUTOTUNE
        self.preprocessing_fn = preprocessing_fn

    def prepare(self, ds, df, shuffle=False, augment=False):
        '''
        Defines the pipeline for loading and preprocessing each item in a TF dataset

        :param ds: The TF dataset to define the pipeline for
        :param df: The DataFrame corresponding to ds
        :param shuffle: A boolean to decide if shuffling is desired
        :param augment: A boolean to decide if augmentation is desired

        :return: A TF dataset with either preprocessed data or a full pipeline for eventual preprocessing
        '''
        
        # Shuffle the dataset
        if shuffle:
            shuffle_val = len(df)
            ds = ds.shuffle(shuffle_val)

        # Load the videos and create their labels as a one-hot vector
        ds = ds.map(parse_tf, num_parallel_calls=self.autotune)
        
        # Define batch size
        ds = ds.batch(self.batch_size, num_parallel_calls=self.autotune)

        # Optionally apply a series of augmentations
        if augment:
            ds = ds.map(lambda x, y: (augment_clip(x), y), num_parallel_calls=self.autotune)
        
        # Map the preprocessing (scaling, resizing) function to each element
        ds = ds.map(lambda x, y: (self.preprocessing_fn(x), y), num_parallel_calls=self.autotune)

        # Allows later elements to be prepared while the current element is being processed
        ds = ds.prefetch(buffer_size=self.autotune)

        return ds


class FlowPreprocessor:

    '''
    Preprocessor for datasets consisting of optical flow videos (2-channel) and their corresponding labels
    '''

    def __init__(self, preprocessing_fn):
        self.batch_size = cfg['TRAIN']['PARAMS']['BATCH_SIZE']
        self.autotune = tf.data.AUTOTUNE
        self.preprocessing_fn = preprocessing_fn

    def prepare(self, ds, df, shuffle=False, augment=False):

        '''
        Defines the pipeline for loading and preprocessing each item in a TF dataset

        :param ds: The TF dataset to define the pipeline for
        :param df: The DataFrame corresponding to ds
        :param shuffle: A boolean to decide if shuffling is desired
        :param augment: A boolean to decide if augmentation is desired

        :return: A TF dataset with either preprocessed data or a full pipeline for eventual preprocessing
        '''

        # Shuffle the dataset
        if shuffle:
            shuffle_val = len(df)
            ds = ds.shuffle(shuffle_val)

        # Load the videos and create their labels as a one-hot vector
        ds = ds.map(parse_flow, num_parallel_calls=self.autotune)

        # Define batch size
        ds = ds.batch(self.batch_size, num_parallel_calls=self.autotune)

        # Optionally apply a series of augmentations
        #if augment:
            #ds = ds.map(lambda x, y: (augment_clip(x), y), num_parallel_calls=self.autotune)

        # Map the preprocessing (scaling, resizing) function to each element
        ds = ds.map(lambda x, y: (self.preprocessing_fn(x), y), num_parallel_calls=self.autotune)

        # Allows later elements to be prepared while the current element is being processed
        ds = ds.prefetch(buffer_size=self.autotune)

        return ds
