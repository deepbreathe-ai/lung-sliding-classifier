import cv2
import numpy as np
import pandas as pd
import os
import yaml
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.ops import random_ops
import tensorflow_addons as tfa
import tensorflow_io as tfio

cfg = yaml.full_load(open(os.getcwd() + '/../config.yml', 'r'))


def random_flip_left_right_clip(x):
    # print(x.shape)
    r = random_ops.random_uniform_int([1], 0, 2)
    if tf.math.equal(tf.constant([1]), r):
        x = tf.image.flip_left_right(x)
    return x


def random_flip_up_down_clip(x):
    r = random_ops.random_uniform_int([1], 0, 2)
    if tf.math.equal(tf.constant([1]), r):
        x = tf.image.flip_up_down(x)
    return x


def random_rotate_clip(x):
    angle = random_ops.random_uniform([], -1.57, 1.57)
    x = tfa.image.rotate(x, angle)
    return x


def random_shift_clip(x):
    r = random_ops.random_uniform_int([1], 0, 2)
    dx = 0.0
    dy = 0.0
    if tf.math.equal(tf.constant([1]), r):
        h, w = x.shape[1], x.shape[2]  # (clip_length, h, w, channels)
        dx = random_ops.random_uniform([], -0.25, 0.25) * h
        dy = random_ops.random_uniform([], -0.0,
                                       0.2) * w  # No upwards shift since many areas of interest are near the top
    translations = [[dx, dy]] * x.shape[0]
    x = tfa.image.translate(x, translations)
    return x


def random_shear_clip(x):
  r = random_ops.random_uniform_int([1], 0, 2)
  level = 0.0
  if tf.math.equal(tf.constant([1]), r):
    level = random_ops.random_uniform([], 0.0, 0.25)  # may need adjustment
  replace = tf.constant([0.0, 0.0, 0.0])
  x = tf.map_fn(lambda x1: tfa.image.shear_x(x1, level, replace), x)
  x = tf.map_fn(lambda x1: tfa.image.shear_y(x1, level, replace), x)
  return x


class Preprocessor:

    def __init__(self, preprocess_fn):
        self.batch_size = cfg['TRAIN']['PARAMS']['BATCH_SIZE']
        self.autotune = tf.data.AUTOTUNE
        self.input_scaler = preprocess_fn

    def prepare(self, ds, shuffle=False, augment=False):

        ds = ds.map(self._parse_fn, num_parallel_calls=self.autotune)

        if shuffle:
            ds = ds.shuffle(1000)  # val should be dataset size - NEEDS ADJUSTMENT?

        ds = ds.batch(self.batch_size)

        if augment:
            ds = ds.map(lambda x, y: (augment_clip(x), y), num_parallel_calls=self.autotune)

        return ds.prefetch(buffer_size=self.autotune)

    def _parse_fn(self, filename, label):
        s = tf.io.read_file(filename)
        vid = tfio.experimental.ffmpeg.decode_video(s)
        print(vid)
        print('yup')
        return vid, tf.one_hot(label, 2)  # hardcoded as binary, can change
    

def augment_clip(x):
    print('nay')
    x = tf.cast(x, tf.float32)  # idk if needed
    x = tf.divide(x, tf.constant(255.0))  # Normalize
    x = tf.map_fn(lambda x1: tf.image.random_brightness(x1, max_delta=0.2), x)  # delta might need tuning
    x = tf.map_fn(lambda x1: tf.image.random_hue(x1, max_delta=0.5), x)  # delta might need tuning
    x = tf.map_fn(lambda x1: tf.image.random_contrast(x1, 0.5, 0.9), x)
    x = tf.map_fn(lambda x1: random_flip_left_right_clip(x1), x)
    x = tf.map_fn(lambda x1: random_flip_up_down_clip(x1), x)
    x = tf.map_fn(lambda x1: random_rotate_clip(x1), x)
    #x = tf.map_fn(lambda x1: random_shift_clip(x1), x)
    x = tf.map_fn(lambda x1: random_shear_clip(x1), x)
    print('yay')
    return x

