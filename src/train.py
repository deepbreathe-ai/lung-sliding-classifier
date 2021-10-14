'''
Script for training experiments, including model training, hyperparameter search, cross validation, etc
'''

import yaml
import os
import pandas as pd
import tensorflow as tf
from preprocessor import Preprocessor

cfg = yaml.full_load(open(os.path.join(os.getcwd(), '../config.yml'), 'r'))['TRAIN']


def train_model(model_def, hparams, preprocessing_fn=(lambda x: x), save_weights=False, log_dir=None, verbose=True):
    # Read in train, val, test dataframes
    CSVS_FOLDER = cfg['PATHS']['CSVS']
    train_df = pd.read_csv(os.path.join(CSVS_FOLDER, 'train.csv'))
    val_df = pd.read_csv(os.path.join(CSVS_FOLDER, 'val.csv'))
    test_df = pd.read_csv(os.path.join(CSVS_FOLDER, 'test.csv'))

    # Create TF datasets for training, validation and test sets
    train_set = tf.data.Dataset.from_tensor_slices((train_df['filename'].tolist(), train_df['label']))
    val_set = tf.data.Dataset.from_tensor_slices((val_df['filename'].tolist(), val_df['label']))
    test_set = tf.data.Dataset.from_tensor_slices((test_df['filename'].tolist(), test_df['label']))

    preprocessor = Preprocessor(preprocessing_fn)

    train_set = preprocessor.prepare(train_set, shuffle=True, augment=True)
    val_set = preprocessor.prepare(val_set, shuffle=False, augment=False)
    test_set = preprocessor.prepare(test_set, shuffle=False, augment=False)

    print(train_set.element_spec)
    print(val_set.element_spec)
    print(test_set.element_spec)

train_model(None, None)
