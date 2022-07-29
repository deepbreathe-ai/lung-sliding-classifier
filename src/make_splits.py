import os
import sys
import pandas as pd
import yaml
import random
from sklearn.model_selection import train_test_split

cfg = yaml.full_load(open(os.path.join(os.getcwd(), 'config.yml'), 'r'))


def df_splits_two_stream(df, flow_df, train, val, test, random_state=cfg['TRAIN']['SPLITS']['RANDOM_SEED']):
    '''
    Splits dataframe into train, val, and test splits, selecting only clips present in both input dataframes

    :param df: Dataframe of regular mini-clip ids
    :param flow_df: Dataframe of optical flow mini-clip ids
    :param train: 0 < Float < 1 representing the training fraction
    :param val:   0 < Float < 1 representing the validation fraction
    :param test:  0 < Float < 1 representing the test fraction
    :param random_state: An integer for setting the random seed

    :return: Dataframe with split labels
    '''

    # Keep only mini-clips existing for regular frames and flow frames in df
    drop_indices = []
    for index, row in df.iterrows():
        if not (row['id'] in list(flow_df['id'])):
            drop_indices.append(index)
    df = df.drop(drop_indices)

    # Make splits from remaining mini-clips
    df = df_splits(df, train, val, test, random_state)

    return df


def df_splits(df, train, val, test, random_state=cfg['TRAIN']['SPLITS']['RANDOM_SEED']):
    '''
    Splits the dataframe into train, val and test (adds a split column with train=0, val=1, test=2).

    :param df: The sliding or no_sliding dataframe to adjust
    :param train: 0 < Float < 1 representing the training fraction
    :param val:   0 < Float < 1 representing the validation fraction
    :param test:  0 < Float < 1 representing the test fraction
    :param random_state: An integer for setting the random seed

    :return: Dataframe with split labels
    '''

    patient_ids = pd.unique(df['patient_id'])

    if random_state == -1:
        random_state = random.randint(0, 1000)

    if test == 0:
        splits = train_test_split(patient_ids, train_size=train, random_state=random_state)
        splits.append([])
    else:
        splits = train_test_split(patient_ids, train_size=train, random_state=random_state)
        val_test_splits = train_test_split(splits[1], train_size=(val / (val + test)), shuffle=False)
        splits[1] = val_test_splits[0]
        splits.append(val_test_splits[1])

    # Add split labels to dataframe
    split_labels = []
    for index, row in df.iterrows():
        curr_id = row['patient_id']
        if curr_id in splits[1]:
            split_labels.append(1)
        elif curr_id in splits[2]:
            split_labels.append(2)
        else:
            split_labels.append(0)

    df['split'] = split_labels

    return df


def minority_oversample(sliding_df, no_sliding_df, split):
    '''
    Duplicates the no_sliding_df until it's approximately the same size as sliding_df

    :param sliding_df: The dataframe storing the sliding data
    :param no_sliding_df: The dataframe storing the non-sliding data
    :param split: Int of 0=train, 1=val, 2=test

    :return: the new no_sliding df for the given split
    '''
    
    n1 = len(sliding_df[sliding_df['split']==split])
    new_no_sliding_df = no_sliding_df[no_sliding_df['split']==split].sample(n=1)
    n2 = len(new_no_sliding_df)
    while n2 < n1:
        new_no_sliding_df = pd.concat([new_no_sliding_df, no_sliding_df[no_sliding_df['split']==split].sample(n=100)])
        n2 = len(new_no_sliding_df)
    return new_no_sliding_df


# Import split params and check validity
train_prop = cfg['TRAIN']['SPLITS']['TRAIN']
val_prop = cfg['TRAIN']['SPLITS']['VAL']
test_prop = cfg['TRAIN']['SPLITS']['TEST']

s = sum([train_prop, val_prop, test_prop])
if (s > 1.0) or (train_prop < 0) or (val_prop < 0) or (test_prop < 0):
    print('Invalid splits given in config file')
    sys.exit()

# CSV paths
csv_path = os.path.join(os.getcwd(), 'data/', cfg['PREPROCESS']['PATHS']['CSVS_OUTPUT'])
if cfg['PREPROCESS']['PATHS']['CSVS_OUTPUT'].split('\\')[0] == 'generalizability':
    csv_path = os.path.join(os.getcwd(), cfg['PREPROCESS']['PATHS']['CSVS_OUTPUT'])

flow = cfg['PREPROCESS']['PARAMS']['FLOW']

sliding_path = ''
no_sliding_path = ''
flow_sliding_path = ''
flow_no_sliding_path = ''

if not (flow == 'No'):
    flow_sliding_path = os.path.join(csv_path, 'sliding_flow_mini_clips.csv')
    flow_no_sliding_path = os.path.join(csv_path, 'no_sliding_flow_mini_clips.csv')
if not (flow == 'Yes'):
    sliding_path = os.path.join(csv_path, 'sliding_mini_clips.csv')
    no_sliding_path = os.path.join(csv_path, 'no_sliding_mini_clips.csv')

# Input csvs of mini-clips
if sliding_path:
    sliding_df = pd.read_csv(sliding_path)
    sliding_df = sliding_df.sort_values(by=['patient_id'])
    no_sliding_df = pd.read_csv(no_sliding_path)
    no_sliding_df = no_sliding_df.sort_values(by=['patient_id'])
if flow_sliding_path:
    flow_sliding_df = pd.read_csv(flow_sliding_path)
    flow_sliding_df = flow_sliding_df.sort_values(by=['patient_id'])
    flow_no_sliding_df = pd.read_csv(flow_no_sliding_path)
    flow_no_sliding_df = flow_no_sliding_df.sort_values(by=['patient_id'])

# Determine splits, add to previously declared dataframes
# Only keep 'sliding_df' and 'no_sliding_df' variables past this (flow df var names not needed)
if flow == 'No':
    sliding_df = df_splits(sliding_df, train_prop, val_prop, test_prop)
    no_sliding_df = df_splits(no_sliding_df, train_prop, val_prop, test_prop)
elif flow == 'Yes':
    sliding_df = df_splits(flow_sliding_df, train_prop, val_prop, test_prop)
    no_sliding_df = df_splits(flow_no_sliding_df, train_prop, val_prop, test_prop)
else:
    sliding_df = df_splits_two_stream(sliding_df, flow_sliding_df, train_prop, val_prop, test_prop)
    no_sliding_df = df_splits_two_stream(no_sliding_df, flow_no_sliding_df, train_prop, val_prop, test_prop)
    
# Add label to dataframes.
l1 = [1] * len(sliding_df)
sliding_df['label'] = l1
l0 = [0] * len(no_sliding_df)
no_sliding_df['label'] = l0

# Add file path to dataframes
npz_dir = ''
if flow == 'Yes':
    npz_dir = cfg['PREPROCESS']['PATHS']['FLOW_NPZ']
else:
    npz_dir = cfg['PREPROCESS']['PATHS']['NPZ']

paths1 = []
for index, row in sliding_df.iterrows():
    paths1.append(os.path.join(os.getcwd(), 'data/', npz_dir, 'sliding/', row['id'] + '.npz'))
sliding_df['filename'] = paths1

paths0 = []
for index, row in no_sliding_df.iterrows():
    paths0.append(os.path.join(os.getcwd(), 'data/', npz_dir, 'no_sliding/', row['id'] + '.npz'))
no_sliding_df['filename'] = paths0

# Add flow paths if two-stream
if flow == 'Both':

    npz_dir = cfg['PREPROCESS']['PATHS']['FLOW_NPZ']

    paths1 = []
    for index, row in sliding_df.iterrows():
        paths1.append(os.path.join(os.getcwd(), 'data/', npz_dir, 'sliding/', row['id'] + '.npz'))
    sliding_df['flow_filename'] = paths1

    paths0 = []
    for index, row in no_sliding_df.iterrows():
        paths0.append(os.path.join(os.getcwd(), 'data/', npz_dir, 'no_sliding/', row['id'] + '.npz'))
    no_sliding_df['flow_filename'] = paths0

# Duplicate the no_sliding_df until it's approximately the same size as sliding_df, if desired
increase = cfg['TRAIN']['PARAMS']['INCREASE']
if increase:
    new_no_sliding_train_df = minority_oversample(sliding_df, no_sliding_df, 0)
    no_sliding_df = pd.concat([no_sliding_df[no_sliding_df['split'] != 0], new_no_sliding_train_df])

# Print the proportion of each split
print('Train: Sliding=={}, No Sliding=={}'.format(len(sliding_df[sliding_df['split']==0]), len(no_sliding_df[no_sliding_df['split']==0])))
print('Val: Sliding=={}, No Sliding=={}'.format(len(sliding_df[sliding_df['split']==1]), len(no_sliding_df[no_sliding_df['split']==1])))
print('Test: Sliding=={}, No Sliding=={}'.format(len(sliding_df[sliding_df['split']==2]), len(no_sliding_df[no_sliding_df['split']==2])))

# Vertically concatenate dataframes
final_df = pd.concat([sliding_df, no_sliding_df])

# Shuffle
final_df = final_df.sample(frac=1, random_state=2)

# Split into train_df, val_df, test_df
train_df = final_df[final_df['split']==0]
val_df = final_df[final_df['split']==1]
test_df = final_df[final_df['split']==2]

# Write each to csv
csv_dir = os.getcwd() + cfg['TRAIN']['PATHS']['CSVS']
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

train_df_path = ''
val_df_path = ''
test_df_path = ''

if flow == 'Yes':
    train_df_path = os.path.join(csv_dir, 'flow_train.csv')
    val_df_path = os.path.join(csv_dir, 'flow_val.csv')
    test_df_path = os.path.join(csv_dir, 'flow_test.csv')
else:
    train_df_path = os.path.join(csv_dir, 'train.csv')
    val_df_path = os.path.join(csv_dir, 'val.csv')
    test_df_path = os.path.join(csv_dir, 'test.csv')

train_df.to_csv(train_df_path, index=False)
val_df.to_csv(val_df_path, index=False)
test_df.to_csv(test_df_path, index=False)
