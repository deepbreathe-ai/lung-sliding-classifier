import os
import sys
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

cfg = yaml.full_load(open(os.path.join(os.getcwd(), '../config.yml'), 'r'))


def df_splits(df, train, val, test, random_state=cfg['TRAIN']['PATHS']['RANDOM_SEED']):
    patient_ids = pd.unique(df['patient_id'])
    splits = train_test_split(patient_ids, train_size=train, random_state=random_state)
    val_test_splits = train_test_split(splits[1], train_size=(val/(val+test)), shuffle=False)
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
    return


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
sliding_path = os.path.join(csv_path, 'sliding_mini_clips.csv')
no_sliding_path = os.path.join(csv_path, 'no_sliding_mini_clips.csv')

# Input csvs of mini-clips
sliding_df = pd.read_csv(sliding_path)
sliding_df = sliding_df.sort_values(by=['patient_id'])
no_sliding_df = pd.read_csv(no_sliding_path)
no_sliding_df = no_sliding_df.sort_values(by=['patient_id'])

# Determine splits, add to previously declared dataframes
df_splits(sliding_df, train_prop, val_prop, test_prop)
df_splits(no_sliding_df, train_prop, val_prop, test_prop)

# Add label to dataframes
l1 = [1] * len(sliding_df)
sliding_df['label'] = l1
l0 = [0] * len(no_sliding_df)
no_sliding_df['label'] = l0

# Add file path to dataframes
npz_dir = cfg['PREPROCESS']['PATHS']['NPZ']

paths1 = []
for index, row in sliding_df.iterrows():
    paths1.append(os.path.join(os.getcwd(), 'data\\', npz_dir, 'sliding\\', row['id'] + '.npz'))
sliding_df['filename'] = paths1

paths0 = []
for index, row in no_sliding_df.iterrows():
    paths0.append(os.path.join(os.getcwd(), 'data\\', npz_dir, 'no_sliding\\', row['id'] + '.npz'))
no_sliding_df['filename'] = paths0

# Vertically concatenate dataframes
final_df = pd.concat([sliding_df, no_sliding_df])

# Split into train_df, val_df, test_df
train_df = final_df[final_df['split']==0]
val_df = final_df[final_df['split']==1]
test_df = final_df[final_df['split']==2]

# Write each to csv
csv_dir = cfg['TRAIN']['PATHS']['CSVS']
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

train_df_path = os.path.join(csv_dir, 'train.csv')
train_df.to_csv(train_df_path, index=False)

val_df_path = os.path.join(csv_dir, 'val.csv')
val_df.to_csv(val_df_path, index=False)

test_df_path = os.path.join(csv_dir, 'test.csv')
test_df.to_csv(test_df_path, index=False)
