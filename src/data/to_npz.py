import cv2
import numpy as np
import pandas as pd
import os
from utils import refresh_folder
import yaml

cfg = yaml.full_load(open(os.path.join(os.getcwd(),"../../config.yml"), 'r'))['PREPROCESS']


def video_to_frames_strided(path, orig_id, patient_id, df_rows, stride=cfg['PARAMS']['STRIDE'], seq_length=cfg['PARAMS']['WINDOW'], resize=cfg['PARAMS']['IMG_SIZE'], write_path=''):

  '''
  Converts a LUS video file to mini-clips with specified sequence length and stride size
  :param path: Path to video file to be converted
  :param orig_id: ID of the video file to be converted
  :param patient_id: Patient ID corresponding to the video file
  :param df_rows: list of (mini-clip_ID, patient_ID), updated in this function, and later downloaded
  :param stride: Size of stride that constitutes sequential frames in each mini-clip
  :param seq_length: Length of each mini-clip
  :param resize: [width, height], dimensions to resize frames to before saving
  :param write_path: Path to directory where output mini-clips are saved
  '''

  #TODO: IF YOU CAN DETERMINE FRAMES IN VIDEO FIRST, WE CAN CHECK VALIDITY CONDITION HERE

  cap = cv2.VideoCapture(path)
  frames = []
  for i in range(stride):
    frames.append([])

  index = 0  # Position in 'frames' array where the next frame is to be appended

  try:
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      frame = cv2.resize(frame, tuple(resize))
      frames[index].append(frame)
      index = (index + 1) % stride

  finally:
    cap.release()

  # Clip is invalid if necessary sequence length is longer than mini-clips created with given stride length
  # Last index guaranteed to have sequence of equal length or shorter by 1
  if len(frames[-1]) < seq_length:
    return

  # Split into mini-clips with given sequence length, update df rows, and write npz files
  # The id of the xth mini-clip from a main clip is the id of the main clip with _x appended to it
  counter = 1
  for set in frames:  # iterate through each 'set' of frames (which make up 1+ mini-clips)
    num_mini_clips = len(set) // seq_length  # rounded down
    for i in range(num_mini_clips):
      df_rows.append([orig_id + '_' + str(counter), patient_id])
      vid = np.array(set[i*seq_length:i*seq_length+seq_length])

      fps = 30
      size = (vid.shape[1], vid.shape[2])
      out = cv2.VideoWriter(write_path + '_' + str(counter) + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

      for i in range(len(vid)):
        out.write(vid[i])
      out.release()
      #np.savez(write_path + '_' + str(counter), frames=set[i*seq_length:i*seq_length+seq_length])
      counter += 1

  return


def video_to_frames_contig(path, orig_id, patient_id, df_rows, seq_length=cfg['PARAMS']['WINDOW'], resize=cfg['PARAMS']['IMG_SIZE'], write_path=''):

  '''
  Converts a LUS video file to contiguous-frame mini-clips with specified sequence length
  :param path: Path to video file to be converted
  :param orig_id: ID of the video file to be converted
  :param patient_id: Patient ID corresponding to the video file
  :param df_rows: list of (mini-clip_ID, patient_ID), updated in this function, and later downloaded
  :param seq_length: Length of each mini-clip
  :param resize: [width, height], dimensions to resize frames to before saving
  :param write_path: Path to directory where output mini-clips are saved
  '''

  counter = seq_length
  mini_clip_num = 1  # nth mini-clip being made from the main clip
  cap = cv2.VideoCapture(path)
  frames = []
  try:
    while True:

      # When seq_length frames have been read, update df rows and write npz files
      # The id of the xth mini-clip from a main clip is the id of the main clip with _x appended to it
      if counter == 0:
        df_rows.append([orig_id + '_' + str(mini_clip_num), patient_id])  # append to what will make output dataframes

        vid = np.array(frames)

        fps = 30
        size = (vid.shape[1], vid.shape[2])
        out = cv2.VideoWriter(write_path + '_' + str(counter) + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

        for i in range(len(vid)):
          out.write(vid[i])
        out.release()


        counter = seq_length
        mini_clip_num += 1
        frames = []

      ret, frame = cap.read()
      if not ret:
        break

      frame = cv2.resize(frame, tuple(resize))
      frames.append(frame)

      counter -= 1

  finally:
    cap.release()

  return


def video_to_npz(path, orig_id, patient_id, df_rows, write_path='', method=cfg['PARAMS']['METHOD']):

  '''
  Converts a LUS video file to mini-clips
  :param path: Path to video file to be converted
  :param orig_id: ID of the video file to be converted
  :param patient_id: Patient ID corresponding to the video file
  :param df_rows: list of (mini-clip_ID, patient_ID), updated in this function, and later downloaded
  :param write_path: Path to directory where output mini-clips are saved
  :param method: Method of frame extraction for mini-clips, either 'Contiguous' or ' Stride'
  '''

  if method == 'Contiguous':
    video_to_frames_contig(path, orig_id, patient_id, df_rows, write_path=write_path)
  else:
    video_to_frames_strided(path, orig_id, patient_id, df_rows, write_path=write_path)


# Setup directories, from config file
npz_folder = cfg['PATHS']['NPZ']
refresh_folder(npz_folder)

sliding_npz_folder = os.path.join(npz_folder, 'sliding/')
os.makedirs(sliding_npz_folder)

no_sliding_npz_folder = os.path.join(npz_folder, 'no_sliding/')
os.makedirs(no_sliding_npz_folder)

masked_folder = cfg['PATHS']['MASKED_VIDEOS']
masked_sliding_folder = os.path.join(masked_folder, 'sliding/')
masked_no_sliding_folder = os.path.join(masked_folder, 'no_sliding/')

# Each element is (mini-clip_id, patient_id) for download as csv
df_rows_sliding = []
df_rows_no_sliding = []

# Load original csvs (queried from database) - patient ids are needed from these
csv_out_folder = cfg['PATHS']['CSVS_OUTPUT']
sliding_df = pd.read_csv(os.path.join(csv_out_folder, 'sliding.csv'))
no_sliding_df = pd.read_csv(os.path.join(csv_out_folder, 'no_sliding.csv'))

# Iterate through clips and extract & download mini-clips
for file in os.listdir(masked_sliding_folder):
  f = os.path.join(masked_sliding_folder, file)
  patient_id = ((sliding_df[sliding_df['id'] == file[:-4]])['patient_id']).values[0]
  video_to_npz(f, orig_id=file[:-4], patient_id=patient_id, df_rows=df_rows_sliding, write_path=(sliding_npz_folder + file[:-4]))

for file in os.listdir(masked_no_sliding_folder):
  f = os.path.join(masked_no_sliding_folder, file)
  patient_id = ((no_sliding_df[no_sliding_df['id'] == file[:-4]])['patient_id']).values[0]
  video_to_npz(f, orig_id=file[:-4], patient_id=patient_id, df_rows=df_rows_no_sliding, write_path=(no_sliding_npz_folder + file[:-4]))

# Download dataframes linking mini-clip ids and patient ids as csv files
out_df_sliding = pd.DataFrame(df_rows_sliding, columns=['id', 'patient_id'])
csv_out_path_sliding = os.path.join(csv_out_folder, 'sliding_mini_clips.csv')
out_df_sliding.to_csv(csv_out_path_sliding, index=False)

out_df_no_sliding = pd.DataFrame(df_rows_no_sliding, columns=['id', 'patient_id'])
csv_out_path_no_sliding = os.path.join(csv_out_folder, 'no_sliding_mini_clips.csv')
out_df_no_sliding.to_csv(csv_out_path_no_sliding, index=False)
