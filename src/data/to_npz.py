import cv2
import numpy as np
import pandas as pd
import os
import yaml
from utils import refresh_folder

cfg = yaml.full_load(open(os.path.join(os.getcwd(),"../../config.yml"), 'r'))['PREPROCESS']


# CURRENTLY NOT UPDATED (Since we are doing the downsampling for 60 FPS videos)
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
      np.savez(write_path + '_' + str(counter), frames=set[i*seq_length:i*seq_length+seq_length])
      counter += 1

  return


def video_to_frames_downsampled(orig_id, patient_id, df_rows, cap, fr, seq_length=cfg['PARAMS']['WINDOW'], resize=cfg['PARAMS']['IMG_SIZE'], write_path=''):

  '''
  Converts a LUS video file to mini-clips downsampled to 30 FPS with specified sequence length

  :param orig_id: ID of the video file to be converted
  :param patient_id: Patient ID corresponding to the video file
  :param df_rows: list of (mini-clip_ID, patient_ID), updated in this function, and later downloaded
  :param cap: Captured video of full clip, returned by cv2.VideoCapture()
  :param fr: Frame rate (integer) of original clip - MUST be divisible by 30
  :param seq_length: Length of each mini-clip
  :param resize: [width, height], dimensions to resize frames to before saving
  :param write_path: Path to directory where output mini-clips are saved
  '''

  # Check validity of frame rate param
  assert(isinstance(fr, int))
  assert(fr % 30 == 0)

  frames = []
  stride = fr // 30

  index = 0  # Position in 'frames' array where the next frame is to be appended

  cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  new_width = tuple(resize)[0]
  height_resize = int((cap_height / cap_width) * new_width)
  pad_top = int((tuple(resize)[1] - height_resize) / 2)
  pad_bottom = tuple(resize)[1] - height_resize - pad_top

  try:
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      if index == 0:  # Add every nth frame only
        frame = cv2.resize(frame, (new_width, height_resize))
        frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        frames.append(frame)
      index = (index + 1) % stride

  finally:
    cap.release()

  # Assemble and save mini-clips from extracted frames
  counter = 1
  num_mini_clips = len(frames) // seq_length
  for i in range(num_mini_clips):
    df_rows.append([orig_id + '_' + str(counter), patient_id])
    np.savez(write_path + '_' + str(counter), frames=frames[i * seq_length:i * seq_length + seq_length])
    counter += 1

  return


def video_to_frames_contig(orig_id, patient_id, df_rows, cap, seq_length=cfg['PARAMS']['WINDOW'], resize=cfg['PARAMS']['IMG_SIZE'], write_path=''):

  '''
  Converts a LUS video file to contiguous-frame mini-clips with specified sequence length

  :param orig_id: ID of the video file to be converted
  :param patient_id: Patient ID corresponding to the video file
  :param df_rows: list of (mini-clip_ID, patient_ID), updated in this function, and later downloaded
  :param cap: Captured video of full clip, returned by cv2.VideoCapture()
  :param seq_length: Length of each mini-clip
  :param resize: [width, height], dimensions to resize frames to before saving
  :param write_path: Path to directory where output mini-clips are saved
  '''

  counter = seq_length
  mini_clip_num = 1  # nth mini-clip being made from the main clip
  frames = []

  cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  new_width = tuple(resize)[0]
  height_resize = int((cap_height / cap_width) * new_width)
  pad_top = int((tuple(resize)[1] - height_resize) / 2)
  pad_bottom = tuple(resize)[1] - height_resize - pad_top

  try:
    while True:

      # When seq_length frames have been read, update df rows and write npz files
      # The id of the xth mini-clip from a main clip is the id of the main clip with _x appended to it
      if counter == 0:
        df_rows.append([orig_id + '_' + str(mini_clip_num), patient_id])  # append to what will make output dataframes
        np.savez(write_path + '_' + str(mini_clip_num), frames=frames)  # output
        counter = seq_length
        mini_clip_num += 1
        frames = []

      ret, frame = cap.read()
      if not ret:
        break

      frame = cv2.resize(frame, (new_width, height_resize))
      frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
      frames.append(frame)

      counter -= 1

  finally:
    cap.release()

  return


def flow_frames_to_npz_downsampled(path, orig_id, patient_id, df_rows, fr, seq_length=cfg['PARAMS']['WINDOW'], resize=cfg['PARAMS']['IMG_SIZE'], write_path=''):

  '''
  Converts a directory of x and y flow frames to mini-clips downsampled to 30 FPS, with flows stacked on axis = -1

  :param path: Path to directory containing flow frames
  :param orig_id: ID of the LUS video file associated with the flow frames
  :param patient_id: Patient ID corresponding to the video file
  :param df_rows: list of (mini-clip_ID, patient_ID), updated in this function, and later downloaded
  :param fr: Frame rate (in FPS) of original LUS clip
  :param seq_length: Length of each mini-clip
  :param resize: [width, height], dimensions to resize frames to before saving
  :param write_path: Path to directory where output mini-clips are saved
  '''

  frames_x = []
  frames_y = []

  stride = fr // 30

  # Read all flow frames
  for file in os.listdir(path):
    frame = cv2.imread(os.path.join(path, file), 0)
    frame = cv2.resize(frame, tuple(resize))
    ind = int(file[7:12])  # flow frame number
    if (ind - 1) % stride == 0:  # take every nth flow frame only
      if '_x_' in file:
        frames_x.append(frame)
      else:
        frames_y.append(frame)

  counter = 1
  num_mini_clips = len(frames_x) // seq_length

  # Stack x and y flow (making 2 channels) and save mini-clip sequences
  for i in range(num_mini_clips):
    df_rows.append([orig_id + '_' + str(counter), patient_id])
    x_seq = np.array(frames_x[i * seq_length:i * seq_length + seq_length])
    y_seq = np.array(frames_y[i * seq_length:i * seq_length + seq_length])
    np.savez(write_path + '_' + str(counter), frames=np.stack((x_seq, y_seq), axis=-1))
    counter += 1

  return


def flow_frames_to_npz_contig(path, orig_id, patient_id, df_rows, seq_length=cfg['PARAMS']['WINDOW'], resize=cfg['PARAMS']['IMG_SIZE'], write_path=''):

  '''
  Converts a directory of x and y flow frames to contiguous mini-clips, with flows stacked on axis = -1

  :param path: Path to directory containing flow frames
  :param orig_id: ID of the LUS video file associated with the flow frames
  :param patient_id: Patient ID corresponding to the video file
  :param df_rows: list of (mini-clip_ID, patient_ID), updated in this function, and later downloaded
  :param seq_length: Length of each mini-clip
  :param resize: [width, height], dimensions to resize frames to before saving
  :param write_path: Path to directory where output mini-clips are saved
  '''

  frames_x = []
  frames_y = []

  # Read all flow frames
  for file in os.listdir(path):
    frame = cv2.imread(os.path.join(path, file), 0)
    frame = cv2.resize(frame, tuple(resize))
    if '_x_' in file:
      frames_x.append(frame)
    else:
      frames_y.append(frame)

  counter = 1
  num_mini_clips = len(frames_x) // seq_length

  # Stack x and y flow (making 2 channels) and save mini-clip sequences
  for i in range(num_mini_clips):
    df_rows.append([orig_id + '_' + str(counter), patient_id])
    x_seq = np.array(frames_x[i * seq_length:i * seq_length + seq_length])
    y_seq = np.array(frames_y[i * seq_length:i * seq_length + seq_length])
    np.savez(write_path + '_' + str(counter), frames=np.stack((x_seq, y_seq), axis=-1))
    counter += 1

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

  flow = cfg['PARAMS']['FLOW']
  
  cap = cv2.VideoCapture(path)
  fr = round(cap.get(cv2.CAP_PROP_FPS))

  # Disregard clips with undesired frame rate
  if not (fr % 30 == 0):
    return

  if method == 'Contiguous':
    if fr == 30:
      video_to_frames_contig(orig_id, patient_id, df_rows, cap, write_path=write_path)
    else:
      video_to_frames_downsampled(orig_id, patient_id, df_rows, cap, fr, write_path=write_path)
  else:
    raise Exception('Stride method not yet implemented!')


flow = cfg['PARAMS']['FLOW']

# Setup directories, from config file
masked_folder = cfg['PATHS']['MASKED_VIDEOS']
masked_sliding_folder = os.path.join(masked_folder, 'sliding/')
masked_no_sliding_folder = os.path.join(masked_folder, 'no_sliding/')

npz_folder = ''
sliding_npz_folder = ''
no_sliding_npz_folder = ''

if not (flow == 'Yes'):
  npz_folder = cfg['PATHS']['NPZ']
  refresh_folder(npz_folder)

  sliding_npz_folder = os.path.join(npz_folder, 'sliding/')
  os.makedirs(sliding_npz_folder)

  no_sliding_npz_folder = os.path.join(npz_folder, 'no_sliding/')
  os.makedirs(no_sliding_npz_folder)

flow_folder = ''
flow_sliding_folder = ''
flow_no_sliding_folder = ''

flow_npz_folder = ''
flow_sliding_npz_folder = ''
flow_no_sliding_npz_folder = ''

if not (flow == 'No'):
  flow_folder = cfg['PATHS']['FLOW_VIDEOS']
  flow_sliding_folder = os.path.join(flow_folder, 'sliding/')
  flow_no_sliding_folder = os.path.join(flow_folder, 'no_sliding/')

  flow_npz_folder = cfg['PATHS']['FLOW_NPZ']
  refresh_folder(flow_npz_folder)

  flow_sliding_npz_folder = os.path.join(flow_npz_folder, 'sliding/')
  os.makedirs(flow_sliding_npz_folder)

  flow_no_sliding_npz_folder = os.path.join(flow_npz_folder, 'no_sliding/')
  os.makedirs(flow_no_sliding_npz_folder)

# Each element is (mini-clip_id, patient_id) for download as csv
df_rows_sliding = []
df_rows_no_sliding = []

# Load original csvs (queried from database) - patient ids are needed from these
csv_out_folder = cfg['PATHS']['CSVS_OUTPUT']
sliding_df = pd.read_csv(os.path.join(csv_out_folder, 'sliding.csv'))
no_sliding_df = pd.read_csv(os.path.join(csv_out_folder, 'no_sliding.csv'))

sliding_fps_df = pd.read_csv(os.path.join(csv_out_folder, 'sliding_frame_rates.csv'))
no_sliding_fps_df = pd.read_csv(os.path.join(csv_out_folder, 'no_sliding_frame_rates.csv'))

# Iterate through clips and extract & download mini-clips

if flow == 'Yes':

  for id in os.listdir(flow_sliding_folder):
    path = os.path.join(flow_sliding_folder, id)
    patient_id = ((sliding_df[sliding_df['id'] == id])['patient_id']).values[0]
    fr = ((sliding_fps_df[sliding_fps_df['id'] == id])['frame_rate']).values[0]
    if fr == 30:
      flow_frames_to_npz_contig(path, orig_id=id, patient_id=patient_id, df_rows=df_rows_sliding,
                                write_path=(flow_sliding_npz_folder + id))
    else:
      flow_frames_to_npz_downsampled(path, orig_id=id, patient_id=patient_id, df_rows=df_rows_sliding, fr=fr,
                                     write_path=(flow_sliding_npz_folder + id))

  for id in os.listdir(flow_no_sliding_folder):
    path = os.path.join(flow_no_sliding_folder, id)
    patient_id = ((no_sliding_df[no_sliding_df['id'] == id])['patient_id']).values[0]
    fr = ((no_sliding_fps_df[no_sliding_fps_df['id'] == id])['frame_rate']).values[0]
    if fr == 30:
      flow_frames_to_npz_contig(path, orig_id=id, patient_id=patient_id, df_rows=df_rows_no_sliding,
                                write_path=(flow_no_sliding_npz_folder + id))
    else:
      flow_frames_to_npz_downsampled(path, orig_id=id, patient_id=patient_id, df_rows=df_rows_no_sliding, fr=fr,
                                     write_path=(flow_no_sliding_npz_folder + id))

elif flow == 'No':

  for file in os.listdir(masked_sliding_folder):
    f = os.path.join(masked_sliding_folder, file)
    patient_id = ((sliding_df[sliding_df['id'] == file[:-4]])['patient_id']).values[0]
    video_to_npz(f, orig_id=file[:-4], patient_id=patient_id, df_rows=df_rows_sliding,
                 write_path=(sliding_npz_folder + file[:-4]))

  for file in os.listdir(masked_no_sliding_folder):
    f = os.path.join(masked_no_sliding_folder, file)
    patient_id = ((no_sliding_df[no_sliding_df['id'] == file[:-4]])['patient_id']).values[0]
    video_to_npz(f, orig_id=file[:-4], patient_id=patient_id, df_rows=df_rows_no_sliding,
                 write_path=(no_sliding_npz_folder + file[:-4]))

else:

  raise Exception('Two-stream preprocessing pipeline not yet implemented!')

# Download dataframes linking mini-clip ids and patient ids as csv files
out_df_sliding = pd.DataFrame(df_rows_sliding, columns=['id', 'patient_id'])
out_df_no_sliding = pd.DataFrame(df_rows_no_sliding, columns=['id', 'patient_id'])

csv_out_path_sliding = ''
csv_out_path_no_sliding = ''

if flow == 'Yes':
  csv_out_path_sliding = os.path.join(csv_out_folder, 'sliding_flow_mini_clips.csv')
  csv_out_path_no_sliding = os.path.join(csv_out_folder, 'no_sliding_flow_mini_clips.csv')
elif flow == 'No':
  csv_out_path_sliding = os.path.join(csv_out_folder, 'sliding_mini_clips.csv')
  csv_out_path_no_sliding = os.path.join(csv_out_folder, 'no_sliding_mini_clips.csv')
else:
  raise Exception('Two-stream preprocessing pipeline not yet implemented!')

out_df_sliding.to_csv(csv_out_path_sliding, index=False)
out_df_no_sliding.to_csv(csv_out_path_no_sliding, index=False)
