import cv2
import numpy as np
import pandas as pd
import os
import yaml
from utils import refresh_folder

cfg = yaml.full_load(open(os.path.join(os.getcwd(),"../../config.yml"), 'r'))['PREPROCESS']


def video_to_frames_downsampled(orig_id, patient_id, df_rows, cap, fr, seq_length=cfg['PARAMS']['WINDOW'],
                                resize=cfg['PARAMS']['IMG_SIZE'], write_path='', base_fr=cfg['PARAMS']['BASE_FR']):
    '''
    Converts a LUS video file to mini-clips downsampled to specified FPS with specified sequence length

    :param orig_id: ID of the video file to be converted
    :param patient_id: Patient ID corresponding to the video file
    :param df_rows: list of (mini-clip_ID, patient_ID), updated in this function, and later downloaded
    :param cap: Captured video of full clip, returned by cv2.VideoCapture()
    :param fr: Frame rate (integer) of original clip - MUST be divisible by base frame rate
    :param seq_length: Length of each mini-clip
    :param resize: [width, height], dimensions to resize frames to before saving
    :param write_path: Path to directory where output mini-clips are saved
    :param base_fr: Base frame rate to downsample to
    '''

    # Check validity of frame rate param
    assert (isinstance(fr, int))
    assert (fr % base_fr == 0)

    frames = []
    stride = fr // base_fr

    index = 0  # Position in 'frames' array where the next frame is to be appended

    cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if cap_width == 0 or cap_height == 0:
        return

    new_width = None
    new_height = None

    if cap_width > cap_height:
        new_width = tuple(resize)[0]
        height_resize = int((cap_height / cap_width) * new_width)
        pad_top = int((tuple(resize)[1] - height_resize) / 2)
        pad_bottom = tuple(resize)[1] - height_resize - pad_top
        pad_left, pad_right = 0, 0
    else:
        new_height = tuple(resize)[1]
        width_resize = int((cap_width / cap_height) * new_height)
        pad_left = int((tuple(resize)[0] - width_resize) / 2)
        pad_right = tuple(resize)[0] - width_resize - pad_left
        pad_top, pad_bottom = 0, 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if index == 0:  # Add every nth frame only
                if new_width:
                    frame = cv2.resize(frame, (new_width, height_resize))
                else:
                    frame = cv2.resize(frame, (width_resize, new_height))
                frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
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


def video_to_frames_contig(orig_id, patient_id, df_rows, cap, seq_length=cfg['PARAMS']['WINDOW'],
                           resize=cfg['PARAMS']['IMG_SIZE'], write_path=''):
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

    if cap_width == 0 or cap_height == 0:
        return

    new_width = None
    new_height = None

    if cap_width > cap_height:
        new_width = tuple(resize)[0]
        height_resize = int((cap_height / cap_width) * new_width)
        pad_top = int((tuple(resize)[1] - height_resize) / 2)
        pad_bottom = tuple(resize)[1] - height_resize - pad_top
        pad_left, pad_right = 0, 0
    else:
        new_height = tuple(resize)[1]
        width_resize = int((cap_width / cap_height) * new_height)
        pad_left = int((tuple(resize)[0] - width_resize) / 2)
        pad_right = tuple(resize)[0] - width_resize - pad_left
        pad_top, pad_bottom = 0, 0

    try:
        while True:

            # When seq_length frames have been read, update df rows and write npz files
            # The id of the xth mini-clip from a main clip is the id of the main clip with _x appended to it
            if counter == 0:
                df_rows.append(
                    [orig_id + '_' + str(mini_clip_num), patient_id])  # append to what will make output dataframes
                np.savez(write_path + '_' + str(mini_clip_num), frames=frames)  # output
                counter = seq_length
                mini_clip_num += 1
                frames = []

            ret, frame = cap.read()
            if not ret:
                break

            if new_width:
                frame = cv2.resize(frame, (new_width, height_resize))
            else:
                frame = cv2.resize(frame, (width_resize, new_height))
            frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,
                                       value=[0, 0, 0])
            frames.append(frame)

            counter -= 1

    finally:
        cap.release()

    return


def flow_frames_to_npz_downsampled(path, orig_id, patient_id, df_rows, fr, seq_length=cfg['PARAMS']['WINDOW'],
                                   resize=cfg['PARAMS']['IMG_SIZE'], write_path='', base_fr=cfg['PARAMS']['BASE_FR']):
    '''
    Converts a directory of x and y flow frames to mini-clips, downsampled, with flows stacked on axis = -1

    :param path: Path to directory containing flow frames
    :param orig_id: ID of the LUS video file associated with the flow frames
    :param patient_id: Patient ID corresponding to the video file
    :param df_rows: list of (mini-clip_ID, patient_ID), updated in this function, and later downloaded
    :param fr: Frame rate (in FPS) of original LUS clip
    :param seq_length: Length of each mini-clip
    :param resize: [width, height], dimensions to resize frames to before saving
    :param write_path: Path to directory where output mini-clips are saved
    :param base_fr: Base frame rate to downsample to
    '''

    frames_x = []
    frames_y = []

    stride = fr // base_fr

    # Read all flow frames
    for file in os.listdir(path):
        frame = cv2.imread(os.path.join(path, file), 0)
        orig_width = frame.shape[1]  # height dimension should be first
        orig_height = frame.shape[0]
        if orig_width > orig_height:
            new_width = tuple(resize)[0]
            height_resize = int((orig_height / orig_width) * new_width)
            pad_top = int((tuple(resize)[1] - height_resize) / 2)
            pad_bottom = tuple(resize)[1] - height_resize - pad_top
            pad_left, pad_right = 0, 0
            frame = cv2.resize(frame, (new_width, height_resize))
        else:
            new_height = tuple(resize)[1]
            width_resize = int((orig_width / orig_height) * new_height)
            pad_left = int((tuple(resize)[0] - width_resize) / 2)
            pad_right = tuple(resize)[0] - width_resize - pad_left
            pad_top, pad_bottom = 0, 0
            frame = cv2.resize(frame, (width_resize, new_height))
        frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,
                                   value=[0, 0, 0])
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


def flow_frames_to_npz_contig(path, orig_id, patient_id, df_rows, seq_length=cfg['PARAMS']['WINDOW'],
                              resize=cfg['PARAMS']['IMG_SIZE'], write_path=''):
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
        orig_width = frame.shape[1]  # height dimension should be first
        orig_height = frame.shape[0]
        if orig_width > orig_height:
            new_width = tuple(resize)[0]
            height_resize = int((orig_height / orig_width) * new_width)
            pad_top = int((tuple(resize)[1] - height_resize) / 2)
            pad_bottom = tuple(resize)[1] - height_resize - pad_top
            pad_left, pad_right = 0, 0
            frame = cv2.resize(frame, (new_width, height_resize))
        else:
            new_height = tuple(resize)[1]
            width_resize = int((orig_width / orig_height) * new_height)
            pad_left = int((tuple(resize)[0] - width_resize) / 2)
            pad_right = tuple(resize)[0] - width_resize - pad_left
            pad_top, pad_bottom = 0, 0
            frame = cv2.resize(frame, (width_resize, new_height))
        frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,
                                   value=[0, 0, 0])
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


def video_to_npz(path, orig_id, patient_id, df_rows, write_path='', method=cfg['PARAMS']['METHOD'], fr=None,
                 base_fr=cfg['PARAMS']['BASE_FR']):
    '''
    Converts a LUS video file to mini-clips

    :param path: Path to video file to be converted
    :param orig_id: ID of the video file to be converted
    :param patient_id: Patient ID corresponding to the video file
    :param df_rows: list of (mini-clip_ID, patient_ID), updated in this function, and later downloaded
    :param write_path: Path to directory where output mini-clips are saved
    :param method: Method of frame extraction for mini-clips, either 'Contiguous' or ' Stride'
    :param base_fr: Base frame rate to downsample to
    '''

    cap = cv2.VideoCapture(path)

    if fr is None:
        fr = round(cap.get(cv2.CAP_PROP_FPS)) \
            # Disregard clips with undesired frame rate, only if frame rate not passed in (passed in = override checking)
        if not (fr % base_fr == 0):
            return

    if method == 'Contiguous':
        if fr == base_fr:
            video_to_frames_contig(orig_id, patient_id, df_rows, cap, write_path=write_path)
        else:
            video_to_frames_downsampled(orig_id, patient_id, df_rows, cap, fr, write_path=write_path)
    else:
        raise Exception('Stride method not yet implemented!')


flow = cfg['PARAMS']['FLOW']

if not (flow == 'Yes'):
    input_folder = cfg['PATHS']['EXTRA_CROPPED_VIDEOS']
    npz_folder = cfg['PATHS']['EXTRA_NPZ']
    refresh_folder(npz_folder)

if not (flow == 'No'):
    flow_input_folder = cfg['PATHS']['EXTRA_FLOW_VIDEOS']
    flow_npz_folder = cfg['PATHS']['EXTRA_FLOW_NPZ']
    refresh_folder(flow_npz_folder)

csv_out_folder = cfg['PATHS']['CSVS_OUTPUT']

# Read frame rate csv
fr_path = os.path.join(csv_out_folder, 'extra_clip_frame_rates.csv')
fr_df = pd.read_csv(fr_path)

# Each element is (mini-clip_id) for download as csv
df_rows = []
df_rows_flow = []

base_fr = cfg['PARAMS']['BASE_FR']

if not (flow == 'No'):

    for id in os.listdir(flow_input_folder):
        path = os.path.join(flow_input_folder, id)
        fr = ((fr_df[fr_df['id'] == id])['frame_rate']).values[0]
        fr = int(round(fr / base_fr) * base_fr)  # Cast to nearest multiple of base frame rate
        if fr == 0:
            fr = base_fr
        if fr == base_fr:
            flow_frames_to_npz_contig(path, orig_id=id, patient_id='N/A', df_rows=df_rows_flow,
                                      write_path=(flow_npz_folder + id))
        else:
            flow_frames_to_npz_downsampled(path, orig_id=id, patient_id='N/A', df_rows=df_rows_flow, fr=fr,
                                           write_path=(flow_npz_folder + id))

if not (flow == 'Yes'):

    for file in os.listdir(input_folder):
        f = os.path.join(input_folder, file)
        fr = ((fr_df[fr_df['id'] == file[:-4]])['frame_rate']).values[0]
        fr = int(round(fr / base_fr) * base_fr)  # Cast to nearest multiple of base frame rate
        if fr == 0:
            fr = base_fr
        video_to_npz(f, orig_id=file[:-4], patient_id='N/A', df_rows=df_rows, write_path=(npz_folder + file[:-4]),
                     fr=fr)

# Download dataframes linking mini-clip ids and patient ids as csv files
if df_rows:
    csv_out_path = os.path.join(csv_out_folder, 'extra_mini_clips.csv')
    out_df = pd.DataFrame(df_rows, columns=['id', 'patient_id'])
    out_df.to_csv(csv_out_path, index=False)
if df_rows_flow:
    csv_out_path_flow = os.path.join(csv_out_folder, 'extra_flow_mini_clips.csv')
    out_df_flow = pd.DataFrame(df_rows_flow, columns=['id', 'patient_id'])
    out_df_flow.to_csv(csv_out_path_flow, index=False)