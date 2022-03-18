import os
import yaml
import cv2
import numpy as np
import argparse
from tqdm import tqdm

from preprocessor import get_middle_pixel_index

cfg = yaml.full_load(open(os.path.join(os.getcwd(), "config.yml"), 'r'))

# import sys
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))
# from src.preprocessor import get_middle_pixel_index
# from ..preprocessor import get_middle_pixel_index


def get_mmode(filename):
    """
    Get M-mode image from the specified npz file
    :param filename: file name of npz file to use for M-mode reconstruction
    :return: The M-mode image produced
    """
    loaded = np.load(filename)
    bounding_box = None
    if cfg['PREPROCESS']['PARAMS']['USE_BOUNDING_BOX']:
        clip, bounding_box, height_width = loaded['frames'], loaded['bounding_box'], loaded['height_width']
    else:
        clip, height_width = loaded['frames'], loaded['height_width']

    # Extract m-mode
    num_frames, new_height, new_width = clip.shape[0], clip.shape[1], clip.shape[2]
    method = cfg['TRAIN']['M_MODE_SLICE_METHOD']
    middle_pixel = get_middle_pixel_index(clip[0], bounding_box, height_width, method=method)

    # Fix bad bounding box
    if middle_pixel == 0:
        middle_pixel = new_width // 2
    three_slice = clip[:, :, middle_pixel - 1:middle_pixel + 2, 0]
    mmode = np.median(three_slice, axis=2).T
    mmode_image = mmode.reshape((new_height, num_frames, 1)).astype('float32')

    return cv2.convertScaleAbs(cv2.cvtColor(mmode_image, cv2.COLOR_GRAY2BGR))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make m-mode images from npz files for training explainability GAN")
    parser.add_argument('output_dir', help='Folder where output m-mode images will be stored')
    parser.add_argument('npzs_dir', nargs='+',
                        help='Path to directory(s) containing npz files. Must provide at least 1 directory')
    args = parser.parse_args()

    for npz_dir in args.npzs_dir:
        print("Processing files in directory ", npz_dir, "\n")
        npz_files = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if
                     os.path.isfile(os.path.join(npz_dir, f)) and '.npz' in f]
        for f in tqdm(npz_files):
            mmode = get_mmode(f)
            out_path = os.path.join(args.output_dir, os.path.basename(os.path.splitext(f)[0]) + '.png')
            cv2.imwrite(out_path, mmode)
