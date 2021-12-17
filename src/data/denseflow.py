# Largely sourced from https://github.com/qijiezhao/py-denseflow

import os
import numpy as np
import cv2
import argparse
import imageio
import yaml
from PIL import Image
from multiprocessing import Pool

videos_root = ""
data_root = ""
new_dir = ""

num_vals = 0  # Count total number of individual flow pixels (x or y, not both) across all videos
sum_x = 0
sum_y = 0
mean_x = 0
mean_y = 0
squared_diff_sum_x = 0
squared_diff_sum_y = 0
std_x = 0
std_y = 0

cfg = yaml.full_load(open(os.path.join(os.getcwd(), "../../config.yml"), 'r'))['PREPROCESS']


def to_img(raw_flow, bound):
    '''
    Scales input pixels to 0-255 within given bounds
    :param raw_flow: input raw pixel value (not in 0-255)
    :param bound: upper and lower bound (-bound, bound)
    :return: pixel value scale from 0 to 255
    '''

    flow = raw_flow
    flow[flow > bound] = bound
    flow[flow < -bound] = -bound
    flow -= -bound
    flow *= (255 / float(2 * bound))
    return flow


def save_flows(flows, save_dir, num, bound):
    '''
    Saves optical flow images
    :param flows: contains flow_x and flow_y
    :param save_dir: save_dir name (always equal to the video id)
    :param num: the save id, which belongs one of the extracted frames
    :param bound: set the bounds to flow images
    '''

    global mean_x, mean_y, std_x, std_y

    # Project to standard Gaussian
    flow_x = (flows[..., 0] - mean_x) / std_x
    flow_y = (flows[..., 1] - mean_y) / std_y

    # rescale to 0~255 with the bound setting
    flow_x = to_img(flow_x, bound)
    flow_y = to_img(flow_y, bound)

    new_dir = save_dir[:save_dir.index('/') + 1]
    save_dir = save_dir[save_dir.index('/') + 1:]

    if not os.path.exists(os.path.join(data_root, new_dir, save_dir)):
        os.makedirs(os.path.join(data_root, new_dir, save_dir))

    # save the flows
    save_x = os.path.join(data_root, new_dir, save_dir, 'flow_x_{:05d}.jpg'.format(num))
    save_y = os.path.join(data_root, new_dir, save_dir, 'flow_y_{:05d}.jpg'.format(num))
    flow_x = flow_x.astype(np.uint8)
    flow_y = flow_y.astype(np.uint8)
    flow_x_img = Image.fromarray(flow_x)
    flow_y_img = Image.fromarray(flow_y)
    imageio.imwrite(save_x, flow_x_img)
    imageio.imwrite(save_y, flow_y_img)


def dense_flow(args, run_num):
    '''
    Extracts dense_flow images from specified video
    :param args: the detailed arguments:
        video_root: Root directory where the video is held
        video_name: Name of video for which flow is to be extracted
        save_dir: Destination path's final directory
        step: Number of frames between each two extracted frames
        bound: Upper and lower bounds
        crop: Boolean, whether or not cropping to pleural line was used
    '''

    global num_vals, sum_x, sum_y, mean_x, mean_y, squared_diff_sum_x, squared_diff_sum_y

    video_root, video_name, save_dir, step, bound, crop = args
    video_path = os.path.join(video_root, video_name)

    print(video_path)

    try:
        videocapture = cv2.VideoCapture(video_path)
    except Exception as e:
        print('{} read error! {}'.format(video_name, e))
        return

    len_frame = videocapture.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_num = 0
    image, prev_image, gray, prev_gray = None, None, None, None
    num0 = 0

    if crop:
        cap_width = videocapture.get(cv2.CAP_PROP_FRAME_WIDTH)
        cap_height = videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        new_width = cfg['PARAMS']['FLOW_CROP_WIDTH']  # HAVE TO MAKE THIS ONLY RUN IF CROP IS TRUE
        height_resize = int((cap_height / cap_width) * new_width)

    while True:
        frame = videocapture.read()[1]
        if num0 >= len_frame:
            break
        num0 += 1
        if frame_num == 0:
            prev_image = frame
            if crop:
                prev_image = cv2.resize(prev_image, (new_width, height_resize))
            prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_RGB2GRAY)
            frame_num += 1

            # to pass the out of stepped frames
            step_t = step
            while step_t > 1:
                videocapture.read()
                num0 += 1
                step_t -= 1
            continue

        image = frame
        if crop:
            image = cv2.resize(image, (new_width, height_resize))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        frame_0 = prev_gray
        frame_1 = gray

        dtvl1 = cv2.createOptFlow_DualTVL1()  # default choose the tvl1 algorithm
        flowDTVL1 = dtvl1.calc(frame_0, frame_1, None)

        if run_num == 1:  # Update sum for mean calculation
            num_vals += flowDTVL1.shape[0] * flowDTVL1.shape[1]
            sum_x += np.sum(flowDTVL1[..., 0])
            sum_y += np.sum(flowDTVL1[..., 1])
        elif run_num == 2:  # Update squared point/mean differences for STD calculation
            squared_diff_sum_x += np.sum((flowDTVL1[..., 0] - mean_x) ** 2)
            squared_diff_sum_y += np.sum((flowDTVL1[..., 1] - mean_y) ** 2)
        else:  # Save standardized flows
            if 'no_sliding' in video_path:
                changed_save_dir = os.path.join('no_sliding/', save_dir)
            else:
                changed_save_dir = os.path.join('sliding/', save_dir)
            save_flows(flowDTVL1, changed_save_dir, frame_num, bound)  # save flow frames

        prev_gray = gray
        frame_num += 1

        # to pass the out of stepped frames
        step_t = step
        while step_t > 1:
            videocapture.read()
            num0 += 1
            step_t -= 1


def get_video_list(video_root, subfolder):
    '''
    Retrieves sorted list of video names from videos_root directory
    :return: Tuple of (sorted video name list, number of videos in list)
    '''

    video_list = []
    for video_ in os.listdir(video_root):
        video_list.append(os.path.join(subfolder, video_))  # add sliding/no sliding before vid name
    video_list.sort()
    return video_list, len(video_list)


def parse_args():
    '''
    Parses command line arguments
    :return: Parsed arguments
    '''

    parser = argparse.ArgumentParser(description="densely extract the video frames and optical flows")
    parser.add_argument('--videos_root', default='', type=str)
    parser.add_argument('--data_root', default='/n/zqj/video_classification/data', type=str)
    # parser.add_argument('--new_dir', default='flows', type=str)
    parser.add_argument('--crop', default='False', type=str)
    parser.add_argument('--num_workers', default=4, type=int, help='num of workers to act multi-process')
    parser.add_argument('--step', default=1, type=int, help='gap frames')
    parser.add_argument('--bound', default=3.5, type=float, help='set the maximum of optical flow')
    parser.add_argument('--s_', default=0, type=int, help='start id')
    parser.add_argument('--e_', default=13320, type=int, help='end id')
    parser.add_argument('--mode', default='run', type=str, help='set \'run\' if debug done, otherwise, set debug')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    # specify the augments
    videos_root = args.videos_root
    data_root = args.data_root
    num_workers = args.num_workers
    step = args.step
    bound = args.bound
    s_ = args.s_
    e_ = args.e_
    # new_dir = args.new_dir
    mode = args.mode
    crop = True if (args.crop == 'True') else False

    video_root = args.videos_root  # Cropped or masked clip root folder
    root1 = os.path.join(video_root, 'sliding/')
    root2 = os.path.join(video_root, 'no_sliding/')

    # get video list
    list1, len1 = get_video_list(root1, 'sliding/')
    list2, len2 = get_video_list(root2, 'no_sliding/')
    video_list = list1 + list2
    len_videos = len1 + len2

    print('find {} videos.'.format(len_videos))
    flows_dirs = [video.split('.')[0][video.split('.')[0].index('/') + 1:] for video in
                  video_list]  # Names of videos, to become folders for flow frames
    print('get videos list done! ')

    # First run - get mean of all values (separate for x and y)
    print('Getting mean')
    for i in range(len_videos):
        dense_flow((video_root, video_list[i], flows_dirs[i], step, bound, crop), 1)

    mean_x = sum_x / num_vals
    mean_y = sum_y / num_vals

    print('Mean x: ' + str(mean_x))
    print('Mean y: ' + str(mean_y))

    # Second run - get standard deviation of all values (separate for x and y)
    print('Getting standard deviation')
    for i in range(len_videos):
        dense_flow((video_root, video_list[i], flows_dirs[i], step, bound, crop), 2)

    std_x = (squared_diff_sum_x / num_vals) ** 0.5
    std_y = (squared_diff_sum_y / num_vals) ** 0.5

    print('Std x: ' + str(std_x))
    print('Std y: ' + str(std_y))

    # Third run - scale flow frames by normalizing (standard Gaussian)
    for i in range(len_videos):
        dense_flow((video_root, video_list[i], flows_dirs[i], step, bound, crop), 3)

    '''
    pool = Pool(num_workers)
    if mode == 'run':
        for i in range(len(video_list)):
            dense_flow((video_root, video_list[i], flows_dirs[i], step, bound))
        #pool.map(dense_flow, zip(video_root, video_list, flows_dirs, [step]*len(video_list), [bound]*len(video_list)))
    else:  #mode=='debug
        dense_flow((videos_root, video_list[0], flows_dirs[0], step, bound))
    '''