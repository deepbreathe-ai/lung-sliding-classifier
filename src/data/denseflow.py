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

cfg = yaml.full_load(open(os.path.join(os.getcwd(),"../../config.yml"), 'r'))['PREPROCESS']

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
    flow *= (255/float(2*bound))
    return flow


def save_flows(flows, save_dir, num, bound):

    '''
    Saves optical flow images

    :param flows: contains flow_x and flow_y
    :param save_dir: save_dir name (always equal to the video id)
    :param num: the save id, which belongs one of the extracted frames
    :param bound: set the bounds to flow images
    '''

    # rescale to 0~255 with the bound setting
    flow_x = to_img(flows[..., 0], bound)
    flow_y = to_img(flows[..., 1], bound)
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


def dense_flow(args):

    '''
    Extracts dense_flow images from specified video

    :param args: the detailed arguments:
        video_name: Name of video for which flow is to be extracted
        save_dir: Destination path's final directory
        step: Number of frames between each two extracted frames
        bound: Upper and lower bounds
    '''

    video_root, video_name, save_dir, step, bound = args
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

    cap_width = videocapture.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap_height = videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    new_width = cfg['PARAMS']['FLOW_CROP_WIDTH']
    height_resize = int((cap_height / cap_width) * new_width)
    
    while True:
        frame = videocapture.read()[1]
        if num0 >= len_frame:
            break
        num0 += 1
        if frame_num == 0:
            prev_image = frame
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
        image = cv2.resize(image, (new_width, height_resize))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        frame_0 = prev_gray
        frame_1 = gray
        
        dtvl1 = cv2.createOptFlow_DualTVL1()  # default choose the tvl1 algorithm
        flowDTVL1 = dtvl1.calc(frame_0, frame_1, None)
        save_flows(flowDTVL1, save_dir, frame_num, bound)  # save flow frames
        
        prev_gray = gray
        frame_num += 1

        # to pass the out of stepped frames
        step_t = step
        while step_t > 1:
            videocapture.read()
            num0 += 1
            step_t -= 1


def get_video_list():

    '''
    Retrieves sorted list of video names from videos_root directory

    :return: Tuple of (sorted video name list, number of videos in list)
    '''

    video_list = []
    for video_ in os.listdir(videos_root):
        video_list.append(video_)
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
    parser.add_argument('--new_dir', default='flows', type=str)
    parser.add_argument('--num_workers', default=4, type=int, help='num of workers to act multi-process')
    parser.add_argument('--step', default=1, type=int, help='gap frames')
    parser.add_argument('--bound', default=15, type=int, help='set the maximum of optical flow')
    parser.add_argument('--s_', default=0, type=int, help='start id')
    parser.add_argument('--e_', default=13320, type=int, help='end id')
    parser.add_argument('--mode', default='run', type=str, help='set \'run\' if debug done, otherwise, set debug')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    #specify the augments
    videos_root = args.videos_root
    data_root = args.data_root
    num_workers = args.num_workers
    step = args.step
    bound = args.bound
    s_ = args.s_
    e_ = args.e_
    new_dir = args.new_dir
    mode = args.mode

    #get video list
    video_list, len_videos = get_video_list()

    print('find {} videos.'.format(len_videos))
    flows_dirs = [video.split('.')[0] for video in video_list]
    print('get videos list done! ')

    video_root = args.videos_root

    pool = Pool(num_workers)
    if mode == 'run':
        for i in range(len(video_list)):
            dense_flow((video_root, video_list[i], flows_dirs[i], step, bound))
        #pool.map(dense_flow, zip(video_root, video_list, flows_dirs, [step]*len(video_list), [bound]*len(video_list)))
    else:  #mode=='debug
        dense_flow((videos_root, video_list[0], flows_dirs[0], step, bound))
