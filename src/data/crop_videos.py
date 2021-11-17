import time
import os
from shutil import rmtree
import tensorflow as tf
import pandas as pd
import cv2
import numpy as np

PATH_TO_SAVED_MODEL = 'exported-models/ssd_resnet50/saved_model'

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))



sliding_df, no_sliding_df = pd.read_csv('csvs/sliding.csv'), pd.read_csv('csvs/no_sliding.csv')




def refresh_folder(path):
    '''
    Ensures the folder exists and is empty
    :param path: Path to the folder of interest
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        rmtree(path)
        os.makedirs(path)



def add_boxes(df):
    ymins, xmins, ymaxs, xmaxs = [], [], [], []
    height, width = None, None 
    
    for index, row in df.iterrows():

        video_path = row['local_path']
        
        cap = cv2.VideoCapture(video_path)
        boxes = []
        
        while(cap.isOpened()):
            ret, frame = cap.read()
            
            if ret == True:
                image_np = frame
                height, width = image_np.shape[0], image_np.shape[1]
                input_tensor = tf.convert_to_tensor(image_np)
                input_tensor = input_tensor[tf.newaxis, ...]
                detections = detect_fn(input_tensor)
                box = detections['detection_boxes'][0, 0, :]
                boxes.append(box)
                
            else:
                break
        
        cap.release()
        all_boxes = np.array(boxes)
        ymin, xmin, ymax, xmax = min(all_boxes[:, 0]), min(all_boxes[:, 1]), max(all_boxes[:, 2]), max(all_boxes[:, 3])
        
        ymin, xmin, ymax, xmax = int(ymin*height), int(xmin*width), int(ymax*height), int(xmax*width)
        ymins.append(ymin)
        xmins.append(xmin)
        ymaxs.append(ymax)
        xmaxs.append(xmax)
    
    df['ymin'], df['xmin'], df['ymax'], df['xmax'] = ymins, xmins, ymaxs, xmaxs
add_boxes(sliding_df)
add_boxes(no_sliding_df)

refresh_folder('cropped_videos/')
refresh_folder('cropped_videos/sliding/')
refresh_folder('cropped_videos/no_sliding/')

new_videos = []

def crop(df):
    cropped_paths = []
    for index, row in df.iterrows():
        first_time = True
        video_path = row['local_path']
        video = []
        new_path = 'cropped_videos/' + video_path.split('/')[1] + '/' + video_path.split('/')[2]
        cropped_paths.append(new_path)
        ymin, xmin, ymax, xmax = row['ymin'], row['xmin'], row['ymax'], row['xmax']
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = None
        
        while True:
            ret, frame = cap.read()
            
            if ret == True:
                cropped_frame = frame[ymin:ymax, xmin:xmax, :]
                if first_time:
                    first_time = False
                    size = (cropped_frame.shape[1], cropped_frame.shape[0])
                    out = cv2.VideoWriter(new_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
                
                out.write(cropped_frame)
            
            else:
                break
        cap.release()
        out.release()
    df['cropped_path'] = pd.Series(cropped_paths)
    
crop(sliding_df)
crop(no_sliding_df)

no_sliding_df.to_csv('csvs/new_no_sliding.csv', index=False)
sliding_df.to_csv('csvs/new_sliding.csv', index=False)