import yaml
import os
from utils import refresh_folder

# Load dictionary of constants stored in config.yml
cfg = yaml.full_load(open(os.path.join(os.getcwd(), "../../config.yml"), 'r'))['PREPROCESS']

# Obtain input folder paths where the masked videos are stored
masked_folder = cfg['PATHS']['MASKED_VIDEOS']
masked_sliding_folder = os.path.join(masked_folder, 'sliding/')
masked_no_sliding_folder = os.path.join(masked_folder, 'no_sliding/')

# These input folders need to exist for this to work, so we terminate if they don't
if (not os.path.exists(masked_folder)) or (not os.path.exists(masked_sliding_folder)) or (not os.path.exists(masked_no_sliding_folder)):
    raise NotADirectoryError('You need to specify a folder for masked clips in the config file!')

# Ensure all contents in the folder for outputting the flow clips are deleted, but that the folder structure exist
flow_folder = cfg['PATHS']['FLOW_VIDEOS']
refresh_folder(flow_folder)

flow_sliding_folder = os.path.join(flow_folder, 'sliding/')
os.makedirs(flow_sliding_folder)

flow_no_sliding_folder = os.path.join(flow_folder, 'no_sliding/')
os.makedirs(flow_no_sliding_folder)

# Call optical flow tool
os.system('python denseflow.py --videos_root=' + masked_folder + ' --data_root=' + flow_folder + ' --num_workers=1')
