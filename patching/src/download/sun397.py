import glob
import os
from distutils.dir_util import mkpath
from distutils.file_util import copy_file
import logging
import pandas as pd
# Define the root directory for the splits
root_dir = "sun397"
os.makedirs(root_dir, exist_ok=True)
# Define the subdirectories for train, test, and valid
splits = ['Training', 'Testing']
label_counters = {}
imagefolder = 'D:/Source/patching/data/SUN397'
labelfolder = 'D:/Source/patching/data/SUN397'
for split in splits:
    # Iterate over each entry in the dataset split
    splitlist = glob.glob(f'{labelfolder}/{split}*')

    for entry in splitlist:
        imagelist = pd.read_csv(entry, names=['filepath'])
        for idx, row in imagelist.iterrows():
            imagepath = row['filepath']
            source_path = f'{imagefolder}/{imagepath}'.replace('/', os.sep)
            target_path = f'{root_dir}/{split}/{imagepath}'.replace('/', os.sep)
            mkpath(os.path.dirname(target_path))
            if os.path.exists(source_path):
                copy_file(source_path, os.path.dirname(target_path))
            else:
                logging.info(f'not exist : {source_path}')