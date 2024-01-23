import glob
import os
from distutils.dir_util import mkpath
from distutils.file_util import move_file, copy_file

import pandas as pd
from PIL import Image
from datasets import load_dataset
# Define the root directory for the splits
root_dir = "dtd"
os.makedirs(root_dir, exist_ok=True)
# Define the subdirectories for train, test, and valid
splits = ['train', 'test', 'val']
label_counters = {}
imagefolder = 'D:/Source/patching/data/dtd/images'
labelfolder = 'D:/Source/patching/data/dtd/labels'
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
                copy_file(source_path, os.path.dirname(target_path) )
            else:
                print(f'not exist : {source_path}')