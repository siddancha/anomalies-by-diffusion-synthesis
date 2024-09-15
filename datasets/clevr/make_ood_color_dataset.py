#!/usr/bin/env python

"""
The CLEVR universe contains objects of 
• Shapes    (3): ['cube', 'cylinder', 'sphere']
• Sizes     (2): ['large', 'small']
• Materials (2): ['metal', 'rubber']
• Colors    (8): ['blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow']
"""

from pathlib import Path
from tqdm import tqdm
from termcolor import cprint
from typing import List
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from datasets.clevr.clevr_stats import get_train_val_scenes, COLORS

root_dir = Path.cwd()
data_dir = root_dir / 'data'
home_dir = data_dir / 'CLEVR_v1.0'

all_scenes = get_train_val_scenes()
assert len(all_scenes) == 70000 + 15000

# Count the number of images without OOD colors
ood_colors = {'red', 'yellow', 'brown'}
ind_colors = COLORS - ood_colors
cprint(f"OOD colors: {ood_colors}", "yellow")

train_scenes, test_scenes = [], []
for scene in tqdm(all_scenes):
    objects = scene['objects']
    obj_colors = { obj['color'] for obj in objects }
    
    # scene is IND if any OOD-colored objects are present
    if len(obj_colors & ood_colors) == 0:
        train_scenes.append(scene)
    
    # add to test set if at least 2 OOD colors and 2 IND colors and at-most
    # 7 objects
    if len(obj_colors & ood_colors) >= 2 and \
       len(obj_colors & ind_colors) >= 2 and \
       len(objects) <= 6:
        test_scenes.append(scene)

print(f'Number of IND (train) scenes: {len(train_scenes)} / {len(all_scenes)}')
print(f'Number of test scenes       : {len(test_scenes)} / {len(all_scenes)}')

def make_and_show_historgam(data: List, title: str):
    nObjLabels = np.arange(3, 11)
    plt.hist(data, align='left', edgecolor = "black", bins=nObjLabels,
             weights=np.ones(len(data)) / len(data))
    plt.title(title, fontsize='xx-large')
    plt.xlabel('Number of objects', fontsize='x-large')
    plt.ylabel('Number of scenes', fontsize='x-large')
    plt.xticks(nObjLabels, fontsize='x-large')
    plt.yticks(fontsize='x-large')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.tight_layout()
    plt.show()

# histogram of number of objects in train set scenes
train_num_objs = [len(scene['objects']) for scene in train_scenes]
make_and_show_historgam(train_num_objs, 'Train set')

# histograms of number of objects in test set scenes
test_num_objs = [len(scene['objects']) for scene in test_scenes]
make_and_show_historgam(test_num_objs, 'Test set')

def make_symlink_dir(scenes: List, dest_dir: Path):
    dest_dir.mkdir(parents=True)
    for scene in tqdm(scenes):
        split, img_filename = scene['split'], scene['image_filename']
        src_img: Path = home_dir / 'images' / split / img_filename
        dest_img: Path = dest_dir / img_filename
        dest_img.symlink_to(src_img.resolve())

no_ood_color_dir = home_dir / 'ddpm_train_sets' / 'no_ood_color'
make_symlink_dir(train_scenes, no_ood_color_dir / 'train')  # train images
make_symlink_dir(test_scenes, no_ood_color_dir / 'test')  # test images
