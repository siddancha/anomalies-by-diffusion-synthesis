#!/usr/bin/env python

"""
Note: must be run from the root of the repository.
"""

from pathlib import Path
from termcolor import cprint

root_dir = Path.cwd()
data_dir = root_dir / 'data'
ddpm_train_sets_dir = data_dir / 'RUGD' / 'ddpm_train_sets'

# RUGD trail-15
mask_file = root_dir / 'scripts' / 'rugd_trail_15_no_building.txt'
src_dir = data_dir / 'RUGD' / 'RUGD_frames-with-annotations' / 'trail-15'
dest_dir = ddpm_train_sets_dir / 'trail_15'
dest_dir.mkdir(parents=True)
with open(mask_file, 'r') as f:
    for line in f:
        line = line.strip()
        src_img = src_dir / f'{line}.png'
        dest_img = dest_dir / f'{line}.png'
        assert src_img.exists(), f'{src_img} does not exist'
        dest_img.symlink_to(src_img)

# RUGD trail
mask_file = root_dir / 'scripts' / 'rugd_trail_no_vehicle.txt'
src_dir = data_dir / 'RUGD' / 'RUGD_frames-with-annotations' / 'trail'
dest_dir = ddpm_train_sets_dir / 'trail'
dest_dir.mkdir(parents=True)
with open(mask_file, 'r') as f:
    for line in f:
        line = line.strip()
        src_img = src_dir / f'{line}.png'
        dest_img = dest_dir / f'{line}.png'
        assert src_img.exists(), f'{src_img} does not exist'
        dest_img.symlink_to(src_img)

cprint('Successfully created datasets!', 'green', attrs=['bold'])
