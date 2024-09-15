#!/usr/bin/env python

import numpy as np
from pathlib import Path
from termcolor import cprint
from tqdm import tqdm
from PIL import Image as PilImage

"""
Note: must be run from the root of the repository.
"""

root_dir = Path.cwd()
data_dir = root_dir / 'data'
ddpm_train_sets_dir = data_dir / 'RUGD' / 'ddpm_train_sets'

CLASSES = ["void"] + \
    [
        'dirt', 'sand', 'grass', 'tree', 'pole', 'water', 'sky', 'vehicle',
        'container/generic-object', 'asphalt', 'gravel', 'building',
        'mulch', 'rock-bed', 'log', 'bicycle', 'person', 'fence', 'bush',
        'sign', 'rock', 'bridge', 'concrete', 'picnic-table'
    ]
OOD_CLASSES = \
    [
        'vehicle', 'container/generic-object', 'building', 'bicycle',
        'person', 'bridge', 'picnic-table'
    ]
OOD_CLASS_INDICES = [CLASSES.index(cls_name) for cls_name in OOD_CLASSES]
PALETTE = [[0, 0, 0]] + \
    [
        [108, 64, 20], [255, 229, 204], [0, 102, 0], [0, 255, 0],
        [0, 153, 153], [0, 128, 255], [0, 0, 255], [255, 255, 0],
        [255, 0, 127], [64, 64, 64], [255, 128, 0], [255, 0, 0],
        [153, 76, 0], [102, 102, 0], [102, 0, 0], [0, 255, 128],
        [204, 153, 255], [102, 0, 204], [255, 153, 204], [0, 102, 102],
        [153, 204, 255], [102, 255, 255], [101, 101, 11], [114, 85, 47]
    ]
SPLITS = dict()
SPLITS['train'] = \
    [
        'park-2', 'trail', 'trail-3', 'trail-4', 'trail-6', 'trail-9',
        'trail-10', 'trail-11', 'trail-12', 'trail-14', 'trail-15'
    ]
SPLITS['val'] = \
    [
        'park-8', 'trail-5'
    ]
SPLITS['test'] = \
    [
        'creek', 'park-1', 'trail-7', 'trail-13'
    ]
    
def convert_label_img_rgb_to_id(rgb_img):
    # set default label id to 255
    id_img = 255 * np.ones(shape=rgb_img.shape[:2], dtype=np.uint8)
    
    # match pixels to ids via their RGB values
    for id, rgb in enumerate(PALETTE):
        id_img[(rgb_img == rgb).all(axis=2)] = id
    
    # sanity check to ensure that each pixel has been matched
    assert (id_img < len(CLASSES)).all()

    return id_img

def ood_mask_img_from_label_idx_img(label_idx_img):
    """
    Pixel mask corresponding to whether each pixel is ID (False) or OOD (True).
    """
    ood_mask_img = np.isin(label_idx_img, OOD_CLASS_INDICES)  # (H, W)
    return ood_mask_img

def contains_ood_pixels(ood_mask) -> bool:
    """
    Whether the mask contains any OOD labels.
    Returns true if at least 100 pixels belong to OOD classes.
    """
    return ood_mask.sum() >= 100

def main():
    scenes = SPLITS['train'] + SPLITS['val'] + SPLITS['test']
    assert len(scenes) == 17

    for scene in scenes:
        # source folders for the scene
        src_images_dir = data_dir / 'RUGD' / 'RUGD_frames-with-annotations' / scene
        src_labels_dir = data_dir / 'RUGD' / 'RUGD_annotations' / scene

        assert src_images_dir.exists(), f'{src_images_dir} does not exist'
        assert src_labels_dir.exists(), f'{src_labels_dir} does not exist'

        print(f"\nConverting labels and copying images for [{scene}] ...")
        for src_label_file in tqdm(list(src_labels_dir.glob('*.png'))):
            png_file_name = src_label_file.name

            # convert label file from rgb to indices
            label_img_rgb = np.asarray(PilImage.open(src_label_file))
            label_img_idx = convert_label_img_rgb_to_id(label_img_rgb)

            # whether this (label, image) pair is OOD
            ood_mask_img = ood_mask_img_from_label_idx_img(label_img_idx)
            is_ood = contains_ood_pixels(ood_mask_img)

            # determine split
            if is_ood:
                split = 'ood'  # put all OOD images in one split
            elif scene in SPLITS['train']:
                split = 'id/train'
            elif scene in SPLITS['val']:
                split = 'id/val'
            elif scene in SPLITS['test']:
                split = 'id/test'
            else:
                raise AssertionError()

            # destination folders
            dst_images_dir = ddpm_train_sets_dir / 'full' / 'images' / split / scene
            # dst_labels_dir = ddpm_train_sets_dir / 'full' / 'labels' / split / scene
            dst_ood_mask_dir = ddpm_train_sets_dir / 'full' / 'masks' / split / scene
            # dst_images_dir = data_dir / 'converted_ood' / 'images' / split / scene
            # dst_labels_dir = data_dir / 'converted_ood' / 'annotations' / split / scene
            dst_images_dir.mkdir(parents=True, exist_ok=True)
            # dst_labels_dir.mkdir(parents=True, exist_ok=True)
            dst_ood_mask_dir.mkdir(parents=True, exist_ok=True)

            # copy camera image via symlink
            src_image_file = src_images_dir / png_file_name
            dst_image_file = dst_images_dir / png_file_name
            assert src_image_file.exists(), f'{src_image_file} does not exist'
            dst_image_file.symlink_to(src_image_file)

            # save label file
            # dst_pil_img = PilImage.fromarray(label_img_idx).convert('L')  # 8-bit pixels, black and white
            # dst_pil_img.save(dst_labels_dir / png_file_name)

            # save mask file
            dst_pil_img = PilImage.fromarray(ood_mask_img).convert('1')  # 1-bit pixels, grayscale
            dst_pil_img.save(dst_ood_mask_dir / png_file_name)

    cprint('Successfully converted the full dataset!', 'green', attrs=['bold'])

if __name__ == '__main__':
    main()
