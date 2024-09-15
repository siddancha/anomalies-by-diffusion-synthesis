#!/usr/bin/env python

import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image as PilImage

"""
The dataset folder should be structured as follows:

data/RELLIS
├── train.lst
├── val.lst
├── test.lst
├── 00000
      ├── pylon_camera_node/             -- directory containing ".jpg" files from the color camera.
      ├── pylon_camera_node_label_id/    -- directory containing ".png" label id images, labelled with class ids from 0 to 34.
"""

# Note: must be run from the root of the repository.
root_dir = Path.cwd()
data_dir = root_dir / 'data'
ddpm_train_sets_dir = data_dir / 'RELLIS' / 'ddpm_train_sets'

# This is based on the ontology and label-mapping provided in RELLIS' "ontology.yaml"
# https://drive.google.com/file/d/1K8Zf0ju_xI5lnx3NTDLJpVTs59wmGPI6/view
LABEL_MAP =   {
    0: 0,    # void
    1: 1,    # dirt
    3: 2,    # grass
    4: 3,    # tree
    5: 4,    # pole
    6: 5,    # water
    7: 6,    # sky
    8: 7,    # vehicle
    9: 8,    # object
    10: 9,   # asphalt
    12: 10,  # building
    15: 11,   # log
    17: 12,   # person
    18: 13,   # fence
    19: 14,   # bush
    23: 15,  # concrete
    27: 16,  # barrier
    31: 17,  # puddle
    33: 18,  # mud
    34: 19,  # rubble
}

CLASSES = ["void"] + \
    [
        'dirt', 'grass', 'tree', 'pole', 'water', 'sky', 'vehicle', 'object',
        'asphalt', 'building', 'log', 'person', 'fence', 'bush', 'concrete',
        'barrier', 'puddle', 'mud', 'rubble'
    ]
OOD_CLASSES = \
    [
        'vehicle', 'person', 'barrier', 'rubble'
    ]
OOD_CLASS_INDICES = [CLASSES.index(cls_name) for cls_name in OOD_CLASSES]

def remap_label_ids(label_id_img):
    # set default label id to 255
    mapped_id_img = 255 * np.ones(label_id_img.shape, dtype=np.uint8)
    
    # map label ids
    for k, v in LABEL_MAP.items():
        mapped_id_img[label_id_img == k] = v
    
    # sanity check to ensure that each pixel has been matched
    assert (mapped_id_img != 255).all()
    
    return mapped_id_img

class FileID:
    def __init__(self, file_id):
        file_id = file_id[5:]  # remove the fixed word "frame"
        self.frame_id, stamp = file_id.split('-')  # get frame id
        self.sec, self.msec = stamp.split('_')

    def __eq__(self, other):
        return self.frame_id == other.frame_id and \
               self.sec  == other.sec and \
               self.msec == other.msec

    def __str__(self):
        return f"frame{self.frame_id}-{self.sec}_{self.msec}"

def read_split_file(split):
    data_root = Path("data/RELLIS")
    scene_file_ids = []

    def get_scene_id_file_id(text):
        scene_id, _, file_name = text.split('/')
        file_id = FileID(file_name.split('.')[0])
        return scene_id, file_id
    
    with open(data_root / f"{split}.lst", 'r') as f:
        for line in f.readlines():
            filename1, filename2 = line.split()
            scene_id, file_id = get_scene_id_file_id(filename1)
            assert (scene_id, file_id) == get_scene_id_file_id(filename2)
            scene_file_ids.append((scene_id, file_id))

    return scene_file_ids

def ood_mask_img_from_label_idx_img(label_idx_img):
    """
    Pixel mask corresponding to whether each pixel is ID (False) or OOD (True).
    """
    ood_mask_img = np.isin(label_idx_img, OOD_CLASS_INDICES)  # (H, W)
    return ood_mask_img

def contains_ood_pixels(ood_mask) -> bool:
    """
    Whether the mask contains any OOD labels.
    Returns true if at least a fraction of pixels belong to OOD classes.
    """
    return ood_mask.mean() >= 0.00025

def main():
    scenes = ['00000', '00001', '00002', '00003', '00004']
    assert len(scenes) == 5

    stats = dict()

    for scene in scenes:
        stats[scene] = dict(id=0, ood=0)

        # source folders for the scene
        src_images_dir = data_dir / 'RELLIS' / scene / 'pylon_camera_node'
        src_labels_dir = data_dir / 'RELLIS' / scene / 'pylon_camera_node_label_id'

        assert src_images_dir.exists(), f'{src_images_dir} does not exist'
        assert src_labels_dir.exists(), f'{src_labels_dir} does not exist'

        print(f"\nConverting labels and copying images for scene [{scene}] ...")
        for src_label_file in tqdm(list(src_labels_dir.glob('*.png'))):
            png_file_name = src_label_file.name
            jpg_file_name = png_file_name.replace('.png', '.jpg')

            # convert label file from rgb to indices
            label_img_idx = np.asarray(PilImage.open(src_label_file))
            label_img_idx = remap_label_ids(label_img_idx)

            # whether this (label, image) pair is OOD
            ood_mask_img = ood_mask_img_from_label_idx_img(label_img_idx)
            is_ood = contains_ood_pixels(ood_mask_img)

            # determine split
            if is_ood:
                split = 'ood'  # put all OOD images in one split
                stats[scene]['ood'] += 1
            else:
                split = 'id'  # put all ID images in one split
                stats[scene]['id'] += 1

            # destination folders
            dst_images_dir = ddpm_train_sets_dir / 'full' / 'images' / split / scene
            dst_ood_mask_dir = ddpm_train_sets_dir / 'full' / 'masks' / split / scene
            dst_images_dir.mkdir(parents=True, exist_ok=True)
            dst_ood_mask_dir.mkdir(parents=True, exist_ok=True)

            # copy camera image via symlink
            src_image_file = src_images_dir / jpg_file_name
            dst_image_file = dst_images_dir / jpg_file_name
            assert src_image_file.exists(), f'{src_image_file} does not exist'
            dst_image_file.symlink_to(src_image_file)

            # save label file
            # dst_pil_img = PilImage.fromarray(label_img_idx).convert('L')  # 8-bit pixels, black and white
            # dst_pil_img.save(dst_labels_dir / png_file_name)

            # save mask file
            dst_pil_img = PilImage.fromarray(ood_mask_img).convert('1')  # 1-bit pixels, grayscale
            dst_pil_img.save(dst_ood_mask_dir / png_file_name)


    # aggregate and print statistics
    stats['all'] = {
        'id': sum([stats[scene]['id'] for scene in scenes]),
        'ood': sum([stats[scene]['ood'] for scene in scenes]),
    }
    print(stats)


if __name__ == '__main__':
    main()
