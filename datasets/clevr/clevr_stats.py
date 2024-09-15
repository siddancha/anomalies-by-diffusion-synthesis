#!/usr/bin/env python

# The CLEVR universe contains objects of 
SHAPES    = {'cube', 'cylinder', 'sphere'}  # 3
SIZES     = {'large', 'small'}  # 2
MATERIALS = {'metal', 'rubber'}  # 2
COLORS    = {'blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow'}  # 8

import json
from pathlib import Path
from termcolor import cprint
from tqdm import tqdm
from prettytable import PrettyTable

def get_train_val_scenes() -> list:
    root_dir = Path.cwd()
    data_dir = root_dir / 'data'
    home_dir = data_dir / 'CLEVR_v1.0'

    # train scenes
    train_scenes_json = home_dir / 'scenes' / 'CLEVR_train_scenes.json'
    with open(train_scenes_json) as f:
        train_scenes = json.load(f)['scenes']
    assert len(train_scenes) == 70000
    cprint("Loaded information of 70,000 train scenes.", 'green')

    # val scenes
    val_scenes_json = home_dir / 'scenes' / 'CLEVR_val_scenes.json'
    with open(val_scenes_json) as f:
        val_scenes = json.load(f)['scenes']
    assert len(val_scenes) == 15000
    cprint("Loaded information of 15,000 validation scenes.", 'green')

    return train_scenes + val_scenes

if __name__ == '__main__':
    scenes = get_train_val_scenes()

    colors, materials, sizes, shapes, nObjects = {}, {}, {}, {}, {}

    def add_to_dict(d: dict, key: str):
        if key not in d:
            d[key] = 0
        d[key] += 1

    for scene in tqdm(scenes):
        for object in scene['objects']:
            add_to_dict(colors, object['color'])
            add_to_dict(materials, object['material'])
            add_to_dict(shapes, object['shape'])
            add_to_dict(sizes, object['size'])
            add_to_dict(nObjects, len(scene['objects']))

    def print_dict(name: str, d: dict):
        keys, items = zip(*sorted(d.items()))
        table = PrettyTable(keys)
        table.add_row(items)
        print(f'\n{name}:')
        print(table)

    print_dict('COLORS', colors)
    print_dict('MATERIALS', materials)
    print_dict('SHAPES', shapes)
    print_dict('SIZES', sizes)
    print_dict('NUM-OBJECTS', nObjects)
