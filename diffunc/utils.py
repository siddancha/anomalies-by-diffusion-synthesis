import numpy as np
from functools import partial, cached_property
import gdown
import json
from pathlib import Path
import PIL.Image
import random
from termcolor import cprint
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torchvision import transforms as T

from denoising_diffusion_pytorch.denoising_diffusion_pytorch import convert_image_to_fn

def set_random_seed(seed: int):
    # https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_input_img(path: Path,
                   image_size: Optional[Tuple[int, int]]=None,
                   convert_image_to: str='RGB') \
    -> Tensor:
    """
    Returns unnormalized image with intensities in [0, 1]
    
    Args:
        path (Path): path to input image
        image_size ([Tuple[int, int]], optional): image size as (H, W).
        convert_image_to (str, optional): format to convert PIL image to.
    
    Returns:
        (Tensor, shape=(1, 3, H, W)): image with intensities in [0, 1].
    """
    maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) \
        if convert_image_to is not None else nn.Identity()
    
    # input image transformation
    transform = T.Compose([
        T.Lambda(maybe_convert_fn),
        T.Resize(image_size) if image_size is not None else nn.Identity(),
        T.ToTensor()
    ])

    # load input image
    img = PIL.Image.open(path)
    img: Tensor = transform(img)  # (3, H, W)
    img = img.unsqueeze(0)  # (1, 3, H, W)
    img = img.to(torch.cuda.current_device())  # (1, 3, H, W)
    return img

def download_from_cloud(cloud_path: str) -> Path:
    home = Path(__file__).parent.parent / 'cloud'
    with open(home / 'files.json', 'r') as f:
        file_bank = json.load(f)
    
    if cloud_path not in file_bank:
        raise ValueError(f'\'{cloud_path}\' is not available on cloud storage. Please check cloud/files.json for available files.')

    # convert to real path
    disk_path: Path = home / cloud_path

    if disk_path.exists():
        print(f'File {cloud_path} already downloaded.')
    else:
        disk_path.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(url=file_bank[cloud_path],
                       output=str(disk_path),
                       quiet=False,
                       fuzzy=True)

    return disk_path

def get_mit_logo_square(size: Optional[int]=32) -> PIL.Image.Image:
    """
    Squared and resized version of the MIT logo, optimized for 32x32 resolution.
    """
    path = Path(__file__).parent / 'media/mit_logo.png'
    logo = PIL.Image.open(path).convert("RGB")
    arr = np.asarray(logo)

    # reduce spacing between the logo to get it in a 1:1 aspect ratio
    keep_rows = [
                    (0,   181),
    ]
    keep_cols = [
                    (  0,  10),  # whitespace
                    ( 35,  55),  # M1
                    (  0,   8),  # whitespace
                    ( 70,  90),  # M2
                    (  0,   8),  # whitespace
                    ( 35,  55),  # M3
                    (  0,   9),  # whitespace
                    (148, 168),  # I
                    (  0,   9),  # whitespace
                    (185, 205),  # T1
                    (206, 234),  # T2
                    (   0,  9),  # whitespace
    ]
    keep_rows = sum([list(range(s, e)) for (s, e) in keep_rows], [])
    keep_cols = sum([list(range(s, e)) for (s, e) in keep_cols], [])
    arr = arr[keep_rows, :, :][:, keep_cols, :]
    assert arr.shape[0] == arr.shape[1], f"Expected square image, got {arr.shape}"

    logo = PIL.Image.fromarray(arr)

    if size is not None:
        logo = logo.resize((size, size), PIL.Image.NEAREST)

    return logo

def log(msg: str, type: Literal['INFO', 'WARN', 'ERROR']='INFO') -> None:
        color = 'green' if type == 'INFO' else 'yellow' if type == 'WARN' else 'red'
        cprint(f'[{type}] {msg}', color)

class cached_property_readonly (cached_property, property):
    """
    Read only version of functools.cached_property.

    Reference: https://github.com/dask/dask/issues/7076

    Instead of implementing __set__(), this class just also subclasses from
    property, which makes it read-only. An additional advantage of doing this
    is that IDEs like VSCode seem to identify subclasses of @property as
    properties, but not subclasses of @cached_property.

    NOTE: order of inheritance matters: first should be cached_property, then
    property.
    """

# =============================================================================
# region Unprocessed
# =============================================================================

def viz_image(img: Tensor, is_normalized: bool):
    img = img.clone()
    if img.dim() == 4:
        assert img.shape[0] == 1, 'only support batch size 1'
        img.squeeze_(0)  # (3, H, W)

    if is_normalized:
        img = diffusion.unnormalize(img)  # convert [0, 1]
    img = img.clamp(0, 1)  # threshold to [0, 1]
    img = img.permute((1, 2, 0))  # (H, W, 3)
    img = img.cpu().numpy()  # (H, W, 3)
    img = (img * 255).astype(np.uint8)  # (H, W, 3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
    return img

def viz_colormap(img: Tensor, normalize=True):
    img = img.clone()
    if img.dim() == 4:
        assert img.shape[0] == 1, 'only support batch size 1'
        img.squeeze_(0)  # (1, H, W) or (H, W)

    if img.dim() == 3:
        assert img.shape[0] == 1, 'only supports one channel'
        img.squeeze_(0)  # (H, W)

    if normalize:
        img -= img.min()
        img /= img.max()

    img = img.cpu().numpy()  # (H, W)
    img = (img * 255).astype(np.uint8)  # (H, W)
    img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
    return img

def show_and_save_gif(imgs: List[np.ndarray], save_path: str, time_in_ms=2000,
                      hold_last_frame_time_in_ms=0, loop=0):
    """
    Online GIF analysis tool: https://onlinegiftools.com/analyze-gif
    loop=0     # loop endlessly
    loop=k     # loop k+1 times
    loop=None  # loop only once
    """    
    # show images using cv2
    # all frames
    delay_in_ms = max(time_in_ms // len(imgs), 1)
    for img in imgs:
        img = cv2.resize(img, (0, 0), fx=3, fy=3)
        cv2.imshow('', img)
        cv2.waitKey(delay_in_ms)
    # last frame
    if hold_last_frame_time_in_ms > 0:
        cv2.imshow('', img)
        cv2.waitKey(hold_last_frame_time_in_ms)
    cv2.destroyAllWindows()

    # save gif using imageio, not more than max_num_imgs images
    MAX_NUM_IMGS = 50
    num_imgs = min(len(imgs), MAX_NUM_IMGS)
    indices = np.linspace(0, len(imgs) - 1, num_imgs, dtype='int')
    imgs = [imgs[i] for i in indices]
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]  # imageio requires this
    print(f"\033[0;32mSaving GIF to \"{save_path}\" ... ", end='', flush=True)
    # a minimum of 10ms is required for imageio/PIL.
    # in fact, it uses increments of 10ms only!
    delay_in_ms = max(time_in_ms // len(imgs), 10)
    num_extra_last_frame = int(hold_last_frame_time_in_ms / time_in_ms * len(imgs))
    imgs = imgs + [imgs[-1]] * num_extra_last_frame
    kwargs = dict(duration=delay_in_ms, save_all=False, optimize=True)
    if loop is not None:
        kwargs['loop'] = loop
    iio.mimsave(save_path, imgs, format='GIF', **kwargs)
    print("done.\033[0;m")

def make_gif_from_images_from_disk(img_dir: Path, time_in_ms=2000,
                                   hold_last_frame_time_in_ms=0, loop=0):
    # Read images from disk
    imgs = []
    for img_file in img_dir.glob('*.png'):
        img = cv2.imread(str(img_file))
        imgs.append(img)

    # save gif using imageio
    save_path = img_dir / 'out.gif'
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]  # imageio requires this
    print(f"\033[0;32mSaving GIF to \"{save_path}\" ... ", end='', flush=True)
    # a minimum of 10ms is required for imageio/PIL.
    # in fact, it uses increments of 10ms only!
    delay_in_ms = max(time_in_ms // len(imgs), 10)
    num_extra_last_frame = int(hold_last_frame_time_in_ms / time_in_ms * len(imgs))
    imgs = imgs + [imgs[-1]] * num_extra_last_frame
    kwargs = dict(duration=delay_in_ms, save_all=False, optimize=True)
    if loop is not None:
        kwargs['loop'] = loop
    iio.mimsave(save_path, imgs, format='GIF', **kwargs)
    print("done.\033[0;m")

def show(canvas: np.ndarray, resize=2, wait=0):
    canvas = cv2.resize(canvas, (0, 0), fx=resize, fy=resize)
    cv2.imshow('', canvas)
    cv2.waitKey(wait)
    cv2.destroyAllWindows()
    # mmcv.imshow(img, win_name='', wait_time=5000)

# endregion
# =============================================================================
