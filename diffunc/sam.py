import numpy as np
import PIL.Image
import torch
from torch import Tensor
import torchvision.transforms as T
from typing import Dict, Optional, Tuple

from segment_anything import SamAutomaticMaskGenerator

from .utils import cached_property_readonly

class Segmentation:
    def __init__(self,
                 mask_generator: SamAutomaticMaskGenerator,
                 image: PIL.Image.Image,
                 resize: Optional[Tuple[int, int]]=None,
                 min_mask_area_frac: float=0) -> Tensor:
        """
        Args:
            mask_generator: SamAutomaticMaskGenerator
            image (PIL.Image.Image, W, H) image.
            resize (Tuple[int, int], optional): (H, W) resize image to this size.
            min_mask_area_frac (float): minimum fraction of the mask area. If a
                segmentation mask has an area-fraction with area smaller than this
                value, then it is discarded.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        logger: Dict[str, float] = dict()

        # generate masks
        image = np.array(image)  # (H, W, 3) dtype=np.uint8
        masks = mask_generator.generate(image)
        logger['sam'] = len(masks)

        # sort segs in *decreasing* order of area
        masks = sorted(masks, key=lambda m: m['area'], reverse=True)

        # batch segs into a single tensor
        self.masks = []
        for mask in masks:
            mask: np.ndarray = mask['segmentation']  # (H, W) dtype=np.bool
            mask: Tensor = torch.from_numpy(mask)  # (H, W) dtype=torch.bool
            mask = mask.to(device=self.device)
            self.masks.append(mask)
        self.masks = torch.stack(self.masks, dim=0)  # (N, H, W) dtype=torch.bool

        # resize
        if resize is not None:
            self.masks = T.Resize(
                resize,
                interpolation=T.InterpolationMode.NEAREST_EXACT,
                antialias=True)(self.masks)  # (N, H, W) dtype=torch.bool

        # union of segmentation masks, which is an cumulative logical-or
        union_masks = self.masks.cummax(dim=0).values  # (N, H, W) dtype=torch.bool

        # disjoint mask = the current mask is true AND the previous union is false
        self.masks[1:] = self.masks[1:] & (~union_masks[:-1])  # (N, H, W) dtype=torch.bool

        # only keep non-empty seg masks
        non_empty = self.masks.amax(dim=(1, 2))  # (N,) dtype=torch.bool
        self.masks = self.masks[non_empty]  # (N, H, W) dtype=torch.bool
        logger['disjoint'] = len(self.masks)

        # filter masks by area
        non_small = self.masks.float().mean(dim=(1, 2)) >= min_mask_area_frac  # (N,) dtype=torch.bool
        self.masks = self.masks[non_small]  # (N, H, W) dtype=torch.bool
        logger['area'] = len(self.masks)

        print(f"SAM output {logger['sam']} masks. "
            f"{logger['disjoint']} were kept after removing mask intersections. "
            f"{logger['area']} were kept after filtering for minimum area.")

    @cached_property_readonly
    def lookup(self) -> Tensor:
        """
        Returns:
            (Tensor, dtype=int, shape=(H, W)): the index of the segment for
                each pixel. Range is [0, N).

            NOTE: background is assigned an index of 0 corresponding to the
            first mask. This is because torch.gather() only accepts indices in
            [0, N). To get foreground/background masks, use self.foreground and
            self.background.
        """
        # background is assigned an index of 0 i.e. the first mask
        # to 
        lookup = torch.zeros(size=self.masks.shape[1:], 
                             dtype=torch.long, device=self.device)  # (H, W)
        for i, mask in enumerate(self.masks):
            lookup[mask] = i
        return lookup

    @cached_property_readonly
    def sizes(self) -> Tensor:
        """
        Returns:
            (Tensor, dtype=int, shape=(N,)): the number of pixels in each segment.
        """
        return self.masks.sum(dim=(-2, -1))  # (N,)

    @cached_property_readonly
    def foreground(self) -> Tensor:
        return self.masks.any(dim=0)  # (H, W)

    @cached_property_readonly
    def background(self) -> Tensor:
        return ~self.foreground  # (H, W)

    @cached_property_readonly
    def count(self):
        return len(self.masks)

    def regularize(self, input: Tensor, fill: float) -> Tensor:
        """
        Args:
            unc (Tensor, dtype=float, shape=(..., H, W)): uncertainty map for each
                segmentation mask.
        """
        if input.shape[-2:] != self.masks.shape[-2:]:
            input = T.Resize(self.masks.shape[-2:])(input)  # (..., H, W)
        
        result = torch.ones_like(input) * fill  # (..., H, W)
        for mask in self.masks:
            result[..., mask] = input[..., mask].mean(dim=-1, keepdim=True)
        
        return result

def show_image_and_masks(image: PIL.Image.Image,
                         seg: Segmentation,
                         bg_alpha: float = 1,
                         fg_alpha: float = 0.35) -> PIL.Image.Image:
    """
    Args:
        image (PIL.Image.Image, W, H) image.
        seg (Segmentation, num=N, size=(H, W)): N disjoint segmentation masks.
    """
    if seg.count == 0:
        return

    N, H, W = seg.masks.shape

    # automatically resize background to the size of the masks
    if image.size != (W, H):
        image = image.resize((W, H))

    # rescale background intensities
    img_np = np.array(image)  # (H, W, 3) dtype=np.uint8
    img_np = (img_np * bg_alpha).astype(np.uint8)  # (H, W, 3) dtype=np.uint8
    image = PIL.Image.fromarray(img_np)  # (H, W, 3) dtype=np.uint8

    # convert seg masks to RGBA uint8 images
    masks = seg.masks.unsqueeze(-1)  # (N, H, W, 1) dtype=torch.bool
    masks = masks.expand(-1, -1, -1, 4)  # (N, H, W, 4) dtype=torch.bool
    rand_color = torch.cat([
        torch.rand((N, 1, 1, 3)),  # random RGB values
        torch.full((N, 1, 1, 1), fill_value=fg_alpha),  # fixed fg_alpha value
    ], dim=-1)  # (N, 1, 1, 4) dtype=float
    rand_color = (rand_color * 255).byte()  # (N, 1, 1, 4) dtype=torch.byte
    rand_color = rand_color.to(device=masks.device)  # (N, 1, 1, 4) dtype=torch.byte
    masks = masks * rand_color  # (N, H, W, 4) dtype=torch.byte

    # paste each mask seg onto image
    for mask_th in masks:
        mask_np = mask_th.cpu().numpy()  # (H, W, 4) dtype=np.uint8
        mask_im = PIL.Image.fromarray(mask_np)  # (H, W) dtype=np.uint8

        # https://stackoverflow.com/a/5324782
        image.paste(mask_im, (0, 0), mask_im)

    return image
