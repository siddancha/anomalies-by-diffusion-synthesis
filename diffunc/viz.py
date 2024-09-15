import io
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

def colormap(img: Tensor, normalize=True) -> Tensor:
    """
    Args:
        img (Tensor, dtype=float, shape=(*BS, H, W)): input grayscale image.
        normalize (bool): if True, normalize image to [0, 1] using image's min
            and max values.
    
    Returns:
        (Tensor, dtype=float, shape=(*BS, 3, H, W)): RGB image with values in [0, 1]
    """
    img = img.clone()
    
    if normalize:
        img -= img.min()
        img /= img.max()

    # https://discuss.pytorch.org/t/pytorch-colormap-gather-operation/106662/2

    # float tensor, values lie between 0.0 and 1.0
    color_map = torch.tensor(cm.viridis.colors, dtype=float, device=img.device)  # (256, 3)

    # long tensor, values between 0 and 255
    img = (img * 255).long()  # (*BS, H, W)

    # float tensor, values between 0.0 and 1.0
    output = color_map[img]  # (*BS, H, W, 3)
    
    # permute RGB channel
    batch_dims = range(len(output.shape) - 3)
    output = output.permute(*batch_dims, -1, -3, -2)  # (*BS, 3, H, W)
    return output

def use_latex_in_plots():
    plt.rc('text', usetex=True)
    plt.rc('mathtext', fontset='stix')
    plt.rc('font', family='STIXGeneral')
    plt.rc('text.latex', preamble=r'\usepackage{amssymb}')

def plt_to_img(dpi=100):
    with io.BytesIO() as buff:
        plt.gcf().savefig(buff, format='png', dpi=dpi)
        buff.seek(0)
        im = plt.imread(buff)
        im = (255 * im).astype(np.uint8)
        return im
