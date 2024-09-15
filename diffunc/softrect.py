import torch
from torch import Tensor
import torch.nn.functional as F

def rect_inlier(x: Tensor,
                center: Tensor,
                width: float) -> Tensor:
    """
    The hard rectangular inlier function based on the rectangular function
    (https://en.wikipedia.org/wiki/Rectangular_function).

    It also goes by other names: the boxcar function (https://en.wikipedia.org/wiki/Boxcar_function)
    at A=1, or the top-hat function (https://en.wikipedia.org/wiki/Top-hat_filter).

    Args:
        x (Tensor, dtype=float): input.
        center (Tensor, dtype=float): center of the window.
        width (float): width of the window.
    """
    edge1 = center - 0.5 * width
    edge2 = center + 0.5 * width

    step1 = (x >= edge1).float()
    step2 = (x <= edge2).float()

    return step1 * step2

def rect_outlier(x: Tensor,
                 center: Tensor,
                 width: float) -> Tensor:
    """
    The hard rectangular outlier function, which is the complement of the hard
    rectangular inlier function.

    Args:
        x (Tensor, dtype=float): input.
        center (Tensor, dtype=float): center of the window.
        width (float): width of the window.
    """
    edge1 = center - 0.5 * width
    edge2 = center + 0.5 * width

    step1 = (x <= edge1).float()
    step2 = (x >= edge2).float()

    return step1 + step2

def softrect_inlier(x: Tensor,
                    center: Tensor,
                    width: float,
                    steepness: float) -> Tensor:
    """
    The soft-rectangular function, which is a soft (differentiable) version
    of the rectangular function (https://en.wikipedia.org/wiki/Rectangular_function).

    It also goes by other names: the boxcar function (https://en.wikipedia.org/wiki/Boxcar_function)
    at A=1, or the top-hat function (https://en.wikipedia.org/wiki/Top-hat_filter).

    The rectangular function can be produced by multiplying two Heaviside step
    functions (https://en.wikipedia.org/wiki/Heaviside_step_function) that are
    mirrored and separated. Since the sigmoid function is a soft version of the
    Heaviside step function, we can produce a soft version of the rectangular
    function by multiplying two sigmoid functions that are mirrored and separated.
    
    Args:
        x (Tensor, dtype=float): input.
        center (Tensor, dtype=float): center of the window.
        width (float): width of the window.
        steepness (float): steepness of the rising and falling edge of the
            soft-rectangular function. The greater the steepness, the more
            closely it will resemble the rectangular function (steepness = inf)
            makes the soft-rectangular function equal to the rectangular
            function.
    """
    edge1 = center - 0.5 * width
    edge2 = center + 0.5 * width

    #    ┌───────
    # ───┘ 
    sig1 = F.sigmoid(+steepness * (x - edge1))
    
    # ───────┐
    #        └───
    sig2 = F.sigmoid(-steepness * (x - edge2))  # sig(-x) = 1 - sig(x)

    #    ┌───┐
    # ───┘   └───
    return sig1 * sig2

def log_softrect_inlier(x: Tensor,
                        center: Tensor,
                        width: float,
                        steepness: float) -> Tensor:
    """
    Logarithm of the soft-rectangular inlier function.

    Args:
        x (Tensor, dtype=float): input.
        center (Tensor, dtype=float): center of the window.
        width (float): width of the window.
        steepness (float): steepness of the rising and falling edge of the
            soft-rectangular function. The greater the steepness, the more
            closely it will resemble the rectangular function (steepness = inf)
            makes the soft-rectangular function equal to the rectangular
            function.
    """
    edge1 = center - 0.5 * width
    edge2 = center + 0.5 * width

    #    ┌───────
    # ───┘ 
    log_sig1 = F.logsigmoid(+steepness * (x - edge1))

    # ───────┐
    #        └───
    log_sig2 = F.logsigmoid(-steepness * (x - edge2))

    #    ┌───┐
    # ───┘   └───
    return log_sig1 + log_sig2

def softrect_outlier(x: Tensor,
                     center: float=0,
                     width: float=1,
                     steepness: float=50) -> Tensor:
    """
    Soft-rectangular outlier function.
    The hard rectangular outlier function is the complement of the hard
    rectangular function -- which is 1 outside the window and 0 inside.

    Args:
        x (Tensor, dtype=float): input.
        center (float): center of the window.
        width (float): width of the window.
        steepness (float): steepness of the rising and falling edge of the
            soft-rectangular function. The greater the steepness, the more
            closely it will resemble the rectangular function (steepness = inf)
            makes the soft-rectangular function equal to the rectangular
            function.
    """
    edge1 = center - 0.5 * width
    edge2 = center + 0.5 * width

    # ───┐       
    #    └───────
    sig1 = F.sigmoid(-steepness * (x - edge1))

    #        ┌───
    # ───────┘   
    sig2 = F.sigmoid(+steepness * (x - edge2))

    # ───┐   ┌───
    #    └───┘   
    return sig1 + sig2

def log_softrect_outlier(x: Tensor,
                         center: Tensor,
                         width: float,
                         steepness: float) -> Tensor:
    """
    Logarithm of the soft-rectangular outlier function.
    The hard rectangular outlier function is the complement of the hard
    rectangular function -- which is 1 outside the window and 0 inside.

    Args:
        x (Tensor, dtype=float): input.
        center (Tensor, dtype=float): center of the window.
        width (float): width of the window.
        steepness (float): steepness of the rising and falling edge of the
            soft-rectangular function. The greater the steepness, the more
            closely it will resemble the rectangular function (steepness = inf)
            makes the soft-rectangular function equal to the rectangular
            function.
    """
    edge1 = center - 0.5 * width
    edge2 = center + 0.5 * width

    # ───┐       
    #    └───────
    log_sig1 = F.logsigmoid(-steepness * (x - edge1))

    #        ┌───
    # ───────┘   
    log_sig2 = F.logsigmoid(+steepness * (x - edge2))

    # ───┐   ┌───
    #    └───┘   
    return torch.logaddexp(log_sig1, log_sig2)
