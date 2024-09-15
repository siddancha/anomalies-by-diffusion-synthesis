"""
Run `pytest` in command line to run test
"""

import torch
from diffunc.softrect import (
    softrect_inlier,
    log_softrect_inlier,
    softrect_outlier,
    log_softrect_outlier,
)

def test_log_softrect_inlier():
    x = torch.linspace(-2, 3, 100, dtype=torch.float)
    list_center = [-1, 0, 0.2]
    list_steepness = [1, 5, 10, 20]
    list_width = [0.1, 0.2]
    for center in list_center:
        for steepness in list_steepness:
            for width in list_width:
                ans1 = torch.log(softrect_inlier(x, center, width, steepness))
                assert torch.isfinite(ans1).all(), f'ans1: {ans1} is not finite'

                ans2 = log_softrect_inlier(x, center, width, steepness)
                assert torch.isfinite(ans1).all(), f'ans2: {ans2} is not finite'

                def error_msg(default_msg):
                    return  default_msg + '\n\n' + \
                            f'center: {center}\n' + \
                            f'steepness: {steepness}\n' + \
                            f'width: {width}\n' + \
                            f'{ans1} did not match {ans2}'
                torch.testing.assert_close(ans1, ans2, msg=error_msg)

def test_log_softrect_outlier():
    x = torch.linspace(-2, 3, 100, dtype=torch.float)
    list_center = [-1, 0, 0.2]
    list_steepness = [1, 5, 10, 20]
    list_width = [0.1, 0.2]
    for center in list_center:
        for steepness in list_steepness:
            for width in list_width:
                ans1 = torch.log(softrect_outlier(x, center, width, steepness))
                assert torch.isfinite(ans1).all(), f'ans1: {ans1} is not finite'

                ans2 = log_softrect_outlier(x, center, width, steepness)
                assert torch.isfinite(ans1).all(), f'ans2: {ans2} is not finite'

                def error_msg(default_msg):
                    return  default_msg + '\n\n' + \
                            f'center: {center}\n' + \
                            f'steepness: {steepness}\n' + \
                            f'width: {width}\n' + \
                            f'{ans1} did not match {ans2}'
                torch.testing.assert_close(ans1, ans2, msg=error_msg)
