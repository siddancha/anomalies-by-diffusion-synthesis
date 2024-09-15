"""
Run `pytest` in command line to run test
"""

import torch
from diffunc.unit_distributions import ExponentialDistribution

def test_logpdf():
    xs = torch.linspace(0, 1, 100)
    λs = torch.linspace(0.1, 95, 100)
    for x in xs:
        for λ in λs:
            dist = ExponentialDistribution(λ)
            ans1 = torch.log(dist.pdf(x))
            ans2 = dist.log_pdf(x)
            def error_msg(default_msg):
                return  default_msg + f'\nλ: {λ}'
            torch.testing.assert_close(ans1, ans2, msg=error_msg)

def test_invcdf():
    xs = torch.linspace(0, 1, 100)
    λs = torch.linspace(0.1, 20, 100)
    for x in xs:
        for λ in λs:
            dist = ExponentialDistribution(λ)
            ans1 = dist.inv_cdf(dist.cdf(x))
            torch.testing.assert_close(
                ans1, x,
                atol=1e-1, rtol=1.3e-6,
                msg=lambda default_msg:
                    f'inv_cdf(cdf(x)) == x\n{default_msg}\n'
                    f'x: {x:.3f}, λ: {λ}, cdf({x}): {dist.cdf(x)}, inv_cdf({dist.cdf(x)}): {ans1:.3f}')

            ans2 = dist.cdf(dist.inv_cdf(x))
            torch.testing.assert_close(ans2, x, msg=
                lambda default_msg: f'cdf(inv_cdf(x)) == x\n{default_msg}\nλ: {λ}')
