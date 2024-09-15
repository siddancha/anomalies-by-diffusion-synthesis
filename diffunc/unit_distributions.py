import math
import torch
from torch import Tensor
import torch.nn.functional as F

# =============================================================================
# region Exponential distribution over [0, 1]
# =============================================================================

# https://github.com/pytorch/pytorch/issues/39242#issuecomment-1349202573 
def log1mexp(x: Tensor) -> Tensor:
    """Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    """
    mask = -math.log(2) < x  # x < 0
    return torch.where(
        mask,
        (-x.expm1()).log(),
        (-x.exp()).log1p(),
    )

class ExponentialDistribution:
    def __init__(self, λ: float, device=torch.device('cpu')):
        """
        The exponential probability distribution, restricted to [0, 1].

        p(x) = Z exp(-λx) for x \in [0, 1] where the normalizing constant Z is
           Z = λ / (1 - exp(-λ))
        
        Args:
            λ (float): rate parameter of the exponential distribution, lying in (0, inf).
        """
        if isinstance(λ, torch.Tensor):
            self.λ = λ.to(device=device)
        else:
            self.λ = torch.tensor(λ, dtype=torch.float, device=device)
    
    def pdf(self, input: Tensor):
        """
        Args:
            input (Tensor, dtype=float): input lying in [0, 1].

        Returns:
            (Tensor, dtype=float): p(x).
        """
        Z = self.λ / (1 - torch.exp(-self.λ))
        return Z * torch.exp(-self.λ * input)

    def log_pdf(self, input: Tensor):
        """
        A more numerically stable version of computing the log of the PDF
        than simply torch.log(self.pdf(input)).

        Args:
            input (Tensor, dtype=float): input lying in [0, 1].

        Returns:
            (Tensor, dtype=float): log p(x).
        """
        logZ = torch.log(self.λ) - log1mexp(-self.λ)
        return -self.λ * input + logZ

    def cdf(self, input: Tensor):
        """
        The cumulative density.
        
        P(x) = (1 - exp(-λx)) / (1 - exp(-λ))

        Args:
            input (Tensor, dtype=float): input lying in [0, 1].

        Returns:
            (Tensor, dtype=float): P(x).
        """
        # return (1 - torch.exp(-self.λ * input)) / (1 - torch.exp(-self.λ))
        # return torch.expm1(-self.λ * input) / torch.expm1(-self.λ)
        return torch.exp(log1mexp(-self.λ * input) - log1mexp(-self.λ))

    def inv_cdf(self, input: Tensor):
        """
        The inverse-CDF.

        P⁻¹(x) = -(1 / λ) log (1 - x (1 - exp(-λ))) 

        Args:
            input (Tensor, dtype=float, shape=(*B)): input lying in [0, 1].

        Returns:
            (Tensor, dtype=float, shape=(*B)): P⁻¹(x).
        """
        # return -(1 / self.λ) * torch.log(1 - input * (1 - torch.exp(-self.λ)))
        return -(1 / self.λ) * log1mexp(torch.log(input) + log1mexp(-self.λ))

# endregion
# =============================================================================
# region Beta distribution
# =============================================================================

# TODO: **IMPORTANT**
# Beta's log_prob function calls log on the input. Since the input comes from
# soft_pixel_match_fraction, we should call log_soft_pixel_match_fraction()
# instead of log(soft_pixel_match_fraction()).
def log_prob_beta_prior(input: Tensor, mode: float, concentration: float):
    """
    Args:
        input (Tensor, dtype=float, shape=(*B)): input lying in [0, 1].
        mode (float): mean parameter of the beta distribution, lying in (0, 1).
        concentration (float): concentration of beta distribution, lying in (0, inf).
    
    Returns:


    Reference: https://en.wikipedia.org/wiki/Beta_distribution#Mode_and_concentration
    We use the notation from the Wikipedia article.
    """
    ω, c = mode, concentration
    κ = c + 2

    α = ω * (κ - 2) + 1
    β = (1 - ω) * (κ - 2) + 1

    return torch.distributions.beta.Beta(α, β).log_prob(input)  # (B,)

# endregion
# =============================================================================
