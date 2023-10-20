from torch.distributions import Distribution, Gamma, constraints
from torch.distributions import Poisson as PoissonTorch
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)

# this code is taken from scvitools

class NegativeBinomial(Distribution):
    r"""Negative binomial distribution.

    (`mu`, `theta`)
    These parameters respectively
    control the mean and inverse dispersion of the distribution.

    Samples are generated as follows:

    1. :math:`w \sim \textrm{Gamma}(\underbrace{\theta}_{\text{shape}}, \underbrace{\theta/\mu}_{\text{rate}})`
    2. :math:`x \sim \textrm{Poisson}(w)`

    Parameters
    ----------
    mu
        Mean of the distribution.
    theta
        Inverse dispersion.
    scale
        Normalized mean expression of the distribution.
    validate_args
        Raise ValueError if arguments do not match constraints
    """

    arg_constraints = {
        "mu": constraints.greater_than_eq(0),
        "theta": constraints.greater_than_eq(0),
        "scale": constraints.greater_than_eq(0),
    }
    support = constraints.nonnegative_integer

    def __init__(
        self,
        mu: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        validate_args: bool = False,
    ):
        self._eps = 1e-8

        mu, theta = broadcast_all(mu, theta)

        self.mu = mu
        self.theta = theta
        self.scale = scale
        super().__init__(validate_args=validate_args)

    @property
    def mean(self):
        return self.mu

    @property
    def variance(self):
        return self.mean + (self.mean**2) / self.theta

    # @torch.inference_mode()
    # def sample(
    #     self,
    #     sample_shape: Optional[Union[torch.Size, Tuple]] = None,
    # ) -> torch.Tensor:
    #     """Sample from the distribution."""
    #     sample_shape = sample_shape or torch.Size()
    #     gamma_d = self._gamma()
    #     p_means = gamma_d.sample(sample_shape)
    #
    #     # Clamping as distributions objects can have buggy behaviors when
    #     # their parameters are too high
    #     l_train = torch.clamp(p_means, max=1e8)
    #     counts = PoissonTorch(
    #         l_train
    #     ).sample()  # Shape : (n_samples, n_cells_batch, n_vars)
    #     return counts

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            try:
                self._validate_sample(value)
            except ValueError:
                warnings.warn(
                    "The value argument must be within the support of the distribution",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )

        return log_nb_positive(value, mu=self.mu, theta=self.theta, eps=self._eps)

    # def _gamma(self):
    #     return _gamma(self.theta, self.mu)


class ZeroInflatedNegativeBinomial(NegativeBinomial):
    r"""Zero-inflated negative binomial distribution.

    (`mu`, `theta`)
    These parameters respectively
    control the mean and inverse dispersion of the distribution.


    Parameters
    ----------
    mu
        Mean of the distribution.
    theta
        Inverse dispersion.
    zi_logits
        Logits scale of zero inflation probability.
    scale
        Normalized mean expression of the distribution.
    validate_args
        Raise ValueError if arguments do not match constraints
    """

    arg_constraints = {
        "mu": constraints.greater_than_eq(0),
        "theta": constraints.greater_than_eq(0),
        "zi_logits": constraints.real,
        "scale": constraints.greater_than_eq(0),
    }
    support = constraints.nonnegative_integer

    def __init__(
        self,
        mu: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        zi_logits: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        validate_args: bool = False,
    ):
        super().__init__(
            mu=mu,
            theta=theta,
            scale=scale,
            validate_args=validate_args,
        )
        self.zi_logits, self.mu, self.theta = broadcast_all(
            zi_logits, self.mu, self.theta
        )

    @property
    def mean(self):
        pi = self.zi_probs
        return (1 - pi) * self.mu

    @property
    def variance(self):
        raise NotImplementedError

    @lazy_property
    def zi_logits(self) -> torch.Tensor:
        """ZI logits."""
        return probs_to_logits(self.zi_probs, is_binary=True)

    @lazy_property
    def zi_probs(self) -> torch.Tensor:
        return logits_to_probs(self.zi_logits, is_binary=True)

    # @torch.inference_mode()
    # def sample(
    #     self,
    #     sample_shape: Optional[Union[torch.Size, Tuple]] = None,
    # ) -> torch.Tensor:
    #     """Sample from the distribution."""
    #     sample_shape = sample_shape or torch.Size()
    #     samp = super().sample(sample_shape=sample_shape)
    #     is_zero = torch.rand_like(samp) <= self.zi_probs
    #     samp_ = torch.where(is_zero, torch.zeros_like(samp), samp)
    #     return samp_

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Log probability."""
        try:
            self._validate_sample(value)
        except ValueError:
            warnings.warn(
                "The value argument must be within the support of the distribution",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )
        return log_zinb_positive(value, self.mu, self.theta, self.zi_logits, eps=1e-08)
