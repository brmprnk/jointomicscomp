import torch.nn.functional as F
from typing import Optional, Tuple, Union
import torch
from torch.distributions import Distribution, Gamma, constraints
from torch.distributions import Poisson as PoissonTorch
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)
import warnings
#import settings
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
        mu,
        theta,
        scale=None,
        validate_args=False
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

    def log_prob(self, value):
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
        mu,
        theta,
        zi_logits,
        scale=None,
        validate_args=False,
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
    def zi_logits(self):
        """ZI logits."""
        return probs_to_logits(self.zi_probs, is_binary=True)

    @lazy_property
    def zi_probs(self):
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

    def log_prob(self, value):
        """Log probability."""
        return log_zinb_positive(value, self.mu, self.theta, self.zi_logits, eps=1e-08)


class NegativeBinomialMixture(Distribution):
    """Negative binomial mixture distribution.

    See :class:`~scvi.distributions.NegativeBinomial` for further description
    of parameters.

    Parameters
    ----------
    mu1
        Mean of the component 1 distribution.
    mu2
        Mean of the component 2 distribution.
    theta1
        Inverse dispersion for component 1.
    mixture_logits
        Logits scale probability of belonging to component 1.
    theta2
        Inverse dispersion for component 1. If `None`, assumed to be equal to `theta1`.
    validate_args
        Raise ValueError if arguments do not match constraints
    """

    arg_constraints = {
        "mu1": constraints.greater_than_eq(0),
        "mu2": constraints.greater_than_eq(0),
        "theta1": constraints.greater_than_eq(0),
        "mixture_probs": constraints.half_open_interval(0.0, 1.0),
        "mixture_logits": constraints.real,
    }
    support = constraints.nonnegative_integer

    def __init__(
        self,
        mu1: torch.Tensor,
        mu2: torch.Tensor,
        theta1: torch.Tensor,
        mixture_logits: torch.Tensor,
        theta2: Optional[torch.Tensor] = None,
        validate_args: bool = False,
    ):
        (
            self.mu1,
            self.theta1,
            self.mu2,
            self.mixture_logits,
        ) = broadcast_all(mu1, theta1, mu2, mixture_logits)

        super().__init__(validate_args=validate_args)

        if theta2 is not None:
            self.theta2 = broadcast_all(mu1, theta2)
        else:
            self.theta2 = None

    @property
    def mean(self):
        pi = self.mixture_probs
        return pi * self.mu1 + (1 - pi) * self.mu2

    @lazy_property
    def mixture_probs(self) -> torch.Tensor:
        return logits_to_probs(self.mixture_logits, is_binary=True)

    # @torch.inference_mode()
    # def sample(
    #     self,
    #     sample_shape: Optional[Union[torch.Size, Tuple]] = None,
    # ) -> torch.Tensor:
    #     """Sample from the distribution."""
    #     sample_shape = sample_shape or torch.Size()
    #     pi = self.mixture_probs
    #     mixing_sample = torch.distributions.Bernoulli(pi).sample()
    #     mu = self.mu1 * mixing_sample + self.mu2 * (1 - mixing_sample)
    #     if self.theta2 is None:
    #         theta = self.theta1
    #     else:
    #         theta = self.theta1 * mixing_sample + self.theta2 * (1 - mixing_sample)
    #     gamma_d = _gamma(theta, mu)
    #     p_means = gamma_d.sample(sample_shape)
    #
    #     # Clamping as distributions objects can have buggy behaviors when
    #     # their parameters are too high
    #     l_train = torch.clamp(p_means, max=1e8)
    #     counts = PoissonTorch(
    #         l_train
    #     ).sample()  # Shape : (n_samples, n_cells_batch, n_features)
    #     return counts

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
        return log_mixture_nb(
            value,
            self.mu1,
            self.mu2,
            self.theta1,
            self.theta2,
            self.mixture_logits,
            eps=1e-08,
        )



def log_nb_positive(
    x,
    mu,
    theta,
    eps=1e-8,
    log_fn=torch.log,
    lgamma_fn=torch.lgamma,
):
    """Log likelihood (scalar) of a minibatch according to a nb model.

    Parameters
    ----------
    x
        data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    theta
        inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    eps
        numerical stability constant
    log_fn
        log function
    lgamma_fn
        log gamma function
    """
    log = log_fn
    lgamma = lgamma_fn
    log_theta_mu_eps = log(theta + mu + eps)
    res = (
        theta * (log(theta + eps) - log_theta_mu_eps)
        + x * (log(mu + eps) - log_theta_mu_eps)
        + lgamma(x + theta)
        - lgamma(theta)
        - lgamma(x + 1)
    )

    return res


def log_zinb_positive(
    x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, pi: torch.Tensor, eps=1e-8
):
    """Log likelihood (scalar) of a minibatch according to a zinb model.

    Parameters
    ----------
    x
        Data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    theta
        inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    pi
        logit of the dropout parameter (real support) (shape: minibatch x vars)
    eps
        numerical stability constant

    Notes
    -----
    We parametrize the bernoulli using the logits, hence the softplus functions appearing.
    """
    # theta is the dispersion rate. If .ndimension() == 1, it is shared for all cells (regardless of batch or labels)
    if theta.ndimension() == 1:
        theta = theta.view(
            1, theta.size(0)
        )  # In this case, we reshape theta for broadcasting

    # Uses log(sigmoid(x)) = -softplus(-x)
    softplus_pi = F.softplus(-pi)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero

    return res



def log_mixture_nb(
    x: torch.Tensor,
    mu_1: torch.Tensor,
    mu_2: torch.Tensor,
    theta_1: torch.Tensor,
    theta_2: torch.Tensor,
    pi_logits: torch.Tensor,
    eps=1e-8,
):
    """Log likelihood (scalar) of a minibatch according to a mixture nb model.

    pi_logits is the probability (logits) to be in the first component.
    For totalVI, the first component should be background.

    Parameters
    ----------
    x
        Observed data
    mu_1
        Mean of the first negative binomial component (has to be positive support) (shape: minibatch x features)
    mu_2
        Mean of the second negative binomial (has to be positive support) (shape: minibatch x features)
    theta_1
        First inverse dispersion parameter (has to be positive support) (shape: minibatch x features)
    theta_2
        Second inverse dispersion parameter (has to be positive support) (shape: minibatch x features)
        If None, assume one shared inverse dispersion parameter.
    pi_logits
        Probability of belonging to mixture component 1 (logits scale)
    eps
        Numerical stability constant
    """
    if theta_2 is not None:
        log_nb_1 = log_nb_positive(x, mu_1, theta_1)
        log_nb_2 = log_nb_positive(x, mu_2, theta_2)
    # this is intended to reduce repeated computations
    else:
        theta = theta_1
        if theta.ndimension() == 1:
            theta = theta.view(
                1, theta.size(0)
            )  # In this case, we reshape theta for broadcasting

        log_theta_mu_1_eps = torch.log(theta + mu_1 + eps)
        log_theta_mu_2_eps = torch.log(theta + mu_2 + eps)
        lgamma_x_theta = torch.lgamma(x + theta)
        lgamma_theta = torch.lgamma(theta)
        lgamma_x_plus_1 = torch.lgamma(x + 1)

        log_nb_1 = (
            theta * (torch.log(theta + eps) - log_theta_mu_1_eps)
            + x * (torch.log(mu_1 + eps) - log_theta_mu_1_eps)
            + lgamma_x_theta
            - lgamma_theta
            - lgamma_x_plus_1
        )
        log_nb_2 = (
            theta * (torch.log(theta + eps) - log_theta_mu_2_eps)
            + x * (torch.log(mu_2 + eps) - log_theta_mu_2_eps)
            + lgamma_x_theta
            - lgamma_theta
            - lgamma_x_plus_1
        )

    logsumexp = torch.logsumexp(torch.stack((log_nb_1, log_nb_2 - pi_logits)), dim=0)
    softplus_pi = F.softplus(-pi_logits)

    log_mixture_nb = logsumexp - softplus_pi

    return log_mixture_nb
