from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.autograd import Variable


class MVAE(nn.Module):
    def __init__(self, use_mixture=False, latent_dim=100, use_cuda=False):
        super(MVAE, self).__init__()
        # define q(z|x_i) for i = 1...2
        self.ge_encoder = Encoder(latent_dim)
        self.me_encoder = Encoder(latent_dim)

        # define p(x_i|z) for i = 1...2
        self.ge_decoder = Decoder(latent_dim)
        self.me_decoder = Decoder(latent_dim)

        # define q(z|x) = q(z|x_1)...q(z|x_6)
        self.experts = ProductOfExperts()
        self.mixture = MixtureOfExperts()
        self.use_mixture = use_mixture
        if self.use_mixture:
            print("Using Mixture-of-Experts Model")
        else:
            print("Using Product-of-Experts Model")

        self.latent_dim = latent_dim
        self.use_cuda = use_cuda

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:  # return mean during inference
            return mu

    def forward(self, ge=None, me=None):

        if self.use_mixture:
            mu, logvar = self.get_mixture_params(ge=ge, me=me)
        else:
            mu, logvar = self.get_product_params(ge=ge, me=me)

        # re-parameterization trick to sample
        z = self.reparameterize(mu, logvar)

        # reconstruct inputs based on sample
        ge_recon = self.ge_decoder(z)
        me_recon = self.me_decoder(z)

        return ge_recon, me_recon, mu, logvar

    def get_product_params(self, ge=None, me=None):
        # define universal expert
        batch_size = get_batch_size(ge, me)
        # initialize the universal prior expert
        mu, logvar = prior_expert((1, batch_size, self.latent_dim))

        if ge is not None:
            ge_mu, ge_logvar = self.ge_encoder(ge)

            mu = torch.cat((mu, ge_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, ge_logvar.unsqueeze(0)), dim=0)

        if me is not None:
            me_mu, me_logvar = self.me_encoder(me)

            mu = torch.cat((mu, me_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, me_logvar.unsqueeze(0)), dim=0)

        # product of experts to combine Gaussian's
        mu, logvar = self.experts(mu, logvar)

        return mu, logvar

    def get_mixture_params(self, ge=None, me=None):

        mu = []
        logvar = []

        if ge is not None:
            ge_mu, ge_logvar = self.ge_encoder(ge)

            mu.append(ge_mu.unsqueeze(0))
            logvar.append(ge_logvar.unsqueeze(0))

        if me is not None:
            me_mu, me_logvar = self.me_encoder(me)

            mu.append(me_mu.unsqueeze(0))
            logvar.append(me_logvar.unsqueeze(0))

        # mixture of experts to combine Gaussian's
        mu, logvar = self.mixture(mu, logvar)

        return mu, logvar


def get_batch_size(ge, me):
    if ge is None:
        return me.size(0)
    else:
        return ge.size(0)


class Encoder(nn.Module):
    """Parametrizes q(z|x).

    We will use this for every q(z|x_i) for all i.

    @param latent_dim: integer
                      number of latent dimensions
    """

    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        input_size = 5000
        hidden_dims = [256]

        modules = []
        if hidden_dims is None:
            hidden_dims = [256]

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_size, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU()
                )
            )

        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        self.latent_dim = latent_dim

    def forward(self, x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var


class Decoder(nn.Module):
    """Parametrizes p(x|z).

    We will use this for every p(x_i|z) for all i.

    @param latent_dim: integer
                      number of latent dimension
    """

    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        input_size = 5000
        hidden_dims = [256]

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[-1]),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.ReLU()
        )
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], input_size),
            nn.Sigmoid())

    def forward(self, z):
        # the input will be a vector of size |latent_dim|
        result = self.decoder(z)
        result = self.final_layer(result)

        return result


class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    """

    @classmethod
    def forward(cls, mu, logvar, eps=1e-8):
        """
        @param mu: M x D for M experts
        @param logvar: M x D for M experts
        @param eps: A small constant
        @return:
        """
        var = torch.exp(logvar) + eps
        T = 1 / (var + eps)  # precision of i-th Gaussian expert at point x
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1 / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar


class MixtureOfExperts(nn.Module):
    """Return parameters for mixture of independent experts.
    See https://papers.nips.cc/paper/2019/file/0ae775a8cb3b499ad1fca944e6f5c836-Paper.pdf for equations.
    https://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables
    Z = W + X + Y gives Z ~ N(mu_w + mu_x + mu_y, var_x^2 + var_x^2 + var_y^2)
    """

    @classmethod
    def forward(cls, mu, logvar, eps=1e-8):
        """
        @param mu: list of M x D for M experts
        @param logvar: list of M x D for M experts
        @param eps: A small constant
        @return:
        """
        # Combine Gaussians

        # Add mu's
        mu = torch.cat(mu, dim=0)
        mx_mu = torch.sum(mu, dim=0) / 3  # For 3 VAE's

        # Add variances squared
        var_squared = []
        for i in range(len(logvar)):
            var = torch.exp(logvar[i]) + eps
            var_squared.append(torch.square(var))

        # Divide by 3 for mean, then take log variance
        mx_var = torch.cat(var_squared, dim=0)
        mx_var = torch.sum(mx_var, dim=0) / 3
        mx_logvar = torch.log(mx_var + eps)

        return mx_mu, mx_logvar


def prior_expert(size, use_cuda=False):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).

    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu = Variable(torch.zeros(size))
    logvar = Variable(torch.log(torch.ones(size)))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar
