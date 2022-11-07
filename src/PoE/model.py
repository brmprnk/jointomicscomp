from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal, Independent, Beta

from src.nets import CrossGeneratingVariationalAutoencoder, ProbabilisticFullyConnectedModule, identity


class PoE(nn.Module):
    def __init__(self, args):
        super(PoE, self).__init__()

        device = torch.device('cuda') if torch.cuda.is_available() and args['cuda'] else torch.device('cpu')
        self.to(device)


        latent_dim = [int(k) for k in args['latent_dim'].split('-')]

        # define q(z|x_i) for i = 1...2
        self.omic1_encoder = Encoder(latent_dim[-1], args['num_features1'], latent_dim[:-1], device, args['dropout_probability'], args['use_batch_norm'])
        self.omic2_encoder = Encoder(latent_dim[-1], args['num_features2'], latent_dim[:-1], device, args['dropout_probability'], args['use_batch_norm'])

        # define p(x_i|z) for i = 1...2
        if 'n_categories1' in args:
            self.omic1_decoder = Decoder(latent_dim[-1], args['num_features1'], latent_dim[:-1][::-1], device, args['likelihood1'], args['dropout_probability'], args['use_batch_norm'], args['n_categories1'])
        else:
            self.omic1_decoder = Decoder(latent_dim[-1], args['num_features1'], latent_dim[:-1][::-1], device, args['likelihood1'], args['dropout_probability'], args['use_batch_norm'])

        if 'n_categories2' in args:
            self.omic2_decoder = Decoder(latent_dim[-1], args['num_features2'], latent_dim[:-1][::-1], device, args['likelihood2'], args['dropout_probability'], args['use_batch_norm'], args['n_categories2'])
        else:
            self.omic2_decoder = Decoder(latent_dim[-1], args['num_features2'], latent_dim[:-1][::-1], device, args['likelihood2'], args['dropout_probability'], args['use_batch_norm'])

        # define q(z|x) = q(z|x_1)...q(z|x_6)
        self.experts = ProductOfExperts()
        # use MMVAE model for MoE

        self.latent_dim = latent_dim[-1]
        self.use_cuda = args['cuda']
        self.training = False

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:  # return mean during inference
            return mu

    def extract(self, omic1=None, omic2=None):

        mu, logvar = self.get_product_params(omic1=omic1, omic2=omic2)

        # re-parameterization trick to sample
        z = self.reparameterize(mu, logvar)

        return z

    def forward(self, omic1=None, omic2=None):

        mu, logvar = self.get_product_params(omic1=omic1, omic2=omic2)

        # re-parameterization trick to sample
        z = self.reparameterize(mu, logvar)

        # reconstruct inputs based on sample
        omic1_recon = self.omic1_decoder(z)
        omic2_recon = self.omic2_decoder(z)

        return omic1_recon, omic2_recon, mu, logvar

    def get_product_params(self, omic1=None, omic2=None):
        # define universal expert
        batch_size = get_batch_size(omic1, omic2)
        use_cuda   = next(self.parameters()).is_cuda  # check if CUDA
        # initialize the universal prior expert
        mu, logvar = prior_expert((1, batch_size, self.latent_dim), use_cuda=use_cuda)

        if omic1 is not None:
            omic1_mu, omic1_logvar = self.omic1_encoder(omic1)

            mu = torch.cat((mu, omic1_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, omic1_logvar.unsqueeze(0)), dim=0)

        if omic2 is not None:
            omic2_mu, omic2_logvar = self.omic2_encoder(omic2)

            mu = torch.cat((mu, omic2_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, omic2_logvar.unsqueeze(0)), dim=0)

        # product of experts to combine Gaussian's
        mu, logvar = self.experts(mu, logvar)

        return mu, logvar

    def embedAndReconstruct(self, omic1, omic2):
        assert self.training == False
        with torch.no_grad():
            z, _ = self.get_product_params(omic1=omic1, omic2=omic2)

            (x1_hat, x2_cross_hat, _, _) = self.forward(omic1=omic1)
            (x1_cross_hat, x2_hat, _, _) = self.forward(omic2=omic2)

            z1, _ = self.get_product_params(omic1=omic1, omic2=None)

            z2, _ = self.get_product_params(omic1=None, omic2=omic2)


            return z, z1, z2, x1_hat, x2_hat, x1_cross_hat, x2_cross_hat



def get_batch_size(omic1, omic2):
    if omic1 is None:
        return omic2.size(0)
    else:
        return omic1.size(0)


class Encoder(nn.Module):
    """Parametrizes q(z|x).

    We will use this for every q(z|x_i) for all i.

    @param latent_dim: integer
                      number of latent dimensions
    """

    def __init__(self, latent_dim, num_features, hidden_dims, device, dropoutP=0., use_batch_norm=True):
        super(Encoder, self).__init__()

        self.to(device)

        input_size = num_features


        modules = []

        if use_batch_norm:
            for h_dim in hidden_dims:
                modules.append(
                    nn.Sequential(
                        nn.Dropout(dropoutP),
                        nn.Linear(input_size, h_dim),
                        nn.BatchNorm1d(h_dim),
                        nn.ReLU()
                    )
                )
                input_size = h_dim
        else:
            for h_dim in hidden_dims:
                modules.append(
                    nn.Sequential(
                        nn.Dropout(dropoutP),
                        nn.Linear(input_size, h_dim),
                        nn.ReLU()
                    )
                )
                input_size = h_dim


        if len(hidden_dims) == 0:
            hidden_dims = [input_size]


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

    def __init__(self, latent_dim, num_features, hidden_dims, device, likelihood='normal', dropoutP=0., use_batch_norm=True, n_categories=None):
        super(Decoder, self).__init__()
        self.to(device)

        self.distribution = likelihood
        self.num_features = num_features
        self.n_categories = n_categories

        input_size = latent_dim


        modules = []

        if use_batch_norm:
            for h_dim in hidden_dims:
                modules.append(
                    nn.Sequential(
                        nn.Dropout(dropoutP),
                        nn.Linear(input_size, h_dim),
                        nn.BatchNorm1d(h_dim),
                        nn.ReLU()
                    )
                )
                input_size = h_dim
        else:
            for h_dim in hidden_dims:
                modules.append(
                    nn.Sequential(
                        nn.Dropout(dropoutP),
                        nn.Linear(input_size, h_dim),
                        nn.ReLU()
                    )
                )
                input_size = h_dim


        if len(hidden_dims) == 0:
            hidden_dims = [input_size]

        self.decoder = nn.Sequential(*modules)

        if self.distribution != 'categorical':
            self.final_layer = nn.Sequential(nn.Linear(hidden_dims[-1], 2 * num_features))
        else:
            self.final_layer = nn.Sequential(nn.Linear(hidden_dims[-1], n_categories * num_features))

        if self.distribution == 'normal':
        	self.p1Transform = identity
        	self.p2Transform = torch.exp
        	self.D = torch.distributions.Normal
        elif self.distribution == 'beta':
        	self.p1Transform = torch.exp
        	self.p2Transform = torch.exp
        	self.D = torch.distributions.Beta
        elif self.distribution == 'categorical':
        	self.D = torch.distributions.Categorical
        elif self.distribution == 'nb':
            self.p1Transform = torch.exp
            self.p2Transform = torch.nn.Sigmoid()
            self.D = torch.distributions.NegativeBinomial

        else:
        	raise NotImplementedError('%s not supported. Use: \'normal\' or \'beta\'' % self.distribution)

    def forward(self, z):
        # the input will be a vector of size |latent_dim|
        result = self.decoder(z)
        result = self.final_layer(result)

        if self.distribution != 'categorical':
        	p1 = self.p1Transform(result[:, :self.num_features])
        	p2 = self.p2Transform(result[:, self.num_features:])

        	result = Independent(self.D(p1, p2), 0)
        else:

            result = torch.reshape(result, (-1, self.num_features, self.n_categories))

            result = self.D(logits=result)

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
        # explanation for two gaussians here: https://ccrma.stanford.edu/~jos/sasp/Product_Two_Gaussian_PDFs.html
        var = torch.exp(logvar) + eps
        T = 1 / (var + eps)  # precision of i-th Gaussian expert at point x
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1 / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar



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
