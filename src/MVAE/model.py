from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.autograd import Variable

from src.nets import MultiOmicVAE


class MVAE(nn.Module):
    def __init__(self, use_mixture=False, latent_dim=100, use_cuda=False):
        super(MVAE, self).__init__()
        # define q(z|x_i) for i = 1...2
        self.omic1_encoder = Encoder(latent_dim)
        self.omic2_encoder = Encoder(latent_dim)

        # define p(x_i|z) for i = 1...2
        self.omic1_decoder = Decoder(latent_dim)
        self.omic2_decoder = Decoder(latent_dim)

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
        self.training = False

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:  # return mean during inference
            return mu

    def extract(self, omic1=None, omic2=None):

        if self.use_mixture:
            mu, logvar = self.get_mixture_params(omic1=omic1, omic2=omic2)
        else:
            mu, logvar = self.get_product_params(omic1=omic1, omic2=omic2)

        # re-parameterization trick to sample
        z = self.reparameterize(mu, logvar)

        return z

    def forward(self, omic1=None, omic2=None):

        if self.use_mixture:
            mu, logvar = self.get_mixture_params(omic1=omic1, omic2=omic2)
        else:
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
        # initialize the universal prior expert
        mu, logvar = prior_expert((1, batch_size, self.latent_dim))

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

    def get_mixture_params(self, omic1=None, omic2=None):

        mu = []
        logvar = []

        if omic1 is not None:
            omic1_mu, omic1_logvar = self.omic1_encoder(omic1)

            mu.append(omic1_mu.unsqueeze(0))
            logvar.append(omic1_logvar.unsqueeze(0))

        if omic2 is not None:
            omic2_mu, omic2_logvar = self.omic2_encoder(omic2)

            mu.append(omic2_mu.unsqueeze(0))
            logvar.append(omic2_logvar.unsqueeze(0))

        # mixture of experts to combine Gaussian's
        mu, logvar = self.mixture(mu, logvar)

        return mu, logvar


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


class MixtureOfExperts(MultiOmicVAE):
    """Return parameters for mixture of independent experts.
    See https://papers.nips.cc/paper/2019/file/0ae775a8cb3b499ad1fca944e6f5c836-Paper.pdf for equations.
    https://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables
    Z = W + X + Y gives Z ~ N(mu_w + mu_x + mu_y, var_x^2 + var_x^2 + var_y^2)
    """

    def __init__(self, input_dim1, input_dim2, enc_hidden_dim=[100], dec_hidden_dim=[], loss1='bce', loss2='bce',
                 use_batch_norm=False, dropoutP=0.0, optimizer_name='Adam', encoder1_lr=1e-4, decoder1_lr=1e-4,
                 enc1_lastActivation='none', enc1_outputScale=1., encoder2_lr=1e-4, decoder2_lr=1e-4,
                 enc2_lastActivation='none', enc2_outputScale=1., beta=1.0):
        super(MixtureOfExperts, self).__init__(input_dim1, input_dim2, enc_hidden_dim, dec_hidden_dim, loss1, loss2,
                                               use_batch_norm, dropoutP,
                                           optimizer_name, encoder1_lr, decoder1_lr, enc1_lastActivation, enc1_outputScale,
                                           encoder2_lr, decoder2_lr, enc2_lastActivation, enc2_outputScale, beta)

        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2


        self.pz = torch.distributions.Laplace(torch.zeros(1, enc_hidden_dim[-1]), torch.ones(1, enc_hidden_dim[-1]))
        # Using this, loss is ignored
        self.px_z = torch.distributions.Normal
        self.qz_x = torch.distributions.Laplace(torch.zeros(1, enc_hidden_dim[-1]), torch.ones(1, enc_hidden_dim[-1]))

    def forward(self, x, K):
        # qz_x in MoE
        x1, x2 = x

        z1 = self.encoder(x1)
        z1sample = z1.rsample(torch.Size([K]))  # zs from MoE

        x1_hat = self.decoder(z1sample)

        z2 = self.encoder2(x2)
        z2sample = z2.rsample(torch.Size([K]))  # zs from MoE
        x2_hat = self.decoder2(z2sample)

        # From Mixture of Experts
        qz_xs = [z1, z2]
        zss = [z1sample, z2sample]

        # star_of_z2samples = []
        # for i in range(K):
        #     star_of_z2samples.append(self.decoder(z2sample[i]))

        x1_cross_hat = self.decoder(z2sample)

        x2_cross_hat = self.decoder2(z1sample)

        px_zs = [[torch.distributions.Normal(x1_hat, torch.ones(self.input_dim1)), torch.distributions.Normal(x2_cross_hat, self.input_dim2)],
                 [torch.distributions.Normal(x1_cross_hat, torch.ones(self.input_dim1)), torch.distributions.Normal(x2_hat, self.input_dim2)]]

        return qz_xs, px_zs, zss

    def compute_microbatch_split(self, x, K):
        """ Checks if batch needs to be broken down further to fit in memory. """
        B = x[0].size(0) if is_multidata(x) else x.size(0)
        S = sum([1.0 / (K * prod(_x.size()[1:])) for _x in x]) if is_multidata(x) \
            else 1.0 / (K * prod(x.size()[1:]))
        S = int(1e8 * S)  # float heuristic for 12Gb cuda memory
        assert (S > 0), "Cannot fit individual data in memory, consider smaller K"
        return min(B, S)

    def _m_iwae(self, x, K=1):
        """IWAE estimate for log p_\theta(x) for multi-modal vae -- fully vectorised"""
        qz_xs, px_zs, zss = self.forward(x, K)
        lws = []
        for r, qz_x in enumerate(qz_xs):
            lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1)
            lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(zss[r]).sum(-1) for qz_x in qz_xs]))
            lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                         .mul(model.vaes[d].llik_scaling).sum(-1)
                     for d, px_z in enumerate(px_zs[r])]
            lpx_z = torch.stack(lpx_z).sum(0)
            lw = lpz + lpx_z - lqz_x
            lws.append(lw)
        return torch.cat(lws)  # (n_modality * n_samples) x batch_size, batch_size

    def compute_loss(model, x, K=1):
        """Computes iwae estimate for log p_\theta(x) for multi-modal vae """
        S = self.compute_microbatch_split(x, K)
        x_split = zip(*[_x.split(S) for _x in x])
        lw = [_m_iwae(model, _x, K) for _x in x_split]
        lw = torch.cat(lw, 1)  # concat on batch
        return log_mean_exp(lw).sum()


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
