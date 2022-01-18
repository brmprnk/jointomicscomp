from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from  torch.distributions import Laplace, Independent
from src.nets import CrossGeneratingVariationalAutoencoder

def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))

def _m_dreg_looser(model, x1, x2):
    """DREG estimate for log p_\theta(x) for multi-modal vae -- fully vectorised
    This version is the looser bound---with the average over modalities outside the log
    """
    qz_xs, px_zs, zss = model(x1, x2)

    qz_xs_ = [Independent(Laplace(qz_xs[0].base_dist.loc.detach(), qz_xs[0].base_dist.scale.detach()), 0), Independent(Laplace(qz_xs[1].base_dist.loc.detach(), qz_xs[1].base_dist.scale.detach()), 0)]

    # replaced by the thing above
    #qz_xs_ = [vae.qz_x(*[p.detach() for p in vae.qz_x_params]) for vae in model.vaes]

    lpz_1 = model.pz.log_prob(zss[0])
    lpz_2 = model.pz.log_prob(zss[1])

    lqz_x_1 = log_mean_exp(torch.stack([qz_x_.log_prob(zss[0]) for qz_x_ in qz_xs_]))
    lqz_x_2 = log_mean_exp(torch.stack([qz_x_.log_prob(zss[1]) for qz_x_ in qz_xs_]))

    x = (x1, x2)

    lpx_z_1 = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1).mul(model.vaes[d].llik_scaling).sum(-1) for d, px_z in enumerate(px_zs[0])]
    lpx_z_2 = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1).mul(model.vaes[d].llik_scaling).sum(-1) for d, px_z in enumerate(px_zs[1])]






    for r, vae in enumerate(model.vaes):
        lpz = model.pz.log_prob(zss[r]).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_x_.log_prob(zss[r]).sum(-1) for qz_x_ in qz_xs_]))
        lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                     .mul(model.vaes[d].llik_scaling).sum(-1)
                 for d, px_z in enumerate(px_zs[r])]
        lpx_z = torch.stack(lpx_z).sum(0)
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)
    return torch.stack(lws), torch.stack(zss)



class MixtureOfExperts(CrossGeneratingVariationalAutoencoder):
    """Return parameters for mixture of independent experts.

    https://arxiv.org/pdf/1911.03393.pdf
    """

    def __init__(self, input_dim1, input_dim2, enc_hidden_dim=[100], dec_hidden_dim=[], loss1='bce', loss2='bce',
                 use_batch_norm=False, dropoutP=0.0, optimizer_name='Adam', encoder1_lr=1e-4, decoder1_lr=1e-4,
                 enc1_lastActivation='none', enc1_outputScale=1., encoder2_lr=1e-4, decoder2_lr=1e-4,
                 enc2_lastActivation='none', enc2_outputScale=1., enc_distribution='laplace', beta=1.0, K=20):
        super(MixtureOfExperts, self).__init__(input_dim1, input_dim2, enc_hidden_dim, dec_hidden_dim, loss1, loss2,
                                               use_batch_norm, dropoutP,
                                           optimizer_name, encoder1_lr, decoder1_lr, enc1_lastActivation, enc1_outputScale,
                                           encoder2_lr, decoder2_lr, enc2_lastActivation, enc2_outputScale, enc_distribution, beta)

        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2

        # nr of samples to draw for IWAE
        self.K = K

        self.pz = Independent(Laplace(torch.zeros(enc_hidden_dim[-1]), torch.ones(enc_hidden_dim[-1])), 1)
        grad = {'requires_grad': False}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, enc_hidden_dim[-1]), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, enc_hidden_dim[-1]), requires_grad=False)  # logvar
        ])

        # Using this, loss is ignored
        self.px_z = torch.distributions.Normal
        self.qz_x = torch.distributions.Laplace(torch.zeros(1, enc_hidden_dim[-1]), torch.ones(1, enc_hidden_dim[-1]))

    @property
    def pz_params(self):
        eta = 1e-6
        return self._pz_params[0], \
               F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(1) + eta

    def forward(self, x1, x2):
        # qz_x in MoE

        z1 = self.encoder(x1)
        z1sample = z1.rsample(torch.Size([self.K]))  # zs from MoE

        x1_hat = self.decoder(z1sample)

        z2 = self.encoder2(x2)
        z2sample = z2.rsample(torch.Size([self.K]))  # zs from MoE
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

    def log_mean_exp(self, value, dim=0, keepdim=False):
        return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))

    def is_multidata(dataB):
        return isinstance(dataB, list) or isinstance(dataB, tuple)

    def compute_microbatch_split(self, x):
        """ Checks if batch needs to be broken down further to fit in memory.

        Found in MMVAE/src/objectives.py
        """
        B = x[0].size(0)
        S = sum([1.0 / (self.K * np.prod(_x.size()[1:])) for _x in x])
        S = int(1e8 * S)  # float heuristic for 12Gb cuda memory
        assert (S > 0), "Cannot fit individual data in memory, consider smaller K"
        return min(B, S)


    def _m_dreg_looser(model, x1, x2):
        """DREG estimate for log p_\theta(x) for multi-modal vae -- fully vectorised
        This version is the looser bound---with the average over modalities outside the log
        """
        qz_xs, px_zs, zss = model(x1, x2)
        qz_xs_ = [vae.qz_x(*[p.detach() for p in vae.qz_x_params]) for vae in model.vaes]
        lws = []
        for r, vae in enumerate(model.vaes):
            lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1)
            lqz_x = log_mean_exp(torch.stack([qz_x_.log_prob(zss[r]).sum(-1) for qz_x_ in qz_xs_]))
            lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                         .mul(model.vaes[d].llik_scaling).sum(-1)
                     for d, px_z in enumerate(px_zs[r])]
            lpx_z = torch.stack(lpx_z).sum(0)
            lw = lpz + lpx_z - lqz_x
            lws.append(lw)
        return torch.stack(lws), torch.stack(zss)


    def compute_loss(model, x1, x2):
        """Computes dreg estimate for log p_\theta(x) for multi-modal vae
        This version is the looser bound---with the average over modalities outside the log

        This function is called m_dreg_looser in the MMVAE repo's objective.py

        """
        S = model.compute_microbatch_split((x1, x2))
        x_split = zip(*[_x.split(S) for _x in (x1, x2)])
        lw, zss = zip(*[_m_dreg_looser(model, _x[0], _x[1]) for _x in x_split])
        lw = torch.cat(lw, 2)  # concat on batch
        zss = torch.cat(zss, 2)  # concat on batch
        with torch.no_grad():
            grad_wt = (lw - torch.logsumexp(lw, 1, keepdim=True)).exp()
            if zss.requires_grad:
                zss.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
        return (grad_wt * lw).mean(0).sum()


    def evaluate(self, x1, x2):
        metrics = dict()

        metrics['loss'] = 0.

        return metrics


if __name__ == '__main__':
    x1 = torch.rand(100,3)
    x2 = torch.rand(100,3)

    model = MixtureOfExperts(3, 3, enc_hidden_dim=[5])

    l = model.compute_loss(x1, x2)
