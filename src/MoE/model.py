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
from src.nets import CrossGeneratingVariationalAutoencoder, getPointEstimate

def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))




class MixtureOfExperts(CrossGeneratingVariationalAutoencoder):
    """Return parameters for mixture of independent experts.

    https://arxiv.org/pdf/1911.03393.pdf
    """

    def __init__(self, input_dims, enc_hidden_dim=[100], dec_hidden_dim=[], likelihoods='normal',
                 use_batch_norm=False, dropoutP=0.0, optimizer_name='Adam', encoder_lr=1e-4, decoder_lr=1e-4,
                 enc_distribution='laplace', beta=1.0, K=20, llik_scaling=None, n_categories=None):
        print(n_categories)
        super(MixtureOfExperts, self).__init__(input_dims, enc_hidden_dim, dec_hidden_dim, likelihoods, use_batch_norm, dropoutP, optimizer_name, encoder_lr, decoder_lr, enc_distribution, beta, n_categories=n_categories)

        if llik_scaling is None:
            self.llik_scaling = torch.ones(self.n_modalities).double().to(self.device)
        else:
            self.llik_scaling = llik_scaling

        self.input_dims = input_dims
        # nr of samples to draw for IWAE
        self.K = K

        self.pz = Independent(Laplace(torch.zeros(enc_hidden_dim[-1]).to(torch.device('cuda:0')), torch.ones(enc_hidden_dim[-1]).to(torch.device('cuda:0'))), 1)

        assert enc_distribution == 'laplace'


    def forward(self, x):
        qz_xs = []
        zss = []

        px_zs = [[None for _ in range(self.n_modalities)] for _ in range(self.n_modalities)]

        for m, (enc, dec) in enumerate(zip(self.encoders, self.decoders)):
            qz_x = enc(x[m])
            zs = qz_x.rsample(torch.Size([self.K]))
            px_z = [dec(zi) for zi in zs]

            qz_xs.append(qz_x)
            zss.append(zs)
            px_zs[m][m] = px_z

        for e, zs in enumerate(zss):
            for d, dec in enumerate(self.decoders):
                if e != d:
                    px_zs[e][d] = [dec(zi) for zi in zs]



        return qz_xs, px_zs, zss


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


    def compute_loss(self, x):
        """Computes dreg estimate for log p_\theta(x) for multi-modal vae
        This version is the looser bound---with the average over modalities outside the log

        This function is called m_dreg_looser in the MMVAE repo's objective.py

        """
        S = self.compute_microbatch_split(x)

        x_split = zip(*[_x.split(S) for _x in x])
        lw, zss = zip(*[self._m_dreg_looser(_x) for _x in x_split])
        lw = torch.cat(lw, 2)  # concat on batch
        zss = torch.cat(zss, 2)  # concat on batch

        with torch.no_grad():
            grad_wt = (lw - torch.logsumexp(lw, 1, keepdim=True)).exp()

            if zss.requires_grad:
                zss.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
        return -1 * (grad_wt * lw).mean(0).mean(-1).sum()


    def _m_dreg_looser(self, x):
        """DREG estimate for log p_\theta(x) for multi-modal vae -- fully vectorised
        This version is the looser bound---with the average over modalities outside the log

        adapted from MMVAE repo's objective.py
        """
        qz_xs, px_zs, zss = self.forward(x)

        qz_xs_ = [Independent(Laplace(qz_xs[i].base_dist.loc.detach(), qz_xs[i].base_dist.scale.detach()), 0) for i in range(self.n_modalities)]

        lws = []

        for i in range(self.n_modalities):
            lpz = self.pz.log_prob(zss[i])

            lqz_x = log_mean_exp(torch.stack([qz_x_.log_prob(zss[i]) for qz_x_ in qz_xs_])).sum(-1)

            lpx_z = [torch.stack([pp.log_prob(x[d]).mul(self.llik_scaling[d]).sum(-1) for pp in px_z]) for d, px_z in enumerate(px_zs[i])]
            #lpx_z = []

            lpx_z = torch.stack(lpx_z).sum(0)

            lw = lpz - lqz_x + lpx_z

            lws.append(lw)



        return torch.stack(lws), torch.stack(zss)



    def evaluate(self, x):
        metrics = {}

        with torch.no_grad():
            S = self.compute_microbatch_split(x)
            assert S == x[0].shape[0]

            qz_xs, px_zs, zss = self.forward(x)

            zmean = [zi.mean for zi in qz_xs]
            zstd = [zi.stddev for zi in qz_xs]

            x_hat = [[dec(zi) for dec in self.decoders] for zi in zmean]

            qz_xs_ = [Independent(Laplace(qz_xs[i].base_dist.loc.detach(), qz_xs[i].base_dist.scale.detach()), 0) for i in range(self.n_modalities)]

            lws = []

            loss = 0.0

            for i in range(self.n_modalities):
                lpz = self.pz.log_prob(zss[i])

                lqz_x = log_mean_exp(torch.stack([qz_x_.log_prob(zss[i]) for qz_x_ in qz_xs_])).sum(-1)

                lpx_z = [torch.stack([pp.log_prob(x[d]).mul(self.llik_scaling[d]).sum(-1) for pp in px_z]) for d, px_z in enumerate(px_zs[i])]
                #lpx_z = []

                lpx_z = torch.stack(lpx_z).sum(0)

                lw = lpz - lqz_x + lpx_z

                lws.append(lw)

            # lw = torch.cat(lw, 2)  # concat on batch
            # zss = torch.cat(zss, 2)  # concat on batch

            grad_wt = (lw - torch.logsumexp(lw, 1, keepdim=True)).exp()

            loss = -grad_wt * lw

            metrics['lw'] = lw.mean(0).mean(-1).sum()
            metrics['grad_wt'] = grad_wt.mean(0).mean(-1).sum()
            metrics['loss'] = loss.mean(0).mean(-1).sum()

            x_hat = [[dec(zi) for dec in self.decoders] for zi in zmean]

            for i in range(self.n_modalities):
                klkey = 'KL/%d' % (i+1)
                for j in range(self.n_modalities):
                    llkey = 'LL%d/%d' % (j+1, i+1)
                    msekey = 'MSE%d/%d' % (j+1, i+1)

                    metrics[llkey] = torch.mean(torch.sum(x_hat[i][j].log_prob(x[j]), 1)).item()
                    metrics[msekey] = torch.mean(torch.sum(torch.nn.MSELoss(reduction='none')(getPointEstimate(x_hat[i][j]), x[j]), 1)).item()


                # lpz = [self.pz.log_prob(zi) for zi in zmean]
                #
                # lqz_x_1 = log_mean_exp(torch.stack([qz_x_.log_prob(zss[0]) for qz_x_ in qz_xs_])).sum(-1)
                # lqz_x_2 = log_mean_exp(torch.stack([qz_x_.log_prob(zss[1]) for qz_x_ in qz_xs_])).sum(-1)
                #
                # metrics['KL/1'] = -torch.mean(lpz_1 - lqz_x_1)


        return metrics
