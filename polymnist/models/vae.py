# Base VAE class definition

import torch
import torch.nn as nn
import math
from utils import get_mean
import numpy as np
import torch.nn.functional as F
class VAE(nn.Module):
    def __init__(self, prior_dist, likelihood_dist, post_dist, enc, dec, params, use_gs, tau):
        super(VAE, self).__init__()
        self.pu = prior_dist
        self.px_u = likelihood_dist
        self.qu_x = post_dist
        self.qw_x = post_dist
        self.enc = enc
        self.dec = dec
        self.modelName = None
        self.params = params
        self._pu_params = None  # defined in subclass
        self._qu_x_params = None  # populated in `forward`
        self.llik_scaling = 1.0
        self._pw_params = None # defined in subclass
        self.sftmx = nn.Softmax(dim=-1)
        self.use_gs = use_gs
        self.tau = tau

    @property
    def pu_params(self):
        return self._pu_params

    @property
    def pw_params(self):
        X = self.means(self.idle_input)
        X = X.detach()
        qw_x_mean, qw_x_lv = self.enc.generate_qwx_params(X)
        return qw_x_mean, qw_x_lv

    @property
    def qu_x_params(self):
        if self._qu_x_params_mean is None:
            raise NameError("qz_x params not initalised yet!")
        return self._qu_x_params_mean, self._qu_x_params_lv

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        # handle merging individual datasets appropriately in sub-class
        raise NotImplementedError

    def forward(self, x, K=1):
        self.out_w, self.out_z, self._qu_x_params_mean, self._qu_x_params_lv = self.enc(x)
        qu_x = self.qu_x(self._qu_x_params_mean, self._qu_x_params_lv)
        us = qu_x.rsample(torch.Size([K]))
        px_u = self.px_u(*self.dec(us))
        return qu_x, px_u, us

    def sample_w_uncon(self, N):
        indx = torch.randint(0, self.number_components, (N,))
        X = self.means(self.idle_input)[indx]
        # calculate params for given data
        qw_x_mean, qw_x_lv = self.enc.generate_qwx_params(X)
        latents_w = self.qw_x(qw_x_mean, qw_x_lv).rsample().view(N,1,-1)
        return latents_w

    def sample_w_con(self, K, out_w):
        BS = out_w.size()[0]

        out_w_re = out_w.reshape(out_w.size()[0], 1, -1)
        X = self.means(self.idle_input)
        X_re = X.reshape(1, X.size()[0], X.size()[1])
        distance = torch.sum((out_w_re - X_re)**2, 2)
        if self.use_gs:
            idx = self.sftmx(-distance/self.tau)
            idx = idx.detach()
            X_nn = idx.matmul(X)
        else:
            nearest_neighbor = torch.argmin(distance, 1)
            X_nn = X[nearest_neighbor]

        # calculate params for given data
        qw_x_mean, qw_x_lv = self.enc.generate_qwx_params(X_nn)
        latents_w = self.qw_x(qw_x_mean, qw_x_lv).rsample(torch.Size([K])).view(K,BS,-1)
        return latents_w

    def sample_w_con_fid(self, K, out_w):
        BS = out_w.size()[0]

        X = self.means(self.idle_input)
        indx = torch.randint(0, self.number_components, (BS*K,))
        X_nn = X[indx]

        # calculate params for given data
        qw_x_mean, qw_x_lv = self.enc.generate_qwx_params(X_nn)
        latents_w = self.qw_x(qw_x_mean, qw_x_lv).rsample().view(K,BS,-1)
        return latents_w

    def log_p_w(self, ws):
        C = self.number_components
        q_w = self.qw_x(*self.pw_params)
        # expand z
        w_expand = ws.view(-1,32).unsqueeze(1)

        a = q_w.log_prob(w_expand).sum(-1) - math.log(C)
        a_max, _ = torch.max(a, 1)  # MB x 1

        # calculte log-sum-exp
        log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))  # MB x 1
        return log_prior.view(ws.size()[0],ws.size()[1])

    # def generate(self, N, K):   # Not exposed as here we only train multimodal VAES
    #     self.eval()
    #     with torch.no_grad():
    #         pu = self.pu(*self.pu_params)
    #         latents = pu.rsample(torch.Size([N]))
    #         px_u = self.px_u(*self.dec(latents))
    #         data = px_u.sample(torch.Size([K]))
    #     return data.view(-1, *data.size()[3:])

    # def reconstruct(self, data):  # Not exposed as here we only train multimodal VAES
    #     self.eval()
    #     with torch.no_grad():
    #         qu_x = self.qu_x(*self.enc(data))
    #         latents = qu_x.rsample()  # no dim expansion
    #         px_u = self.px_u(*self.dec(latents))
    #         recon = get_mean(px_u)
    #     return recon
