# PolyMNIST-PolyMNIST multi-modal model specification
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from numpy import sqrt, prod
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from datasets_PolyMNIST import PolyMNISTDataset
from .mmvae import MMVAE
from .vae_polymnist import PolyMNIST
from torch.autograd import Variable
from utils import Constants
import math

class NonLinear(nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation=None):
        super(NonLinear, self).__init__()

        self.activation = activation
        self.linear = nn.Linear(int(input_size), int(output_size), bias=bias)

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation( h )

        return h

def normal_init(m, mean=0., std=0.01):
    m.weight.data.normal_(mean, std)

class PolyMNIST_5modalities(MMVAE):
    def __init__(self, params, device):
        super(PolyMNIST_5modalities, self).__init__(device, dist.Laplace, params, PolyMNIST(params,device,0.0027,0.0005, False), 
                                                    PolyMNIST(params,device,0.011,0.002, False), PolyMNIST(params,device,0.0034,0.0007, False), 
                                                    PolyMNIST(params,device,0.0007,0.0001, True, 1.0), PolyMNIST(params,device,-0.0109,0.002, False))
        self._pu_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim_w + params.latent_dim_z), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim_w + params.latent_dim_z), requires_grad=False)  # logvar
        ])
        # self._pz_params = nn.ParameterList([
        #     nn.Parameter(torch.zeros(1, params.latent_dim_z), requires_grad=False),  # mu
        #     nn.Parameter(torch.zeros(1, params.latent_dim_z), requires_grad=False)  # logvar
        # ])
        # REMOVE LLIK SCALING
        # self.vaes[0].llik_scaling = prod(self.vaes[1].dataSize) / prod(self.vaes[0].dataSize) \
            # if params.llik_scaling == 0 else params.llik_scaling
        self.modelName = 'polymnist-5modalities'
        # Fix model names for indiviudal models to be saved
        for idx, vae in enumerate(self.vaes):
            vae.modelName = 'polymnist_m'+str(idx)
            vae.llik_scaling = 1.0
        self.tmpdir = params.tmpdir
        self.number_components = 20
        self.input_size = 12544
        self.device = device
        self.pseudoinputs_mean = 0.0
        self.pseudoinputs_std = 0.002
        self.fc_mu_z = nn.Linear(12544, params.latent_dim_z)
        self.fc_lv_z = nn.Linear(12544, params.latent_dim_z)
        self.add_pseudo_outz_inputs()

    @property
    def pu_params(self):
        return self._pu_params[0], F.softmax(self._pu_params[1], dim=1) * self._pu_params[1].size(-1)

    def log_prob_z(self, zs):
        C = self.number_components
        X = self.means_z(self.idle_outz_input)
        qz_x_mean, qz_x_lv = self.generate_pzx_params(X)
        q_z = self.pz(qz_x_mean, qz_x_lv)
        # expand z
        z_expand = zs.view(-1,32).unsqueeze(1)

        a = q_z.log_prob(z_expand).sum(-1) - math.log(C)
        a_max, _ = torch.max(a, 1)  # MB x 1

        # calculte log-sum-exp
        log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))  # MB x 1
        return log_prior.view(zs.size()[0],zs.size()[1])

    #@property
    #def pu_params(self):
    #    return self._pu_params[0], F.softplus(self._pu_params[1]) + Constants.eta

    #def setTmpDir(self, tmpdir):
    #    self.tmpdir = tmpdir

    def add_pseudo_outz_inputs(self):
        nonlinearity = nn.Hardtanh(min_val=-1.0, max_val=1.0)
        self.means_z = NonLinear(self.number_components, self.input_size, bias=False, activation=nonlinearity)

        # init pseudo-inputs
        normal_init(self.means_z.linear, self.pseudoinputs_mean, self.pseudoinputs_std)

        # create an idle input for calling pseudo-inputs
        self.idle_outz_input = Variable(torch.eye(self.number_components, self.number_components), requires_grad=False)
        self.idle_outz_input = self.idle_outz_input.to(self.device)

    def generate_pzx_params(self, out_z):
        lv_z = self.fc_lv_z(out_z)
        return self.fc_mu_z(out_z), F.softmax(lv_z, dim=-1) * lv_z.size(-1) + Constants.eta

    def getDataLoaders(self, batch_size, shuffle=True, device='cuda'):
        tx = transforms.ToTensor()
        unim_train_datapaths = [self.tmpdir+"/PolyMNIST/train/" + "m" + str(i) for i in [0, 1, 2, 3, 4]]
        unim_test_datapaths = [self.tmpdir+"/PolyMNIST/test/" + "m" + str(i) for i in [0, 1, 2, 3, 4]]
        dataset_PolyMNIST_train = PolyMNISTDataset(unim_train_datapaths, transform=tx)
        dataset_PolyMNIST_test = PolyMNISTDataset(unim_test_datapaths, transform=tx)
        kwargs = {} if device == 'cpu' else {'num_workers': 2, 'pin_memory': True}
        train = DataLoader(dataset_PolyMNIST_train, batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(dataset_PolyMNIST_test, batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train, test

    def generate(self):
        N = 100
        outputs = []
        samples_list = super(PolyMNIST_5modalities, self).generate(N)
        for i, samples in enumerate(samples_list):
            samples = samples.data.cpu()
            samples = samples.view(N, *samples.size()[1:])
            outputs.append(make_grid(samples, nrow=int(sqrt(N))))
        return outputs

    def generate_for_calculating_unconditional_coherence(self, N):
        samples_list = super(PolyMNIST_5modalities, self).generate(N)
        return [samples.data.cpu() for samples in samples_list]

    def generate_for_fid(self, savedir, num_samples, tranche):
        N = num_samples
        samples_list = super(PolyMNIST_5modalities, self).generate(N)
        for i, samples in enumerate(samples_list):
            samples = samples.data.cpu()
            for image in range(samples.size(0)):
                save_image(samples[image, :, :, :], '{}/random/m{}/{}_{}.png'.format(savedir, i, tranche, image))

    def reconstruct_for_fid(self, data, savedir, i):
        recons_mat = super(PolyMNIST_5modalities, self).reconstruct_and_cross_reconstruct_fid([d for d in data])
        for r, recons_list in enumerate(recons_mat):
            for o, recon in enumerate(recons_list):
                recon = recon.squeeze(0).cpu()
                for image in range(recon.size(0)):
                    save_image(recon[image, :, :, :],
                                '{}/m{}/m{}/{}_{}.png'.format(savedir, r,o, image, i))

    def cross_generate(self, data):
        N = 10
        recon_triess = [[[] for i in range(N)] for j in range(N)]
        outputss = [[[] for i in range(N)] for j in range(N)]
        for i in range(10):
            recons_mat = super(PolyMNIST_5modalities, self).reconstruct_and_cross_reconstruct_fid([d[:N] for d in data])
            for r, recons_list in enumerate(recons_mat):
                for o, recon in enumerate(recons_list):
                      recon = recon.squeeze(0).cpu()
                      recon_triess[r][o].append(recon)
        for r, recons_list in enumerate(recons_mat):
            for o, recon in enumerate(recons_list):
                outputss[r][o] = make_grid(torch.cat([data[r][:N].cpu()]+recon_triess[r][o]), nrow=N)
        return outputss

def resize_img(img, refsize):
    return F.pad(img, (2, 2, 2, 2)).expand(img.size(0), *refsize)
