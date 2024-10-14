import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, RandomSampler
from torchvision.transforms.v2.functional import hflip, vflip, rotate, center_crop
from torchvision.models import convnext_tiny, resnet50
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url
from timm.models.layers import trunc_normal_, DropPath
import lightning.pytorch as pl
import wandb

from nbodykit.lab import *
import PiInTheSky.binnedEstimatorFlatSky as binEstFlatSky
import Pk_library as PKL

class LSSDataset(Dataset):
    def __init__(
        self,
        df,
        data_dir,
        transform
    ):
        super().__init__()
        self.df = df
        self.data_dir = data_dir
        self.transform = transform

        w_dict = {}
        for z in df['redshift'].unique():
            w = len(df) / len(df[df['redshift']==z])
            w_dict[z] = w
        w_dict = {k: v / min(w_dict.values()) for k, v in w_dict.items()}
        w_val_dict = {0: 2, 0.5: 1, 1: 1, 2: 1, 3: 0}

        self.w_dict = w_dict
        self.w_val_dict = w_val_dict

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        z = torch.tensor([self.df.iloc[idx, 2]]).to(torch.float32)
        omega_m = torch.tensor([self.df.iloc[idx, 3]]).to(torch.float32)
        omega_b = torch.tensor([self.df.iloc[idx, 4]]).to(torch.float32)
        h = torch.tensor([self.df.iloc[idx, 5]]).to(torch.float32)
        n_s = torch.tensor([self.df.iloc[idx, 6]]).to(torch.float32)
        sigma_8 = torch.tensor([self.df.iloc[idx, 7]]).to(torch.float32)
        w = self.w_dict[z.cpu().numpy()[0]]
        w_val = self.w_val_dict[z.cpu().numpy()[0]]

        pair_and_fft = torch.load(f'{self.data_dir}/pairs_pt/{self.df.iloc[idx, 0]}')
        pair = pair_and_fft[0].unsqueeze(1).to(torch.float32)
        fft = pair_and_fft[1]

        if self.transform:
            rotate_and_flip = torch.rand(3)
            translate = torch.randint(low=0, high=pair.size(-1), size=(2,))

            pair = rotate(pair, angle=int(rotate_and_flip[0]//0.25*90))

            pair = (rotate_and_flip[1]//0.5)*pair + (1-rotate_and_flip[1]//0.5)*hflip(pair)
            pair = (rotate_and_flip[2]//0.5)*pair + (1-rotate_and_flip[2]//0.5)*vflip(pair)

            pair = torch.cat((pair[:,:,translate[0]:,:], pair[:,:,:translate[0],:]), dim=-2)
            pair = torch.cat((pair[:,:,:,translate[1]:], pair[:,:,:,:translate[1]]), dim=-1)
        
        return pair[0], pair[1], fft[0], fft[1], z, omega_m, omega_b, h, n_s, sigma_8, w, w_val

class LSSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df_train,
        df_test,
        data_dir,
        val_size,
        batch_size
    ):
        super().__init__()
        self.df_train = df_train
        self.df_test = df_test
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_size = val_size

    def setup(self, stage):
        if stage == 'fit' or stage == 'validate' or stage is None:
            data_full = LSSDataset(self.df_train, self.data_dir, transform=True)
            self.data_train, self.data_val = random_split(data_full, [1-self.val_size, self.val_size])

        if stage == 'test' or stage == 'predict':
            self.data_test = LSSDataset(self.df_test, self.data_dir, transform=True)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, drop_last=True)

    def predict_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, drop_last=True)

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channels_last', 'channels_first']:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == 'channels_first':
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=9, stride=1, padding=4, padding_mode='circular', groups=dim)
        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, padding_mode='circular', groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)

        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        y = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = self.drop_path(x)
        return y + x

class DownsampleBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = LayerNorm(in_dim, eps=1e-6, data_format='channels_first')
        self.conv = nn.Conv2d(in_dim, out_dim , kernel_size=2, stride=2)
        self.convnext1 = ConvNeXtBlock(out_dim)
        self.convnext2 = ConvNeXtBlock(out_dim)
        self.convnext3 = ConvNeXtBlock(out_dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = self.convnext1(x)
        x = self.convnext2(x)
        x = self.convnext3(x)
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = LayerNorm(in_dim, eps=1e-6, data_format='channels_first')
        self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.conv = nn.Conv2d(in_dim+out_dim, out_dim, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.convnext1 = ConvNeXtBlock(out_dim)
        self.convnext2 = ConvNeXtBlock(out_dim)
        self.convnext3 = ConvNeXtBlock(out_dim)

    def forward(self, x, s):
        x = self.norm(x)
        x = self.up(x)
        x = torch.cat([x, s], axis=1)
        x = self.conv(x)
        x = self.convnext1(x)
        x = self.convnext2(x)
        x = self.convnext3(x)
        return x

class ConditioningBlock(nn.Module):
    def __init__(self, hidden_features, out_features):
        super().__init__()
        self.emb1 = nn.Linear(in_features=6, out_features=hidden_features)
        self.emb2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.act = nn.GELU()

    def forward(self, x, l):
        l = self.emb1(l)
        l = self.act(l)
        l = self.emb2(l)

        if len(x.shape) == 4:
            l = l.view(l.size(0), l.size(1), 1, 1).expand(-1, -1, x.size(-2), x.size(-1)).type_as(x)

        x = torch.cat((x, l), dim=1)
        return x

class ConvNeXtUnet(nn.Module):
    def __init__(self, shape):
        super().__init__()
        dims = [64, 128, 256, 512]
        # dims = [96, 192, 384, 768]

        self.cond1 = ConditioningBlock(hidden_features=6, out_features=8)
        self.inputs = nn.Conv2d(1+8, dims[0], kernel_size=3, stride=1, padding=1, padding_mode='circular')

        self.down1 = DownsampleBlock(dims[0], dims[1])
        self.down2 = DownsampleBlock(dims[1], dims[2])
        self.down3 = DownsampleBlock(dims[2], dims[3])

        self.cond2 = ConditioningBlock(hidden_features=8, out_features=16)
        self.bottle1 = ConvNeXtBlock(dims[-1]+16)
        self.bottle2 = ConvNeXtBlock(dims[-1]+16)
        self.bottle3 = ConvNeXtBlock(dims[-1]+16)
        # self.bottle4 = ConvNeXtBlock(dims[-1]+16)

        self.cond3 = ConditioningBlock(hidden_features=8, out_features=16)
        self.up1 = UpsampleBlock(dims[-1]+16+16, dims[-2])
        self.up2 = UpsampleBlock(dims[-2], dims[-3])
        self.up3 = UpsampleBlock(dims[-3], dims[-4])

        self.cond4 = ConditioningBlock(hidden_features=6, out_features=8)
        self.outputs = nn.Conv2d(dims[0]+8, 1, kernel_size=1, stride=1)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, maps, z, omega_m, omega_b, h, n_s, sigma_8):
        labels = torch.cat((z, omega_m, omega_b, h, n_s, sigma_8), dim=1)

        x = self.cond1(maps, labels)
        s0 = self.inputs(x)

        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)

        b0 = self.cond2(s3, labels)
        b1 = self.bottle1(b0)
        b2 = self.bottle2(b1)
        b3 = self.bottle3(b2)
        # b4 = self.bottle4(b3)

        u0 = self.cond3(b3, labels)
        # u0 = self.cond3(b4, labels)
        u1 = self.up1(u0, s2)
        u2 = self.up2(u1, s1)
        u3 = self.up3(u2, s0)

        x = self.cond4(u3, labels)
        x = self.outputs(x)
        return torch.clamp(x, min=-20, max=10)

# class ConvNeXtCritic(nn.Module):
#     def __init__(self, shape):
#         super().__init__()
#         self.cond1 = ConditioningBlock(hidden_features=8, out_features=2)

#         self.model = convnext_tiny(weights='DEFAULT')

#         self.cond2 = ConditioningBlock(hidden_features=8, out_features=16)

#         self.lin = nn.Linear(in_features=1000+shape//4+16, out_features=256)
#         self.act = nn.GELU()
#         self.outputs = nn.Linear(in_features=256, out_features=1)

#     def forward(self, maps, ps_diff, z, omega_m, omega_b, h, n_s, sigma_8):
#         labels = torch.cat((z, omega_m, omega_b, h, n_s, sigma_8), dim=1)

#         x = self.cond1(maps, labels)
#         x = self.model(x)

#         x = torch.cat((x, ps_diff), dim=1)
#         x = self.cond2(x, labels)
#         x = self.lin(x)
#         x = self.act(x)
#         x = self.outputs(x)
#         return x

class ResNetCritic(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.cond1 = ConditioningBlock(hidden_features=8, out_features=2)

        self.model = resnet50(weights='DEFAULT')

        self.cond2 = ConditioningBlock(hidden_features=8, out_features=16)

        self.lin = nn.Linear(in_features=1000+shape//2+16, out_features=256)
        self.act = nn.GELU()
        self.outputs = nn.Linear(in_features=256, out_features=1)

    def forward(self, maps, ps_diff, z, omega_m, omega_b, h, n_s, sigma_8):
        labels = torch.cat((z, omega_m, omega_b, h, n_s, sigma_8), dim=1)

        x = self.cond1(maps, labels)
        x = self.model(x)

        x = torch.cat((x, ps_diff), dim=1)
        x = self.cond2(x, labels)
        x = self.lin(x)
        x = self.act(x)
        x = self.outputs(x)
        return x

class LSSModel(pl.LightningModule):
    def __init__(
        self,
        shape,
        lambda_gp,
        lambda_pixel,
        batch_size,
        lr_factor,
        learning_rate,
        beta_1,
        beta_2,
        gradient_clip_val
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.shape = shape
        self.lambda_gp = lambda_gp
        self.lambda_pixel = lambda_pixel
        self.batch_size = batch_size
        self.lr_factor = lr_factor
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.gradient_clip_val = gradient_clip_val

        self.generator = ConvNeXtUnet(shape)
        self.critic = ResNetCritic(shape)
        # self.critic = ConvNeXtCritic(shape)

        kfreq = (torch.fft.fftfreq(shape) * shape).requires_grad_(requires_grad=True)
        kfreq2D = torch.meshgrid(kfreq, kfreq, indexing='xy')
        knrm = torch.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2).flatten()
        kbins = torch.arange(0.5, shape//2+1, 1.0).requires_grad_(requires_grad=True)
        kvals = 0.5 * (kbins[1:] + kbins[:-1])
        bucketized = torch.bucketize(knrm, boundaries=kbins)
        surface = np.pi * (kbins[1:]**2 - kbins[:-1]**2)

        self.register_buffer('kvals', kvals)
        self.register_buffer('bucketized', bucketized)
        self.register_buffer('surface', surface)

        self.bins = 64
        self.pixel_edges = {0: (-1.4, 1.4), 0.5: (-1.1, 1.1), 1: (-0.8, 0.8), 2: (-0.6, 0.6), 3: (-0.5, 0.5)}
        self.peak_edges = {0: (-0.6, 2.00), 0.5: (-0.5, 1.75), 1: (-0.4, 1.50), 2: (-0.3, 1.25), 3: (-0.2, 1.00)}

        self.BoxSize = 1000.0
        self.kmin = 1e-5
        self.kmax = 1.0

        self.MAS = 'PCS'
        self.pixScale = self.BoxSize/self.shape

        self.limits_max = torch.finfo(torch.float32).max
        self.limits_min = torch.finfo(torch.float32).min
    
    def forward(self, x, z, omega_m, omega_b, h, n_s, sigma_8):
        return self.generator(x, z, omega_m, omega_b, h, n_s, sigma_8)

    def generator_loss(self, y_hat, y, x_ps, y_hat_ps, z, omega_m, omega_b, h, n_s, sigma_8, w):
        loss_c = -self.critic(y_hat, y_hat_ps-x_ps, z, omega_m, omega_b, h, n_s, sigma_8).squeeze()
        loss_p = self.lambda_pixel*torch.mean(F.mse_loss(y_hat, y, reduction='none'), dim=(1, 2, 3)).squeeze()
        loss = loss_c + loss_p
        return torch.clamp(torch.mean(w*loss), min=self.limits_min, max=self.limits_max)

    def critic_loss(self, y_hat, y, y_hat_fft, y_fft, x_ps, y_hat_ps, y_ps, z, omega_m, omega_b, h, n_s, sigma_8, w):
        loss_f = self.critic(y_hat, y_hat_ps-x_ps, z, omega_m, omega_b, h, n_s, sigma_8).squeeze()
        loss_r = -self.critic(y, y_ps-x_ps, z, omega_m, omega_b, h, n_s, sigma_8).squeeze()
        loss_g = self.lambda_gp*self.gradient_penalty(y_hat, y, y_hat_fft, y_fft, x_ps, z, omega_m, omega_b, h, n_s, sigma_8, w).squeeze()
        loss = loss_f + loss_r + loss_g
        return torch.clamp(torch.mean(w*loss), min=self.limits_min, max=self.limits_max)
    
    def gradient_penalty(self, y_hat, y, y_hat_fft, y_fft, x_ps, z, omega_m, omega_b, h, n_s, sigma_8, w):
        alphas = torch.stack([torch.ones(1, self.shape, self.shape).type_as(y)*alpha for alpha in torch.rand(y_hat.size(0))], dim=0).requires_grad_(requires_grad=True)
        interpolation = alphas*y + (1-alphas)*y_hat
        int_fft = alphas.squeeze()*y_fft + (1-alphas).squeeze()*y_hat_fft
        int_ps = self.train_power_spectrum(int_fft)
        logits = self.critic(interpolation, int_ps-x_ps, z, omega_m, omega_b, h, n_s, sigma_8)
        grad_outputs = torch.ones_like(logits)
        gradients = torch.autograd.grad(
            outputs=logits,
            inputs=interpolation,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0].view(y_hat.size(0), -1)
        return (gradients.norm(2, dim=1) - 1)**2
    
    def train_power_spectrum(self, fft):
        amplitudes = torch.square(torch.abs(fft))
        Abins = torch.stack([torch.tensor([amplitudes[i].flatten()[self.bucketized==j].mean() / 10e9 for j in torch.arange(self.kvals.size(0))]) for i in torch.arange(fft.size(0))]).type_as(amplitudes)
        Abins = Abins * self.surface
        return torch.log(Abins).to(torch.float32)

    def validation_loss(self, y_hat, y, w_val):
        loss = torch.abs(y_hat-y) / y
        loss = torch.nan_to_num(loss, posinf=torch.inf, neginf=torch.inf)
        loss = torch.stack([torch.nan_to_num(loss[i], posinf=loss[i, loss[i]!=torch.inf].mean()) for i in torch.arange(y_hat.size(0))]).mean(dim=1)
        return torch.mean(w_val*loss)

    def map_images(self, x, y_hat, y):
        cm = plt.get_cmap('viridis')
        buffer = torch.zeros((y_hat.size(0), self.shape, self.shape//32)).type_as(x)
        maps = torch.cat((x, buffer, y_hat, buffer, y), dim=2)
        maps = (maps-torch.min(maps))/(torch.max(maps)-torch.min(maps))
        maps = cm(maps.cpu().numpy())
        return [Image.fromarray((maps[i]*255).astype(np.uint8)) for i in torch.arange(y_hat.size(0))]

    def bin_midpoints(self, bins, edge_min, edges_max):
        edges = torch.linspace(edge_min, edges_max, bins+1)
        return torch.tensor([(edges[i]+edges[i+1])/2 for i in torch.arange(edges.size(0)-1)])

    def pixel_counts(self, inputs, z):
        edges = [self.pixel_edges[float(z[i])] for i in torch.arange(inputs.size(0))]
        pixel_counts = torch.stack([torch.histc(inputs[i], bins=self.bins, min=edges[i][0], max=edges[i][1]) for i in torch.arange(inputs.size(0))])
        bins = self.bin_midpoints(self.bins, edges[0][0], edges[0][1]).type_as(inputs)
        return bins, pixel_counts

    def peak_counts(self, inputs, z):
        edges = [self.peak_edges[float(z[i])] for i in torch.arange(inputs.size(0))]
        inputs = F.pad(inputs, (1,1,1,1), mode='reflect')
        max_pooled = F.max_pool2d(inputs, kernel_size=3, stride=1, padding=1, ceil_mode=True)
        peaks = [inputs[i, torch.isclose(inputs[i], max_pooled[i])] for i in torch.arange(inputs.size(0))]
        peak_counts = torch.stack([torch.histc(peaks[i], bins=self.bins, min=edges[i][0], max=edges[i][1]) for i in torch.arange(inputs.size(0))])
        bins = self.bin_midpoints(self.bins, edges[0][0], edges[0][1]).type_as(inputs)
        return bins, peak_counts
    
    def phase_distributions(self, inputs):
        edges = (0, 2*np.pi)
        phase = torch.angle(torch.fft.fftshift(torch.fft.fft2(inputs)))
        phase_1 = torch.cat((phase[:,1:,:], torch.unsqueeze(phase[:,0,:], 1)), dim=1)
        d = torch.where(phase>phase_1, phase-phase_1, phase+(2*np.pi-phase_1))
        hist = torch.zeros((inputs.size(0), self.bins)).type_as(inputs)
        for i in torch.arange(inputs.size(0)):
            hist_i = torch.histc(d[i], bins=self.bins, min=edges[0], max=edges[1])
            start = torch.argmax(hist_i)
            hist[i] = torch.cat((hist_i[start:], hist_i[:start]))
        bins = self.bin_midpoints(self.bins, edges[0], edges[1]).type_as(inputs)
        return bins, hist

    def val_power_spectrum(self, inputs):
        field_meshes = [ArrayMesh(inputs[i].cpu().detach().numpy().astype(np.float32), BoxSize=[self.BoxSize, self.BoxSize]) for i in torch.arange(inputs.size(0))]
        spectra = [FFTPower(field_mesh, mode='1d', kmin=self.kmin, kmax=self.kmax) for field_mesh in field_meshes]
        k = spectra[0].power['k'][3:]
        spectra = np.array([spectrum.power['power'].real[3:] for spectrum in spectra])
        return k, spectra
    
    def find_nearest_id(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return int(idx)
    
    def bispectrum(self, inputs, k1, k2):
        binEdges = np.arange(0.005, k1+k2, 0.02)
        fft = torch.fft.fft2(inputs).cpu().detach().numpy()
        inputs = inputs.cpu().detach().numpy().astype(np.float32)
        k3 = np.arange(k2-k1, k2+k1, 0.02)
        theta = np.arccos((k3**2 - k1**2 - k2**2)/(-2*k1*k2))
        theta[0] = 0
        theta[-1] = np.pi
        bin_mid_points = ((binEdges[1:]+binEdges[:-1]) / 2)[:-4]
        arg_k1 = self.find_nearest_id(bin_mid_points, k1)
        arg_k2 = self.find_nearest_id(bin_mid_points, k2)
        EST = binEstFlatSky.binnedEstimator(self.shape, self.shape, self.pixScale, self.pixScale, binEdges=binEdges, invC=0)
        bispectra = np.zeros((inputs.shape[0], len(theta)))
        reduced_bispecta = np.zeros((inputs.shape[0], len(theta)))
        bispecs = []
        pss = []
        for i in range(inputs.shape[0]):
            bispec, _, _ = EST.analyze([fft[i]], calcNorm=0)
            bispecs.append(bispec)
            Pk2D = PKL.Pk_plane(inputs[i], self.BoxSize, self.MAS)
            pss.append(Pk2D)
            for j, k in enumerate(k3):
                arg_k3 = self.find_nearest_id(bin_mid_points, k)
                indices = np.sort([arg_k1, arg_k2, arg_k3])[::-1]
                bispectra[i, j] = bispecs[i][0, 0, 0, indices[0], indices[1], indices[2]]
                k_values = pss[i].k
                ps_values = pss[i].Pk
                arg_k1_ps = self.find_nearest_id(k_values, k1)
                arg_k2_ps = self.find_nearest_id(k_values, k2)
                arg_k3_ps = self.find_nearest_id(k_values, k)
                ps_values_1 = ps_values[arg_k1_ps]
                ps_values_2 = ps_values[arg_k2_ps]
                ps_values_3 = ps_values[arg_k3_ps]
                reduced_bispecta[i, j] = bispectra[i, j] / (ps_values_1*ps_values_2+ps_values_1*ps_values_3+ps_values_2*ps_values_3)
        return theta, bispectra, reduced_bispecta

    def training_step(self, batch, batch_idx):      
        x, y, x_fft, y_fft, z, omega_m, omega_b, h, n_s, sigma_8, w, w_val = batch

        y_hat = self.forward(x, z, omega_m, omega_b, h, n_s, sigma_8)
        y_hat_fft = torch.fft.fft2(torch.exp(y_hat.squeeze()) - 1)

        x_ps = self.train_power_spectrum(x_fft)
        y_hat_ps = self.train_power_spectrum(y_hat_fft)
        y_ps = self.train_power_spectrum(y_fft)

        optimiser_g, optimiser_c = self.optimizers()

        self.toggle_optimizer(optimiser_g)
        loss_g = self.generator_loss(y_hat, y.detach(), x_ps.detach(), y_hat_ps, z, omega_m, omega_b, h, n_s, sigma_8, w)
        self.log('loss_g', loss_g, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        optimiser_g.zero_grad(set_to_none=True)
        self.manual_backward(loss_g, retain_graph=True)
        self.clip_gradients(optimiser_g, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm='norm')
        optimiser_g.step()
        self.untoggle_optimizer(optimiser_g)

        self.toggle_optimizer(optimiser_c)
        loss_c = self.critic_loss(y_hat.detach(), y.detach(), y_hat_fft.detach(), y_fft.detach(), x_ps.detach(), y_hat_ps.detach(), y_ps.detach(), z, omega_m, omega_b, h, n_s, sigma_8, w)
        self.log('loss_c', loss_c, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        optimiser_c.zero_grad(set_to_none=True)
        self.manual_backward(loss_c, retain_graph=True)
        self.clip_gradients(optimiser_c, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm='norm')
        optimiser_c.step()
        self.untoggle_optimizer(optimiser_c)

    def validation_step(self, batch, batch_idx):
        x, y, x_fft, y_fft, z, omega_m, omega_b, h, n_s, sigma_8, w, w_val = batch
        y_hat = self.forward(x, z, omega_m, omega_b, h, n_s, sigma_8)
        x, y_hat, y = x.squeeze(), y_hat.squeeze(), y.squeeze()

        if batch_idx == 0:
            maps = self.map_images(x, y_hat, y)
            self.logger.experiment.log({f'val_map_examples_{self.current_epoch}': [wandb.Image(m) for m in maps]})

        bins, pixel_counts_hat = self.pixel_counts(y_hat, z)
        bins, pixel_counts_target = self.pixel_counts(y, z)
        pixel_counts_loss = self.validation_loss(pixel_counts_hat, pixel_counts_target, w_val)

        x = torch.exp(x) - 1
        y_hat = torch.exp(y_hat) - 1
        y = torch.exp(y) - 1

        bins, peak_counts_hat = self.peak_counts(y_hat, z)
        bins, peak_counts_target = self.peak_counts(y, z)
        peak_counts_loss = self.validation_loss(peak_counts_hat, peak_counts_target, w_val)

        bins, phase_distributions_hat = self.phase_distributions(y_hat)
        bins, phase_distributions_targets = self.phase_distributions(y)
        phase_distributions_loss = self.validation_loss(phase_distributions_hat, phase_distributions_targets, w_val)

        k, power_yhat = self.val_power_spectrum(y_hat)
        k, power_y = self.val_power_spectrum(y)
        power_loss = self.validation_loss(torch.from_numpy(power_yhat).type_as(x), torch.from_numpy(power_y).type_as(x), w_val)

        val_loss = (pixel_counts_loss + peak_counts_loss + phase_distributions_loss + 7*power_loss) / 10
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.batch_size)

    def test_step(self, batch, batch_idx):
        out_dir = f'./out/models/{self.lambda_gp}_{self.lambda_pixel}/test_set/results/'

        x, y, x_fft, y_fft, z, omega_m, omega_b, h, n_s, sigma_8, w, w_val = batch
        y_hat = self.forward(x, z, omega_m, omega_b, h, n_s, sigma_8)
        x, y_hat, y = x.squeeze(), y_hat.squeeze(), y.squeeze()
              
        titles = ['Lognormal']*self.batch_size + ['Prediction']*self.batch_size + ['Quijote']*self.batch_size

        bins, pixel_counts_x = self.pixel_counts(x, z)
        bins, pixel_counts_yhat = self.pixel_counts(y_hat, z)
        bins, pixel_counts_y = self.pixel_counts(y, z)
        pixel_counts = torch.cat((pixel_counts_x, pixel_counts_yhat, pixel_counts_y), dim=0)

        df_pixel_counts = pd.DataFrame(data=pixel_counts.cpu().numpy(), columns=bins.cpu().numpy().tolist())
        df_pixel_counts = df_pixel_counts.assign(label=titles)
        df_pixel_counts.to_csv(f'{out_dir}/pixel_counts.csv', mode='a', index=False, header=(not Path(f'{out_dir}/pixel_counts.csv').is_file()))

        x = torch.exp(x) - 1
        y_hat = torch.exp(y_hat) - 1
        y = torch.exp(y) - 1

        bins, peak_counts_x = self.peak_counts(x, z)
        bins, peak_counts_yhat = self.peak_counts(y_hat, z)
        bins, peak_counts_y = self.peak_counts(y, z)
        peak_counts = torch.cat((peak_counts_x, peak_counts_yhat, peak_counts_y), dim=0)

        df_peak_counts = pd.DataFrame(data=peak_counts.cpu().numpy(), columns=bins.cpu().numpy().tolist())
        df_peak_counts = df_peak_counts.assign(label=titles)
        df_peak_counts.to_csv(f'{out_dir}/peak_counts.csv', mode='a', index=False, header=(not Path(f'{out_dir}/peak_counts.csv').is_file()))

        bins, phase_distributions_x = self.phase_distributions(x)
        bins, phase_distributions_yhat = self.phase_distributions(y_hat)
        bins, phase_distributions_y = self.phase_distributions(y)
        phase_distributions = torch.cat((phase_distributions_x, phase_distributions_yhat, phase_distributions_y), dim=0)

        df_phase_distributions = pd.DataFrame(data=phase_distributions.cpu().numpy(), columns=bins.cpu().numpy().tolist())
        df_phase_distributions = df_phase_distributions.assign(label=titles)
        df_phase_distributions.to_csv(f'{out_dir}/phase_distributions.csv', mode='a', index=False, header=(not Path(f'{out_dir}/phase_distributions.csv').is_file()))

        k, power_x = self.val_power_spectrum(x)
        k, power_yhat = self.val_power_spectrum(y_hat)
        k, power_y = self.val_power_spectrum(y)
        power = np.concatenate((power_x, power_yhat, power_y), axis=0)

        df_power = pd.DataFrame(data=power, columns=k.tolist())
        df_power = df_power.assign(label=titles)
        df_power.to_csv(f'{out_dir}/power_spectrum.csv', mode='a', index=False, header=(not Path(f'{out_dir}/power_spectrum.csv').is_file()))

        k1 = 0.4
        k2 = 0.6
        theta, bi_x, red_x = self.bispectrum(x, k1, k2)
        theta, bi_yhat, red_yhat = self.bispectrum(y_hat, k1, k2)
        theta, bi_y, red_y = self.bispectrum(y, k1, k2)
        bi = np.concatenate((bi_x, bi_yhat, bi_y), axis=0)
        red = np.concatenate((red_x, red_yhat, red_y), axis=0)

        df_bi = pd.DataFrame(data=bi, columns=theta.tolist())
        df_bi = df_bi.assign(label=titles)
        df_bi.to_csv(f'{out_dir}/bispectrum_{k1}_{k2}.csv', mode='a', index=False, header=(not Path(f'{out_dir}/bispectrum_{k1}_{k2}.csv').is_file()))
    
        df_red = pd.DataFrame(data=red, columns=theta.tolist())
        df_red = df_red.assign(label=titles)
        df_red.to_csv(f'{out_dir}/reduced_bispectrum_{k1}_{k2}.csv', mode='a', index=False, header=(not Path(f'{out_dir}/reduced_bispectrum_{k1}_{k2}.csv').is_file()))

        k1 = 0.4
        k2 = 0.4
        theta, bi_x, red_x = self.bispectrum(x, k1, k2)
        theta, bi_yhat, red_yhat = self.bispectrum(y_hat, k1, k2)
        theta, bi_y, red_y = self.bispectrum(y, k1, k2)
        bi = np.concatenate((bi_x, bi_yhat, bi_y), axis=0)
        red = np.concatenate((red_x, red_yhat, red_y), axis=0)

        df_bi = pd.DataFrame(data=bi, columns=theta.tolist())
        df_bi = df_bi.assign(label=titles)
        df_bi.to_csv(f'{out_dir}/bispectrum_{k1}_{k2}.csv', mode='a', index=False, header=(not Path(f'{out_dir}/bispectrum_{k1}_{k2}.csv').is_file()))
    
        df_red = pd.DataFrame(data=red, columns=theta.tolist())
        df_red = df_red.assign(label=titles)
        df_red.to_csv(f'{out_dir}/reduced_bispectrum_{k1}_{k2}.csv', mode='a', index=False, header=(not Path(f'{out_dir}/reduced_bispectrum_{k1}_{k2}.csv').is_file()))

    def configure_optimizers(self):
        optimiser_g = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate*(0.5**0), betas=(self.beta_1, self.beta_2))
        optimiser_c = torch.optim.Adam(self.critic.parameters(), lr=self.lr_factor*self.learning_rate*(0.5**0), betas=(self.beta_1, self.beta_2))
        return [optimiser_g, optimiser_c], []