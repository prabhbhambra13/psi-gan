import os
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from tqdm import tqdm, trange
from itertools import product
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
from torch import nn
from captum.attr import Saliency, NoiseTunnel

from model import LSSModel, LSSDataModule

torch.set_float32_matmul_precision('medium')
torch.cuda.empty_cache()

mpl.rcParams['figure.dpi'] = 300

def calc_discrepancies(model, x, y_hat):
    x = torch.exp(x) - 1
    y_hat = torch.exp(y_hat) - 1

    k, power_x = model.val_power_spectrum(x)
    k, power_y_hat = model.val_power_spectrum(y_hat)

    discrepancies = np.mean(np.abs(power_y_hat - power_x) / power_x, axis=1)

    return discrepancies

class CriticWithPSCalc(nn.Module):
    def __init__(self, model, x):
        super().__init__()
        self.model = model
        self.x = x
    
    def forward(self, y_hat, z, omega_m, omega_b, h, n_s, sigma_8):
        x_fft = torch.fft.fft2(torch.exp(self.x.squeeze(1)) - 1)
        x_ps = self.model.train_power_spectrum(x_fft)
        y_hat_fft = torch.fft.fft2(torch.exp(y_hat.squeeze(1)) - 1)
        y_hat_ps = self.model.train_power_spectrum(y_hat_fft)

        return self.model.critic(y_hat, (y_hat_ps-x_ps), z, omega_m, omega_b, h, n_s, sigma_8).squeeze()

def main():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    model_params = params['model']
    data_params = params['data']
    test_params = params['test']

    shape = model_params['shape']
    lambda_gp = model_params['lambda_gp']
    lambda_pixel = model_params['lambda_pixel']
    batch_size = model_params['batch_size']
    lr_factor = model_params['lr_factor']
    learning_rate = model_params['learning_rate']
    beta_1 = model_params['beta_1']
    beta_2 = model_params['beta_2']
    gradient_clip_val = model_params['gradient_clip_val']

    seed = data_params['seed']
    data_dir = data_params['data_dir']
    val_size = data_params['val_size']

    interpolate_z = test_params['interpolate_z']
    n_success = test_params['n_success']

    np.random.seed(seed)
    df = pd.read_csv(f'./out/unseen_cosmology.csv')
    z_dict = {0:4, 0.5:3, 1:2, 2:1, 3:0}

    lh_params = np.loadtxt('./out/latin_hypercube_params.txt')
    omega_m = 0.3175
    omega_b = 0.049
    hubble = 0.6711
    n_s = 0.9624
    sigma_8 = 0.834
    fiducial = np.array([omega_m, omega_b, hubble, n_s, sigma_8])
    sim = df['sim'].unique()[0]

    ckpt_dir = f'{data_dir}/out_625/models/{lambda_gp}_{lambda_pixel}/'
    out_dir = f'./out/models/{lambda_gp}_{lambda_pixel}/'
    redshifts = [0, 0.5, 1, 2, 3]
    for redshift in redshifts:
        Path(f'{out_dir}/xai_figures/{redshift}/').mkdir(exist_ok=True, parents=True)
    Path(f'{out_dir}/example_maps/').mkdir(exist_ok=True, parents=True)

    model = LSSModel.load_from_checkpoint(f'{ckpt_dir}/checkpoints/model.ckpt')
    model.eval()

    condition_grads_total = np.zeros(6)
    k, _ = model.val_power_spectrum(torch.zeros((1, shape, shape)))
    ps_saliency_z = np.zeros((len(redshifts), len(k)))
    
    for z_idx, redshift in enumerate(redshifts):
        df_test = df[(df['sim']==sim) & (df['redshift']==redshift)]
        row = df_test.iloc[0]

        z = torch.stack([torch.tensor([row.redshift]) for i in torch.arange(batch_size)]).to(dtype=torch.float32, device='cuda')
        omega_m = torch.stack([torch.tensor([row.omega_m]) for i in torch.arange(batch_size)]).to(dtype=torch.float32, device='cuda')
        omega_b = torch.stack([torch.tensor([row.omega_b]) for i in torch.arange(batch_size)]).to(dtype=torch.float32, device='cuda')
        h = torch.stack([torch.tensor([row.h]) for i in torch.arange(batch_size)]).to(dtype=torch.float32, device='cuda')
        n_s = torch.stack([torch.tensor([row.n_s]) for i in torch.arange(batch_size)]).to(dtype=torch.float32, device='cuda')
        sigma_8 = torch.stack([torch.tensor([row.sigma_8]) for i in torch.arange(batch_size)]).to(dtype=torch.float32, device='cuda')

        success_x = torch.zeros((n_success, 1, shape, shape))
        success_y_hat = torch.zeros((n_success, 1, shape, shape))
        success_y = torch.zeros(n_success)
        success_discrepancies = torch.ones(n_success) * np.inf
        success_saliency = torch.zeros((n_success, 1, shape, shape))

        df_test = df_test.sample(frac=1).reset_index(drop=True)

        for i in trange(len(df_test)//batch_size):
            with torch.no_grad():
                file_names = [df_test.iloc[idx, 0] for idx in range(i*batch_size, (i+1)*batch_size)]
                slices = [int(file_name.split('_')[2]) for file_name in file_names]
                x = torch.stack([torch.from_numpy(np.load(f'{data_dir.replace("gpu4", "lustre")}/slices_625/npy/{file_name}')).unsqueeze(0) for file_name in file_names]).to(dtype=torch.float32, device='cuda')
                y_hat = model.forward(x, z, omega_m, omega_b, h, n_s, sigma_8)
                discrepancies = calc_discrepancies(model, x.squeeze(), y_hat.squeeze())
                for j in np.argsort(discrepancies):
                    if discrepancies[j] < torch.max(success_discrepancies):
                        success_x[torch.argmax(success_discrepancies)] = x[j]
                        success_y_hat[torch.argmax(success_discrepancies)] = y_hat[j]
                        success_y[torch.argmax(success_discrepancies)] = slices[j]
                        success_discrepancies[torch.argmax(success_discrepancies)] = discrepancies[j]
        
        quijote = np.load(f'{data_dir.replace("gpu4", "lustre")}/raw/{int(sim)}/{z_dict[row.redshift]}.npy')

        for idx in trange(n_success):
            critic = CriticWithPSCalc(model.to(device='cuda'), success_x[idx].to(device='cuda'))

            saliency = Saliency(critic)
            nt = NoiseTunnel(saliency)

            inputs = success_y_hat[idx].unsqueeze(0).to(device='cuda'), z[0].unsqueeze(0), omega_m[0].unsqueeze(0), omega_b[0].unsqueeze(0), h[0].unsqueeze(0), n_s[0].unsqueeze(0), sigma_8[0].unsqueeze(0)
            grads = nt.attribute(inputs, nt_type='smoothgrad_sq', nt_samples=256, nt_samples_batch_size=batch_size, stdevs=0.2)

            maps = np.array([success_y_hat.detach().cpu().numpy()[idx,0], np.clip(np.log(quijote[int(success_y[idx]):int(success_y[idx])+32].mean(axis=0)+1), -15, 7.5)])
            cm = plt.get_cmap('viridis')
            maps = cm((maps-np.min(maps))/(np.max(maps)-np.min(maps)))
            smoothed_saliency = gaussian_filter(grads[0].squeeze().detach().cpu().numpy(), sigma=2)
            success_saliency[idx] = grads[0]
            diff_map = success_y_hat.detach().cpu().numpy()[idx,0] - np.clip(np.log(quijote[int(success_y[idx]):int(success_y[idx])+32].mean(axis=0)+1), -15, 7.5)

            cm = plt.get_cmap('jet')
            diff_map_cm = cm((diff_map-np.min(diff_map))/(np.max(diff_map)-np.min(diff_map)))
           
            fig, axs = plt.subplots(1, 4, figsize=(20, 6), layout='constrained')
            axs[0].imshow(maps[1])
            axs[0].set_xlabel('$N$-body Simulation', fontsize=32)
            axs[1].imshow(maps[0])
            axs[1].set_xlabel('Psi-GAN Emulation', fontsize=32)
            axs[2].imshow(smoothed_saliency, cmap='jet')
            axs[2].set_xlabel('Saliency Map', fontsize=32)
            axs[3].imshow(diff_map, cmap='jet')
            axs[3].set_xlabel('Difference Map', fontsize=32)
            for i in range(4):
                axs[i].set_xticks([])
                axs[i].set_yticks([])

            plt.savefig(f'{out_dir}/xai_figures/{redshift}/critic_{idx}.png')
            plt.close()

            condition_grads = np.zeros(6)
            for i in range(6):
                condition_grads[i] = grads[i+1].item()
            condition_grads = condition_grads / condition_grads.sum()
            condition_grads_total += condition_grads / (n_success * len(redshifts))

        _, ps_saliency = model.val_power_spectrum(success_saliency.squeeze())
        ps_saliency_z[z_idx] = ps_saliency.mean(axis=0)

    np.save(f'{out_dir}/xai_figures/ps_saliency_z.npy', ps_saliency_z)
    np.save(f'{out_dir}/xai_figures/condition_grads_total.npy', condition_grads_total)
    ps_saliency_z = np.load(f'{out_dir}/xai_figures/ps_saliency_z.npy')
    condition_grads_total = np.load(f'{out_dir}/xai_figures/condition_grads_total.npy')

    fig, axs = plt.subplots(len(redshifts), 1, figsize=(5, 5), sharex=True, layout='constrained')
    cm = plt.get_cmap('plasma')
    colours = cm(np.array(redshifts)/np.max(np.array(redshifts)))
    for z_idx, redshift in enumerate(redshifts):
        axs[z_idx].loglog(k, ps_saliency_z[z_idx]/np.max(ps_saliency_z[z_idx]), label=f'$z={redshifts[z_idx]}$', color=colours[z_idx])
        axs[z_idx].legend(fontsize='large', loc='lower left')
        for pixel in range(1, 5):
            axs[z_idx].axvline(x=shape/1000/pixel, linestyle='--', color='gray')
        axs[z_idx].set_yticks([], [])

    fig.supylabel(r'Normalised Power Spectrum $[h^{-1}~\rm{Mpc}]^2$', fontsize='large')
    fig.supxlabel(r'k $[h~\rm{Mpc}^{-1}]$', fontsize='large')
    plt.savefig(f'{out_dir}/xai_figures/xai_power_spectrum.png')
    plt.close()

    condition_labels = np.array([r'$z$', r'$\Omega_{\rm{m}}$', r'$\Omega_{\rm{b}}$', r'$h$', r'$n_{\rm{s}}$', r'$\sigma_{8}$'])
    fig = plt.figure(layout='constrained')
    plt.bar(condition_labels, condition_grads_total, label=condition_labels, color='tab:blue')
    plt.ylabel('Relative Saliency', fontsize='large')
    plt.xticks(fontsize='large')
    plt.savefig(f'{out_dir}/xai_figures/xai_conditions.png')
    plt.close()

    for sim in [663, 815, 1586]:
        df_ex = pd.DataFrame(columns=df.columns)
        for redshift in redshifts:
            if sim == 663:
                df = pd.read_csv(f'./out/data.csv')
                df_ex = pd.concat((df_ex, df[(df['sim']==sim) & (df['redshift']==redshift)].head(1)))
            else:
                df = pd.read_csv(f'./out/unseen_cosmology.csv')
                df_ex = pd.concat((df_ex, df[(df['sim']==sim) & (df['redshift']==redshift)].head(1)))
        
        if sim != 663:
            slices = [int(df_ex.iloc[z].file_name.split('_')[2]) for z in range(len(redshifts))]
            quijote_stack = [np.clip(np.log(np.load(f'{data_dir.replace("gpu4", "lustre")}/raw/{int(sim)}/{z_dict[redshifts[z]]}.npy')[slices[z]:slices[z]+32].mean(axis=0)+1), -15, 7.5) for z in range(len(redshifts))]

        row = df_ex.iloc[0]
        file_names = [df_ex.iloc[int(i), 0] for i in torch.arange(len(df_ex))]
        z = torch.stack([torch.tensor([redshift]) for redshift in redshifts]).to(dtype=torch.float32, device='cuda')
        omega_m = torch.stack([torch.tensor([row.omega_m]) for i in torch.arange(len(df_ex))]).to(dtype=torch.float32, device='cuda')
        omega_b = torch.stack([torch.tensor([row.omega_b]) for i in torch.arange(len(df_ex))]).to(dtype=torch.float32, device='cuda')
        h = torch.stack([torch.tensor([row.h]) for i in torch.arange(len(df_ex))]).to(dtype=torch.float32, device='cuda')
        n_s = torch.stack([torch.tensor([row.n_s]) for i in torch.arange(len(df_ex))]).to(dtype=torch.float32, device='cuda')
        sigma_8 = torch.stack([torch.tensor([row.sigma_8]) for i in torch.arange(len(df_ex))]).to(dtype=torch.float32, device='cuda')

        with torch.no_grad():
            if sim == 663:
                x = torch.stack([torch.from_numpy(np.load(f'{data_dir.replace("gpu4", "lustre")}/slices_625/npy/{file_name.replace("pt", "npy")}'))[0].unsqueeze(0) for file_name in file_names]).to(dtype=torch.float32, device='cuda')
                y = torch.stack([torch.from_numpy(np.load(f'{data_dir.replace("gpu4", "lustre")}/slices_625/npy/{file_name.replace("pt", "npy")}'))[1].unsqueeze(0) for file_name in file_names]).to(dtype=torch.float32, device='cuda')
            else:
                x = torch.stack([torch.from_numpy(np.load(f'{data_dir.replace("gpu4", "lustre")}/slices_625/npy/{file_name.replace("pt", "npy")}')).unsqueeze(0) for file_name in file_names]).to(dtype=torch.float32, device='cuda')
                y = torch.stack([torch.from_numpy(quijote_stack[z]) for z in range(len(redshifts))]).to(dtype=torch.float32, device='cuda')
            y_hat = model.forward(x, z, omega_m, omega_b, h, n_s, sigma_8)

        maps = model.map_images(x.squeeze(), y_hat.squeeze(), y.squeeze())
        maps = [np.array(map_i) for map_i in maps]
        
        fig, axs = plt.subplots(5, 3, figsize=(10, 18))
        for i in range(5):
            redshift = redshifts[i]
            axs[i, 0].set_ylabel(f'$z={redshift}$', fontsize='xx-large')
            axs[i, 0].imshow(maps[i][:,:512])
            axs[i, 1].imshow(maps[i][:,512+16:512+512+16])
            axs[i, 2].imshow(maps[i][:,-512:])
            for j in range(3):
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
        axs[4, 0].set_xlabel('Lognormal Approximation', fontsize='xx-large')
        axs[4, 1].set_xlabel('Psi-GAN Emulation', fontsize='xx-large')
        axs[4, 2].set_xlabel('$N$-body Simulation', fontsize='xx-large')
        plt.suptitle(f'Simulation #{int(sim)}\n'
            fr'$\Omega_{{\rm{{m}}}}={omega_m[0].item():.4f}, \Omega_{{\rm{{b}}}}={omega_b[0].item():.4f}, h={h[0].item():.4f}, n_{{\rm{{s}}}}={n_s[0].item():.4f}, \sigma_{{8}}={sigma_8[0].item():.4f}$',
            fontsize='xx-large'
        )
        plt.tight_layout()
        plt.savefig(f'{out_dir}/example_maps/examples_{int(sim)}.png')
        plt.close()

    df = pd.read_csv(f'./out/data.csv')
    sims = [663, 542, 950, 1913, 652]
    desc = [r'"Fiducial"', r'Low $\Omega_{\rm{m}}$, Low $\sigma_{8}$', r'Low $\Omega_{\rm{m}}$, High $\sigma_{8}$', r'High $\Omega_{\rm{m}}$, Low $\sigma_{8}$', r'High $\Omega_{\rm{m}}$, High $\sigma_{8}$']
    df_ex = pd.DataFrame(columns=df.columns)
    for sim in sims:
        df_ex = pd.concat((df_ex, df[(df['sim']==sim) & (df['redshift']==redshift)].head(1)))

    file_names = [df_ex.iloc[int(i)].file_name for i in torch.arange(len(df_ex))]
    z = torch.stack([torch.tensor([0]) for i in torch.arange(len(df_ex))]).to(dtype=torch.float32, device='cuda')
    omega_m = torch.stack([torch.tensor([df.iloc[int(i)].omega_m]) for i in torch.arange(len(df_ex))]).to(dtype=torch.float32, device='cuda')
    omega_b = torch.stack([torch.tensor([df.iloc[int(i)].omega_b]) for i in torch.arange(len(df_ex))]).to(dtype=torch.float32, device='cuda')
    h = torch.stack([torch.tensor([df.iloc[int(i)].h]) for i in torch.arange(len(df_ex))]).to(dtype=torch.float32, device='cuda')
    n_s = torch.stack([torch.tensor([df.iloc[int(i)].n_s]) for i in torch.arange(len(df_ex))]).to(dtype=torch.float32, device='cuda')
    sigma_8 = torch.stack([torch.tensor([df.iloc[int(i)].sigma_8]) for i in torch.arange(len(df_ex))]).to(dtype=torch.float32, device='cuda')

    with torch.no_grad():
        x = torch.stack([torch.from_numpy(np.load(f'{data_dir.replace("gpu4", "lustre")}/slices_625/npy/{file_name.replace("pt", "npy")}'))[0].unsqueeze(0) for file_name in file_names]).to(dtype=torch.float32, device='cuda')
        y = torch.stack([torch.from_numpy(np.load(f'{data_dir.replace("gpu4", "lustre")}/slices_625/npy/{file_name.replace("pt", "npy")}'))[1].unsqueeze(0) for file_name in file_names]).to(dtype=torch.float32, device='cuda')
        y_hat = model.forward(x, z, omega_m, omega_b, h, n_s, sigma_8)
    
    maps = [np.array(model.map_images(x.squeeze()[i].unsqueeze(0), y_hat.squeeze()[i].unsqueeze(0), y.squeeze()[i].unsqueeze(0))[0]) for i in range(len(sims))]
    # maps = model.map_images(x.squeeze(), y_hat.squeeze(), y.squeeze())
    # maps = [np.array(map_i) for map_i in maps]

    fig, axs = plt.subplots(5, 3, figsize=(10, 17))
    for i in range(5):
        axs[i, 0].set_ylabel(desc[i], fontsize='xx-large')
        axs[i, 0].imshow(maps[i][:,:512])
        axs[i, 1].imshow(maps[i][:,512+16:512+512+16])
        axs[i, 2].imshow(maps[i][:,-512:])
        for j in range(3):
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    axs[4, 0].set_xlabel('Lognormal Approximation', fontsize='xx-large')
    axs[4, 1].set_xlabel('Psi-GAN Emulation', fontsize='xx-large')
    axs[4, 2].set_xlabel('$N$-body Simulation', fontsize='xx-large')

    plt.tight_layout()
    plt.savefig(f'{out_dir}/example_maps/examples_cosmo.png')
    plt.close()

if __name__ == '__main__':
    main()