import os
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from itertools import product, combinations_with_replacement

import torch

from model import LSSModel, LSSDataModule

os.environ['SLURM_JOB_NAME'] = 'bash'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
torch.set_float32_matmul_precision('medium')
torch.cuda.empty_cache()

def calc_discrepancies(model, x, y_hat):
    x = torch.exp(x) - 1
    y_hat = torch.exp(y_hat) - 1

    k, power_x = model.val_power_spectrum(x)
    k, power_y_hat = model.val_power_spectrum(y_hat)

    discrepancies = np.mean(np.abs(power_y_hat - power_x) / power_x, axis=1)

    return discrepancies

def calc_and_save_metrics(model, x, z_bins, titles, out_dir, test_sim, redshift):
    bins, pixel_counts = model.pixel_counts(x, z_bins)
    df_pixel_counts = pd.DataFrame(data=pixel_counts.cpu().numpy(), columns=bins.cpu().numpy().tolist())
    df_pixel_counts = df_pixel_counts.assign(label=titles)
    df_pixel_counts.to_csv(f'{out_dir}/unseen_cosmology_{test_sim}/results/z={redshift if redshift==0.5 else int(redshift)}_pixel_counts.csv', mode='a', index=False, header=(not Path(f'{out_dir}/unseen_cosmology_{test_sim}/results/z={redshift if redshift==0.5 else int(redshift)}_pixel_counts.csv').is_file()))

    x = torch.exp(x) - 1

    bins, peak_counts = model.peak_counts(x, z_bins)
    df_peak_counts = pd.DataFrame(data=peak_counts.cpu().numpy(), columns=bins.cpu().numpy().tolist())
    df_peak_counts = df_peak_counts.assign(label=titles)
    df_peak_counts.to_csv(f'{out_dir}/unseen_cosmology_{test_sim}/results/z={redshift if redshift==0.5 else int(redshift)}_peak_counts.csv', mode='a', index=False, header=(not Path(f'{out_dir}/unseen_cosmology_{test_sim}/results/z={redshift if redshift==0.5 else int(redshift)}_peak_counts.csv').is_file()))

    bins, phase_distributions = model.phase_distributions(x)
    df_phase_distributions = pd.DataFrame(data=phase_distributions.cpu().numpy(), columns=bins.cpu().numpy().tolist())
    df_phase_distributions = df_phase_distributions.assign(label=titles)
    df_phase_distributions.to_csv(f'{out_dir}/unseen_cosmology_{test_sim}/results/z={redshift if redshift==0.5 else int(redshift)}_phase_distributions.csv', mode='a', index=False, header=(not Path(f'{out_dir}/unseen_cosmology_{test_sim}/results/z={redshift if redshift==0.5 else int(redshift)}_phase_distributions.csv').is_file()))

    k, power = model.val_power_spectrum(x)
    df_power = pd.DataFrame(data=power, columns=k.tolist())
    df_power = df_power.assign(label=titles)
    df_power.to_csv(f'{out_dir}/unseen_cosmology_{test_sim}/results/z={redshift if redshift==0.5 else int(redshift)}_power_spectrum.csv', mode='a', index=False, header=(not Path(f'{out_dir}/unseen_cosmology_{test_sim}/results/z={redshift if redshift==0.5 else int(redshift)}_power_spectrum.csv').is_file()))

    k_values = [0.05, 0.2, 0.4, 0.6]
    for k1, k2 in list(combinations_with_replacement(k_values, 2))[2:4]:
        theta, bi, red = model.bispectrum(x, k1, k2)
        df_bi = pd.DataFrame(data=bi, columns=theta.tolist())
        df_bi = df_bi.assign(label=titles)
        df_bi.to_csv(f'{out_dir}/unseen_cosmology_{test_sim}/results/z={redshift if redshift==0.5 else int(redshift)}_bispectrum_{k1}_{k2}.csv', mode='a', index=False, header=(not Path(f'{out_dir}/unseen_cosmology_{test_sim}/results/z={redshift if redshift==0.5 else int(redshift)}_bispectrum_{k1}_{k2}.csv').is_file()))
        df_red = pd.DataFrame(data=red, columns=theta.tolist())
        df_red = df_red.assign(label=titles)
        df_red.to_csv(f'{out_dir}/unseen_cosmology_{test_sim}/results/z={redshift if redshift==0.5 else int(redshift)}_reduced_bispectrum_{k1}_{k2}.csv', mode='a', index=False, header=(not Path(f'{out_dir}/unseen_cosmology_{test_sim}/results/z={redshift if redshift==0.5 else int(redshift)}_reduced_bispectrum_{k1}_{k2}.csv').is_file()))

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

    batch_size = 16

    np.random.seed(seed)
    df = pd.read_csv(f'./out/unseen_cosmology.csv')
    df = df.drop_duplicates()
    df.to_csv(f'./out/unseen_cosmology.csv', index=False)
    z_dict = {0:4, 0.5:3, 1:2, 2:1, 3:0}

    ckpt_dir = f'{data_dir}/out_625/models/{lambda_gp}_{lambda_pixel}/'
    out_dir = f'./out/models/{lambda_gp}_{lambda_pixel}/'
    data_dir = data_dir.replace('gpu4', 'lustre')

    model = LSSModel.load_from_checkpoint(f'{ckpt_dir}/checkpoints/model.ckpt')
    # model = LSSModel.load_from_checkpoint(f'{ckpt_dir.replace("out", "nottur_out")}/checkpoints/epoch=0.ckpt')
    model.eval()

    for test_sim, redshift in product(df['sim'].unique(), df['redshift'].unique()):
        df_test = df[(df['sim']==test_sim) & (df['redshift']==redshift)]
        row = df_test.iloc[0]

        Path(f'{out_dir}/unseen_cosmology_{int(test_sim)}/results').mkdir(exist_ok=True, parents=True)
        with open(f'{out_dir}/unseen_cosmology_{int(test_sim)}/test_case_info.txt', 'w') as f:
            f.write(f'Specific cosmology #{int(test_sim)} reserved for testing, unseen by training process.')
            f.write('\n')
            f.write(','.join(df.columns[3:]))
            f.write('\n')
            f.write(','.join(str(i) for i in list(row[3:])))

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

        df_test = df_test.sample(frac=1).reset_index(drop=True)

        for i in trange(len(df_test)//batch_size):
            with torch.no_grad():
                file_names = [df_test.iloc[idx, 0] for idx in range(i*batch_size, (i+1)*batch_size)]
                slices = [int(file_name.split('_')[2]) for file_name in file_names]
                x = torch.stack([torch.from_numpy(np.load(f'{data_dir}/slices_625/npy/{file_name}')).unsqueeze(0) for file_name in file_names]).to(dtype=torch.float32, device='cuda')
                y_hat = model.forward(x, z, omega_m, omega_b, h, n_s, sigma_8)
                discrepancies = calc_discrepancies(model, x.squeeze(), y_hat.squeeze())
                for j in np.argsort(discrepancies):
                    if discrepancies[j] < torch.max(success_discrepancies):
                        success_x[torch.argmax(success_discrepancies)] = x[j]
                        success_y_hat[torch.argmax(success_discrepancies)] = y_hat[j]
                        success_y[torch.argmax(success_discrepancies)] = slices[j]
                        success_discrepancies[torch.argmax(success_discrepancies)] = discrepancies[j]

        quijote = np.load(f'{data_dir}/raw/{int(row.sim)}/{z_dict[redshift]}.npy')

        for i in trange(n_success//batch_size):
            with torch.no_grad():
                x = success_x[i*batch_size:(i+1)*batch_size]
                y_hat = success_y_hat[i*batch_size:(i+1)*batch_size]
                y = torch.stack([torch.from_numpy(np.clip(np.log(quijote[int(idx):int(idx)+32].mean(axis=0)+1), -15, 7.5)).unsqueeze(0) for idx in success_y[i*batch_size:(i+1)*batch_size]]).to(dtype=torch.float32, device='cuda')

                calc_and_save_metrics(model, x.squeeze(), z, ['Lognormal']*batch_size, out_dir, int(test_sim), redshift)
                calc_and_save_metrics(model, y_hat.squeeze(), z, ['Prediction']*batch_size, out_dir, int(test_sim), redshift)
                calc_and_save_metrics(model, y.squeeze(), z, ['Quijote']*batch_size, out_dir, int(test_sim), redshift)
        
if __name__ == '__main__':
    main()