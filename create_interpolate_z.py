import numpy as np
import pandas as pd
from pathlib import Path
import gc
import yaml
from scipy.interpolate import interp1d
from nbodykit.lab import ArrayMesh, FFTPower, FFTCorr, cosmology, LinearMesh
from nbodykit.cosmology.correlation import pk_to_xi, xi_to_pk
from classy import Class
from tqdm import tqdm, trange
from itertools import product

def calc_ps_from_field_nbodykit(field, BoxSize=[1000, 1000], kmin=0.025, kmax=1.20):
    field_mesh = ArrayMesh(field, BoxSize=BoxSize)
    r_2d = FFTPower(field_mesh, mode='1d', kmin=kmin, kmax=kmax)
    return r_2d.power

def remove_nans(x):
    return x[~np.isnan(x)]

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

data_dir = data_dir.replace('gpu4', 'lustre')

Path(f'./out/').mkdir(exist_ok=True, parents=True)
if not Path(f'./out/interpolate_z.csv').is_file():
    df = pd.DataFrame(data=None, columns=['file_name', 'sim', 'redshift', 'omega_m', 'omega_b', 'h', 'n_s', 'sigma_8'])
    df.to_csv(f'./out/interpolate_z.csv', index=False)

lh_params = np.loadtxt('./out/latin_hypercube_params.txt')
omega_m = 0.3175
omega_b = 0.049
hubble = 0.6711
n_s = 0.9624
sigma_8 = 0.834
fiducial = np.array([omega_m, omega_b, hubble, n_s, sigma_8])

z_dict = {0:4, 0.5:3, 1:2, 2:1, 3:0}
snapshots = np.array([0, 0.5, 1, 2, 3])
k0 = 1e-5
kmin = 0.025

fudge = True
max_fudges = 100

sim = np.argmin(np.mean(np.abs(lh_params-fiducial)/fiducial, axis=1))

hubble = lh_params[sim, 2]
omega_b = lh_params[sim, 1]
omega_cdm = lh_params[sim, 0]-omega_b
n_s = lh_params[sim, 3]
sigma_8 = lh_params[sim, 4]
parameters = [omega_b*hubble**2, omega_cdm*hubble**2, hubble, n_s, sigma_8]
cosmo_params = {
    'omega_b': parameters[0],
    'omega_cdm': parameters[1],
    'h': parameters[2],
    'n_s': parameters[3],
    'sigma8': parameters[4],
    'output': 'mPk',
    'non linear': 'hmcode',
    'z_max_pk': 50,
    'P_k_max_h/Mpc': 100
}
cosmo_class = Class()
cosmo_class.set(cosmo_params)
cosmo_class.compute()

box_0 = np.load(f'{data_dir}/raw/{sim}/4.npy')
box_05 = np.load(f'{data_dir}/raw/{sim}/3.npy')
box_1 = np.load(f'{data_dir}/raw/{sim}/2.npy')
ics = np.load(f'{data_dir}/raw/{sim}/ICs.npy')

box_0 = np.concatenate((box_0, box_0[0:64]), axis=0)
box_05 = np.concatenate((box_05, box_05[0:64]), axis=0)
box_1 = np.concatenate((box_1, box_1[0:64]), axis=0)
ics = np.concatenate((ics, ics[0:64]), axis=0)

for z in interpolate_z:
    print(f'Generating lognormal maps for cosmology #{sim} at redshift {z}.')
    Path(f'{data_dir}/slices_625/npy/{sim}/{z}').mkdir(exist_ok=True, parents=True)

    kmaxes = np.concatenate((np.arange(1.200, 1.300, 0.005), np.arange(1.195, 1.100, -0.005))) # h/Mpc

    z_lower = np.max(snapshots[snapshots<z])
    z_upper = np.min(snapshots[snapshots>z])

    box_lower = np.load(f'{data_dir}/raw/{sim}/{z_dict[z_lower]}.npy')
    box_upper = np.load(f'{data_dir}/raw/{sim}/{z_dict[z_upper]}.npy')

    for i in trange(0, shape, shape//64):
        for j in np.arange(0, shape, step=shape//64):
            for kmax in kmaxes:
                P2D = calc_ps_from_field_nbodykit(np.zeros((shape, shape)), kmin=kmin, kmax=kmax)
                k_values = remove_nans(P2D['k'])
                P2D_0 = calc_ps_from_field_nbodykit(box_0[i:i+32].mean(axis=0), kmin=kmin, kmax=kmax)
                P2D_05 = calc_ps_from_field_nbodykit(box_05[i:i+32].mean(axis=0), kmin=kmin, kmax=kmax)
                P2D_1 = calc_ps_from_field_nbodykit(box_1[i:i+32].mean(axis=0), kmin=kmin, kmax=kmax)
                P2D_lower = calc_ps_from_field_nbodykit(box_lower[i:i+32].mean(axis=0), kmin=kmin, kmax=kmax)
                P2D_upper = calc_ps_from_field_nbodykit(box_upper[i:i+32].mean(axis=0), kmin=kmin, kmax=kmax)
                ps_0 = P2D_0['power'].real
                ps_05 = P2D_05['power'].real
                ps_1 = P2D_1['power'].real
                ps_lower = P2D_lower['power'].real
                ps_upper = P2D_upper['power'].real
                ps_values = ps_lower - (ps_lower-ps_upper) * (ps_0-ps_05) / (ps_0-ps_1)
                ps_values = remove_nans(ps_values)
            
                k_values_theory = np.logspace(np.log10(k0*hubble), np.log10(k_values[0]*hubble)-1e-10, 500)
                power_spectrum_theory = np.array([cosmo_class.pk(k_value_theory, z) for k_value_theory in k_values_theory])
                k_values_theory /= hubble
                power_spectrum_theory *= hubble**3

                k_values_interp = np.concatenate([k_values_theory, k_values])
                ps_values_interp = np.concatenate([power_spectrum_theory*ps_values[0]/power_spectrum_theory[-1], ps_values])

                n_points = 5000
                f2 = interp1d(k_values_interp, ps_values_interp, kind='linear', fill_value='extrapolate')
                k_values_ = np.logspace(np.log10(k_values_interp[0]+1e-10), np.log10(k_values[-1]-1e-10), n_points)
                power_spectrum = np.array([f2(k_value_) for k_value_ in k_values_])

                cf_class = pk_to_xi(k_values_, power_spectrum, extrap=True)
                r = np.logspace(-5, 5, int(1e5))

                def cf_g(r):
                    return np.log(1+cf_class(r))
                
                Pclass_g = xi_to_pk(r, cf_g(r))

                g_field = LinearMesh(Pclass_g, Nmesh=[shape, shape], BoxSize=1000, seed=np.random.randint(1e9), unitary_amplitude=True).preview() - 1
                g_fft = np.fft.fft2(g_field)
                ic_fft = np.fft.fft2(ics[j:j+32].mean(axis=0))
                g_field = np.real_if_close(np.fft.ifft2(ic_fft / np.abs(ic_fft) * np.abs(g_fft)))
                g_stddev = np.std(g_field.flatten())
                ln_field = np.exp(g_field-g_stddev**2/2)-1

                ln_ps = calc_ps_from_field_nbodykit(ln_field, kmin=k0, kmax=kmax)
                k_values_gen = ln_ps['k']

                target_ps = np.array([f2(k_value_gen) for k_value_gen in k_values_gen])

                if np.isnan(np.sum(ln_ps['power'].real)):
                    continue
                
                if fudge:
                    power_spectrum_adjusted = power_spectrum
                    fudge_done = False
                    threshold = 0.001
                    fudge_count = 0

                    while fudge_done == False:
                        test_adjust = target_ps/ln_ps['power'].real

                        k_val_interp = interp1d(k_values_gen, test_adjust, kind='linear', fill_value='extrapolate')
                        k_pivot = k_values_[np.argwhere(k_values_ >= k_values_gen[0])[:-1]]
                        interpolated_k_part = k_val_interp(k_pivot)[:, 0]
                        interpolated_k_part = np.concatenate((interpolated_k_part, interpolated_k_part[-1:]))

                        pivot = int(np.argwhere(k_values_ >= k_values_gen[0])[0])

                        right_hand_side_power_spectrum = np.multiply(power_spectrum_adjusted[pivot:], interpolated_k_part)
                        power_spectrum_adjusted = np.concatenate((power_spectrum_adjusted[:pivot]*right_hand_side_power_spectrum[np.argwhere(k_pivot <= kmin)].mean()/power_spectrum_adjusted[:pivot][-1], right_hand_side_power_spectrum))

                        cf_class = pk_to_xi(k_values_, power_spectrum_adjusted, extrap=True)
                        r = np.logspace(-5, 5, int(1e5))

                        def cf_g(r):
                            return np.log(1+cf_class(r))

                        Pclass_g = xi_to_pk(r, cf_g(r))

                        g_field = LinearMesh(Pclass_g, Nmesh=[shape, shape], BoxSize=1000, seed=np.random.randint(1e9), unitary_amplitude=True).preview() - 1
                        g_fft = np.fft.fft2(g_field)
                        g_field = np.real_if_close(np.fft.ifft2(ic_fft / np.abs(ic_fft) * np.abs(g_fft)))
                        g_stddev = np.std(g_field.flatten())
                        ln_field = np.exp(g_field-g_stddev**2/2)-1

                        ln_ps = calc_ps_from_field_nbodykit(ln_field, kmin=k0, kmax=kmax)
                        k_values_gen = ln_ps['k']

                        max_discrepancy = np.max(np.abs(ln_ps['power'].real[3:] - target_ps[3:]) / ln_ps['power'].real[3:])

                        if np.isnan(max_discrepancy):
                            break

                        if max_discrepancy <= threshold:
                            fudge_done = True
                            g_field = np.clip(np.log(ln_field+1), -30, 15)
                            file_name = f'{sim}/{z}/ln_field_{i}_{j}.npy'
                            np.save(f'{data_dir}/slices_625/npy/{file_name}', g_field)
                            d = [[file_name, int(sim), z, lh_params[sim, 0], lh_params[sim, 1], lh_params[sim, 2], lh_params[sim, 3], lh_params[sim, 4]]]
                            df = pd.DataFrame(d, columns=['file_name', 'sim', 'redshift', 'omega_m', 'omega_b', 'h', 'n_s', 'sigma_8'])
                            df.to_csv(f'./out/interpolate_z.csv', mode='a', index=False, header=False)
                            break
                        else:
                            fudge_count += 1
                        if fudge_count >= max_fudges:
                            break
                        
                    if fudge_done:
                        break

    print(f'Saved lognormal sims for cosmology #{sim} at redshift {z}.')