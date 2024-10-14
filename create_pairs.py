'''
Script to take as input N-body fields, and create dataset of pairs of 
highly-correlated lognormal and N-body slices. Code is not extremely factorised, 
and requires some pre-computed density fields to be run smoothly.
Adapted from the available one by D. Piras from 05/07/2021.
'''
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

Path(f'{data_dir}/slices_625/npy/').mkdir(exist_ok=True, parents=True)
Path(f'./out/').mkdir(exist_ok=True, parents=True)
if not Path(f'./out/data.csv').is_file():
    df = pd.DataFrame(data=None, columns=['file_name', 'sim', 'redshift', 'omega_m', 'omega_b', 'h', 'n_s', 'sigma_8'])
    df.to_csv(f'./out/data.csv', index=False)

lh_params = np.loadtxt('./out/latin_hypercube_params.txt')

np.random.seed(seed)
test_range = [(lh_params[:,i].min()+0.2*(lh_params[:,i].max()-lh_params[:,i].min()), lh_params[:,i].max()-0.2*(lh_params[:,i].max()-lh_params[:,i].min())) for i in range(lh_params.shape[1])]
test_options = lh_params[
    (lh_params[:,0]>test_range[0][0]) &
    (lh_params[:,0]<test_range[0][1]) & 
    (lh_params[:,2]>test_range[2][0]) &
    (lh_params[:,2]<test_range[2][1]) &
    (lh_params[:,4]>test_range[4][0]) &
    (lh_params[:,4]<test_range[4][1])
]
test_sims = np.random.randint(0, len(test_options), size=2)
test_sims = [np.argwhere(lh_params==test_options[i])[0,0] for i in test_sims]

k0 = 1e-5 # h/Mpc
kmin = 0.025 # h/Mpc

# some nans will happen because fewer fudges are needed; we'll remove these nans when training, they are caused by numerical errors due to high noise at low k end
fudge = True
max_fudges = 100

n_sim_start = 0 # Which sim to start generating pairs from.
n_sims = 2000 # How many sims to generate pairs for. This will take a long time, so it's recommended to run this in batches over multiple parallel processess.
z_dict = {4:0, 3:0.5, 2:1, 1:2, 0:3} # Mapping dictionary {snapshot_number: redshift value}.
zs = [4, 3, 2, 1, 0] # Which snapshots to generate pairs for.

for sim, z in product(list(range(n_sim_start, n_sim_start+n_sims)), zs):
    if sim in test_sims:
        print(f'{sim} is a reserved cosmology for testing. Skipping generating paris.')
        continue
    else:
        print(f'Running sim #{sim} at redshift {z_dict[z]}.')

    Path(f'{data_dir}/slices_625/npy/{sim}/{z_dict[z]}').mkdir(exist_ok=True, parents=True)
    d = []
    nan_count = 0
    save_count = 0

    df_path_z0 = f'{data_dir}/raw/{sim}/{z}.npy'
    df_path_z1 = f'{data_dir}/raw/{sim}/ICs.npy'
    output_train = np.load(df_path_z0)
    input_train = np.load(df_path_z1)

    cosmo_class = Class()
    hubble = lh_params[sim, 2]
    omega_b = lh_params[sim, 1]
    omega_cdm = lh_params[sim, 0]-omega_b
    ns = lh_params[sim, 3]
    sigma8 = lh_params[sim, 4]
    parameters = [omega_b*hubble**2, omega_cdm*hubble**2, hubble, ns, sigma8]
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

    cosmo_class.set(cosmo_params)
    try:
        cosmo_class.compute()
        # fails sometimes since omega_b is too high...
    except:
        # if it fails, just use fiducial cosmology; results do not change much.
        print(f'Cosmology failed for simulation no. {sim} at redshift {z_dict[z]}.')
        omega_b = 0.049
        omega_cdm = lh_params[sim, 0]-omega_b
        parameters = [omega_b*hubble**2, omega_cdm*hubble**2, hubble, ns, sigma8]
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
        cosmo_class.set(cosmo_params)
        cosmo_class.compute()

    kmaxes = np.concatenate((np.arange(1.200, 1.600, 0.010), np.arange(1.190, 0.800, -0.010))) if z == 4 else np.concatenate((np.arange(1.200, 1.300, 0.005), np.arange(1.195, 1.100, -0.005))) # h/Mpc

    for i in trange(0, shape, shape//16):
        slice_625 = output_train[i:i+32].mean(axis=0)
        ics_625 = input_train[i:i+32].mean(axis=0)
        # calculate ps of current slice
        for kmax in kmaxes:
            P2D = calc_ps_from_field_nbodykit(slice_625, kmin=kmin, kmax=kmax)
            
            k_values = remove_nans(P2D['k'])
            ps_values = remove_nans(P2D['power'].real)

            # we also add some values from k0 to kmin, from the theory
            k_values_theory = np.logspace(np.log10(k0*hubble), np.log10(k_values[0]*hubble)-1e-10, 500) # 1/Mpc
            power_spectrum_theory = np.array([cosmo_class.pk(k_value_theory, z) for k_value_theory in k_values_theory])
            k_values_theory /= hubble # h/Mpc
            power_spectrum_theory *= hubble**3 # Mpc/h**3

            k_values_interp = np.concatenate([k_values_theory, k_values])
            ps_values_interp = np.concatenate([power_spectrum_theory*ps_values[0]/power_spectrum_theory[-1], ps_values])

            n_points = 5000
            f2 = interp1d(k_values_interp, ps_values_interp, kind='linear')
            k_values_ = np.logspace(np.log10(k_values_interp[0]+1e-10), np.log10(k_values[-1]-1e-10), n_points)
            power_spectrum = np.array([f2(k_value_) for k_value_ in k_values_])

            cf_class = pk_to_xi(k_values_, power_spectrum, extrap=True)
            r = np.logspace(-5, 5, int(1e5))

            def cf_g(r):
                return np.log(1+cf_class(r))
            
            # then it should be easy to use the same transformation as above, just inverse, to obtain a Gaussian-like power spectrum
            Pclass_g = xi_to_pk(r, cf_g(r))

            # using the input slice at z=127 to create the input dataset to U-net
            # we combine the phase information there with the power from the 'theory'
            g_field = LinearMesh(Pclass_g, Nmesh=[shape, shape], BoxSize=1000, seed=np.random.randint(1e9), unitary_amplitude=True).preview() - 1
            ft_g = np.fft.fftn(g_field)
            abses_g = np.abs(ft_g)

            ft_IC = np.fft.fftn(ics_625)
            abses_ic = np.abs(ft_IC)
            ft_mixed = ft_IC / abses_ic * abses_g
            # note that this inverse transform yields some 1e-17 imaginary parts, which I think are just
            # numerical errors; np.real_if_close takes care of that.
            g_field_mixed = np.real_if_close(np.fft.ifftn(ft_mixed), tol=1000)
            # finally, take the LN off this Gaussian map
            gaussian_stddev = np.std(g_field_mixed.flatten())
            ln_field_mixed = np.exp(g_field_mixed-gaussian_stddev**2/2)-1
            # now we have the input and output boxes; we need to shift them, and then normalise each slice
            # before that, we calculate the power spectrum of each input and output box
            # since we usually need to add the fudge here
            
            in_ps = calc_ps_from_field_nbodykit(ln_field_mixed, kmin=k0, kmax=kmax)
            out_ps = calc_ps_from_field_nbodykit(slice_625, kmin=k0, kmax=kmax) 
            k_values_in = in_ps['k']
            k_values_out = out_ps['k']

            if np.isnan(np.sum(in_ps['power'].real)):
                # breaking even before fudging, for nans due to kmax
                continue

            if fudge:
                power_spectrum_adjusted = power_spectrum
                fudge_done = False # we fudge until we are within the threshold level
                threshold = 0.001 # 0.1% in this case
                fudge_count = 0
                while fudge_done == False:
                    test_adjust = out_ps['power'].real/in_ps['power'].real 
                    
                    k_val_interp = interp1d(k_values_in, test_adjust, kind='linear', fill_value='extrapolate')
                    k_pivot = k_values_[np.argwhere(k_values_ >= k_values_in[0])[:-1]]
                    interpolated_k_part = k_val_interp(k_pivot)[:, 0]
                    interpolated_k_part = np.concatenate((interpolated_k_part, interpolated_k_part[-1:]))

                    pivot = int(np.argwhere(k_values_ >= k_values_in[0])[0])

                    right_hand_side_power_spectrum = np.multiply(power_spectrum_adjusted[pivot:], interpolated_k_part)
                    # for some reason, while correct this seems to give some nans every now and then, so be careful about it
                    power_spectrum_adjusted = np.concatenate((power_spectrum_adjusted[:pivot]*right_hand_side_power_spectrum[np.argwhere(k_pivot <= kmin)].mean()/power_spectrum_adjusted[:pivot][-1], right_hand_side_power_spectrum))

                    # same as above, using the adjusted ps
                    cf_class = pk_to_xi(k_values_, power_spectrum_adjusted, extrap=True)
                    r = np.logspace(-5, 5, int(1e5))

                    def cf_g(r):
                        return np.log(1+cf_class(r))

                    # then it should be easy to use the same transformation as above, just inverse, to obtain a Gaussian-like power spectrum
                    Pclass_g = xi_to_pk(r, cf_g(r))

                    # using the input slice at z=127 to create the input dataset to U-net
                    # we combine the phase information there with the power from the 'theory'
                    g_field = LinearMesh(Pclass_g, Nmesh=[shape, shape], BoxSize=1000, seed=np.random.randint(1e9), unitary_amplitude=True).preview() - 1 
                    
                    ft_g = np.fft.fftn(g_field)
                    abses_g = np.abs(ft_g)
                    ft_mixed = ft_IC / abses_ic * abses_g
                    # note that this inverse transform yields some 1e-17 imaginary parts, which I think are just
                    # numerical errors; np.real_if_close takes care of that.
                    g_field_mixed = np.real_if_close(np.fft.ifftn(ft_mixed), tol=1000)
                    # finally, take the LN off this Gaussian map
                    gaussian_stddev = np.std(g_field_mixed.flatten())
                    ln_field_mixed = np.exp(g_field_mixed-gaussian_stddev**2/2)-1

                    in_ps = calc_ps_from_field_nbodykit(ln_field_mixed, kmin=k0, kmax=kmax)
                    k_values_in = in_ps['k']
                   
                    # we check if the maximum discrepancy is less than 1%, otherwise keep on fudging
                    # in some cases there will be discrepancies at low k which are hard to reduce, so
                    # we set a maximum of 100 fudges. We only look at k > 0.2, since we never use the information below that.
                    max_discrepancy = np.max(np.abs(out_ps['power'].real[3:] - in_ps['power'].real[3:]) / in_ps['power'].real[3:])
                    if np.isnan(max_discrepancy):
                        break
                    if max_discrepancy <= threshold:
                        fudge_done = True
                    else:
                        fudge_count += 1
                    if fudge_count >= max_fudges:
                        break

            if not np.isnan(np.sum(in_ps['power'].real)):
                break

        out_dataset = np.stack([ln_field_mixed, slice_625], axis=0)
        pair = np.log(out_dataset+1)
        if not(np.isnan(pair.sum())) and not(np.isnan(np.sum(in_ps['power'].real))) and max_discrepancy <= threshold:
            file_name = f'{sim}/{z_dict[z]}/pair_{save_count}.npy'
            np.save(f'{data_dir}/slices_625/npy/{file_name}', pair)
            save_count += 1
            d.append([file_name, int(sim), z_dict[z], lh_params[sim, 0], lh_params[sim, 1], lh_params[sim, 2], lh_params[sim, 3], lh_params[sim, 4]])
        else:
            nan_count += 1
    
    del output_train
    del input_train
    gc.collect()

    df = pd.DataFrame(d, columns=['file_name', 'sim', 'redshift', 'omega_m', 'omega_b', 'h', 'n_s', 'sigma_8'])
    df.to_csv(f'./out/data.csv', mode='a', index=False, header=False)
    print(f'{nan_count} NaN found in total for sim {sim} at redshift {z_dict[z]}.')