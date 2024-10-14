'''
This script generates 3D coarse density fields from the simulation snapshots 
Adapted from the available one by D. Piras from 05/07/2021
'''
import sys
import yaml
from pathlib import Path
from mpi4py import MPI
import numpy as np
import readgadget
import MAS_library as MASL

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

data_dir = params['data']['data_dir']
data_dir = f'{data_dir}/raw'.replace('gpu4', 'lustre')
grid = params['model']['shape']
ptypes = [1]
num_id = int(sys.argv[1])

def compute_df(snapshot, ptypes, grid, file_out):
    if not Path(f'{snapshot}.0').is_file() and not Path(f'{snapshot}.0.hdf5').is_file():
        return 0
    df = MASL.density_field_gadget(
        snapshot,
        ptypes,
        grid,
        MAS='PCS',
        do_RSD=False,
        axis=0,
        verbose=True
    )
    df = df/np.mean(df, dtype=np.float64) - 1.0
    np.save(file_out, df)

for z in [0, 1, 2, 3, 4, 'ICs']:
    folder_in = f'{data_dir}/{num_id}/'
    file_out = f'{data_dir}/{num_id}/{z}.npy'

    if z == 'ICs':
        snapshot = f'{folder_in}/ICs/ics'
    else:
        snapshot = f'{folder_in}/snapdir_00{z}/snap_00{z}'

    compute_df(snapshot, ptypes, grid, file_out)
    comm.Barrier()