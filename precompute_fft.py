import os
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

import torch

os.environ['SLURM_JOB_NAME'] = 'bash'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
torch.set_float32_matmul_precision('medium')
torch.cuda.empty_cache()

def main():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    model_params = params['model']
    data_params = params['data']

    batch_size = model_params['batch_size']
    data_dir = data_params['data_dir']
    batch_size = batch_size * 128

    df = pd.read_csv(f'./out/data.csv')
    df = df.drop_duplicates()
    df.to_csv(f'./out/data.csv', index=False)

    for sim in df['sim'].unique():
        for z in [3, 2, 1, 0.5, 0]:
            Path(f'{data_dir}/pairs_pt/{sim}/{z}').mkdir(exist_ok=True, parents=True)

    data_min = 0
    data_max = 0

    for i in trange(len(df)//batch_size):
        file_name = [df.iloc[idx, 0] for idx in range(i*batch_size, (i+1)*batch_size)]
        pair = torch.stack([torch.from_numpy(np.load(f'{data_dir.replace("gpu4", "lustre")}/slices_625/npy/{file_name[j]}')) for j in range(len(file_name))]).to(device='cuda')
        pair = torch.clamp(pair, min=-30, max=15)
        fft = torch.fft.fft2(torch.exp(pair) - 1)
        fft_save = [torch.save(torch.stack([pair[j], fft[j]]), f'{data_dir}/pairs_pt/{file_name[j].replace("npy", "pt")}') for j in torch.arange(pair.size(0))]
        if torch.isnan(pair).any() or torch.isinf(pair).any():
            print(f'Nan/Inf in data: {df.iloc[i, 0]}')
        elif torch.isnan(fft).any() or torch.isinf(fft).any():
            print(f'Nan/Inf in fft: {df.iloc[i, 0]}')
        else:
            if torch.min(pair) < data_min:
                data_min = torch.min(pair)
            if torch.max(pair) > data_max:
                data_max = torch.max(pair)

    if len(df) % batch_size != 0:
        i = len(df) % batch_size
        file_name = [df.iloc[idx, 0] for idx in range(len(df)-i, len(df))]
        pair = torch.stack([torch.from_numpy(np.load(f'{data_dir.replace("gpu4", "lustre")}/slices_625/npy/{file_name[j]}')) for j in range(len(file_name))]).to(device='cuda')
        pair = torch.clamp(pair, min=-30, max=15)
        fft = torch.fft.fft2(torch.exp(pair) - 1)
        fft_save = [torch.save(torch.stack([pair[j], fft[j]]), f'{data_dir}/pairs_pt/{file_name[j].replace("npy", "pt")}') for j in torch.arange(pair.size(0))]
        if torch.isnan(pair).any() or torch.isinf(pair).any():
            print(f'Nan/Inf in data: {file_name}')
        elif torch.isnan(fft).any() or torch.isinf(fft).any():
            print(f'Nan/Inf in fft: {file_name}')
        else:
            if torch.min(pair) < data_min:
                data_min = torch.min(pair)
            if torch.max(pair) > data_max:
                data_max = torch.max(pair)

    print(f'max = {data_max}')
    print(f'min = {data_min}')

    df['file_name'] = df['file_name'].apply(lambda x: x.replace('npy','pt'))
    df = df.drop_duplicates()
    df.to_csv(f'./out/data.csv', index=False)

if __name__ == '__main__':
    main()