import os
import yaml
from pathlib import Path
import numpy as np
import pandas as pd

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, BatchSizeFinder
from lightning.pytorch.loggers import WandbLogger

from model import LSSModel, LSSDataModule

torch.set_float32_matmul_precision('medium')
torch.cuda.empty_cache()

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
    df = pd.read_csv(f'./out/data.csv')
    df_test = pd.DataFrame(columns=df.columns)
    for z in df['redshift'].unique():
        df_test = pd.concat((df_test, df[df['redshift']==z].sample(shape*4, replace=False, random_state=seed)))
    df_train = df.drop(df_test.index)

    ckpt_dir = f'{data_dir}/out_625/models/{lambda_gp}_{lambda_pixel}/'
    checkpoints = Path(f'{ckpt_dir}/checkpoints/')
    checkpoints.mkdir(exist_ok=True, parents=True)
    max_epochs = 10
    epoch = len(list(checkpoints.iterdir()))
    ckpt_path = str(list(checkpoints.iterdir())[-1]) if epoch!=0 else None

    model = LSSModel(
        shape=shape,
        lambda_gp=lambda_gp,
        lambda_pixel=lambda_pixel,
        batch_size=batch_size,
        lr_factor=lr_factor,
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        gradient_clip_val=gradient_clip_val
    )

    dm = LSSDataModule(
        df_train=df_train,
        df_test=df_test,
        data_dir=data_dir,
        val_size=val_size,
        batch_size=batch_size
    )

    checkpoint = ModelCheckpoint(
        dirpath=str(checkpoints),
        filename='{epoch}',
        save_top_k=-1
    )

    wandb_logger = WandbLogger(
        save_dir=f'./out/',
        project=f'ConvNeXtUnet_625',
        log_model=True
    )

    callbacks = [checkpoint]

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        devices=-1,
        precision=32,
        callbacks=callbacks,
        logger=wandb_logger,
        profiler='simple'
    )

    trainer.fit(model, dm, ckpt_path=ckpt_path)
    Path(f'{ckpt_dir}/checkpoints/model.ckpt').write_bytes(Path(f'{checkpoint.best_model_path}').read_bytes())

if __name__ == '__main__':
    main()