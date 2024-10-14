# Psi-GAN: A power-spectrum-informed generative adversarial network for the emulation of large-scale structure maps across cosmologies and redshifts.

This repository contains the code used in the paper [Psi-GAN: A power-spectrum-informed generative adversarial network for the emulation of large-scale structure maps across cosmologies and redshifts](https://arxiv.org/abs/2410.07349).

## Dependencies

Due to `numpy` dependency conflicts between `pytorch` and `nbodykit`, this project requires two separate environments. One is used for generating data (using the `nbodykit` package), and the other for training (via `pytorch`).

### `nbodykit env`
- `python 3.10.6`
- `numpy 1.22.4`
- `pandas 2.0.3`
- `scipy 1.11.1`
- `tqdm 4.65.0`
- `classy 3.2.0`
- `Pylians 0.11`
- `nbodykit 0.3.15`

### `pytorch env`
- `python 3.8.17`
- `pyyaml 6.0`
- `numpy 1.24.4`
- `pandas 2.0.3`
- `pytorch 2.1.0`
- `lightning 2.2.1`
- `tqdm 4.65.0`
- `captum 0.7.0`
- `seaborn 0.12.2`
- `nbodykit 0.3.15`

The package `PiInTheSky` (written by William Coulton) was used for bispectra estimation, however this code is not yet publically available. Bispectra are only computed in testing this model, and so `PiInTheSky` is not necessary for training the model. If you plan to run the code, please either remove the relevant parts containing bispectrum estimation, or contact William Coulton. Any use of the `PiInTheSky` code should cite the following papers in which the code was developed: [1810.02374](https://arxiv.org/abs/1810.02374), [1901.04515](https://arxiv.org/abs/1901.04515).
 
## Data

The following data set was used for this project (this repository includes a script to download all relevant data via [`globus`](https://www.globus.org/)):

- [Quijote simulations: Latin-hypercubes](https://quijote-simulations.readthedocs.io/en/latest/LH.html)

## Usage

Please find below a description of each step required to reproduce our study.

### `params.yaml`

Most user inputs are handled by editing the relevant entry in the `params.yaml` file. A description for each entry is provided:

- `model`
    - `shape` - The resolution of the dark matter overdensity fields.
    - `lambda_gp` - The hyperparameter controlling the level of gradient penalty regularisation used when training.
    - `lambda_pixel` - The hyperparamter controlling the level of pixel matching regularisation (i.e. the $l^{2}$ loss) when training.
    - `batch_size` - The batch size to use when training.
    - `lr_factor` - The factor $f$ applied to the learning rate of the critic.
    - `learning_rate` - The base learning rate used by the generator.
    - `beta_1` - The first decay parameter used in the Adam optimiser.
    - `beta_2` - The second decay parameter used in the Adam optimiser.
    - `gradient_clip_val` - The value to clip gradients to when training.
- `data`
    - `seed` - The seed to use for all random processes.
    - `data_dir` - The directory where all data is stored.
    - `val_size` - The fraction of data to be used for validation.
- `test`
    - `interpolate_z` - A list of redshift values to use as test cases for interpolating redshift.
    - `n_success` - How many examples to use for plotting test results.

### `save_quijote.sh`

This script uses [`globus`](https://www.globus.org/) to download raw data from the Latin-hypercube suite of the Quijote simulations (you will need to set up a `globus` account and create an endpoint), and then processes saves each simulation box as a `.npy` file by calling `save_quijote.py`.

This script requires you to manually enter your endpoint as `ep2`, as well as manually setting file paths throughout the script.

This script uses the `nbodykit env`.

### `create_pairs.py`, `create_interpolate_z.py`, and `create_unseen_cosmology.py`

These scripts are used to slice the $N$-body simulation boxes into 2-dimensional maps, and then create their corresponding lognormal counterparts.

`create_pairs.py` creates the main dataset used for training, validation, and the randomised test set. `create_interpolate_z.py` creates the test set for redshift interpolation, and `create_unseen_cosmology.py` creates the test set for interpolating cosmology.

These scripts uses the `nbodykit env`.

### `precompute_fft.py`

This script pre-computes fast Fourier transforms for all lognormal/$N$-body pairs created by `create_pairs.py` (so that we can calculate their power spectra for the critic during training) and saves the results as `.pt` files. This facilitates faster training as we will not need to compute the fast Fourier transforms each epoch during training, and loading `.pt` files is faster than `.npy` files.

This script uses the `pytorch env`.

### `model.py`

This file defines the model and data module classes which are then imported into subsequent scripts.

### `train.py`

This script trains and validates the model using a `pytorch lightning` Trainer. The script is currently set up to log training via [`Weights and Biases`](https://wandb.ai/), however this can either be removed or changed to your logger of choice (this will also require changing or removing the image logging in the `validation_step` method of the `LSSModel` class in `model.py`).

The best model is saved as `model.ckpt`. The script is set up to resume training from the last completed epoch if interrupted.

This script uses the `pytorch env`.

### `test_set.py`, `test_interpolate_z.py`, and `test_unseen_cosmology.py`

These scripts are used to run all test cases. `test_set.py` tests the model on the randomised test set (i.e. within the domain of the model's training data), while `test_interpolate_z.py`, and `test_unseen_cosmology.py` test the model on its ability to interpolate redshift and cosmology, respectively.

These scripts uses the `pytorch env`.

### `test_xai.py`

This script is used to run saliency mapping tests on the model outputs.

This script uses the `pytorch env`.