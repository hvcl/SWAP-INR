# SWAP-INR Reproduction Example
Due to supplementary material size limitations, the provided model weights were obtained using downsampled versions of the ROCOM and Ionization datasets. 

The original datasets used in the paper were substantially larger than those used for the experiments in this repository. 

| Dataset | Original dataset size<br>(used in the paper) | Downsampled dataset size<br>(used in this repository) | 
|:--|--:|--:| 
| ROCOM | 1,943,807 KB | 80,995 KB | 
| Ionization | 450,469 KB | 11,270 KB | 

> **Note:** Results reproduced using the provided model weights and datasets may differ from those reported in the paper.


### Downsmapled datasets results
The results obtained using the downsampled datasets in this repository may differ from those reported in the paper. 

| Dataset | Source | PSNR | NRMSE | R² | 
|:--|:--|--:|--:|--:| 
| ROCOM | Paper | 47.41 | 0.0073 | 0.9782 | 
| ROCOM | This repository (downsampled data) | 43.06 | 0.0108 | 0.9421 | 
| Ionization | Paper | 50.25 | 0.0032 | 0.9905 | 
| Ionization | This repository (downsampled data) | 53.66 | 0.0021 | 0.9810 | 

> **Note:** The model weights and experimental results in this repository were obtained using downsampled versions of the ROCOM and Ionization datasets because of supplementary-material size limitations. Therefore, the results may not exactly match those reported in the paper. All reported results were obtained using half-precision (FP16)


# Run

Run all commands from this directory:

```bash
cd code/
```

## Train

```bash
python train.py --dataset ROCOM_down_down --num_var 8 --Shared_hidden 64 --Branch_hidden 32 --name ROCOM_down_down_test --device cuda
```

or:

```bash
python train.py --dataset ionization_down_down --num_var 4 --Shared_hidden 64 --Branch_hidden 24 --name ionization_down_down_test --device cuda
```

The `--name` argument is required. Training saves checkpoints and pruned models under:

```text
Exp/<dataset>/<name>/
```

Useful optional arguments:

```bash
python train.py \
  --dataset ROCOM_down_down \
  --name experiment_name \
  --device cuda \
  --batch_size 100000 \
  --base_num_epochs 2000 \
  --on_num_epochs 1000
```

Use `--device cpu` if CUDA is not available.

## Inference

Run inference with the same `--dataset`, `--name`, and model architecture arguments used for training:

### ROCOM
```bash
python inference.py --dataset ROCOM_down_down --num_var 8 --Shared_hidden 64 --Branch_hidden 32 --name ROCOM_down_down_test --device cuda
```
To run with the half-precision model:
```bash
python inference.py --dataset ROCOM_down_down --num_var 8 --Shared_hidden 64 --Branch_hidden 32 --name ROCOM_down_down_test --use_half --device cuda
```


### Ionization
```bash
python inference.py --dataset ionization_down_down --num_var 4 --Shared_hidden 64 --Branch_hidden 24 --name ionization_down_down_test --device cuda
```


To run with the half-precision model:

```bash
python inference.py --dataset ionization_down_down --num_var 4 --Shared_hidden 64 --Branch_hidden 24 --name ionization_down_down_test --use_half --device cuda
```
