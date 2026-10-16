<div align="center">

# SWAP-INR: Implicit Neural Representation with Switchable Attention Paths for Time-varying Multivariate Scientific Data Compression

**Accepted at WACV 2027**  
**Paper:** Coming soon

</div>

## Overview

<p align="center">
  <img src="SWAP_INR_overview.png" alt="Overview of the proposed SWAP-INR architecture" width="100%">
</p>

**Figure 1. Overview of the proposed SWAP-INR.** Given a spatiotemporal coordinate $(x, y, z, t)$, the network first applies positional encoding. The encoded features are processed by MLP layers shared across variables, equipped with **A²-SIREN**, which employs a learnable parameter $\boldsymbol{\alpha}$ for adaptive frequency modulation (bottom left). The representations are then split into two branches: a non-attention path (red) and an attention path (blue). The attention branch models cross-variable interactions using a cross-variable attention layer (purple). Finally, the **variable-driven path switching** block selects the appropriate representation for each variable to reconstruct high-fidelity multivariate outputs.

## Main Results

**Quantitative comparison across six time-varying multivariate scientific datasets.** Higher PSNR, R², and compression ratio (CR) are better; lower NRMSE and model size are better.

<table>
  <thead>
    <tr><th>Dataset</th><th>Method</th><th>PSNR ↑</th><th>NRMSE ↓</th><th>R² ↑</th><th>CR ↑</th><th>Size (KB) ↓</th><th>Dataset</th><th>Method</th><th>PSNR ↑</th><th>NRMSE ↓</th><th>R² ↑</th><th>CR ↑</th><th>Size (KB) ↓</th></tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="11"><strong>ROCOM</strong></td>
      <td align="left">SIREN</td>
      <td align="right">42.34</td>
      <td align="right">0.0114</td>
      <td align="right">0.9323</td>
      <td align="right">3,311</td>
      <td align="right">587</td>
      <td rowspan="11"><strong>CUPID</strong></td>
      <td align="left">SIREN</td>
      <td align="right">21.46</td>
      <td align="right">0.1657</td>
      <td align="right">0.5585</td>
      <td align="right">2,634</td>
      <td align="right">626</td>
    </tr>
    <tr>
      <td align="left">NeurComp</td>
      <td align="right">42.66</td>
      <td align="right">0.0105</td>
      <td align="right">0.9398</td>
      <td align="right">3,346</td>
      <td align="right">581</td>
      <td align="left">NeurComp</td>
      <td align="right">24.63</td>
      <td align="right">0.0850</td>
      <td align="right">0.7002</td>
      <td align="right">2,626</td>
      <td align="right">628</td>
    </tr>
    <tr>
      <td align="left">CoordNet</td>
      <td align="right">43.49</td>
      <td align="right">0.0105</td>
      <td align="right">0.9392</td>
      <td align="right">3,351</td>
      <td align="right">580</td>
      <td align="left">CoordNet</td>
      <td align="right">23.87</td>
      <td align="right">0.0914</td>
      <td align="right">0.6540</td>
      <td align="right">2,609</td>
      <td align="right">632</td>
    </tr>
    <tr>
      <td align="left">Instant-NGP</td>
      <td align="right">30.28</td>
      <td align="right">0.0356</td>
      <td align="right">0.4391</td>
      <td align="right">3,346</td>
      <td align="right">581</td>
      <td align="left">Instant-NGP</td>
      <td align="right">19.61</td>
      <td align="right">0.1791</td>
      <td align="right">0.3948</td>
      <td align="right">2,561</td>
      <td align="right">644</td>
    </tr>
    <tr>
      <td align="left">fV-SRN*</td>
      <td align="right">40.24</td>
      <td align="right">0.0125</td>
      <td align="right">0.8764</td>
      <td align="right">3,351</td>
      <td align="right">580</td>
      <td align="left">fV-SRN*</td>
      <td align="right">13.99</td>
      <td align="right">0.2362</td>
      <td align="right">-13.4032</td>
      <td align="right">2,618</td>
      <td align="right">630</td>
    </tr>
    <tr>
      <td align="left">WIRE</td>
      <td align="right">25.09</td>
      <td align="right">0.0966</td>
      <td align="right">-0.0004</td>
      <td align="right">3,351</td>
      <td align="right">580</td>
      <td align="left">WIRE</td>
      <td align="right">16.92</td>
      <td align="right">0.2595</td>
      <td align="right">0.0000</td>
      <td align="right">2,630</td>
      <td align="right">627</td>
    </tr>
    <tr>
      <td align="left">AFNet</td>
      <td align="right">44.19</td>
      <td align="right">0.0094</td>
      <td align="right">0.9480</td>
      <td align="right">3,351</td>
      <td align="right">580</td>
      <td align="left">AFNet</td>
      <td align="right">48.38</td>
      <td align="right">0.0044</td>
      <td align="right">0.9695</td>
      <td align="right">2,622</td>
      <td align="right">629</td>
    </tr>
    <tr>
      <td align="left">APMGSRN*</td>
      <td align="right">38.21</td>
      <td align="right">0.0171</td>
      <td align="right">0.8383</td>
      <td align="right">3,351</td>
      <td align="right">580</td>
      <td align="left">APMGSRN*</td>
      <td align="right">19.25</td>
      <td align="right">0.1789</td>
      <td align="right">-0.1587</td>
      <td align="right">2,618</td>
      <td align="right">630</td>
    </tr>
    <tr>
      <td align="left">SL²A-INR</td>
      <td align="right">43.39</td>
      <td align="right">0.0113</td>
      <td align="right">0.9258</td>
      <td align="right">3,334</td>
      <td align="right">583</td>
      <td align="left">SL²A-INR</td>
      <td align="right">46.77</td>
      <td align="right">0.0051</td>
      <td align="right">0.9679</td>
      <td align="right">2,626</td>
      <td align="right">628</td>
    </tr>
    <tr>
      <td align="left">MC-INR</td>
      <td align="right">33.61</td>
      <td align="right">0.0255</td>
      <td align="right">0.5857</td>
      <td align="right">3,328</td>
      <td align="right">584</td>
      <td align="left">MC-INR</td>
      <td align="right">46.89</td>
      <td align="right">0.0086</td>
      <td align="right">0.9579</td>
      <td align="right">2,440</td>
      <td align="right">676</td>
    </tr>
    <tr>
      <td align="left"><strong>Ours</strong></td>
      <td align="right"><strong>47.41</strong></td>
      <td align="right"><strong>0.0073</strong></td>
      <td align="right"><strong>0.9782</strong></td>
      <td align="right"><strong>3,351</strong></td>
      <td align="right"><strong>580</strong></td>
      <td align="left"><strong>Ours</strong></td>
      <td align="right"><strong>50.29</strong></td>
      <td align="right"><strong>0.0035</strong></td>
      <td align="right"><strong>0.9904</strong></td>
      <td align="right"><strong>2,634</strong></td>
      <td align="right"><strong>626</strong></td>
    </tr>
    <tr>
      <td rowspan="11"><strong>Synthetic</strong></td>
      <td align="left">SIREN</td>
      <td align="right">70.31</td>
      <td align="right">0.0003</td>
      <td align="right">0.9245</td>
      <td align="right">7,793</td>
      <td align="right">316</td>
      <td rowspan="11"><strong>SMR Simulation</strong></td>
      <td align="left">SIREN</td>
      <td align="right">33.71</td>
      <td align="right">0.0277</td>
      <td align="right">0.2851</td>
      <td align="right">665</td>
      <td align="right">493</td>
    </tr>
    <tr>
      <td align="left">NeurComp</td>
      <td align="right">68.72</td>
      <td align="right">0.0004</td>
      <td align="right">0.8830</td>
      <td align="right">7,868</td>
      <td align="right">313</td>
      <td align="left">NeurComp</td>
      <td align="right">33.15</td>
      <td align="right">0.0297</td>
      <td align="right">0.1884</td>
      <td align="right">665</td>
      <td align="right">493</td>
    </tr>
    <tr>
      <td align="left">CoordNet</td>
      <td align="right">72.14</td>
      <td align="right">0.0003</td>
      <td align="right">0.9755</td>
      <td align="right">7,843</td>
      <td align="right">314</td>
      <td align="left">CoordNet</td>
      <td align="right">34.03</td>
      <td align="right">0.0267</td>
      <td align="right">0.3411</td>
      <td align="right">666</td>
      <td align="right">492</td>
    </tr>
    <tr>
      <td align="left">Instant-NGP</td>
      <td align="right">35.22</td>
      <td align="right">0.0776</td>
      <td align="right">0.0280</td>
      <td align="right">7,601</td>
      <td align="right">324</td>
      <td align="left">Instant-NGP</td>
      <td align="right">30.16</td>
      <td align="right">0.0850</td>
      <td align="right">0.0356</td>
      <td align="right">635</td>
      <td align="right">516</td>
    </tr>
    <tr>
      <td align="left">fV-SRN*</td>
      <td align="right">64.10</td>
      <td align="right">0.0007</td>
      <td align="right">0.8038</td>
      <td align="right">7,463</td>
      <td align="right">330</td>
      <td align="left">fV-SRN*</td>
      <td align="right">29.60</td>
      <td align="right">0.0472</td>
      <td align="right">-3.1898</td>
      <td align="right">655</td>
      <td align="right">500</td>
    </tr>
    <tr>
      <td align="left">WIRE</td>
      <td align="right">34.90</td>
      <td align="right">0.0802</td>
      <td align="right">-0.0470</td>
      <td align="right">7,868</td>
      <td align="right">313</td>
      <td align="left">WIRE</td>
      <td align="right">29.98</td>
      <td align="right">0.0911</td>
      <td align="right">-0.0024</td>
      <td align="right">663</td>
      <td align="right">494</td>
    </tr>
    <tr>
      <td align="left">AFNet</td>
      <td align="right">74.60</td>
      <td align="right">0.0002</td>
      <td align="right">0.9657</td>
      <td align="right">7,793</td>
      <td align="right">316</td>
      <td align="left">AFNet</td>
      <td align="right">35.80</td>
      <td align="right">0.0190</td>
      <td align="right">0.3564</td>
      <td align="right">666</td>
      <td align="right">492</td>
    </tr>
    <tr>
      <td align="left">APMGSRN*</td>
      <td align="right">60.93</td>
      <td align="right">0.0011</td>
      <td align="right">0.7438</td>
      <td align="right">7,463</td>
      <td align="right">330</td>
      <td align="left">APMGSRN*</td>
      <td align="right">30.51</td>
      <td align="right">0.0534</td>
      <td align="right">-0.6753</td>
      <td align="right">655</td>
      <td align="right">500</td>
    </tr>
    <tr>
      <td align="left">SL²A-INR</td>
      <td align="right">73.00</td>
      <td align="right">0.0002</td>
      <td align="right">0.9338</td>
      <td align="right">7,793</td>
      <td align="right">316</td>
      <td align="left">SL²A-INR</td>
      <td align="right">35.10</td>
      <td align="right">0.0210</td>
      <td align="right">0.3108</td>
      <td align="right">667</td>
      <td align="right">491</td>
    </tr>
    <tr>
      <td align="left">MC-INR</td>
      <td align="right">66.69</td>
      <td align="right">0.0006</td>
      <td align="right">0.9616</td>
      <td align="right">7,696</td>
      <td align="right">320</td>
      <td align="left">MC-INR</td>
      <td align="right">33.92</td>
      <td align="right">0.0259</td>
      <td align="right">0.2826</td>
      <td align="right">655</td>
      <td align="right">500</td>
    </tr>
    <tr>
      <td align="left"><strong>Ours</strong></td>
      <td align="right"><strong>75.00</strong></td>
      <td align="right"><strong>0.0002</strong></td>
      <td align="right"><strong>0.9736</strong></td>
      <td align="right"><strong>7,868</strong></td>
      <td align="right"><strong>313</strong></td>
      <td align="left"><strong>Ours</strong></td>
      <td align="right"><strong>38.17</strong></td>
      <td align="right"><strong>0.0153</strong></td>
      <td align="right"><strong>0.4494</strong></td>
      <td align="right"><strong>667</strong></td>
      <td align="right"><strong>491</strong></td>
    </tr>
    <tr>
      <td rowspan="11"><strong>Combustion</strong></td>
      <td align="left">SIREN</td>
      <td align="right">30.62</td>
      <td align="right">0.0296</td>
      <td align="right">0.8967</td>
      <td align="right">989</td>
      <td align="right">256</td>
      <td rowspan="11"><strong>Ionization</strong></td>
      <td align="left">SIREN</td>
      <td align="right">43.12</td>
      <td align="right">0.0078</td>
      <td align="right">0.9664</td>
      <td align="right">2,011</td>
      <td align="right">224</td>
    </tr>
    <tr>
      <td align="left">NeurComp</td>
      <td align="right">32.22</td>
      <td align="right">0.0245</td>
      <td align="right">0.9140</td>
      <td align="right">997</td>
      <td align="right">254</td>
      <td align="left">NeurComp</td>
      <td align="right">45.67</td>
      <td align="right">0.0056</td>
      <td align="right">0.9777</td>
      <td align="right">2,002</td>
      <td align="right">225</td>
    </tr>
    <tr>
      <td align="left">CoordNet</td>
      <td align="right">31.09</td>
      <td align="right">0.0280</td>
      <td align="right">0.9014</td>
      <td align="right">981</td>
      <td align="right">258</td>
      <td align="left">CoordNet</td>
      <td align="right">39.28</td>
      <td align="right">0.0135</td>
      <td align="right">0.9451</td>
      <td align="right">1,993</td>
      <td align="right">226</td>
    </tr>
    <tr>
      <td align="left">Instant-NGP</td>
      <td align="right">20.86</td>
      <td align="right">0.1033</td>
      <td align="right">0.4767</td>
      <td align="right">977</td>
      <td align="right">259</td>
      <td align="left">Instant-NGP</td>
      <td align="right">21.93</td>
      <td align="right">0.1988</td>
      <td align="right">0.1837</td>
      <td align="right">1,984</td>
      <td align="right">227</td>
    </tr>
    <tr>
      <td align="left">fV-SRN*</td>
      <td align="right">23.53</td>
      <td align="right">0.0726</td>
      <td align="right">0.6768</td>
      <td align="right">920</td>
      <td align="right">275</td>
      <td align="left">fV-SRN*</td>
      <td align="right">22.76</td>
      <td align="right">0.1039</td>
      <td align="right">-0.6598</td>
      <td align="right">1,802</td>
      <td align="right">250</td>
    </tr>
    <tr>
      <td align="left">WIRE</td>
      <td align="right">17.75</td>
      <td align="right">0.1615</td>
      <td align="right">0.0000</td>
      <td align="right">985</td>
      <td align="right">257</td>
      <td align="left">WIRE</td>
      <td align="right">20.94</td>
      <td align="right">0.2469</td>
      <td align="right">-0.0002</td>
      <td align="right">2,011</td>
      <td align="right">224</td>
    </tr>
    <tr>
      <td align="left">AFNet</td>
      <td align="right">28.33</td>
      <td align="right">0.0392</td>
      <td align="right">0.8643</td>
      <td align="right">997</td>
      <td align="right">254</td>
      <td align="left">AFNet</td>
      <td align="right">50.62</td>
      <td align="right">0.0030</td>
      <td align="right">0.9899</td>
      <td align="right">2,011</td>
      <td align="right">224</td>
    </tr>
    <tr>
      <td align="left">APMGSRN*</td>
      <td align="right">21.52</td>
      <td align="right">0.0951</td>
      <td align="right">0.5493</td>
      <td align="right">920</td>
      <td align="right">275</td>
      <td align="left">APMGSRN*</td>
      <td align="right">18.02</td>
      <td align="right">0.1439</td>
      <td align="right">-9.2996</td>
      <td align="right">1,802</td>
      <td align="right">250</td>
    </tr>
    <tr>
      <td align="left">SL²A-INR</td>
      <td align="right">26.02</td>
      <td align="right">0.0525</td>
      <td align="right">0.8025</td>
      <td align="right">981</td>
      <td align="right">258</td>
      <td align="left">SL²A-INR</td>
      <td align="right">42.64</td>
      <td align="right">0.0078</td>
      <td align="right">0.9496</td>
      <td align="right">2,002</td>
      <td align="right">225</td>
    </tr>
    <tr>
      <td align="left">MC-INR</td>
      <td align="right">18.05</td>
      <td align="right">0.1502</td>
      <td align="right">-0.0065</td>
      <td align="right">989</td>
      <td align="right">256</td>
      <td align="left">MC-INR</td>
      <td align="right">20.93</td>
      <td align="right">0.1686</td>
      <td align="right">-0.3931</td>
      <td align="right">1,909</td>
      <td align="right">236</td>
    </tr>
    <tr>
      <td align="left"><strong>Ours</strong></td>
      <td align="right"><strong>31.17</strong></td>
      <td align="right"><strong>0.0278</strong></td>
      <td align="right"><strong>0.9116</strong></td>
      <td align="right"><strong>997</strong></td>
      <td align="right"><strong>254</strong></td>
      <td align="left"><strong>Ours</strong></td>
      <td align="right"><strong>50.25</strong></td>
      <td align="right"><strong>0.0032</strong></td>
      <td align="right"><strong>0.9905</strong></td>
      <td align="right"><strong>2,011</strong></td>
      <td align="right"><strong>224</strong></td>
    </tr>
  </tbody>
</table>

**Note:** Bold rows indicate SWAP-INR (Ours). `*` follows the method notation used in the original paper. The reported values are from the full datasets used in the paper, not from the downsampled reproduction examples below.

## Reproduction with Downsampled Datasets

Due to supplementary-material size limitations, the provided model weights and example datasets for **ROCOM** and **Ionization** use downsampled versions of the original datasets. The original full-resolution datasets used for the paper experiments are substantially larger.

### Dataset Sizes

| Dataset | Original dataset size (paper) | Downsampled dataset size (repository) |
| :-- | --: | --: |
| ROCOM | 1,943,807 KB | 80,995 KB |
| Ionization | 450,469 KB | 11,270 KB |

### Results with Provided Models and Downsampled Data

| Dataset | Evaluation | PSNR ↑ | NRMSE ↓ | $R^2$ ↑ |
| :-- | :-- | --: | --: | --: |
| ROCOM | Paper (original data) | 47.41 | 0.0073 | 0.9782 |
| ROCOM | Repository (downsampled data) | 43.06 | 0.0108 | 0.9421 |
| Ionization | Paper (original data) | 50.25 | 0.0032 | 0.9905 |
| Ionization | Repository (downsampled data) | 53.66 | 0.0021 | 0.9810 |

> **Note:** The supplied checkpoints were obtained using downsampled ROCOM and Ionization data. These results are **not directly comparable** with full-dataset results because the evaluation data differ. All results reported in this reproduction table were obtained using **FP16**.

## Usage

Run the following commands from the `code/` directory:

```bash
cd code/
```

### Training

**ROCOM**

```bash
python train.py --dataset ROCOM_down_down --num_var 8 --Shared_hidden 64 --Branch_hidden 32 --name ROCOM_down_down_test --device cuda
```

**Ionization**

```bash
python train.py --dataset ionization_down_down --num_var 4 --Shared_hidden 64 --Branch_hidden 24 --name ionization_down_down_test --device cuda
```

The `--name` argument is required. Checkpoints and pruned models are saved under:

```text
Exp/<dataset>/<name>/
```

Additional training options:

```bash
python train.py \
  --dataset ROCOM_down_down \
  --name experiment_name \
  --device cuda \
  --batch_size 100000 \
  --base_num_epochs 2000 \
  --on_num_epochs 1000
```

Use `--device cpu` if CUDA is unavailable.

### Inference

Use the **same dataset, experiment name, and model architecture arguments** as in training.

**ROCOM**

```bash
python inference.py --dataset ROCOM_down_down --num_var 8 --Shared_hidden 64 --Branch_hidden 32 --name ROCOM_down_down_test --device cuda
```

FP16 inference:

```bash
python inference.py --dataset ROCOM_down_down --num_var 8 --Shared_hidden 64 --Branch_hidden 32 --name ROCOM_down_down_test --use_half --device cuda
```

**Ionization**

```bash
python inference.py --dataset ionization_down_down --num_var 4 --Shared_hidden 64 --Branch_hidden 24 --name ionization_down_down_test --device cuda
```

FP16 inference:

```bash
python inference.py --dataset ionization_down_down --num_var 4 --Shared_hidden 64 --Branch_hidden 24 --name ionization_down_down_test --use_half --device cuda
```

## Citation

Citation information will be added when the official WACV 2027 paper metadata is available.
