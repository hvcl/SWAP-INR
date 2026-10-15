import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional

import re
import os
import numpy as np
import random
from sklearn.metrics import r2_score



def get_mgrid(sidelen, dim=2, s=1,t=0):
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[1]:s, :sidelen[0]:s], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / (sidelen[1] - 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[0] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[2]:s, :sidelen[1]:s, :sidelen[0]:s], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / (sidelen[2] - 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[0] - 1)
    elif dim == 4:
        pixel_coords = np.stack(np.mgrid[:sidelen[0]:(t+1), :sidelen[3]:s, :sidelen[2]:s, :sidelen[1]:s], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[3] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
        pixel_coords[..., 3] = pixel_coords[..., 3] / (sidelen[1] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)
    pixel_coords -= 0.5
    pixel_coords *= 2.
    
    return pixel_coords

def Subsample(dim, scale):
    coords_indices = []
    for z in range(0,dim[2],scale):
        for y in range(0,dim[1],scale):
            for x in range(0,dim[0],scale):
                index = (((z) * dim[1] + y) * dim[0] + x)
                coords_indices.append(index)
    
    return np.asarray(coords_indices)

def calculate_psnr(original, reconstructed, data_range=None):
    original = np.array(original, dtype=np.float64)
    reconstructed = np.array(reconstructed, dtype=np.float64)
    
    if data_range is None:
        min_val = min(original.min(), reconstructed.min())
        max_val = max(original.max(), reconstructed.max())
        data_range = max_val - min_val
    
    mse = np.mean((original - reconstructed) ** 2)
    
    if mse == 0:
        return float('inf')
    
    psnr_value = 10 * np.log10((data_range ** 2) / mse)
    return psnr_value

def NRMSE(pred: torch.Tensor, target: torch.Tensor, normalization: str = "min-max"):
    mse = F.mse_loss(pred, target)
    rmse = torch.sqrt(mse)

    if normalization == "mean":
        norm_factor = torch.mean(target)
    elif normalization == "min-max":
        norm_factor = target.max() - target.min()
    elif normalization == "euclidean":
        norm_factor = torch.sqrt(torch.sum(target ** 2))
    else:
        raise ValueError("Unsupported normalization type. Choose from ['mean', 'min-max', 'euclidean']")

    return rmse / norm_factor

def compute_r2(gt, pred):
    k = gt.shape[1]
    r2_list = []

    for i in range(k):
        y_true = gt[:, i]
        y_pred = pred[:, i]
        r2 = r2_score(y_true, y_pred)
        r2_list.append(r2)

    return r2_list

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    print(f"Random seed set to {seed}")

@torch.no_grad()
def build_selection_mask_from_both(model, loader, device):
    model.eval()
    sum_base = None
    sum_on = None
    n = 0

    for x, gt in loader:
        x, gt = x.to(device), gt.to(device)
        out_base, out_on = model(x, return_mode="both")

        Lb = F.mse_loss(out_base, gt, reduction="none").mean(dim=0)
        Lo = F.mse_loss(out_on,   gt, reduction="none").mean(dim=0)

        sum_base = Lb.detach().clone() if sum_base is None else (sum_base + Lb.detach())
        sum_on   = Lo.detach().clone() if sum_on   is None else (sum_on   + Lo.detach())
        n += 1

    L_base = sum_base / max(n, 1)
    L_on   = sum_on   / max(n, 1)

    select_on = (L_on < L_base)
    return select_on, L_base, L_on

def _infer_max_hidden_layer_index(sd: Dict[str, torch.Tensor], decoder_prefix: str) -> int:
    pat = re.compile(rf"^{re.escape(decoder_prefix)}\.hidden_layers\.(\d+)\.")
    idxs = []
    for k in sd.keys():
        m = pat.match(k)
        if m:
            idxs.append(int(m.group(1)))
    return max(idxs) if idxs else -1

def _get_select_on_from_state_dict(sd: Dict[str, torch.Tensor], V: int, mask_key: str) -> List[bool]:
    if mask_key not in sd:
        raise KeyError(
            f"Mask buffer '{mask_key}' not found in state_dict. "
            f"Make sure you saved after model.set_decoder_selection(select_on)."
        )
    m = sd[mask_key].detach().cpu()
    if m.dtype == torch.bool:
        mask = m.tolist()
    else:
        mask = (m > 0).tolist()
    if len(mask) != V:
        raise ValueError(f"Mask length != V. mask={len(mask)}, inferred V={V}")
    return mask

def prune_state_dict_by_selected_mask(
    sd_in: Dict[str, torch.Tensor],
    num_vars: int = None,
    attn_layer_index: int = 3,
    mask_key: str = "decoder_select_on",
    keep_mask_key: bool = True,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    
    sd = dict(sd_in)
    V = num_vars
    select_on = _get_select_on_from_state_dict(sd, V, mask_key)

    max_base_h = _infer_max_hidden_layer_index(sd, "decoder_base")
    max_on_h   = _infer_max_hidden_layer_index(sd, "decoder_on")

    removed = 0

    def _drop_prefix(prefix: str):
        nonlocal removed
        keys = [k for k in sd.keys() if k.startswith(prefix)]
        for k in keys:
            sd.pop(k, None)
            removed += 1

    for v in range(V):
        if select_on[v]:
            for li in range(max_base_h + 1):
                _drop_prefix(f"decoder_base.hidden_layers.{li}.layers.{v}.")
            _drop_prefix(f"decoder_base.out_layers.{v}.")
        else:
            for li in range(attn_layer_index + 1, max_on_h + 1):
                _drop_prefix(f"decoder_on.hidden_layers.{li}.layers.{v}.")
            _drop_prefix(f"decoder_on.out_layers.{v}.")

    if not keep_mask_key:
        if mask_key in sd:
            sd.pop(mask_key, None)
            removed += 1

    if verbose:
        print(f"[prune] V={V}, base_hidden_layers_max={max_base_h}, on_hidden_layers_max={max_on_h}")
        print(f"[prune] removed_keys={removed}, remaining_keys={len(sd)}")
        n_on = sum(select_on)
        print(f"[prune] select_on True count = {n_on}/{V}")

    return sd