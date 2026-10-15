import torch
import torch.nn as nn
import torch.nn.functional as F

from model import *
from utils import *
from data import MultivariateDataset

import os
import time
import argparse

p = argparse.ArgumentParser()
p.add_argument('--seed', type=int, default=42, help='random seed')
p.add_argument('--dataset', type=str, default='ionization_down_down', choices=['ionization_down_down', 'ROCOM_down_down'], help='dataset name')
p.add_argument('--batch_size', type=int, default=100000, help='batch size')

p.add_argument('--base_num_epochs', type=int, default=2000, help='Baseline epochs')
p.add_argument('--on_num_epochs', type=int, default=1000, help='Attention on epochs')

p.add_argument('--base_lr', type=float, default=2e-5, help='baseline learning rate')
p.add_argument('--on_lr', type=float, default=2e-5, help='attention on learning rate')
p.add_argument('--alpha_lr', type=float, default=2e-6, help='alpha parameter specific learning rate') 
p.add_argument('--patience', type=int, default=20, help='early stopping patience') 

p.add_argument('--in_features', type=int, default=4, metavar='N', help='input dimension')
p.add_argument('--num_var', type=int, default=8, metavar='N', help='number of variables to train')

p.add_argument('--name', type=str, required=True, help='experiment name (e.g., "exp1")')
p.add_argument('--device', type=str, default="cuda", help='(e.g., "cuda", "cpu")')

p.add_argument('--Shared_hidden', type=int, default=128, metavar='N', help='Shared Encoders hidden dimension')
p.add_argument('--Shared_depth', type=int, default=2, metavar='N', help='Shared Encoders depth')
p.add_argument('--Branch_hidden', type=int, default=64, metavar='N', help='Branch Networks hidden dimension')
p.add_argument('--Branch_depth', type=int, default=6, metavar='N', help='Branch Networks depth')
p.add_argument('--attn_pos', type=int, default=3, metavar='N', help='position of branch decoder attention')
p.add_argument('--init_w0', type=float, default=30.0, metavar='N', help='initial omega_0 value')
p.add_argument('--w0_min', type=float, default=30.0, metavar='N', help='minimum omega_0')
p.add_argument('--w0_max', type=float, default=150.0, metavar='N', help='maximum omega_0')
opt = p.parse_args()


########### config setting ###########
set_seed(opt.seed)
LR_BASE = opt.base_lr
LR_ON = opt.on_lr
ALPHA_LR = opt.alpha_lr
MINIMUM_EPOCH = 100
EPOCHS_BASE = opt.base_num_epochs
EPOCHS_ON = opt.on_num_epochs
data_name = opt.dataset
name = opt.name

print(f"Project Name: {name} | Dataset: {data_name}")
target_dir = f"./Exp/{data_name}/{name}/"
log_dir = os.path.join(target_dir, "logs")

########### make directory ###########
os.makedirs('./Exp', exist_ok=True)
os.makedirs(target_dir, exist_ok=True)

########### model setting ###########
model = AttnINR(
        d_in=opt.in_features, num_vars=opt.num_var,
        cfe_depth=opt.Shared_depth, cfe_hidden=opt.Shared_hidden,
        dec_hidden=opt.Branch_hidden, dec_depth=opt.Branch_depth,
        attn_pos=opt.attn_pos, init_w0=opt.init_w0,
        w0_min=opt.w0_min, w0_max=opt.w0_max
    )
model.to(opt.device)

########### dataset setting ###########
dataset = MultivariateDataset(dataset=data_name)
num_points, num_frames, data_list = dataset.get_data_info()
trainloader = dataset.get_dataloader(batch_size=opt.batch_size, shuffle=True, num_workers=4)


for p in model.decoder_on.parameters():
    p.requires_grad_(False)

alpha_params = []
cfe_params = []
base_decoder_params = []
for param_name, param in model.named_parameters():
    if 'cfe' in param_name and 'alpha' not in param_name:
        cfe_params.append(param)
    elif 'decoder_base' in param_name and 'alpha' not in param_name:
        base_decoder_params.append(param)
    elif ('cfe' in param_name or 'decoder_base' in param_name) and 'alpha' in param_name:
        alpha_params.append(param)

optimizer_base = torch.optim.Adam([
    {'params': cfe_params + base_decoder_params, 'lr': LR_BASE},
    {'params': alpha_params, 'lr': ALPHA_LR}
])

best_base_metric = float("inf")
best_on_metric = float("inf")
best_base_model_state = None
best_on_model_state = None
t_start = time.time()

early_stop_patience = opt.patience
no_improve_count = 0

############## Base Training ##############
for step in range(1, EPOCHS_BASE+1):
    model.train()
    
    epoch_L_base_var_sum = None

    for idx, (input, gt) in enumerate(trainloader):
        input, gt = input.to(opt.device), gt.to(opt.device)
        optimizer_base.zero_grad()

        out_base = model(input, return_mode="base")
        L_base = F.mse_loss(out_base, gt, reduction="none").mean(dim=0)

        loss = L_base.sum()
        loss.backward()
        optimizer_base.step()
        
        if epoch_L_base_var_sum is None:
            epoch_L_base_var_sum = L_base.detach().clone()
        else:
            epoch_L_base_var_sum += L_base.detach()

    epoch_L_base_var_mean = epoch_L_base_var_sum / len(trainloader)
    epoch_base_metric = epoch_L_base_var_mean.mean().item()
    
    print(f"Epoch {step}: L_base={epoch_base_metric:.8f}")
    
    if epoch_base_metric < best_base_metric:
        best_base_metric = epoch_base_metric
        no_improve_count = 0

        best_base_model_state = model.cpu().state_dict()  # save best model state
        torch.save(best_base_model_state, target_dir + "best_base_model.pth")
        model.to(opt.device)
    else:
        if step >= MINIMUM_EPOCH:
            no_improve_count += 1

    if step >= MINIMUM_EPOCH and no_improve_count >= early_stop_patience:
        log_msg = (
            f"Base Decoder Early stopping triggered at epoch {step} "
            f"(no improvement for {early_stop_patience} epochs)"
        )
        break



model.load_state_dict(torch.load(target_dir + "best_base_model.pth"), strict=False)

for p in model.cfe.parameters():
    p.requires_grad_(False)

for p in model.decoder_base.parameters():
    p.requires_grad_(False)

for p in model.decoder_on.parameters():
    p.requires_grad_(True)

model.decoder_on.load_state_dict(
    model.decoder_base.state_dict(),
    strict=False
)

alpha_params = []
on_decoder_params = []
for param_name, param in model.named_parameters():
    if 'decoder_on' in param_name and 'alpha' not in param_name:
        on_decoder_params.append(param)
    elif 'decoder_on' in param_name and 'alpha' in param_name:
        alpha_params.append(param)

optimizer_on = torch.optim.Adam([
    {'params': on_decoder_params, 'lr': LR_ON},
    {'params': alpha_params, 'lr': ALPHA_LR}
])

no_improve_count = 0

############## Attn On Training ##############
for step in range(1, EPOCHS_ON+1):
    model.train()

    epoch_L_on_var_sum = None

    for idx, (input, gt) in enumerate(trainloader):
        input, gt = input.to(opt.device), gt.to(opt.device)
        optimizer_on.zero_grad()

        out_on = model(input, return_mode="on")
        L_on = F.mse_loss(out_on, gt, reduction="none").mean(dim=0)

        loss = L_on.sum()
        loss.backward()
        optimizer_on.step()
        
        if epoch_L_on_var_sum is None:
            epoch_L_on_var_sum = L_on.detach().clone()
        else:
            epoch_L_on_var_sum += L_on.detach()

    epoch_L_on_mean = epoch_L_on_var_sum / len(trainloader)
    epoch_on_metric = epoch_L_on_mean.mean().item()
    
    print(f"Epoch {step}: L_on={epoch_on_metric:.8f}")
    
    if epoch_on_metric < best_on_metric:
        best_on_metric = epoch_on_metric
        no_improve_count = 0

        best_on_model_state = model.cpu().state_dict()  # save best model state
        torch.save(best_on_model_state, target_dir + "best_on_model.pth")
        model.to(opt.device)
    else:
        if step >= MINIMUM_EPOCH:
            no_improve_count += 1

    if step >= MINIMUM_EPOCH and no_improve_count >= early_stop_patience:
        log_msg = (
            f"On Decoder Early stopping triggered at epoch {step} "
            f"(no improvement for {early_stop_patience} epochs)"
        )
        break


############## pruning & quantization ##############
model.load_state_dict(torch.load(target_dir + "best_on_model.pth", map_location=opt.device), strict=False)
model.to(opt.device)

testloader = dataset.get_dataloader(batch_size=opt.batch_size, shuffle=False, num_workers=4)

select_on, L_base, L_on = build_selection_mask_from_both(model, testloader, opt.device)
model.set_decoder_selection(select_on)

torch.save(model.state_dict(), target_dir + "final_selected_model.pth")
torch.save(model.half().state_dict(), target_dir + "final_selected_model_half.pth")

pruned = prune_state_dict_by_selected_mask(
        torch.load(target_dir + "final_selected_model.pth"),
        attn_layer_index=opt.attn_pos,
        num_vars=len(data_list),
        mask_key="decoder_select_on",
        keep_mask_key=True,
        verbose=True,
    )
torch.save(pruned, target_dir + 'final_selected_model_pruned.pth')

pruned = prune_state_dict_by_selected_mask(
        torch.load(target_dir + "final_selected_model_half.pth"),
        attn_layer_index=opt.attn_pos,
        num_vars=len(data_list),
        mask_key="decoder_select_on",
        keep_mask_key=True,
        verbose=True,
    )
torch.save(pruned, target_dir + 'final_selected_model_half_pruned.pth')