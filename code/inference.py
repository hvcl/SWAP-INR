import torch
import numpy as np
import argparse

from model import *
from utils import *
from data import *

p = argparse.ArgumentParser()
p.add_argument('--seed', type=int, default=42, help='random seed')
p.add_argument('--dataset', type=str, default='ROCOM')
p.add_argument('--batch_size', type=int, default=100000, help='batch size')

p.add_argument('--in_features', type=int, default=4, metavar='N', help='input dimension')
p.add_argument('--num_var', type=int, default=8, metavar='N', help='number of variables to train')

p.add_argument('--name', type=str, required=True, help='experiment name (folder name for saving results)')
p.add_argument('--device', type=str, default="cuda", help='device 설정 (e.g., "cuda", "cpu")')
p.add_argument("--use_half", action="store_true")

p.add_argument('--Shared_hidden', type=int, default=128, metavar='N', help='hidden dimension of Shared Encoder')
p.add_argument('--Shared_depth', type=int, default=2, metavar='N', help='depth of Shared Encoder')
p.add_argument('--Branch_hidden', type=int, default=64, metavar='N', help='hidden dimension of Branch Network')
p.add_argument('--Branch_depth', type=int, default=6, metavar='N', help='depth of Branch Network')
p.add_argument('--attn_pos', type=int, default=3, metavar='N', help='position of attention in branch decoder')
p.add_argument('--init_w0', type=float, default=30.0, metavar='N', help='initial omega_0 value')
p.add_argument('--w0_min', type=float, default=30.0, metavar='N', help='minimum omega_0 value')
p.add_argument('--w0_max', type=float, default=150.0, metavar='N', help='maximum omega_0 value')
opt = p.parse_args()

set_seed(opt.seed)

data_name = opt.dataset
name = opt.name

model = AttnINR(
        d_in=opt.in_features, num_vars=opt.num_var,
        cfe_depth=opt.Shared_depth, cfe_hidden=opt.Shared_hidden,
        dec_hidden=opt.Branch_hidden, dec_depth=opt.Branch_depth,
        attn_pos=opt.attn_pos, init_w0=opt.init_w0,
        w0_min=opt.w0_min, w0_max=opt.w0_max
    )
if opt.use_half:
    model.half()
    model_pth = f'./Exp/{data_name}/{name}/final_selected_model_half_pruned.pth'
else:
    model_pth = f'./Exp/{data_name}/{name}/final_selected_model_pruned.pth'
model.load_state_dict(torch.load(model_pth), strict=False)
model.to(opt.device)
model.eval()

dataset = MultivariateDataset(dataset=data_name)
num_points, num_frames, data_list = dataset.get_data_info()
testloader = dataset.get_dataloader(batch_size=opt.batch_size, shuffle=False, num_workers=4)

predict_list = []
gt_list = []

with torch.no_grad():
    for idx, (input, gt) in enumerate(testloader):
        if opt.use_half:
            input = input.to(opt.device).half()
        else:
            input = input.to(opt.device)
        out = model(input, return_mode="selected")
        predict_list.append(out.cpu())
        gt_list.append(gt)
        
    predict_array = torch.cat(predict_list, dim=0)
    gt_array = torch.cat(gt_list, dim=0)

data_range = gt_array.max()-gt_array.min()

psnr_list = []
nrmse_list = []

for i in range(len(data_list)):
    psnr_value = calculate_psnr(predict_array[...,i], gt_array[...,i], data_range=data_range)
    psnr_list.append(psnr_value)

r2_list = compute_r2(gt=gt_array, pred=predict_array)

for i in range(len(data_list)):
    nrmse_loss = NRMSE(torch.Tensor(predict_array[:,i]), torch.Tensor(gt_array[:,i]))
    nrmse_list.append(nrmse_loss)
    print(f"{data_list[i]} PSNR: {psnr_list[i]:.2f} dB")
    print(f"{data_list[i]} NRMSE loss value: {nrmse_loss:.5e}")
    print(f"{data_list[i]} R2 score: {r2_list[i]:.4f}")
    print()

print(f"Avg PSNR: {np.mean(psnr_list):.2f} dB")
print(f"Avg NRMSE loss value: {np.mean(nrmse_list):.5e}")
print(f"Avg R2 score: {np.mean(r2_list):.4f}")
