import os
import numpy as np
import torch
import cv2

def save_vis_polar(pred, out_dir, name):
    pred = pred.unsqueeze(0)
    pred[:, 0, :, 1:, 1:] = pred[:, 0, :, :-1, :-1].clone()
    pred[:, 1, :, :-1, 1:] = pred[:, 1, :, 1:, :-1].clone()
    pred[:, 2, :, :-1, :-1] = pred[:, 2, :, 1:, 1:].clone()
    pred[:, 3, :, 1:, :-1] = pred[:, 3, :, :-1, 1:].clone()

    pred = pred ** 2.2
    pred[:, :, 0] /= 1.67
    pred[:, :, 2] /= 2.30
    
    s0 = torch.sum(pred,dim=1)/2
    s1 = pred[:,0]-pred[:,2]
    s2 = pred[:,1]-pred[:,3]
    dop = torch.sqrt(s1**2+s2**2)/(s0)
    dop[dop>1] = 1

    s0[:,0] *= 1.67
    s0[:,2] *= 2.30

    s0 = (s0/2)**(1.0/2.2)

    # Normalize to 0  2^14 range and convert to numpy array
    s0 = (s0.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0) * 65535).cpu().numpy().astype(
        np.uint16)
    s0 = cv2.cvtColor(s0, cv2.COLOR_RGB2BGR)
    dop = (dop.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0) * 65535).cpu().numpy().astype(
        np.uint16)
    dop = cv2.cvtColor(dop, cv2.COLOR_RGB2BGR)

    now_dir = f'{out_dir}/{name}'
    os.makedirs(now_dir, exist_ok=True)

    # Save predictions as png
    cv2.imwrite(f'{now_dir}/img_vis_s0.png', s0)
    cv2.imwrite(f'{now_dir}/img_vis_dop.png', dop)
