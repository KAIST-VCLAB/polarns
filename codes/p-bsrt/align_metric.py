import os
import glob
import numpy as np
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import torch.nn as nn
import torch.nn.functional as F
import argparse

from pwcnet.pwcnet import PWCNet
from model.utils.warp import warp
from model.utils.spatial_color_alignment import get_gaussian_kernel, match_colors
from model.model import polar_demosaic
from model.loss import mse_for_properties_from_stokes_no_reduction
import cv2

def cut_image(img):
    return img[24:-24, 24:-24]

def degree2stokes(img_degree):
    # B c H W
    img_degree = torch.reshape(img_degree, [img_degree.shape[0], 4, 3, img_degree.shape[-2], img_degree.shape[-1]]) # B Angles C H W
    s0 = torch.sum(img_degree, dim=1) / 2
    s1 = img_degree[:, 0] - img_degree[:, 2]
    s2 = img_degree[:, 1] - img_degree[:, 3]
    return torch.cat([s0,s1,s2],dim=1)

def batch_unpack_9ch(img_9ch):
    img_9ch = torch.reshape(img_9ch, [img_9ch.shape[0], 3, 3, img_9ch.shape[-2], img_9ch.shape[-1]])
    return img_9ch[:, 0],img_9ch[:, 1],img_9ch[:, 2]

def psnr_ours(mse, peak=1.0):
    return -10 * np.log10(mse / (peak**2))

def imreadmono2tensor(filename):
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    stacked_images = [
        img[1::4, 1::4], # red 0
        img[0::4, 1::4], # red 45
        img[0::4, 0::4], # red 90
        img[1::4, 0::4], # red 135
        img[1::4, 3::4], # green 0
        img[0::4, 3::4], # green 45
        img[0::4, 2::4], # green 90
        img[1::4, 2::4], # green 135
        img[3::4, 1::4], # green 0
        img[2::4, 1::4], # green 45
        img[2::4, 0::4], # green 90
        img[3::4, 0::4], # green 135
        img[3::4, 3::4], # blue 0
        img[2::4, 3::4], # blue 45
        img[2::4, 2::4], # blue 90
        img[3::4, 2::4], # blue 135
    ]
    img = np.stack(stacked_images,axis=0)
    img = torch.from_numpy(img.astype(np.float32)) /65535
    return img

def stokes2degree(img_stokes):
    # B c H W
    s0,s1,s2 = batch_unpack_9ch(img_stokes)  # B Stokes C H W
    img0 = (s0 + s1)/2
    img45 = (s0 + s2)/2
    img90 = (s0 - s1)/2
    img135 = (s0 - s2)/2
    return torch.cat([img0,img45,img90,img135],dim=1)

def stokes2properties(img_9ch):
    # B c H W
    s0,s1,s2 = batch_unpack_9ch(img_9ch)  # B Stokes C H W

    s0[s0<=0] = torch.finfo(torch.float32).eps

    dop = torch.sqrt(s1 ** 2 + s2 ** 2) / (s0)
    aolp = torch.atan2(s2, s1) / 2
    dop[dop < 0] = 0
    dop[dop > 1] = 1
    return torch.cat([s0,dop,aolp],dim=1)

def compute_metrics(gt_folder, rendered_folder, out_file_name="s0"):
    ### Load the alignment network
    sr_factor = 2
    boundary_ignore = 24
    alignment_net = PWCNet(load_pretrained=True,
                        weights_path='./pwcnet/pwcnet-network-default.pth')
    alignment_net = alignment_net.cuda()
    for param in alignment_net.parameters():
        param.requires_grad = False

    gauss_kernel, ksz = get_gaussian_kernel(sd=1.5)


    gt_files = sorted(glob.glob(f'{gt_folder}/*/gt_s0.npy'))
    rendered_files = sorted(glob.glob(f'{rendered_folder}/*/{out_file_name}.npy'))
    assert len(gt_files) == len(rendered_files), "Mismatch in number of gt and rendered files"
    
    loss_fn_lpips = lpips.LPIPS(net='alex').to("cuda" if torch.cuda.is_available() else "cpu")
    
    total_psnr, total_ssim, total_lpips, total_psnr_dolp, total_psnr_aolp = [], [], [], [], []
    
    for gt_file, rendered_file in zip(gt_files, rendered_files):
        assert os.path.basename(os.path.dirname(gt_file)) == os.path.basename(os.path.dirname(rendered_file)), "Mismatch in file names"
        gt_s0 = np.load(gt_file)
        gt_s1 = np.load(gt_file.replace('gt_s0', 'gt_s1')) # H, W, C
        gt_s2 = np.load(gt_file.replace('gt_s0', 'gt_s2'))
        assert np.sum(gt_s0 < 0) == 0, "Negative values in gt_s0"
        gt_stokes = torch.tensor(np.concatenate([gt_s0, gt_s1, gt_s2], axis=-1).transpose(2, 0, 1)).float().cuda().unsqueeze(0)

        rendered_s0 = np.load(rendered_file) # 1 C H W
        rendered_s1 = np.load(rendered_file.replace('s0.npy', 's1.npy'))
        rendered_s2 = np.load(rendered_file.replace('s0.npy', 's2.npy'))
        assert np.sum(rendered_s0 < 0) == 0, "Negative values in gt_s0"
        pred_stokes = torch.tensor(np.concatenate([rendered_s0, rendered_s1, rendered_s2], axis=1)).float().cuda()

        pred = stokes2degree(pred_stokes.float())
        gt = stokes2degree(gt_stokes.float())
        pred_rgb = pred_stokes[:,0:3]
        gt_rgb = gt_stokes[:,0:3]

        burst_input = imreadmono2tensor(gt_file.replace('gt_s0.npy', 'frame_00.png')).unsqueeze(0).cuda() # 1 4 H W

        with torch.no_grad():
            # flow = self.alignment_net(pred_rgb / (pred_rgb.max() + 1e-6), gt_rgb / (gt_rgb.max() + 1e-6))
            flow = alignment_net((pred_rgb / (pred_rgb.max() + 1e-6)).float(), (gt_rgb / (gt_rgb.max() + 1e-6)).float())

        # Warp the prediction to the ground truth coordinates
        pred_warped = warp(pred, flow)

        # Warp the base input frame to the ground truth. This will be used to estimate the color transformation between
        # the input and the ground truth
        ds_factor = 1.0 / float(2.0 * sr_factor)
        flow_ds = F.interpolate(flow, scale_factor=ds_factor, mode='bilinear', recompute_scale_factor=True, align_corners=False) * ds_factor

        burst_0 = (polar_demosaic(burst_input)*2).contiguous()
        burst_0_warped = warp(burst_0, flow_ds)
        frame_gt_ds = F.interpolate(gt_rgb, scale_factor=ds_factor, mode='bilinear', recompute_scale_factor=True, align_corners=False)

        # Match the colorspace between the prediction and ground truth
        pred_warped_m, valid = match_colors(frame_gt_ds, burst_0_warped, pred_warped, ksz,
                                                      gauss_kernel)

        # Ignore boundary pixels if specified
        if boundary_ignore is not None:
            pred_warped_m = pred_warped_m[..., boundary_ignore:-boundary_ignore,
                            boundary_ignore:-boundary_ignore]
            gt_stokes = gt_stokes[..., boundary_ignore:-boundary_ignore, boundary_ignore:-boundary_ignore]

            valid = valid[..., boundary_ignore:-boundary_ignore, boundary_ignore:-boundary_ignore]

        pred_warped_m_stokes = degree2stokes(pred_warped_m)
        mse_s0,mse_dop,bias_dop,mse_aolp = mse_for_properties_from_stokes_no_reduction(pred_warped_m_stokes,gt_stokes)

        pw_s0, _, _ = batch_unpack_9ch(stokes2properties(pred_warped_m_stokes))
        gt_w_s0, _, _ = batch_unpack_9ch(stokes2properties(gt_stokes))

        ss = ssim(pw_s0.contiguous().permute(2, 3, 1, 0).squeeze(-1).cpu().numpy(), gt_w_s0.contiguous().permute(2, 3, 1, 0).squeeze(-1).cpu().numpy(), data_range=1.0, channel_axis=2)
        lp = loss_fn_lpips(pw_s0.contiguous() * 2 - 1, gt_w_s0.contiguous() * 2 - 1).squeeze().cpu().item()
        total_ssim.append(ss)
        total_lpips.append(lp)

        eps = 1e-12
        elem_ratio = mse_s0.numel() / valid.numel()
        mse_s0 = (mse_s0 * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)
        mse_dop = (mse_dop * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)
        mse_aolp = (mse_aolp * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)
        total_psnr.append(psnr_ours(mse_s0.cpu().item()))
        total_psnr_dolp.append(psnr_ours(mse_dop.cpu().item()))
        total_psnr_aolp.append(psnr_ours(mse_aolp.cpu().item(), np.pi))
    
    avg_psnr = np.array(total_psnr).mean()
    avg_ssim = np.array(total_ssim).mean()
    avg_lpips = np.array(total_lpips).mean()
    avg_psnr_dolp = np.array(total_psnr_dolp).mean()
    avg_psnr_aolp = np.array(total_psnr_aolp).mean()
    
    return avg_psnr, avg_ssim, avg_lpips, avg_psnr_dolp, avg_psnr_aolp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute metrics for ground truth and rendered images.")
    parser.add_argument('--gt', type=str, default="/root/data/polar_burst_v2/test", help="Path to the ground truth folder")
    parser.add_argument('--render', type=str, default="/root/data/comparison/polar_real_rgb_pretrain/bsrt", help="Path to the rendered images folder")
    parser.add_argument('--out_file_name', type=str, default="s0", help="Output file name")
    
    args = parser.parse_args()

    avg_psnr, avg_ssim, avg_lpips, avg_psnr_dolp, avg_psnr_aolp = compute_metrics(args.gt, args.render, args.out_file_name)
    
    with open(os.path.join(args.render, 'aligned_summary.txt'), 'w') as f:
        f.write(f"PSNR {avg_psnr:.4f} SSIM {avg_ssim:.4f} LPIPS {avg_lpips:.4f} DoLP {avg_psnr_dolp:.4f} AoLP {avg_psnr_aolp:.4f}\n")

    print(f"PSNR {avg_psnr:.4f} SSIM {avg_ssim:.4f} LPIPS {avg_lpips:.4f} DoLP {avg_psnr_dolp:.4f} AoLP {avg_psnr_aolp:.4f}")

    with open(os.path.join(args.render, 'total_summary_aligned.txt'), 'w') as f:
        f.write(f"{avg_psnr:.2f} & {avg_ssim:.3f} & {avg_lpips:.3f} & {avg_psnr_dolp:.2f} & {avg_psnr_aolp:.2f}\n")
    print(f"{avg_psnr:.2f} & {avg_ssim:.3f} & {avg_lpips:.3f} & {avg_psnr_dolp:.2f} & {avg_psnr_aolp:.2f}\n")