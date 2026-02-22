import os
import glob
import numpy as np
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import argparse

def cut_image(img):
    return img[24:-24, 24:-24]

def psnr_ours(gt, rendered, peak=1.0):
    return -10 * np.log10(np.mean((gt - rendered)**2) / peak**2)

def compute_metrics(gt_folder, rendered_folder):
    gt_files = sorted(glob.glob(f'{gt_folder}/*/gt_s0.npy'))
    rendered_files = sorted(glob.glob(f'{rendered_folder}/*/output_s0.npy'))
    
    assert len(gt_files) == len(rendered_files), f"Mismatch in number of gt {len(gt_files)} and rendered files {len(rendered_files)}"
    
    loss_fn_lpips = lpips.LPIPS(net='alex').to("cuda" if torch.cuda.is_available() else "cpu")
    
    total_psnr, total_ssim, total_lpips = [], [], []
    
    for gt_file, rendered_file in zip(gt_files, rendered_files):
        assert os.path.basename(os.path.dirname(gt_file)) == os.path.basename(os.path.dirname(rendered_file)), "Mismatch in file names"
        gt = np.load(gt_file)
        rendered = np.load(rendered_file)
        rendered = rendered.transpose(2, 3, 1, 0).squeeze(-1)
        gt[gt < 0] = 0
        gt[gt > 1] = 1
        rendered[rendered < 0] = 0
        rendered[rendered > 1] = 1

        assert np.sum(gt < 0) == 0, "Negative values in gt_s0"
        assert np.sum(rendered < 0) == 0, "Negative values in rendered_s0"
        assert gt.shape == rendered.shape, f"Shape mismatch: {gt.shape} vs {rendered.shape}"
        
        gt = cut_image(gt)
        rendered = cut_image(rendered)

        # Compute PSNR
        # psnr_value = psnr(gt, rendered, data_range=1.0)
        psnr_value = psnr_ours(gt, rendered)
        # print(os.path.basename(os.path.dirname(gt_file)), os.path.basename(os.path.dirname(rendered_file)), psnr_value)
        total_psnr.append(psnr_value)
        
        # Compute SSIM
        ssim_value = ssim(gt, rendered, data_range=1.0, channel_axis=2)
        total_ssim.append(ssim_value)
        
        # Compute LPIPS
        gt_tensor = torch.tensor(gt).permute(2, 0, 1).unsqueeze(0).float()
        rendered_tensor = torch.tensor(rendered).permute(2, 0, 1).unsqueeze(0).float()
        
        if torch.cuda.is_available():
            gt_tensor, rendered_tensor = gt_tensor.cuda(), rendered_tensor.cuda()
        
        lpips_value = loss_fn_lpips(gt_tensor, rendered_tensor).item()
        total_lpips.append(lpips_value)
    
    avg_psnr = np.array(total_psnr).mean()
    avg_ssim = np.array(total_ssim).mean()
    avg_lpips = np.array(total_lpips).mean()
    
    return avg_psnr, avg_ssim, avg_lpips

def compute_dop_metrics(gt_folder, rendered_folder):
    gt_files = sorted(glob.glob(f'{gt_folder}/*/gt_s0.npy'))
    rendered_files = sorted(glob.glob(f'{rendered_folder}/*/output_s0.npy'))
    
    assert len(gt_files) == len(rendered_files), "Mismatch in number of gt and rendered files"
    
    loss_fn_lpips = lpips.LPIPS(net='alex').to("cuda" if torch.cuda.is_available() else "cpu")
    
    total_psnr, total_ssim, total_lpips = [], [], []
    
    for gt_file, rendered_file in zip(gt_files, rendered_files):
        assert os.path.basename(os.path.dirname(gt_file)) == os.path.basename(os.path.dirname(rendered_file)), "Mismatch in file names"
        gt_s0 = np.load(gt_file)
        gt_s1 = np.load(gt_file.replace('gt_s0', 'gt_s1'))
        gt_s2 = np.load(gt_file.replace('gt_s0', 'gt_s2'))
        gt_dop = np.sqrt(gt_s1**2+gt_s2**2)/(gt_s0)
        gt_dop[gt_dop>1] = 1
        assert np.sum(gt_s0 < 0) == 0, "Negative values in gt_s0"
        assert np.sum(gt_dop < 0) == 0, "Negative values in gt_dop"

        rendered_s0 = np.load(rendered_file)
        rendered_s1 = np.load(rendered_file.replace('s0.npy', 's1.npy'))
        rendered_s2 = np.load(rendered_file.replace('s0.npy', 's2.npy'))
        rendered_dop = np.sqrt(rendered_s1**2+rendered_s2**2)/(rendered_s0)
        rendered_dop[rendered_dop>1] = 1
        assert np.sum(rendered_s0 < 0) == 0, "Negative values in rendered_s0"
        assert np.sum(rendered_dop < 0) == 0, "Negative values in rendered_dop"
        rendered_dop = rendered_dop.transpose(2, 3, 1, 0).squeeze(-1)

        gt_dop = np.nan_to_num(gt_dop, nan=1.0, posinf=1.0, neginf=0.0)
        rendered_dop = np.nan_to_num(rendered_dop, nan=1.0, posinf=1.0, neginf=0.0)

        assert gt_dop.shape == rendered_dop.shape, f"Shape mismatch: {gt_dop.shape} vs {rendered_dop.shape}"
        
        # Compute PSNR
        # psnr_value = psnr(gt_dop, rendered_dop, data_range=1.0)
        psnr_value = psnr_ours(gt_dop, rendered_dop)
        total_psnr.append(psnr_value)
        
    avg_psnr = np.array(total_psnr).mean()
    
    return avg_psnr

def compute_aolp_metrics(gt_folder, rendered_folder):
    gt_files = sorted(glob.glob(f'{gt_folder}/*/gt_s1.npy'))
    rendered_files = sorted(glob.glob(f'{rendered_folder}/*/output_s1.npy'))
    
    assert len(gt_files) == len(rendered_files), "Mismatch in number of gt and rendered files"
    
    loss_fn_lpips = lpips.LPIPS(net='alex').to("cuda" if torch.cuda.is_available() else "cpu")
    
    total_mae_aolp, total_psnr_aolp = [], []
    
    for gt_file, rendered_file in zip(gt_files, rendered_files):
        assert os.path.basename(os.path.dirname(gt_file)) == os.path.basename(os.path.dirname(rendered_file)), "Mismatch in file names"
        gt_s1 = np.load(gt_file)
        gt_s2 = np.load(gt_file.replace('gt_s1', 'gt_s2'))
        rendered_s1 = np.load(rendered_file)
        rendered_s1 = rendered_s1.transpose(2, 3, 1, 0).squeeze(-1)
        rendered_s2 = np.load(rendered_file.replace('s1.npy', 's2.npy'))
        rendered_s2 = rendered_s2.transpose(2, 3, 1, 0).squeeze(-1)

        assert gt_s1.shape == rendered_s1.shape, f"Shape mismatch: {gt_s1.shape} vs {rendered_s1.shape}"

        gt_aolp = np.arctan2(gt_s2, gt_s1) / 2
        rendered_aolp = np.arctan2(rendered_s2, rendered_s1) / 2

        gt_aolp = cut_image(gt_aolp)
        rendered_aolp = cut_image(rendered_aolp)

        # Compute angle difference
        def diff_angles(angle1,angle2):
            # [-pi,pi]
            return (angle1 - angle2 + torch.pi) % (2 * torch.pi) - torch.pi

        def diff_aolp(aolp1,aolp2):
            return diff_angles(aolp1*2,aolp2*2)/2

        angle_diff = diff_aolp(gt_aolp, rendered_aolp)
        mae_aolp = np.mean(np.abs(angle_diff))
        mse_aolp = np.mean(angle_diff**2)
        psnr_aolp = - 10*np.log10(mse_aolp/(np.pi**2))

        total_mae_aolp.append(mae_aolp)
        total_psnr_aolp.append(psnr_aolp)
    
    avg_mae_aolp = np.array(total_mae_aolp).mean()
    avg_psnr_aolp = np.array(total_psnr_aolp).mean()
    
    return avg_mae_aolp, avg_psnr_aolp


def compute_s1s2_metrics(gt_folder, rendered_folder):
    gt_files = sorted(glob.glob(f'{gt_folder}/*/gt_s1.npy'))
    rendered_files = sorted(glob.glob(f'{rendered_folder}/*/output_s1.npy'))
    
    assert len(gt_files) == len(rendered_files), "Mismatch in number of gt and rendered files"

    total_s1_psnr, total_s2_psnr = [], []
    
    for gt_file, rendered_file in zip(gt_files, rendered_files):
        assert os.path.basename(os.path.dirname(gt_file)) == os.path.basename(os.path.dirname(rendered_file)), "Mismatch in file names"
        gt_s1 = np.load(gt_file)
        gt_s2 = np.load(gt_file.replace('gt_s1', 'gt_s2'))
        rendered_s1 = np.load(rendered_file)
        rendered_s1 = rendered_s1.transpose(2, 3, 1, 0).squeeze(-1)
        rendered_s2 = np.load(rendered_file.replace('s1.npy', 's2.npy'))
        rendered_s2 = rendered_s2.transpose(2, 3, 1, 0).squeeze(-1)

        assert gt_s1.shape == rendered_s1.shape, f"Shape mismatch: {gt_s1.shape} vs {rendered_s1.shape}"

        psnr_s1_value = psnr_ours(gt_s1, rendered_s1, peak=2.0)
        psnr_s2_value = psnr_ours(gt_s2, rendered_s2, peak=2.0)
        total_s1_psnr.append(psnr_s1_value)
        total_s2_psnr.append(psnr_s2_value)

    avg_s1_psnr = np.array(total_s1_psnr).mean()
    avg_s2_psnr = np.array(total_s2_psnr).mean()
    
    return avg_s1_psnr, avg_s2_psnr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute metrics for ground truth and rendered images.")
    parser.add_argument('--gt', type=str, default="/mnt/d/Data/polar-results/comparison/test_48x48_down2_v2", help="Path to the ground truth folder")
    parser.add_argument('--render', type=str, default="/mnt/d/Data/polar-results/comparison/polar_synthetic_rgb_pretrain/bsrt", help="Path to the rendered images folder")
    
    args = parser.parse_args()

    avg_psnr, avg_ssim, avg_lpips = compute_metrics(args.gt, args.render)
    
    with open(os.path.join(args.render, 's0_summary.txt'), 'w') as f:
        f.write(f"PSNR {avg_psnr:.4f} SSIM {avg_ssim:.4f} LPIPS {avg_lpips:.4f}\n")

    print(f"PSNR {avg_psnr:.4f} SSIM {avg_ssim:.4f} LPIPS {avg_lpips:.4f}")

    avg_dop_psnr = compute_dop_metrics(args.gt, args.render)
    
    with open(os.path.join(args.render, 'dop_summary.txt'), 'w') as f:
        f.write(f"PSNR {avg_dop_psnr:.4f}\n")

    print(f"DOP: PSNR {avg_dop_psnr:.4f}")

    avg_mae_aolp, avg_psnr_aolp = compute_aolp_metrics(args.gt, args.render)
    
    with open(os.path.join(args.render, 'aolp_summary.txt'), 'w') as f:
        f.write(f"MAE {avg_mae_aolp:.4f} PSNR {avg_psnr_aolp:.4f}\n")

    print(f"AOLP: MAE {avg_mae_aolp:.4f} PSNR {avg_psnr_aolp:.4f}")

    with open(os.path.join(args.render, 'total_summary_latex_v2.txt'), 'w') as f:
        f.write(f"{avg_psnr:.2f} & {avg_ssim:.3f} & {avg_lpips:.3f} & {avg_dop_psnr:.2f} & {avg_psnr_aolp:.2f}\n")
    print(f"{avg_psnr:.2f} & {avg_ssim:.3f} & {avg_lpips:.3f} & {avg_dop_psnr:.2f} & {avg_psnr_aolp:.2f}\n")