import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from model.utils.spatial_color_alignment import get_gaussian_kernel, match_colors
from model.utils.warp import warp
from model.model import polar_demosaic

class Loss(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, result, gt):
        return F.mse_loss(result, gt)


def new_loss(pred,gt):
    pass

def diff_angles(angle1,angle2):
    # [-pi,pi]
    return (angle1 - angle2 + torch.pi) % (2 * torch.pi) - torch.pi

def diff_aolp(aolp1,aolp2):
    return diff_angles(aolp1*2,aolp2*2)/2

def batch_unpack_9ch(img_9ch):
    img_9ch = torch.reshape(img_9ch, [img_9ch.shape[0], 3, 3, img_9ch.shape[-2], img_9ch.shape[-1]])
    return img_9ch[:, 0],img_9ch[:, 1],img_9ch[:, 2]
def batch_unpack_12ch(img):
    img = torch.reshape(img, [img.shape[0], 4, 3, img.shape[-2], img.shape[-1]])
    return img[:, 0],img[:, 1],img[:, 2],img[:, 3]

def stokes2properties(img_9ch):
    # B c H W
    s0,s1,s2 = batch_unpack_9ch(img_9ch)  # B Stokes C H W

    s0[s0<=0] = torch.finfo(torch.float32).eps

    dop = torch.sqrt(s1 ** 2 + s2 ** 2) / (s0)
    aolp = torch.atan2(s2, s1) / 2
    dop[dop < 0] = 0
    dop[dop > 1] = 1
    return torch.cat([s0,dop,aolp],dim=1)

def properties2stokes(img_properties):
    # B c H W
    s0, dop, aolp = batch_unpack_9ch(img_properties)  # B Stokes C H W

    s1 = s0 * dop * torch.cos(2*aolp)
    s2 = s0 * dop * torch.sin(2*aolp)
    return torch.cat([s0,s1,s2],dim=1)

def degree2stokes(img_degree):
    # B c H W
    img_degree = torch.reshape(img_degree, [img_degree.shape[0], 4, 3, img_degree.shape[-2], img_degree.shape[-1]]) # B Angles C H W
    s0 = torch.sum(img_degree, dim=1) / 2
    s1 = img_degree[:, 0] - img_degree[:, 2]
    s2 = img_degree[:, 1] - img_degree[:, 3]
    return torch.cat([s0,s1,s2],dim=1)

def stokes2degree(img_stokes):
    # B c H W
    s0,s1,s2 = batch_unpack_9ch(img_stokes)  # B Stokes C H W
    img0 = (s0 + s1)/2
    img45 = (s0 + s2)/2
    img90 = (s0 - s1)/2
    img135 = (s0 - s2)/2
    return torch.cat([img0,img45,img90,img135],dim=1)

def mse_for_properties_from_9ch(pred,gt):
    # return mse_s0, mse_dop, bias_dop, mse_aolp
    s0, dop, aolp = batch_unpack_9ch(stokes2properties(pred))
    s0_gt, dop_gt, aolp_gt = batch_unpack_9ch(stokes2properties(gt))
    # return F.mse_loss(s0,s0_gt),F.mse_loss(dop,dop_gt),torch.mean(dop-dop_gt,dim=None),torch.mean(diff_aolp(aolp,aolp_gt)**2,dim=None)
    return F.mse_loss(s0,s0_gt),F.mse_loss(dop,dop_gt),torch.mean(dop-dop_gt),torch.mean(diff_aolp(aolp,aolp_gt)**2)

def mse_for_properties_from_stokes_no_reduction(pred,gt):
    # return mse_s0, mse_dop, bias_dop, mse_aolp
    s0, dop, aolp = batch_unpack_9ch(stokes2properties(pred))
    s0_gt, dop_gt, aolp_gt = batch_unpack_9ch(stokes2properties(gt))
    # return F.mse_loss(s0,s0_gt),F.mse_loss(dop,dop_gt),torch.mean(dop-dop_gt,dim=None),torch.mean(diff_aolp(aolp,aolp_gt)**2,dim=None)
    return F.mse_loss(s0,s0_gt,reduction='none'),F.mse_loss(dop,dop_gt,reduction='none'),dop-dop_gt,diff_aolp(aolp,aolp_gt)**2

def mse_for_properties(pred,gt):
    # return mse_s0, mse_dop, bias_dop, mse_aolp
    s0, dop, aolp = batch_unpack_9ch(pred)
    s0_gt, dop_gt, aolp_gt = batch_unpack_9ch(gt)
    # return F.mse_loss(s0,s0_gt),F.mse_loss(dop,dop_gt),torch.mean(dop-dop_gt,dim=None),torch.mean(diff_aolp(aolp,aolp_gt)**2,dim=None)
    return F.mse_loss(s0,s0_gt),F.mse_loss(dop,dop_gt),torch.mean(dop-dop_gt),torch.mean(diff_aolp(aolp,aolp_gt)**2)

def adaptive_l1(pred,gt):
    s0, s1, s2 = batch_unpack_9ch(pred)
    s0_gt, s1_gt, s2_gt = batch_unpack_9ch(gt)
    s0_std = torch.std(s0)
    s1_std = torch.std(s1)
    s2_std = torch.std(s2)
    return F.l1_loss(s0,s0_gt) + (s0_std/s1_std)* F.l1_loss(s1,s1_gt) + (s0_std/s2_std)* F.l1_loss(s2,s2_gt)

def l1_and_mse(pred,gt):
    # l1 for s0, mse for s1 and s2
    pred = torch.reshape(pred, [pred.shape[0], 3, 3, pred.shape[-2], pred.shape[-1]])  # B Stokes C H W
    s0 = pred[:, 0]
    spol = pred[:, 1:]
    gt = torch.reshape(gt, [gt.shape[0], 3, 3, gt.shape[-2], gt.shape[-1]])  # B Stokes C H W
    s0_gt = gt[:, 0]
    spol_gt = gt[:, 1:]
    return F.l1_loss(s0,s0_gt)+F.mse_loss(spol,spol_gt)

def l1_and_mad(pred,gt):
    # l1 for s0, mad for s1 and s2
    s0,s1,s2 = batch_unpack_9ch(pred)
    s0_gt,s1_gt,s2_gt = batch_unpack_9ch(gt)
    s1_diff_sq = (s1-s1_gt)**2
    s2_diff_sq = (s2-s2_gt)**2
    mad = torch.mean(torch.sqrt(s1_diff_sq+s2_diff_sq))
    return F.l1_loss(s0,s0_gt)+mad

def mse_in_properties(pred,gt):
    s0, dop, aolp = batch_unpack_9ch(pred)
    s0_gt, dop_gt, aolp_gt = batch_unpack_9ch(gt)
    # return F.mse_loss(s0,s0_gt)+F.mse_loss(dop,dop_gt)+torch.mean(diff_aolp(aolp,aolp_gt)**2,dim=None)
    return F.mse_loss(s0,s0_gt)+F.mse_loss(dop,dop_gt)+torch.mean(diff_aolp(aolp,aolp_gt)**2)
def l1_in_properties(pred,gt):
    s0, dop, aolp = batch_unpack_9ch(pred)
    s0_gt, dop_gt, aolp_gt = batch_unpack_9ch(gt)
    # return F.l1_loss(s0,s0_gt)+F.l1_loss(dop,dop_gt)+torch.mean(torch.abs(diff_aolp(aolp,aolp_gt)),dim=None)
    return F.l1_loss(s0,s0_gt)+F.l1_loss(dop,dop_gt)+torch.mean(torch.abs(diff_aolp(aolp,aolp_gt)))

class AlignedL1(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None):
        super().__init__()
        self.sr_factor = sr_factor
        self.boundary_ignore = boundary_ignore
        self.alignment_net = alignment_net

        self.gauss_kernel, self.ksz = get_gaussian_kernel(sd=1.5)

    def forward(self, pred, gt, burst_input):
        # Estimate flow between the prediction and the ground truth

        gt = gt.float()
        pred_rgb = degree2stokes(pred.float())[:,0:3]
        gt_rgb = degree2stokes(gt.float())[:,0:3]

        with torch.no_grad():
            flow = self.alignment_net(pred_rgb / (pred_rgb.max() + 1e-6), gt_rgb / (gt_rgb.max() + 1e-6))
            # flow = self.alignment_net((pred_rgb / (pred_rgb.max() + 1e-6)).float(), (gt_rgb / (gt_rgb.max() + 1e-6)).float())

        # Warp the prediction to the ground truth coordinates
        pred_warped = warp(pred, flow)

        # Warp the base input frame to the ground truth. This will be used to estimate the color transformation between
        # the input and the ground truth
        sr_factor = self.sr_factor
        ds_factor = 1.0 / float(2.0 * sr_factor)
        flow_ds = F.interpolate(flow, scale_factor=ds_factor, mode='bilinear', recompute_scale_factor=True, align_corners=False) * ds_factor

        burst_0 = (burst_input[:, 0, 0:3]+burst_input[:, 0, 3:6]+burst_input[:, 0, 6:9]+burst_input[:, 0, 9:12]).contiguous()/2
        burst_0_warped = warp(burst_0, flow_ds)
        frame_gt_ds = F.interpolate(gt_rgb, scale_factor=ds_factor, mode='bilinear', recompute_scale_factor=True, align_corners=False)

        # Match the colorspace between the prediction and ground truth
        pred_warped_m, valid = match_colors(frame_gt_ds, burst_0_warped, pred_warped, self.ksz,
                                                      self.gauss_kernel)

        # Ignore boundary pixels if specified
        if self.boundary_ignore is not None:
            pred_warped_m = pred_warped_m[..., self.boundary_ignore:-self.boundary_ignore,
                            self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        pred_warped_m = pred_warped_m.contiguous()
        gt = gt.contiguous()
        # Estimate MSE
        l1 = F.l1_loss(pred_warped_m, gt, reduction='none')

        eps = 1e-12
        elem_ratio = l1.numel() / valid.numel()
        l1 = (l1 * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)

        return l1

class AlignedL2(nn.Module):
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None):
        super().__init__()
        self.sr_factor = sr_factor
        self.boundary_ignore = boundary_ignore
        self.alignment_net = alignment_net
        # self.loss_fn = lpips.LPIPS(net='alex').cuda()

        self.gauss_kernel, self.ksz = get_gaussian_kernel(sd=1.5)

    def forward(self, pred, gt, burst_input):
        # Estimate flow between the prediction and the ground truth
        
        gt = gt.float()
        pred_stokes = degree2stokes(pred.float())
        gt_stokes = degree2stokes(gt.float())
        pred_rgb = pred_stokes[:,0:3]
        gt_rgb = gt_stokes[:,0:3]

        with torch.no_grad():
            # flow = self.alignment_net(pred_rgb / (pred_rgb.max() + 1e-6), gt_rgb / (gt_rgb.max() + 1e-6))
            flow = self.alignment_net((pred_rgb / (pred_rgb.max() + 1e-6)).float(), (gt_rgb / (gt_rgb.max() + 1e-6)).float())

        # Warp the prediction to the ground truth coordinates
        pred_warped = warp(pred, flow)

        # Warp the base input frame to the ground truth. This will be used to estimate the color transformation between
        # the input and the ground truth
        sr_factor = self.sr_factor
        ds_factor = 1.0 / float(2.0 * sr_factor)
        flow_ds = F.interpolate(flow, scale_factor=ds_factor, mode='bilinear', recompute_scale_factor=True, align_corners=False) * ds_factor

        burst_0 = (burst_input[:, 0, 0:3]+burst_input[:, 0, 3:6]+burst_input[:, 0, 6:9]+burst_input[:, 0, 9:12]).contiguous()/2
        burst_0_warped = warp(burst_0, flow_ds)
        frame_gt_ds = F.interpolate(gt_rgb, scale_factor=ds_factor, mode='bilinear', recompute_scale_factor=True, align_corners=False)

        # Match the colorspace between the prediction and ground truth
        pred_warped_m, valid = match_colors(frame_gt_ds, burst_0_warped, pred_warped, self.ksz,
                                                      self.gauss_kernel)

        # Ignore boundary pixels if specified
        if self.boundary_ignore is not None:
            pred_warped_m = pred_warped_m[..., self.boundary_ignore:-self.boundary_ignore,
                            self.boundary_ignore:-self.boundary_ignore]
            gt_stokes = gt_stokes[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        # Estimate MSE
        # mse = F.mse_loss(pred_warped_m.contiguous(), gt.contiguous(), reduction='none')

        pred_warped_m_stokes = degree2stokes(pred_warped_m)
        mse_s0,mse_dop,bias_dop,mse_aolp = mse_for_properties_from_stokes_no_reduction(pred_warped_m_stokes,gt_stokes)

        eps = 1e-12
        elem_ratio = mse_s0.numel() / valid.numel()
        mse_s0 = (mse_s0 * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)
        mse_dop = (mse_dop * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)
        bias_dop = (bias_dop * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)
        mse_aolp = (mse_aolp * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)

        #ss = ssim(pred_warped_m.contiguous(), gt.contiguous(), data_range=1.0, size_average=True)

        #lp = self.loss_fn(pred_warped_m.contiguous(), gt.contiguous()).squeeze()

        return mse_s0,mse_dop,bias_dop,mse_aolp#, ss, lp
    