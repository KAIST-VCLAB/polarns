import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
import model.deeprep.deeprepnet as deeprep_nets

def warp(frame,flow):
    H = frame.shape[-2]
    W = frame.shape[-1]
    grid_r,grid_c = torch.meshgrid([torch.arange(H),torch.arange(W)])
    grid = torch.stack([grid_c,grid_r],dim=2).float()
    grid = grid[None].to(flow.device)+torch.permute(flow,[0,2,3,1]) # N 2 H W -> N H W 2

    grid = grid+0.5
    grid[:,:,:,0] = grid[:,:,:,0] * 2 / W - 1
    grid[:,:,:,1] = grid[:,:,:,1] * 2 / H - 1

    return F.grid_sample(frame, grid, mode='bilinear', padding_mode='zeros', align_corners = False)

def flow_rescale(flow,size,func_resize):
    flow_size = flow.shape[-1]
    flow_resized = func_resize(flow)
    return flow_resized*size/flow_size



def simple_polar_demosaic(img):
    return torch.concat([torch.mean(img[:,0:4,:,:],1,True),torch.mean(img[:,4:12,:,:],1,True),torch.mean(img[:,12:16,:,:],1,True)],1)

def simple_demosaic(img):
    return torch.concat([img[:,0,:,:][:,None],torch.mean(img[:,1:3,:,:],1,True),img[:,3,:,:][:,None]],1)

def demosaic(img,pattern='rggb'):
    assert len(pattern) == 4, "The length of Bayer pattern should be 4"
    device = img.device
    pattern_dict = {'r': 0, 'g': 1, 'b': 2}
    patternlist = [pattern_dict[c] for c in pattern]
    H = img.shape[-2]
    W = img.shape[-1]
    output = torch.zeros([img.shape[0],3,2*H,2*W],device=device)
    output_denom = torch.zeros([img.shape[0],3,2*H,2*W],device=device)
    output[:,patternlist[0],0::2,0::2] = img[:,0,:,:]
    output[:,patternlist[1],0::2,1::2] = img[:,1,:,:]
    output[:,patternlist[2],1::2,0::2] = img[:,2,:,:]
    output[:,patternlist[3],1::2,1::2] = img[:,3,:,:]
    output_denom[:,patternlist[0],0::2,0::2] = 1
    output_denom[:,patternlist[1],0::2,1::2] = 1
    output_denom[:,patternlist[2],1::2,0::2] = 1
    output_denom[:,patternlist[3],1::2,1::2] = 1
    filter_rb = torch.tensor([[0.5,1,0.5]],device=device)
    filter_rb = filter_rb * filter_rb.transpose(0,1)
    filter_g = torch.tensor([[0,0.5,0],[0.5,1,0.5],[0,0.5,0]],device=device)
    output[:,0,:,:] = F.conv2d(output[:,0,:,:][:,None],filter_rb[None,None],padding='same').squeeze()
    output_denom[:,0,:,:] = F.conv2d(output_denom[:,0,:,:][:,None],filter_rb[None,None],padding='same').squeeze()
    output[:,1,:,:] = F.conv2d(output[:,1,:,:][:,None],filter_g[None,None],padding='same').squeeze()
    output_denom[:,1,:,:] = F.conv2d(output_denom[:,1,:,:][:,None],filter_g[None,None],padding='same').squeeze()
    output[:,2,:,:] = F.conv2d(output[:,2,:,:][:,None],filter_rb[None,None],padding='same').squeeze()
    output_denom[:,2,:,:] = F.conv2d(output_denom[:,2,:,:][:,None],filter_rb[None,None],padding='same').squeeze()
    return output/output_denom

def polar_demosaic(img,pattern='rggb'):
    return demosaic(torch.concat([torch.mean(img[:,0:4,:,:],1,True),torch.mean(img[:,4:8,:,:],1,True),torch.mean(img[:,8:12,:,:],1,True),torch.mean(img[:,12:16,:,:],1,True)],1),pattern=pattern)


class RGBDemosaic(nn.Module):
    def __init__(self,img_size=[14,96,96]):
        super().__init__()
        self.resize = transforms.Resize(img_size[2]*4)


    def forward(self, frames_batch):
        frames_batch=frames_batch[:,0]
        output = demosaic(frames_batch,device=frames_batch.device,pattern='rggb')
        output = self.resize(output)
        return output


def no_norm_layer(dim):
    return (lambda x : x)

    
class MFIR_wrap(nn.Module):
    def __init__(self):
        super().__init__()
        self.mfir = deeprep_nets.deeprep_sr_iccv21(num_iter=3, enc_dim=64, enc_num_res_blocks=5, enc_out_dim=256,
                                         dec_dim_pre=64, dec_dim_post=32, dec_num_pre_res_blocks=5,
                                         dec_num_post_res_blocks=5,
                                         dec_in_dim=64, dec_upsample_factor=4, gauss_blur_sd=1,
                                         feature_degradation_upsample_factor=2, use_feature_regularization=False,
                                         wp_ref_offset_noise=0.00)


    
    def forward(self, frames_batch):
        x,_ = self.mfir(frames_batch)
        return x

if __name__ == '__main__':
    import cv2
    img_idx = 0
    for i in range(2):
        warp_img = self.demosaic(frame_aligned[:,:,i])[img_idx].cpu()
        cv2.imshow(f'warp{i}', self.raft_resize(warp_img).permute([1, 2, 0]).numpy() ** (1.0 / 2.2))
    for i in range(2):
        warp_img = self.demosaic(frames[:,i])[img_idx].cpu()
        cv2.imshow(f'warp_before{i}', self.raft_resize(warp_img).permute([1, 2, 0]).numpy() ** (1.0 / 2.2))
    cv2.waitKey()

    warp_img = demosaic(frame_warped, device=frame.device, pattern='rggb')[img_idx].cpu()
    cv2.imshow(f'warp_before', self.raft_resize(warp_img).permute([1, 2, 0]).numpy() ** (1.0 / 2.2))
    cv2.waitKey()