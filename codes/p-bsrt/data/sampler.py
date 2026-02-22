import torch
import torch.nn.functional as F
import random
import numpy as np
import cv2
import math
import data.camera_pipeline as rgb2raw
from torchvision import transforms

def random_crop_tensor(image, crop_size):
    shape = image.shape
    h = shape[1]
    w = shape[2]
    if h<crop_size or w<crop_size:
        raise Exception('The image is smaller than the crop size.')
    offset_x = random.randint(0, w-crop_size)
    offset_y = random.randint(0, h-crop_size)
    return image[:,offset_y:offset_y+crop_size,offset_x:offset_x+crop_size]

def random_crop(image, crop_size):
    shape = image.shape
    h = shape[0]
    w = shape[1]
    if h<crop_size or w<crop_size:
        raise Exception('The image is smaller than the crop size.')
    offset_x = random.randint(0, w-crop_size)
    offset_y = random.randint(0, h-crop_size)
    return image[offset_y:offset_y+crop_size,offset_x:offset_x+crop_size,:]

def warp_tensor(img,info,output_size,downsample_factor,batch_size,scale=1,data_generation=True):
    shape = img.shape
    M_list=[]
    # cv2.imshow('original', img[0:3, :, :].squeeze().permute([1, 2, 0]).numpy())
    h = shape[-2]
    w = shape[-1]
    for i in range(batch_size):
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), info[i]['rot_deg'], 1.0)
        M[0, 2] = info[i]['trans_x']*2/w
        M[1, 2] = info[i]['trans_y']*2/h
        M_t = torch.tensor(M*downsample_factor, dtype=torch.float32)
        if data_generation:
            M_t = torch.concat([M_t, torch.tensor([[0, 0, 1]])])
        M_list.append(M_t)

    M_t = torch.stack(M_list,dim=0)
    if len(shape)==3:
        img = img[None].repeat([batch_size,1,1,1])

    if data_generation:
        grid = F.affine_grid(torch.inverse(M_t)[:,:2, :].to(img.device),
                                 [batch_size, shape[0], h, w])
    else:
        M_t[:,:,:2] = M_t[:,:,:2]/(downsample_factor*scale)
        grid = F.affine_grid(M_t.to(img.device),
                                 [batch_size, shape[0], h, w])

    warped = center_crop_tensor(F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros',align_corners = False),output_size)

    if warped.shape[1]==9: #polar rotation
        angles = [np.deg2rad(info[i]['rot_deg']) for i in range(batch_size)]
        angles = np.stack(angles,axis=0)
        angles = torch.from_numpy(angles).to(img.device).reshape([batch_size,1,1,1])
        s1 = warped[:,3:6].clone()
        s2 = warped[:,6:].clone()
        cos_angle = torch.cos(angles*2)
        sin_angle = torch.sin(angles*2)
        warped[:, 3:6] = cos_angle * s1 - sin_angle * s2
        warped[:,6:] = sin_angle * s1 + cos_angle * s2

    return warped

"""
rotation debug code

import cv2
import numpy as np
import torch
x,y = np.meshgrid(np.linspace(-1,1,1001),np.linspace(-1,1,1001))
s1 = x*x - (-y)*(-y)
s2 = 2*x*(-y)
img = np.stack([s1,s2,np.zeros_like(s1)],axis=2)
warped_img = np.zeros_like(img)
warped_img[:,:,0] = np.cos(np.deg2rad(30)) * s1 - np.sin(np.deg2rad(30)) * s2
warped_img[:,:,1] = np.sin(np.deg2rad(30)) * s1 + np.cos(np.deg2rad(30)) * s2
rot_img = cv2.warpAffine(warped_img,cv2.getRotationMatrix2D((500, 500), 15, 1.0),(1001,1001))
cv2.imshow('rot_img',rot_img*0.5+0.5)
cv2.imshow('warped_img',warped_img*0.5+0.5)
cv2.imshow('img',img*0.5+0.5)
cv2.imshow('diff',(rot_img-img)*0.5+0.5)
cv2.waitKey()


"""

def random_warp_tensor(img,max_rotation,max_translation,output_size,downsample_factor,gen_size,info_return=False):
    info=[]
    for i in range(gen_size):
        rot_deg = random.uniform(-max_rotation, max_rotation)
        trans_x,trans_y = random.uniform(-max_translation, max_translation),random.uniform(-max_translation, max_translation)

        info.append({'rot_deg': rot_deg, 'trans_x': trans_x, 'trans_y': trans_y})

    warped = warp_tensor(img,info,output_size,downsample_factor,gen_size,data_generation=True)

    if info_return:
        return warped,info
    return warped

def random_warp(img, max_rotation, max_translation, output_size, scale, info_return=False):
    shape = img.shape
    # cv2.imshow('original', img[0:3, :, :].squeeze().permute([1, 2, 0]).numpy())
    h = shape[0]
    w = shape[1]
    rot_deg = random.uniform(-max_rotation, max_rotation)
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rot_deg, 1.0)
    trans_x,trans_y = random.uniform(-max_translation, max_translation),random.uniform(-max_translation, max_translation)
    M[0, 2] += trans_x
    M[1, 2] += trans_y

    img = cv2.warpAffine(img, M*scale, (int(w * scale), int(h * scale)))
    if info_return:
        return center_crop(img, output_size),{'rot_deg':rot_deg,'trans_x':trans_x,'trans_y':trans_y}
    return center_crop(img, output_size)

def center_crop_tensor(image, crop_size):
    shape = image.shape
    if len(shape) ==3:
        image=image[None]
    h = shape[-2]
    w = shape[-1]
    if h<crop_size or w<crop_size:
        raise Exception('The image is smaller than the crop size.')
    offset_x = (w-crop_size)//2
    offset_y = (h-crop_size)//2
    image = image[:,:,offset_y:offset_y+crop_size,offset_x:offset_x+crop_size]

    if len(shape) ==3:
        image=image.squeeze()
    return image

def center_crop(image, crop_size):
    shape = image.shape
    h = shape[0]
    w = shape[1]
    if h<crop_size or w<crop_size:
        raise Exception('The image is smaller than the crop size.')
    offset_x = (w-crop_size)//2
    offset_y = (h-crop_size)//2
    return image[offset_y:offset_y+crop_size,offset_x:offset_x+crop_size,:]

def cut_boundary(image,bd_pxls):
    shape = image.shape
    h = shape[-2]
    w = shape[-1]
    assert h>bd_pxls*2 and w>bd_pxls*2, 'Image size should be larger than boundaries.'
    image = torch.reshape(image,[torch.prod(torch.tensor(shape[:-2])),shape[-2],shape[-1]])
    image = image[:, bd_pxls:h - bd_pxls, bd_pxls:w - bd_pxls]
    new_shape = list(shape[:-2])
    new_shape.extend([h - bd_pxls * 2, w - bd_pxls * 2])
    image = torch.reshape(image,new_shape)
    return image


def random_noise(image, shot_noise=0.01, read_noise=0.0005):
    variance = image * shot_noise + read_noise
    noise = torch.normal(mean=0, std=variance.sqrt())
    return image + noise

def random_noise_levels():
    """Generates random noise levels from a log-log linear distribution."""
    log_min_shot_noise = math.log(0.0001)
    log_max_shot_noise = math.log(0.012)
    log_shot_noise = random.uniform(log_min_shot_noise, log_max_shot_noise)
    shot_noise = math.exp(log_shot_noise)

    line = lambda x: 2.18 * x + 1.20
    log_read_noise = line(log_shot_noise) + random.gauss(mu=0.0, sigma=0.26)
    read_noise = math.exp(log_read_noise)
    return shot_noise, read_noise

def mosaic_rgb(image,pattern='rggb'):
    shape = image.shape
    if image.dim() == 3:
        image = image.unsqueeze(0)
    assert len(pattern)==4, "The length of Bayer pattern should be 4"
    pattern_dict = {'r':0,'g':1,'b':2}
    patternlist = [pattern_dict[c] for c in pattern]
    stacked_images = [
        image[:, patternlist[0], 0::2, 0::2],
        image[:, patternlist[1], 0::2, 1::2],
        image[:, patternlist[2], 1::2, 0::2],
        image[:, patternlist[3], 1::2, 1::2],
    ]
    image = torch.stack(stacked_images, dim=1)

    if len(shape) == 3:
        return image.view((4, shape[-2] // 2, shape[-1] // 2))
    else:
        return image.view((-1, 4, shape[-2] // 2, shape[-1] // 2))

def mosaic_polar(image):
    shape = image.shape
    if image.dim() == 3:
        image = image.unsqueeze(0)
    stacked_images = [
        image[:, 0, 1::4, 1::4], # red 0
        image[:, 3, 0::4, 1::4], # red 45
        image[:, 6, 0::4, 0::4], # red 90
        image[:, 9, 1::4, 0::4], # red 135
        image[:, 1, 1::4, 3::4], # green 0
        image[:, 4, 0::4, 3::4], # green 45
        image[:, 7, 0::4, 2::4], # green 90
        image[:, 10, 1::4, 2::4], # green 135
        image[:, 1, 3::4, 1::4], # green 0
        image[:, 4, 2::4, 1::4], # green 45
        image[:, 7, 2::4, 0::4], # green 90
        image[:, 10, 3::4, 0::4], # green 135
        image[:, 2, 3::4, 3::4], # blue 0
        image[:, 5, 2::4, 3::4], # blue 45
        image[:, 8, 2::4, 2::4], # blue 90
        image[:, 11, 3::4, 2::4], # blue 135
    ]
    image = torch.stack(stacked_images, dim=1)

    if len(shape) == 3:
        return image.view((16, shape[-2] // 4, shape[-1] // 4))
    else:
        return image.view((-1, 16, shape[-2] // 4, shape[-1] // 4))


def mosaic_polar_raw(image):
    shape = image.shape
    if image.dim() == 3:
        image = image.unsqueeze(0)


    new_images = torch.zeros(shape[0],1,shape[-2],shape[-1])
    new_images[:, 0, 1::4, 1::4] = image[:, 0, 1::4, 1::4] # red 0
    new_images[:, 0, 0::4, 1::4] = image[:, 3, 0::4, 1::4] # red 45
    new_images[:, 0, 0::4, 0::4] = image[:, 6, 0::4, 0::4] # red 90
    new_images[:, 0, 1::4, 0::4] = image[:, 9, 1::4, 0::4] # red 135
    new_images[:, 0, 1::4, 3::4] = image[:, 1, 1::4, 3::4] # green 0
    new_images[:, 0, 0::4, 3::4] = image[:, 4, 0::4, 3::4] # green 45
    new_images[:, 0, 0::4, 2::4] = image[:, 7, 0::4, 2::4] # green 90
    new_images[:, 0, 1::4, 2::4] = image[:, 10, 1::4, 2::4] # green 135
    new_images[:, 0, 3::4, 1::4] = image[:, 1, 3::4, 1::4] # green 0
    new_images[:, 0, 2::4, 1::4] = image[:, 4, 2::4, 1::4] # green 45
    new_images[:, 0, 2::4, 0::4] = image[:, 7, 2::4, 0::4] # green 90
    new_images[:, 0, 3::4, 0::4] = image[:, 10, 3::4, 0::4] # green 135
    new_images[:, 0, 3::4, 3::4] = image[:, 2, 3::4, 3::4] # blue 0
    new_images[:, 0, 2::4, 3::4] = image[:, 5, 2::4, 3::4] # blue 45
    new_images[:, 0, 2::4, 2::4] = image[:, 8, 2::4, 2::4] # blue 90
    new_images[:, 0, 3::4, 2::4] = image[:, 11, 3::4, 2::4] # blue 135

    if len(shape) == 3:
        return new_images.view((1, shape[-2], shape[-1]))
    else:
        return new_images.view((-1, 1, shape[-2], shape[-1]))

def cv_image2tensor(img):
    return torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA if img.shape[2]==4 else cv2.COLOR_BGR2RGB)).permute([2, 0, 1])

def cv_9ch_image2tensor(img):
    return torch.cat([torch.from_numpy(cv2.cvtColor(img[:,:,3*i:3*i+3], cv2.COLOR_BGR2RGB)).permute([2, 0, 1]) for i in range(3)],dim=0)

def tensor2cv_image(img):
    return cv2.cvtColor(img.permute([1,2,0]).numpy(), cv2.COLOR_RGBA2BGRA if img.shape[0]==4 else cv2.COLOR_RGB2BGR)


def stokes2angles(img):
    shape = img.shape
    if len(shape) == 3:
        img = img[None]
    img = torch.cat([img[:, 0:3, :, :]/2 + img[:, 3:6, :, :]/2,
                      img[:, 0:3, :, :]/2 + img[:, 6:9, :, :]/2,
                      img[:, 0:3, :, :]/2 - img[:, 3:6, :, :]/2,
                      img[:, 0:3, :, :]/2 - img[:, 6:9, :, :]/2], dim=1)

    if len(shape) == 3:
        img = img.squeeze()
    return img

class RandomBurstImage(torch.utils.data.Dataset):
    def __init__(self, dataset, num_samples, info_bursts, device='cpu', random_indexing=True, save_warp_info=False):
        self.dataset = dataset
        self.num_samples = num_samples
        assert random_indexing==True or num_samples==len(dataset), 'In the sequential indexing, the number of the samples should be the same with that of the original dataset '
        self.crop_size = info_bursts['crop_size']
        self.max_rotation = info_bursts['max_rotation']
        self.max_translation = info_bursts['max_translation']
        self.downsample_factor = info_bursts['downsample_factor']
        self.burst_size = info_bursts['burst_size']
        self.random_flip = info_bursts['random_flip']
        self.device = device
        self.random_indexing = random_indexing
        if self.max_rotation < 45:
            self.crop_range = np.sqrt(2) * self.crop_size * np.sin(np.deg2rad(self.max_rotation+45))
        else:
            self.crop_range = np.sqrt(2) * self.crop_size
        self.crop_range = np.ceil(self.crop_range + 2*self.max_translation).astype(np.int32)
        self.save_warp_info = save_warp_info

        if self.crop_size % self.downsample_factor != 0:
            raise Exception('The crop size is not a multiple of the downsample factor.')

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if self.random_indexing:
            im_id = random.randint(0, len(self.dataset) - 1)
        else:
            im_id = index
        images = self.dataset[im_id]

        s0 = images['s0']
        s1 = images['s1']
        s2 = images['s2']



        # gt = random_crop(torch.cat([s0,s1,s2],dim=0),self.crop_range)

        gt = random_crop(np.concatenate([s0, s1, s2], axis=2), self.crop_range)

        output_size = self.crop_size//self.downsample_factor
        if not self.save_warp_info:
            burst_images = [cv_9ch_image2tensor(random_warp(gt,self.max_rotation,self.max_translation,output_size,1.0/self.downsample_factor)) for i in range(self.burst_size-1)]
        else:
            burst_images = []
            warp_info = []
            for i in range(self.burst_size-1):
                img,info = random_warp(gt, self.max_rotation, self.max_translation, output_size, 1.0 / self.downsample_factor,info_return=True)
                burst_images.append(cv_9ch_image2tensor(img))
                warp_info.append(info)


        gt = center_crop(gt,self.crop_size) # 9 channel gt
        burst_images.insert(0, cv_9ch_image2tensor(cv2.resize(gt,[output_size,output_size])).to(device=self.device))
        gt = cv_9ch_image2tensor(gt).to(device=self.device)
        gt = stokes2angles(gt)
        gt[gt < 0] = 0
        gt[gt > 1] = 1

        burst_angle_images = [stokes2angles(img.to(device=self.device)) for img in burst_images]

        burst_angle_images = torch.stack(burst_angle_images,dim=0)
        # import os
        # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        # # import matplotlib
        # # matplotlib.use('TkAgg')
        # import matplotlib.pyplot as plt
        # plt.imshow(burst_angle_images[0,0])

        # burst_angle_images[burst_angle_images<0] = 0
        frames = mosaic_polar(burst_angle_images)
        frames[frames < 0] = 0
        shot_noise, read_noise = random_noise_levels()
        frames = random_noise(frames,shot_noise, read_noise)
        frames[frames < 0] = 0
        frames[frames > 1] = 1
        ret_val = {'frames': frames, 'gt': gt}
        if self.save_warp_info:
            ret_val['warp_info'] = warp_info
        return ret_val

class RandomBurstImageTensor(torch.utils.data.Dataset):
    def __init__(self, dataset, num_samples, info_bursts, device='cpu', random_indexing=True, save_warp_info=False):
        self.dataset = dataset
        self.num_samples = num_samples
        assert random_indexing==True or num_samples==len(dataset), 'In the sequential indexing, the number of the samples should be the same with that of the original dataset '
        self.crop_size = info_bursts['crop_size']
        self.max_rotation = info_bursts['max_rotation']
        self.max_translation = info_bursts['max_translation']
        self.downsample_factor = info_bursts['downsample_factor']
        self.burst_size = info_bursts['burst_size']
        self.random_flip = info_bursts['random_flip']
        self.output_size = self.crop_size//self.downsample_factor
        self.device = device
        self.random_indexing = random_indexing
        if self.max_rotation < 45:
            self.crop_range = np.sqrt(2) * self.crop_size * np.sin(np.deg2rad(self.max_rotation+45))
        else:
            self.crop_range = np.sqrt(2) * self.crop_size
        self.crop_range = np.ceil(self.crop_range + 2*self.max_translation).astype(np.int32)
        self.save_warp_info = save_warp_info
        self.resize_gt = transforms.Resize(self.output_size)

        if self.crop_size % self.downsample_factor != 0:
            raise Exception('The crop size is not a multiple of the downsample factor.')

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if self.random_indexing:
            im_id = random.randint(0, len(self.dataset) - 1)
        else:
            im_id = index
        images = self.dataset[im_id]

        # print("s0_device:",self.device)
        s0 = images['s0'].to(self.device)
        s1 = images['s1'].to(self.device)
        s2 = images['s2'].to(self.device)

        gt = random_crop_tensor(torch.cat([s0,s1,s2],dim=0),self.crop_range)

        if self.random_flip and random.randint(0,1):
            gt = torch.flip(gt,[-1])

        if not self.save_warp_info:
            burst_images = random_warp_tensor(gt,self.max_rotation,self.max_translation,self.output_size,1.0/self.downsample_factor,self.burst_size-1)
        else:
            burst_images,warp_info = random_warp_tensor(gt, self.max_rotation, self.max_translation, self.output_size, 1.0 / self.downsample_factor,self.burst_size-1,info_return=True)

        gt = center_crop_tensor(gt,self.crop_size) # 9 channel gt
        burst_images = torch.concat([self.resize_gt(gt[None]),burst_images],dim=0)
        # gt = stokes2angles(gt)
        # gt[gt < 0] = 0
        # gt[gt > 1] = 1
        # gt[3:] -=0.5

        burst_angle_images = stokes2angles(burst_images)

        # import os
        # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        # # import matplotlib
        # # matplotlib.use('TkAgg')
        # import matplotlib.pyplot as plt
        # plt.imshow(burst_angle_images[0,0])

        # burst_angle_images[burst_angle_images<0] = 0
        frames = mosaic_polar(burst_angle_images)
        frames[frames < 0] = 0
        shot_noise, read_noise = random_noise_levels()
        frames = random_noise(frames,shot_noise, read_noise)
        frames[frames < 0] = 0
        frames[frames > 1] = 1
        ret_val = {'frames': frames, 'gt': gt}
        if self.save_warp_info:
            ret_val['warp_info'] = warp_info
        return ret_val


class RandomRGBBurstImage(RandomBurstImage):
    def __getitem__(self, index):
        if self.random_indexing:
            im_id = random.randint(0, len(self.dataset) - 1)
        else:
            im_id = index

        img = self.dataset[im_id]
        img = cv_image2tensor(img).to(device=self.device)

        # unprocssing
        # Approximately inverts global tone mapping.
        img = rgb2raw.invert_smoothstep(img)

        # Inverts gamma compression.
        img = rgb2raw.gamma_expansion(img)

        # Inverts color correction.
        img = rgb2raw.apply_ccm(img, rgb2raw.random_ccm())

        # Approximately inverts white balance and brightening.
        rgb_gain, red_gain, blue_gain = rgb2raw.random_gains()
        img = rgb2raw.safe_invert_gains(img, rgb_gain, red_gain, blue_gain)

        # Clip saturated pixels.
        img = img.clamp(0.0, 1.0)
        img = tensor2cv_image(img.to('cpu'))

        # burst
        gt = random_crop(img, self.crop_range)

        output_size = self.crop_size//self.downsample_factor
        if not self.save_warp_info:
            burst_images = [cv_image2tensor(random_warp(gt,self.max_rotation,self.max_translation,output_size,1.0/self.downsample_factor)) for i in range(self.burst_size-1)]
        else:
            burst_images = []
            warp_info = []
            for i in range(self.burst_size-1):
                img,info = random_warp(gt, self.max_rotation, self.max_translation, output_size, 1.0 / self.downsample_factor,info_return=True)
                burst_images.append(cv_image2tensor(img))
                warp_info.append(info)

        gt = center_crop(gt,self.crop_size)
        burst_images.insert(0, cv_image2tensor(cv2.resize(gt,[output_size,output_size])))
        gt = cv_image2tensor(gt).to(device=self.device)
        gt[gt < 0] = 0
        gt[gt > 1] = 1
        gt = gt

        burst_images = torch.stack(burst_images,dim=0)

        frames = mosaic_rgb(burst_images,'rggb')
        frames[frames < 0] = 0
        shot_noise, read_noise = random_noise_levels()
        frames = frames.to(device=self.device)
        frames = random_noise(frames,shot_noise, read_noise)
        frames[frames < 0] = 0
        frames[frames > 1] = 1
        ret_val = {'frames':frames,'gt':gt}
        if self.save_warp_info:
            ret_val['warp_info'] = warp_info
        return ret_val

class RandomRGBBurstImageTensor(RandomBurstImageTensor):
    def __getitem__(self, index):
        if self.random_indexing:
            im_id = random.randint(0, len(self.dataset) - 1)
        else:
            im_id = index
        img = self.dataset[im_id]
        img = cv_image2tensor(img).to(device=self.device)

        # # unprocssing
        # # Approximately inverts global tone mapping.
        # img = rgb2raw.invert_smoothstep(img)
        #
        # # Inverts gamma compression.
        # img = rgb2raw.gamma_expansion(img)
        #
        # # Inverts color correction.
        # img = rgb2raw.apply_ccm(img, rgb2raw.random_ccm())
        #
        # # Approximately inverts white balance and brightening.
        # rgb_gain, red_gain, blue_gain = rgb2raw.random_gains()
        # img = rgb2raw.safe_invert_gains(img, rgb_gain, red_gain, blue_gain)

        # Clip saturated pixels.
        img = img.clamp(0.0, 1.0)

        gt = random_crop_tensor(img,self.crop_range) ** 2.2

        if self.random_flip and random.randint(0,1):
            gt = torch.flip(gt,[-1])

        if not self.save_warp_info:
            burst_images = random_warp_tensor(gt,self.max_rotation,self.max_translation,self.output_size,1.0/self.downsample_factor,self.burst_size-1)
        else:
            burst_images,warp_info = random_warp_tensor(gt, self.max_rotation, self.max_translation, self.output_size, 1.0 / self.downsample_factor,self.burst_size-1,info_return=True)

        gt = center_crop_tensor(gt,self.crop_size)
        burst_images = torch.concat([self.resize_gt(gt[None]),burst_images],dim=0)
        gt[gt < 0] = 0
        gt[gt > 1] = 1

        # burst_angle_images[burst_angle_images<0] = 0
        frames = mosaic_rgb(burst_images)
        frames[frames < 0] = 0
        shot_noise, read_noise = random_noise_levels()
        frames = random_noise(frames,shot_noise, read_noise)
        frames[frames < 0] = 0
        frames[frames > 1] = 1
        ret_val = {'frames': frames, 'gt': gt}
        if self.save_warp_info:
            ret_val['warp_info'] = warp_info
        return ret_val

class RandomIndexing(torch.utils.data.Dataset):
    def __init__(self, dataset, num_samples, device='cpu', random_indexing=True):
        self.dataset = dataset
        self.num_samples = num_samples
        assert random_indexing==True or num_samples==len(dataset), 'In the sequential indexing, the number of the samples should be the same with that of the original dataset '
        self.device = device
        self.random_indexing = random_indexing

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if self.random_indexing:
            im_id = random.randint(0, len(self.dataset) - 1)
        else:
            im_id = index
        return self.dataset[im_id]


if __name__ == "__main__":
    # from data.dataset import PolarImageDatasetCache
    # polar_dataset = PolarImageDatasetCache("G:/polar-normal/polar_image_dataset/rsp_dataset/rsp_dataset/train_val/gt/train")
    from data.dataset import RGBImageDataset
    polar_dataset = RGBImageDataset("G:/polar-normal/polar_image_dataset/zurich-rgb-full/train")

    info_bursts={}
    info_bursts['max_rotation']= 1
    info_bursts['max_translation'] = 24
    info_bursts['downsample_factor'] = 2
    info_bursts['burst_size'] = 14
    info_bursts['crop_size'] = 192
    polar_image_sampler = RandomRGBBurstImageTensor(polar_dataset,num_samples=len(polar_dataset),info_bursts=info_bursts,device='cpu',random_indexing=False)
    for i in range(5,len(polar_dataset)):
        a = polar_image_sampler[i]
        cv2.imshow('0', a['frames'][1, 0:3, :, :].to('cpu').squeeze().permute([1, 2, 0]).numpy())
        # cv2.imshow('90', a['frames'][1, 3:6, :, :].to('cpu').squeeze().permute([1, 2, 0]).numpy())
        # cv2.imshow('45', a['frames'][1, 6:9, :, :].to('cpu').squeeze().permute([1, 2, 0]).numpy())
        # cv2.imshow('135', a['frames'][1, 9:12, :, :].to('cpu').squeeze().permute([1, 2, 0]).numpy())
        cv2.imshow('gt', a['gt'][0:3, :, :].to('cpu').squeeze().permute([1, 2, 0]).numpy())
    cv2.waitKey(0)
    polar_image_sampler[1]
