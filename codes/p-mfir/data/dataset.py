import torch
import glob
import cv2
import numpy as np
import random
import shutil
import os

def imread_tensor(filename):
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA if img.shape[2]==4 else cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0
    img = torch.from_numpy(img)
    img = torch.permute(img, [2,0,1])
    return img

def imwrite_tensor(img, filename):
    img = torch.permute(img, [1, 2, 0])
    img = img.numpy()*65535
    img = cv2.cvtColor(img.astype(np.uint16), cv2.COLOR_RGBA2BGRA if img.shape[2]==4 else cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)

def imread_16bit(filename):
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    img = img.astype(np.float32) / 65535.0
    return img

def imread_8bit(filename):
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    img = img.astype(np.float32) / 255.0
    return img

def load_npy_tensor(filename):
    img = np.load(filename)
    img = torch.from_numpy(img)
    img = torch.permute(img, [2, 0, 1])
    return img

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


class RGBImageDataset(torch.utils.data.Dataset):
    def __init__(self, path_images):
        self.path_images = path_images
        s0_fn = glob.glob(f"{path_images}/*.jpg")
        fn = [x.replace('\\','/').split('/')[-1] for x in s0_fn]
        self.fn_number = [int(x.split('.')[0]) for x in fn]
        self.fn_number.sort()

    def __len__(self):
        return len(self.fn_number)

    def __getitem__(self, idx):
        return imread_8bit(f'{self.path_images}/{self.fn_number[idx]}.jpg')



class PolarImageDataset(torch.utils.data.Dataset):
    """
    Dataset for rsp dataset
    input : i6bit image rsp dataset (s0,s1,s2)
    output : float32 opencv image dict (s0,s1,s2)
    """
    def __init__(self, path_images):
        self.path_images = path_images
        s0_fn = glob.glob(f"{path_images}/s0gt/*.png")
        fn = [x.replace('\\','/').split('/')[-1] for x in s0_fn]
        self.fn_number = [x.split('_')[0] for x in fn]

    def __len__(self):
        return len(self.fn_number)

    def __getitem__(self, idx):
        s0 = imread_16bit(f'{self.path_images}/s0gt/{self.fn_number[idx]}_s0gt.png')
        s1 = imread_16bit(f'{self.path_images}/s1gt/{self.fn_number[idx]}_s1gt.png') *2 -1
        s2 = imread_16bit(f'{self.path_images}/s2gt/{self.fn_number[idx]}_s2gt.png') *2 -1
        images = {'s0': s0, 's1': s1, 's2': s2}
        return images

class PolarImageDatasetTensor(PolarImageDataset):
    """
    Dataset for rsp dataset
    input : i6bit image rsp dataset (s0,s1,s2)
    output : float32 tensor image dict (s0,s1,s2)
    """
    def __getitem__(self, idx):
        s0 = imread_tensor(f'{self.path_images}/s0gt/{self.fn_number[idx]}_s0gt.png')
        s1 = imread_tensor(f'{self.path_images}/s1gt/{self.fn_number[idx]}_s1gt.png') *2 -1
        s2 = imread_tensor(f'{self.path_images}/s2gt/{self.fn_number[idx]}_s2gt.png') *2 -1
        images = {'s0': s0, 's1': s1, 's2': s2}
        return images

class PolarImageDatasetCache(PolarImageDatasetTensor):
    def __init__(self, path_images, max_size=None):
        super().__init__(path_images)
        self.cache_image = [None]*len(self.fn_number)
        self.max_size = max_size
        self.cache_size = 0

    def __getitem__(self, idx):
        if self.cache_image[idx] is None:
            if self.max_size is None or self.cache_size < self.max_size:
                self.cache_image[idx] = super().__getitem__(idx)
                self.cache_size += 1
            else:
                return super().__getitem__(idx)
        return self.cache_image[idx]


class SavedImageDataset(torch.utils.data.Dataset):
    def __init__(self, path_images, burst_size):
        self.path_images = path_images
        self.burst_size = burst_size
        self.path_each_image = glob.glob(f"{path_images}/*")
        self.path_each_image.sort()

    def __len__(self):
        return len(self.path_each_image)

class PolarSavedImageDataset(SavedImageDataset):
    def __getitem__(self, idx):
        gt = [imread_tensor(f'{self.path_each_image[idx]}/gt_{ch}.png') for ch in range(3)]
        gt = torch.concat(gt,dim=0)
        frames = []
        for idx_burst in range(self.burst_size):
            frame = [imread_tensor(f'{self.path_each_image[idx]}/frame_{idx_burst:02d}_{ch}.png') for ch in range(4)]
            frames.append(torch.concat(frame,dim=0))
        frames = torch.stack(frames, dim=0)

        return {'frames':frames,'gt':gt}

class PolarSavedImageNpyDataset(SavedImageDataset):
    """
    Dataset for cropped polarization dataset
    input : float32 npy gt images (s0,s1,s2) + 16bit mosaic burst images
    output : float32 tensor images dict frame (D C H W), gt (C H W)
    """
    def __getitem__(self, idx):
        gt = [load_npy_tensor(f'{self.path_each_image[idx]}/gt_s{ch}.npy') for ch in range(3)]
        gt = torch.concat(gt,dim=0)
        frames = []
        for idx_burst in range(self.burst_size):
            frame = imreadmono2tensor(f'{self.path_each_image[idx]}/frame_{idx_burst:02d}.png')
            frames.append(frame)
        frames = torch.stack(frames, dim=0)

        return {'frames':frames,'gt':gt}


class PolarSavedImageNpyDatasetCache(PolarSavedImageNpyDataset):
    def __init__(self, path_images, burst_size, max_size=None):
        super().__init__(path_images, burst_size)
        self.cache_image = [None]*super().__len__()
        self.max_size = max_size
        self.cache_size = 0

    def __getitem__(self, idx):
        if self.cache_image[idx] is None:
            if self.max_size is None or self.cache_size < self.max_size:
                self.cache_image[idx] = super().__getitem__(idx)
                self.cache_size += 1
            else:
                return super().__getitem__(idx)
        return self.cache_image[idx]

class RGBSavedImageDataset(SavedImageDataset):
    def __getitem__(self, idx):
        gt = imread_tensor(f'{self.path_each_image[idx]}/gt.png')
        frames = []
        for idx_burst in range(self.burst_size):
            frame = imread_tensor(f'{self.path_each_image[idx]}/frame_{idx_burst:02d}.png')
            frames.append(frame)
        frames = torch.stack(frames, dim=0)

        return {'frames':frames,'gt':gt}

class ValImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, val_image_idx_list):
        self.dataset = dataset
        self.val_image_idx_list = val_image_idx_list
        assert max(val_image_idx_list)<len(dataset) , "Index out of range"

    def __len__(self):
        return len(self.val_image_idx_list)
    def __getitem__(self, idx):
        return self.dataset[self.val_image_idx_list[idx]]

if __name__ == "__main__":
    # polar_dataset = PolarImageDataset("G:/polar-normal/polar_image_dataset/rsp_dataset/rsp_dataset/train_val/gt/train")
    # polar_dataset[0]
    # rgb_dataset = RGBImageDataset("G:/polar-normal/polar_image_dataset/zurich-raw-to-rgb/train/canon")
    # rgb_dataset[0]
    path_images = "G:/polar-normal/polar_image_dataset/zurich-raw-to-rgb/train/canon"
    path_new = "G:/polar-normal/polar_image_dataset/zurich-random/"
    s0_fn = glob.glob(f"{path_images}/*.jpg")
    fn = [x.replace('\\', '/').split('/')[-1] for x in s0_fn]
    fn_number = [int(x.split('.')[0]) for x in fn]
    fn_number.sort()
    size_img = len(fn_number)
    size_test = size_img//10
    size_valid = size_img//10
    random.seed(42)
    random.shuffle(fn_number)
    test_set = fn_number[:size_test]
    valid_set = fn_number[size_test:size_test+size_valid]
    train_set = fn_number[size_test+size_valid:]
    test_set.sort()
    valid_set.sort()
    train_set.sort()

    os.makedirs(f"{path_new}/test/",exist_ok=True)
    for fn_num in test_set:
        shutil.copy(f"{path_images}/{fn_num}.jpg",f"{path_new}/test/")

    os.makedirs(f"{path_new}/valid/", exist_ok=True)
    for fn_num in valid_set:
        shutil.copy(f"{path_images}/{fn_num}.jpg", f"{path_new}/valid/")

    os.makedirs(f"{path_new}/train/", exist_ok=True)
    for fn_num in train_set:
        shutil.copy(f"{path_images}/{fn_num}.jpg", f"{path_new}/train/")

