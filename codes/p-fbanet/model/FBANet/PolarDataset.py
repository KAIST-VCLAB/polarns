import torch
import cv2
import numpy as np
import glob
import colour_demosaicing

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

def imread_polar_mono2bayer_tensor(filename):
    img = imreadmono2tensor(filename)

    bayer_raws = []
    for polar_idx in range(4):
        polar_img = img[polar_idx::4]
        bayer_raw = torch.zeros((polar_img.shape[1] * 2, polar_img.shape[2] * 2), dtype=polar_img.dtype)
        bayer_raw[0::2, 0::2] = polar_img[0]
        bayer_raw[0::2, 1::2] = polar_img[1]
        bayer_raw[1::2, 0::2] = polar_img[2]
        bayer_raw[1::2, 1::2] = polar_img[3]
        bayer_raws.append(bayer_raw)

    bayer_raws = torch.stack(bayer_raws, dim=0)
    return bayer_raws


def align_images(burst, burst_size=14):
    for polar_ch in range(4):
        im1 = burst[0, 3*polar_ch:3*(polar_ch+1)].permute(1, 2, 0).numpy()
        # Convert images to grayscale
        im1_gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
        for burst_idx in range(1, burst_size):
            im2 = burst[burst_idx, 3*polar_ch:3*(polar_ch+1)].permute(1, 2, 0).numpy()
            im2_gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)

            # Find size of image1
            sz = im1.shape

            # Define the motion model
            warp_mode = cv2.MOTION_TRANSLATION
            # warp_mode = cv2.MOTION_HOMOGRAPHY

            # Define 2x3 or 3x3 matrices and initialize the matrix to identity
            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                warp_matrix = np.eye(3, 3, dtype=np.float32)
            else:
                warp_matrix = np.eye(2, 3, dtype=np.float32)

            # Specify the number of iterations.
            number_of_iterations = 100

            # Specify the threshold of the increment
            # in the correlation coefficient between two iterations
            termination_eps = 1e-10

            # Define termination criteria
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

            try:
                # Run the ECC algorithm. The results are stored in warp_matrix.
                (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria)

                if warp_mode == cv2.MOTION_HOMOGRAPHY:
                    # Use warpPerspective for Homography
                    im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1], sz[0]),
                                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                else:
                    # Use warpAffine for Translation, Euclidean and Affine
                    im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]),
                                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    
                burst[burst_idx, 3*polar_ch:3*(polar_ch+1)] = torch.tensor(im2_aligned.astype(np.float32)).permute(2, 0, 1)
            except:
                print("An error occured when ECC not converge")
    return burst
                
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
    def __init__(self, path_images, burst_size=14):
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


class SyntheticPolarBurstVal(torch.utils.data.Dataset):
    """ Synthetic burst validation set. The validation burst have been generated using the same synthetic pipeline as
    employed in SyntheticBurst dataset.
    """
    def __init__(self, root=None, initialize=True):
        """
        args:
            root - Path to root dataset directory
            initialize - boolean indicating whether to load the meta-data for the dataset
        """
        self.root = root
        self.burst_list = glob.glob(f"{root}/*")
        self.burst_list.sort()
        self.burst_size = 14

    def initialize(self):
        pass

    def __len__(self):
        return len(self.burst_list)

    def _read_meta_info(self, index):
        meta_info = {}
        meta_info['burst_name'] = f'{index:07d}'

        return meta_info

    def __getitem__(self, idx):
        """ Generates a synthetic burst
        args:
            index: Index of the burst

        returns:
            burst: LR RAW burst, a torch tensor of shape
                   [14, 16, 48, 48] ==> [B, C, H, W]
                   The 4 channels correspond to 'R', 'G', 'G', and 'B' values in the RGGB bayer mosaick.
            gt : Ground truth linear image ==> [3x3, 48, 48]
            meta_info: Meta info about the burst which can be used to convert gt to sRGB space
        """
        gt = [load_npy_tensor(f'{self.burst_list[idx]}/gt_s{ch}.npy') for ch in range(3)]
        gt = torch.concat(gt,dim=0)
        burst = []
        for idx_burst in range(self.burst_size):
            frame = imread_polar_mono2bayer_tensor(f'{self.burst_list[idx]}/frame_{idx_burst:02d}.png')
            # demosaic the frame
            demosaiced_frames = []
            for polar_img in frame:
                demosaiced_frame = cv2.demosaicing((polar_img.numpy() * 65535.0).astype(np.uint16), cv2.COLOR_BayerRGGB2RGB)
                # demosaiced_frame2 = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(polar_img.numpy())
                # cv2.imwrite('results/temp1.png', (demosaiced_frame[:, :, ::-1]).astype(np.uint16))
                # cv2.imwrite('results/temp2.png', (demosaiced_frame2 * 65535.0).astype(np.uint16))
                demosaiced_frames.append(torch.tensor(demosaiced_frame.astype(np.float32) / 65535.0).permute(2, 0, 1))
            demosaiced_frames = torch.cat(demosaiced_frames, dim=0)
            burst.append(demosaiced_frames)

        burst = torch.stack(burst, dim=0)

        meta_info = self._read_meta_info(idx)
        data = {}
        data['LR'] = burst.float()
        data['HR'] = gt
        data['burst_name'] = meta_info['burst_name']

        return data


class SyntheticPolarBurstAlignedVal(torch.utils.data.Dataset):
    """ Synthetic burst validation set. The validation burst have been generated using the same synthetic pipeline as
    employed in SyntheticBurst dataset.
    """
    def __init__(self, root=None, initialize=True):
        """
        args:
            root - Path to root dataset directory
            initialize - boolean indicating whether to load the meta-data for the dataset
        """
        self.root = root
        self.burst_list = glob.glob(f"{root}/*")
        self.burst_list.sort()
        self.burst_size = 14

    def initialize(self):
        pass

    def __len__(self):
        return len(self.burst_list)

    def _read_meta_info(self, index):
        meta_info = {}
        meta_info['burst_name'] = f'{index:07d}'

        return meta_info

    def __getitem__(self, idx):
        """ Generates a synthetic burst
        args:
            index: Index of the burst

        returns:
            burst: LR RAW burst, a torch tensor of shape
                   [14, 16, 48, 48] ==> [B, C, H, W]
                   The 4 channels correspond to 'R', 'G', 'G', and 'B' values in the RGGB bayer mosaick.
            gt : Ground truth linear image ==> [3x3, 48, 48]
            meta_info: Meta info about the burst which can be used to convert gt to sRGB space
        """

        """
        hhha: Found that when the value is too small, network cannot estimate the correct value.
        """
        gt = [load_npy_tensor(f'{self.burst_list[idx]}/gt_s{ch}.npy') for ch in range(3)]
        gt = torch.concat(gt,dim=0)
        burst = []
        for idx_burst in range(self.burst_size):
            frame = imread_polar_mono2bayer_tensor(f'{self.burst_list[idx]}/frame_{idx_burst:02d}.png')
            # demosaic the frame
            demosaiced_frames = []
            for polar_img in frame:
                demosaiced_frame = cv2.demosaicing((polar_img.numpy() * 65535.0).astype(np.uint16), cv2.COLOR_BayerRGGB2RGB)
                demosaiced_frame = demosaiced_frame.astype(np.float32) / 65535.0
                demosaiced_frame[:, :, 0] *= 1.67
                demosaiced_frame[:, :, 2] *= 2.30
                # demosaiced_frame2 = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(polar_img.numpy())
                # cv2.imwrite('results/temp1.png', (demosaiced_frame[:, :, ::-1]).astype(np.uint16))
                # cv2.imwrite('results/temp2.png', (demosaiced_frame2 * 65535.0).astype(np.uint16))
                demosaiced_frames.append(torch.tensor(demosaiced_frame).permute(2, 0, 1))
            demosaiced_frames = torch.cat(demosaiced_frames, dim=0)
            burst.append(demosaiced_frames)

        burst = torch.stack(burst, dim=0)
        burst = burst ** (1.0 / 2.2)
        burst = align_images(burst)

        meta_info = self._read_meta_info(idx)
        data = {}
        data['LR'] = burst.float()
        data['HR'] = gt
        data['burst_name'] = meta_info['burst_name']

        return data