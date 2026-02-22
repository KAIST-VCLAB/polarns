import os
import torch
from torch import nn
from model import model
from data import dataset,sampler
from torch.utils.data import DataLoader
import cv2
import numpy as np
from local import Config
import pickle as pkl
import argparse
import shutil
from model.loss import *

def test_loop(dataloader, model, loss_fn, device,bd_size,output_format,path_save=None):
    num_samples = len(dataloader.dataset)
    test_loss = 0
    num_samples_count = 0
    with torch.no_grad():
        for item in dataloader:
            input = item['frames'].to(device)

            pred = model(input)
            pred[pred<0] = 0
            pred[pred>1] = 1
            pred_cut = sampler.cut_boundary(pred,bd_size)
            gt = item['gt']
            gt_cut = sampler.cut_boundary(gt,bd_size).to(device)

            if output_format == 'degree':
                gt_ = stokes2degree(gt_cut)
            elif output_format == 'properties':
                gt_ = stokes2properties(gt_cut)
            else:
                gt_ = gt_cut

            test_loss += loss_fn(pred_cut, gt_) * item['gt'].shape[0]

            if output_format == 'degree':
                pred_cut = degree2stokes(pred_cut)
            if output_format != 'properties':
                pred_cut = stokes2properties(pred_cut)
            gt_cut = stokes2properties(gt_cut)
            ## save
            if path_save is not None:
                
                if output_format == 'degree':
                    pred = degree2stokes(pred)
                if output_format != 'properties':
                    pred_stokes = pred
                    pred_properties = stokes2properties(pred_stokes)
                else:
                    pred_properties = pred
                    pred_stokes = properties2stokes(pred_properties)
                
                s0,s1,s2 = batch_unpack_9ch(pred_stokes)
                _,dolp,_ = batch_unpack_9ch(pred_properties)

                counter = num_samples_count
                for iter_image in range(pred.shape[0]):
                    s0_output = s0[iter_image].to('cpu')
                    s1_output = s1[iter_image].to('cpu')
                    s2_output = s2[iter_image].to('cpu')
                    dolp_output = dolp[iter_image].to('cpu')

                    
                    path_now = f'{path_save}/{counter:07d}'
                    os.makedirs(path_now, exist_ok=True)

                    dolp_output[dolp_output<0] = 0
                    dolp_output[dolp_output>1] = 1
                    np.save(f'{path_now}/output_s0.npy', s0_output[None])
                    np.save(f'{path_now}/output_s1.npy', s1_output[None])
                    np.save(f'{path_now}/output_s2.npy', s2_output[None])
                    np.save(f'{path_now}/output_dolp.npy', dolp_output[None])


                    s0_output[s0_output<0] = 0
                    s0_output[s0_output>1] = 1
                    dataset.imwrite_tensor(s0_output**(1/2.2), f'{path_now}/output_vis_s0.png')
                    dataset.imwrite_tensor(dolp_output, f'{path_now}/output_vis_dolp.png')
                    counter += 1

            num_samples_count+=item['gt'].shape[0]
            if num_samples_count % 10 == 0:
                print(f"[{num_samples_count:>5d}/{num_samples:>5d}]")

    test_loss = test_loss.cpu()
    test_loss /= num_samples

    return test_loss

def main(args):
    CONFIG = Config()
    if args.idx_gpu is not None:
        if torch.cuda.device_count()>args.idx_gpu:
            CONFIG.device = [f'cuda:{args.idx_gpu}']
    loss_fn = nn.L1Loss()
    checkpoint = torch.load(CONFIG.path_model_selected,map_location=CONFIG.device[0])
    if CONFIG.burstM:
        polar_burst_model = model.BurstM_wrap(CONFIG.img_size,CONFIG.info_bursts['downsample_factor']).to(CONFIG.device[0])

    polar_burst_model.load_state_dict(checkpoint['model_state_dict'])
    polar_burst_model.eval()

    polar_image_sampler = dataset.PolarSavedImageNpyDataset(CONFIG.path_image_gt,CONFIG.burst_size_val)

    test_dataloader = DataLoader(polar_image_sampler, CONFIG.batch_size_val, num_workers=CONFIG.num_workers)

    test_loop(test_dataloader,polar_burst_model,loss_fn,CONFIG.device[0],CONFIG.bd_size,CONFIG.output_format,CONFIG.path_image_output if CONFIG.save_output_images else None)

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g','--idx_gpu', type=int, help='select index of gpu')
    args = parser.parse_args()
    main(args)
