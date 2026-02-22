import os
import sys
import argparse
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
from model import model
from model.loss import *
from data import dataset,sampler
from torch.utils.data import DataLoader
import glob
import re
from local import Config
from threading import Thread
from queue import Queue
import numpy as np
import shutil
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import importlib
import random

import torch.backends.cudnn as cudnn

from torch.nn.parallel import DistributedDataParallel as DDP

from pwcnet.pwcnet import PWCNet

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def setup(rank, world_size, package, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group(package, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def file_remove(fn_remove):
    if os.path.exists(fn_remove):
        os.remove(fn_remove)

class Trainer():
    def __init__(self,model,loss_fn,optimizer,device,bd_size,output_format):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.bd_size = bd_size
        self.queue_job = Queue(maxsize=1)
        self.queue_output = Queue()
        self.join_signal = False
        self.output_format = output_format
        self.alignment_net = PWCNet(load_pretrained=True,
                           weights_path='./pwcnet/pwcnet-network-default.pth')
        self.alignment_net = self.alignment_net.to(device)
        for param in self.alignment_net.parameters():
            param.requires_grad = False
        
        self.aligned_loss = AlignedL1(alignment_net=self.alignment_net,sr_factor=2, boundary_ignore=self.bd_size)
        self.aligned_l2 = AlignedL2(alignment_net=self.alignment_net,sr_factor=2, boundary_ignore=self.bd_size)


def train_thread_f(trainer):
    torch.cuda.set_device(trainer.device)
    while (trainer.join_signal is False) or (trainer.queue_job.empty() is False):
        job = trainer.queue_job.get()
        item = job['item']
        if item is None:
            return
        batch_num = job['batch_num']
        dataset_size = job['dataset_size'] * trainer.world_size
        input = item['frames'].to(trainer.device)

        pred = trainer.model(input)

        # gt = sampler.cut_boundary(item['gt'],trainer.bd_size).to(trainer.device)
        gt = item['gt'].to(trainer.device)

        if trainer.output_format == 'degree':
            gt = stokes2degree(gt)
        if trainer.output_format == 'properties':
            gt = stokes2properties(gt)

        loss = trainer.aligned_loss(pred, gt, input)

        trainer.optimizer.zero_grad()
        loss.backward()

        trainer.optimizer.step()

        trainer.queue_output.put({'loss':loss.clone().detach(),'loss_sum':loss.item()*len(item['gt'])})
        trainer.queue_job.task_done()
        if trainer.rank == 0 and batch_num % 10 == 0:
            current = (batch_num + 1) * len(item['gt']) * trainer.world_size
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{dataset_size:>5d}]")

def train_loop_multi(dataloader, trainer):
    size = len(dataloader.dataset)
    loss = None
    loss_sum = torch.zeros(1,device=trainer.device)
    for batch, item in enumerate(dataloader):
        trainer.queue_job.put({'item':item,'batch_num':batch,'dataset_size':size})

    trainer.queue_job.join()
    while trainer.queue_output.empty() is False:
        output = trainer.queue_output.get()
        loss = output['loss']
        loss_sum += output['loss_sum']
    assert trainer.queue_output.empty(), 'Trainer queue is not empty.'
    output_loss = loss_sum/size
    return loss,output_loss

def train_loop(dataloader, trainer):
    size = len(dataloader.dataset)
    loss = None
    loss_sum = torch.zeros(1,device=trainer.device)
    for batch, item in enumerate(dataloader):
        input = item['frames'].to(trainer.device)

        pred = trainer.model(input)

        # gt = sampler.cut_boundary(item['gt'],trainer.bd_size).to(trainer.device)
        gt = item['gt'].to(trainer.device)

        if trainer.output_format == 'degree':
            gt = stokes2degree(gt)
        if trainer.output_format == 'properties':
            gt = stokes2properties(gt)

        loss = trainer.aligned_loss(pred, gt, input)

        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()

        loss_sum += loss.item() * len(item['gt'])
        # if batch % 10 == 0:
        #     current = (batch + 1) * len(item['gt'])
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        if trainer.rank == 0 and batch % 10 == 0:
            current = (batch + 1) * len(item['gt']) * trainer.world_size
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size * trainer.world_size:>5d}]")

    output_loss = loss_sum/size
    return loss,output_loss

def test_loop(dataloader, model, loss_fn,device,bd_size,rank,output_format):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    # mse_loss = 0
    # mse_f = nn.MSELoss()
    mse_loss_s0 = 0
    mse_loss_dop = 0
    mse_loss_aolp = 0
    bias_entire_dop = 0
    mse_f = loss_fn[1]
    loss_fn = loss_fn[0]
    with torch.no_grad():
        for item in dataloader:
            input = item['frames'].to(device)
            pred = model(input)
            gt = item['gt'].to(device)

            gt = stokes2degree(gt)
            test_loss += loss_fn(pred, gt, input)*len(item['gt'])
            mse_s0,mse_dop,bias_dop,mse_aolp = mse_f(pred, gt, input)
            mse_loss_s0 += mse_s0*len(item['gt'])
            mse_loss_dop += mse_dop*len(item['gt'])
            bias_entire_dop += bias_dop*len(item['gt'])
            mse_loss_aolp += mse_aolp*len(item['gt'])


    test_loss /= size
    # mse_loss /= size
    mse_loss_s0 /= size
    mse_loss_dop /= size
    bias_entire_dop /= size
    mse_loss_aolp /= size
    # PSNR = - 10*torch.log10(mse_loss)
    PSNR_s0 = - 10*torch.log10(mse_loss_s0)
    PSNR_dop = - 10*torch.log10(mse_loss_dop)
    PSNR_aolp = - 10*torch.log10(mse_loss_aolp/(torch.pi**2))
    if rank == 0:
        # print(f"Avg loss: {test_loss:>8f} PSNR: {PSNR:>8.2f}\n")
        print(f"Avg loss: {test_loss:>8f} PSNR_s0: {PSNR_s0:>8.2f} PSNR_dop: {PSNR_dop:>8.2f} bias_dop: {bias_entire_dop:>2.8f} PSNR_aolp: {PSNR_aolp:>8.2f}\n")
    return test_loss,mse_loss_s0,mse_loss_dop,bias_entire_dop,mse_loss_aolp

def show_val_images(writer, dataloader, model, device, image_type,global_step,output_format):
    with torch.no_grad():
        counter = 0
        for item in dataloader:
            input = item['frames'].to(device)

            pred = model(input)

            if pred.shape[1] > 3: # polar
                if output_format=='degree':
                    pred[pred<0] = 0
                    pred[pred>1] = 1
                    pred = degree2stokes(pred)
                if output_format != 'properties':
                    pred = stokes2properties(pred)
                s0, dop, aolp = batch_unpack_9ch(pred)
                dop[dop>1] = 1
                dop[dop<0] = 0
                writer.add_images(f'images_dop/{counter}', dop, global_step=global_step)
                pred = s0/2
                if image_type == 'polar_real':
                    pred[:,0] *= 1.67
                    pred[:,2] *= 2.30

            pred[pred<0] = 0
            pred[pred>1] = 1
            pred = pred ** (1/2.2)
            writer.add_images(f'images/{counter}',pred,global_step=global_step)
            counter +=1

def show_gt_images(writer, dataloader, image_type, output_format):
    with torch.no_grad():
        counter = 0
        for item in dataloader:
            gt = item['gt']

            if gt.shape[1] > 3: # polar
                if gt.shape[1] == 12:
                    gt[gt<0] = 0
                    gt[gt>1] = 1
                    gt = degree2stokes(gt)
                gt = stokes2properties(gt)
                s0, dop, aolp = batch_unpack_9ch(gt)
                writer.add_images(f'images_dop/{counter}_gt', dop)
                gt = s0/2
                if image_type == 'polar_real':
                    gt[:,0] *= 1.67
                    gt[:,2] *= 2.30

            gt[gt<0] = 0
            gt[gt>1] = 1
            gt = gt ** (1/2.2)
            writer.add_images(f'images/{counter}_gt',gt)
            counter += 1

def train(rank, world_size, CONFIG):

    print(f"divicecount:{torch.cuda.device_count()}")
    torch.cuda.set_device(CONFIG.device[rank])
    print(CONFIG.device[rank])
    torch.cuda.empty_cache()

    init_seeds(rank)

    if world_size>1:
        setup(rank,world_size,CONFIG.ddp_package,CONFIG.port)

    if rank == 0:
        writer = SummaryWriter(f'{CONFIG.path_model}/tensorboard')

    # loss
    if CONFIG.loss == 'l1':
        loss_fn = nn.L1Loss()
        if CONFIG.output_format == 'properties':
            loss_fn = l1_in_properties
    if CONFIG.loss == 'l2':
        loss_fn = nn.MSELoss()
        if CONFIG.output_format == 'properties':
            loss_fn = mse_in_properties


    # model and dataset
    if CONFIG.image_type == 'polar':
        image_dataset = dataset.PolarImageDatasetCache(CONFIG.path_image_train)
        image_dataset_val = dataset.PolarImageDatasetCache(CONFIG.path_image_val)
        sampler_func = sampler.RandomBurstImageTensor
        info_bursts = CONFIG.info_bursts
    if CONFIG.image_type == 'polar_real':
        image_dataset = dataset.PolarSavedImageNpyDatasetCache(CONFIG.path_image_real_train, CONFIG.burst_size[0])
        image_dataset_val = dataset.PolarSavedImageNpyDatasetCache(CONFIG.path_image_real_val, CONFIG.burst_size_val)
        sampler_func = sampler.RandomIndexing
        info_bursts = CONFIG.info_bursts_real

    print('init model')
    if CONFIG.mfir:
        denoiser_model = model.MFIR_wrap().to(CONFIG.device[rank])

    if CONFIG.image_type != 'polar_real':
        image_sampler = sampler_func(image_dataset, num_samples=CONFIG.batch_size[0] * CONFIG.iter_size // world_size,
                                     info_bursts=info_bursts, device=CONFIG.device[rank])

        image_sampler_val = sampler_func(image_dataset_val, num_samples=CONFIG.batch_size_val * CONFIG.iter_size_val // world_size,
                                     info_bursts=info_bursts, device=CONFIG.device[rank])
    else:
        image_sampler = sampler_func(image_dataset, num_samples=CONFIG.batch_size[0] * CONFIG.iter_size // world_size,
                                     device=CONFIG.device[rank])
        image_sampler_val = sampler_func(image_dataset_val, num_samples=CONFIG.batch_size[0] * CONFIG.iter_size_val // world_size,
                                     device=CONFIG.device[rank])
    
    if not CONFIG.debug:
        if CONFIG.image_type != 'polar_real':
            image_sampler_val = dataset.PolarSavedImageNpyDatasetCache(CONFIG.path_image_val_cropped, CONFIG.burst_size_val)
        else:
            image_sampler_val = dataset.PolarSavedImageNpyDatasetCache(CONFIG.path_image_real_val, CONFIG.burst_size_val)

    image_sampler.burst_size = CONFIG.burst_size[0]
    if CONFIG.image_type == 'polar':
        image_sampler_val_image = dataset.ValImageDataset(
            dataset.PolarSavedImageNpyDatasetCache(CONFIG.path_image_val_cropped, CONFIG.burst_size_val),
            CONFIG.polar_val_image_idx_list)
    if CONFIG.image_type == 'polar_real':
        image_sampler_val_image = dataset.ValImageDataset(
            dataset.PolarSavedImageNpyDatasetCache(CONFIG.path_image_real_val, CONFIG.burst_size_val),
            CONFIG.polar_real_val_image_idx_list)

    # optimizer
    print('init optimizer')
    optimizer = torch.optim.Adam(denoiser_model.parameters(), lr=CONFIG.learning_rate*(CONFIG.gamma**CONFIG.init_epoch))

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=CONFIG.gamma)

    # ddp
    if world_size>1:
        dist.barrier()
        print('DDP init')
        denoiser_model = DDP(denoiser_model, device_ids=[CONFIG.device[rank]], find_unused_parameters=True)

    if not CONFIG.init_epoch==0 or CONFIG.is_fine_tuning:
        print('load model')
        if CONFIG.init_epoch==0:
            checkpoint = torch.load(CONFIG.path_base_model,map_location='cpu')
        else:
            path_checkpoints = glob.glob(f'{CONFIG.path_model}/model_{CONFIG.init_epoch:04d}*.pth')
            checkpoint = torch.load(path_checkpoints[0],map_location='cpu')
        if world_size>1:
            denoiser_model.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            denoiser_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            print('Optimizer load failed')
            pass
        del checkpoint

    # dataloader
    train_dataloader = DataLoader(image_sampler,CONFIG.batch_size[0] // world_size,num_workers=CONFIG.num_workers)
    val_dataloader = DataLoader(image_sampler_val,CONFIG.batch_size_val // world_size,num_workers=CONFIG.num_workers)
    val_image_dataloader = DataLoader(image_sampler_val_image,CONFIG.batch_size_val,num_workers=CONFIG.num_workers)

    # trainer
    trainer = Trainer(denoiser_model, loss_fn, optimizer,CONFIG.device[rank],CONFIG.bd_size,CONFIG.output_format)
    trainer.rank = rank
    trainer.world_size = world_size
    trainer.thread = Thread(target=train_thread_f,args=[trainer])
    trainer.thread.start()

    # loss initialization
    min_train_loss = float('inf')
    min_val_loss = float('inf')
    iter_min_train_loss = -1
    iter_min_val_loss = -1
    test_loss = float('inf')

    if rank == 0:
        if os.path.exists(f'{CONFIG.path_model}/loss.txt'):
            with open(f'{CONFIG.path_model}/loss.txt', 'r') as f:
                f.readline()
                for line in f:
                    row = line.split()
                    if row[0] == 'train':
                        iter_min_train_loss = int(row[1])
                        min_train_loss = float(row[2])
                    if row[0] == 'val':
                        iter_min_val_loss = int(row[1])
                        min_val_loss = float(row[2])

    save_and_remove_threads = []

    # train loop
    burst_size_idx = 0
    for t in range(CONFIG.init_epoch,CONFIG.epochs):
        while sum(CONFIG.burst_size_epochs[:burst_size_idx+1])<=t and burst_size_idx<len(CONFIG.burst_size):
            burst_size_idx+=1
            image_sampler.burst_size = CONFIG.burst_size[burst_size_idx]
            train_dataloader = DataLoader(image_sampler, CONFIG.batch_size[burst_size_idx], num_workers=CONFIG.num_workers)
            image_sampler.num_samples = CONFIG.batch_size[burst_size_idx]*CONFIG.iter_size
        if rank == 0:
            print(f"Epoch {t + 1}\n-------------------------------")
        if CONFIG.train_multithread:
            loss, loss_val = train_loop_multi(train_dataloader,trainer)
        else:
            loss, loss_val = train_loop(train_dataloader, trainer)

        scheduler.step()

        if rank == 0:
            for thread in save_and_remove_threads:
                thread.join()
            save_and_remove_threads = []
            model_save = denoiser_model.module if world_size > 1 else denoiser_model
            save_list = []
            remove_list = []
            rewrite_loss = False
            saved_name = None
            if (t + 1)%CONFIG.period_model_save==0:
                save_list.append(f'{CONFIG.path_model}/model_{t + 1:04d}.pth')
            if CONFIG.save_last_iteration:
                save_list.append(f'{CONFIG.path_model}/model_{t + 1:04d}_last.pth')
                remove_list.append(f'{CONFIG.path_model}/model_{t:04d}_last.pth')
            if CONFIG.save_min_train_loss and min_train_loss > loss_val:
                min_train_loss = float(loss_val)
                save_list.append(f'{CONFIG.path_model}/model_{t + 1:04d}_train_loss.pth')
                remove_list.append(f'{CONFIG.path_model}/model_{iter_min_train_loss+1:04d}_train_loss.pth')
                iter_min_train_loss = t
                rewrite_loss = True
            if save_list and saved_name is None:
                saved_name = save_list.pop()
                save_thread = Thread(target=(lambda:torch.save({
                'epoch': t,
                'model_state_dict': model_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, saved_name)))
                save_thread.start()



        if (t + 1)%CONFIG.period_valid==0:
            test_loss,mse_loss_s0,mse_loss_dop,bias_dop,mse_loss_aolp\
                = test_loop(val_dataloader, denoiser_model, [trainer.aligned_loss,trainer.aligned_l2],CONFIG.device[rank],CONFIG.bd_size,rank,CONFIG.output_format)

        if world_size>1:
            dist.reduce(loss_val, 0, op=dist.ReduceOp.SUM)
            if (t + 1)%CONFIG.period_valid==0:
                dist.reduce(test_loss, 0, op=dist.ReduceOp.SUM)
                # dist.reduce(PSNR, 0, op=dist.ReduceOp.SUM)
                dist.reduce(mse_loss_s0, 0, op=dist.ReduceOp.SUM)
                dist.reduce(mse_loss_dop, 0, op=dist.ReduceOp.SUM)
                dist.reduce(bias_dop, 0, op=dist.ReduceOp.SUM)
                dist.reduce(mse_loss_aolp, 0, op=dist.ReduceOp.SUM)

        if rank==0:
            loss_val /= world_size
            writer.add_scalars('loss/obj', {'Train':loss_val}, global_step=t+1)
            if (t + 1)%CONFIG.period_valid==0:
                test_loss /= world_size
                # PSNR /= world_size
                mse_loss_s0 /= world_size
                mse_loss_dop /= world_size
                bias_dop /= world_size
                mse_loss_aolp /= world_size
                PSNR_s0 = - 10*torch.log10(mse_loss_s0)
                PSNR_dop = - 10*torch.log10(mse_loss_dop)
                PSNR_aolp = - 10*torch.log10(mse_loss_aolp/(torch.pi**2))
                writer.add_scalars('loss/obj', {'Validation': test_loss}, global_step=t + 1)
                # writer.add_scalars('loss/PSNR', {'Validation' : PSNR}, global_step=t + 1)
                writer.add_scalars('loss/PSNR', {'s0' : PSNR_s0}, global_step=t + 1)
                writer.add_scalars('loss/PSNR', {'DoLP' : PSNR_dop}, global_step=t + 1)
                writer.add_scalars('loss/PSNR', {'AoLP' : PSNR_aolp}, global_step=t + 1)
                writer.add_scalars('loss/mse', {'s0': mse_loss_s0}, global_step=t + 1)
                writer.add_scalars('loss/mse', {'DoLP': mse_loss_dop}, global_step=t + 1)
                writer.add_scalars('loss/mse', {'AoLP': mse_loss_aolp}, global_step=t + 1)
                writer.add_scalars('loss/bias', {'DoLP': bias_dop}, global_step=t + 1)
                show_val_images(writer,val_image_dataloader,denoiser_model,CONFIG.device[rank],CONFIG.image_type,t + 1,CONFIG.output_format)
            if t==0:
                show_gt_images(writer, val_image_dataloader,CONFIG.image_type,CONFIG.output_format)



            if CONFIG.save_min_val_loss and min_val_loss > test_loss:
                min_val_loss = float(test_loss)
                save_list.append(f'{CONFIG.path_model}/model_{t + 1:04d}_val_loss.pth')
                remove_list.append(f'{CONFIG.path_model}/model_{iter_min_val_loss+1:04d}_val_loss.pth')
                iter_min_val_loss = t
                rewrite_loss = True
            
            if save_list and saved_name is None:
                saved_name = save_list.pop()
                save_thread = Thread(target=(lambda:torch.save({
                'epoch': t,
                'model_state_dict': model_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, saved_name)))
                save_thread.start()


            if rewrite_loss:
                with open(f'{CONFIG.path_model}/loss.txt','w') as f:
                    f.write('name\titer\tvalue\n')
                    if CONFIG.save_min_train_loss:
                        f.write(f'train\t{iter_min_train_loss+1}\t{min_train_loss}\n')
                    if CONFIG.save_min_val_loss:
                        f.write(f'val\t{iter_min_val_loss+1}\t{min_val_loss}\n')
                
            if saved_name is not None:
                save_thread.join()

            for fn_save in save_list:
                save_and_remove_threads.append(
                    Thread(target=(lambda:shutil.copy(saved_name,fn_save))))
                save_and_remove_threads[-1].start()
            
            for fn_remove in remove_list:
                save_and_remove_threads.append(
                    Thread(target=(lambda x:file_remove(x)),args=[fn_remove]))
                save_and_remove_threads[-1].start()

    for thread in save_and_remove_threads:
        thread.join()

    trainer.join_signal = True
    trainer.queue_job.put({'item':None})
    trainer.thread.join()

    if world_size>1:
        dist.barrier()
        cleanup()

def main(args):
    CONFIG = Config()
    if args.idx_gpu is not None:
        if len(args.idx_gpu)>1:
            CONFIG.device = [torch.device(f'cuda:{i}') for i in range(len(args.idx_gpu))]
            # CONFIG.device = [torch.device(f'cuda:{i}') for i in args.idx_gpu]
            # CONFIG.device = [torch.device(f'cuda:{0}')]
        else:
            CONFIG.device = [torch.device(f'cuda:{i}') for i in args.idx_gpu]
    CONFIG.num_threads = len(CONFIG.device)
    if args.port is not None:
        CONFIG.port = args.port

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in args.idx_gpu])
    # importlib.reload(torch)

    os.makedirs(CONFIG.path_model, exist_ok=True)

    # save model codes
    if CONFIG.save_model_code:
        path_save_codes = f'{CONFIG.path_model}/model_codes'
        os.makedirs(path_save_codes, exist_ok=True)
        shutil.copy('./local.py',f'{path_save_codes}')
        shutil.copy('./train_real.py',f'{path_save_codes}')
        shutil.copytree('./data',f'{path_save_codes}/data',dirs_exist_ok=True)
        shutil.copytree('./model',f'{path_save_codes}/model',dirs_exist_ok=True)


    # check load information
    if CONFIG.init_epoch==-1:
        path_checkpoints = glob.glob(f"{CONFIG.path_model}/model_*.pth")
        if not path_checkpoints:
            CONFIG.init_epoch=0
        else:
            path_checkpoints.sort()
            # for fn in path_checkpoints: print(fn) 
            last = re.findall("[0-9]+",path_checkpoints[-1])
            CONFIG.init_epoch=int(last[-1])

    if CONFIG.num_threads == 1:
        train(0,1,CONFIG)
    else:
        try:
            mp.spawn(train,args=(CONFIG.num_threads,CONFIG),nprocs=CONFIG.num_threads,join=True)
            # process_list = []
            # for i in range(CONFIG.num_threads):
            #     os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
            #     os.environ['CUDA_VISIBLE_DEVICES'] = str(args.idx_gpu[i])
            #     importlib.reload(os)
            #     importlib.reload(torch)
            #     process_list.append(mp.spawn(train,args=(CONFIG.num_threads,CONFIG),nprocs=1,join=False))
                
            # for i in range(CONFIG.num_threads):
            #     process_list[i].join()

        except KeyboardInterrupt:
            dist.destroy_process_group()

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g','--idx_gpu', nargs='+', type=int, help='select index of gpu')
    parser.add_argument('-p','--port', type=str, help='port number for DDP')
    args = parser.parse_args()
    main(args)
