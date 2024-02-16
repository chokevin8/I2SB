# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import copy
import argparse
import random
from pathlib import Path
from easydict import EasyDict as edict

import numpy as np

import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.utils.data import DataLoader, Subset
from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu

from logger import Logger
import distributed_util as dist_util
from i2sb import Runner, download_ckpt
from corruption import build_corruption
from dataset import imagenet
from i2sb import ckpt_util

#added
from torch.utils.data import Dataset
from torchvision import transforms as T
import torch.nn as nn
from PIL import Image
from natsort import natsorted

import colored_traceback.always
from ipdb import set_trace as debug

RESULT_DIR = Path("results") # I2SB/results

def set_seed(seed):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def build_subset_per_gpu(opt, dataset, log): # create subset of dataset for distributed training

    n_data = len(dataset)
    n_gpu  = opt.global_size
    n_dump = (n_data % n_gpu > 0) * (n_gpu - n_data % n_gpu)

    # create index for each gpu
    total_idx = np.concatenate([np.arange(n_data), np.zeros(n_dump)]).astype(int)
    idx_per_gpu = total_idx.reshape(-1, n_gpu)[:, opt.global_rank]
    log.info(f"[Dataset] Add {n_dump} data to the end to be devided by {n_gpu=}. Total length={len(total_idx)}!")

    # build subset
    indices = idx_per_gpu.tolist()
    subset = Subset(dataset, indices)
    log.info(f"[Dataset] Built subset for gpu={opt.global_rank}! Now size={len(subset)}!")
    return subset

def collect_all_subset(sample, log):
    batch, *xdim = sample.shape
    gathered_samples = dist_util.all_gather(sample, log)
    gathered_samples = [sample.cpu() for sample in gathered_samples]
    # [batch, n_gpu, *xdim] --> [batch*n_gpu, *xdim]
    return torch.stack(gathered_samples, dim=1).reshape(-1, *xdim)

def build_partition(opt, full_dataset, log):
    n_samples = len(full_dataset)

    part_idx, n_part = [int(s) for s in opt.partition.split("_")]
    assert part_idx < n_part and part_idx >= 0
    assert n_samples % n_part == 0

    n_samples_per_part = n_samples // n_part
    start_idx = part_idx * n_samples_per_part
    end_idx = (part_idx+1) * n_samples_per_part

    indices = [i for i in range(start_idx, end_idx)]
    subset = Subset(full_dataset, indices)
    log.info(f"[Dataset] Built partition={opt.partition}, {start_idx=}, {end_idx=}! Now size={len(subset)}!")
    return subset


def custom_transform(t):
    #convert to tensor
    tensor = T.ToTensor()(t)
    #scale to [-1, 1]
    scaled_tensor = (tensor * 2) - 1
    return scaled_tensor


class MyDataset(Dataset):
    def __init__(self, opt, log, train):
        super().__init__()
        self.dataset_dir = opt.dataset_dir / ('train' if train else 'val')
        self.corrupt_dir = self.dataset_dir / 'HE'  # corrupt to clean -> IHC2HE
        # self.clean_dir = self.dataset_dir / 'IHC'
        self.seg_map_dir = self.dataset_dir / 'HE_segmentation_map'
        self.image_size = opt.image_size

        if opt.image_type == "jpg":
            if os.path.isdir(self.corrupt_dir):
                self.corrupt_fnames = [os.path.join(self.corrupt_dir, x) for x in os.listdir(self.corrupt_dir) if
                                       x.endswith(".jpg")]
            else:
                print(self.corrupt_dir)
                raise IOError('corrupt path must point to a valid directory')
        else:
            if os.path.isdir(self.corrupt_dir):
                self.corrupt_fnames = [os.path.join(self.corrupt_dir, x) for x in os.listdir(self.corrupt_dir) if
                                       x.endswith(".png")]
            else:
                print(self.corrupt_dir)
                raise IOError('corrupt path must point to a valid directory')

        # if os.path.isdir(self.clean_dir):
        #     self.clean_fnames = [os.path.join(self.clean_dir, x) for x in os.listdir(self.clean_dir) if
        #                          x.endswith(".png")]
        # else:
        #     raise IOError('clean path must point to a valid directory')
        #
        if os.path.isdir(self.seg_map_dir):
            self.seg_map_fnames = [os.path.join(self.seg_map_dir, x) for x in os.listdir(self.seg_map_dir) if
                                   x.endswith(".tif")]
        else:
            raise IOError('seg_map path must point to a valid directory')

        if opt.image_type == "jpg":
            self.corrupt_image_fnames = [fname for fname in self.corrupt_fnames if self._file_ext(fname) in '.jpg']
        else:
            self.corrupt_image_fnames = [fname for fname in self.corrupt_fnames if self._file_ext(fname) in '.png']
        self.corrupt_image_fnames = natsorted(self.corrupt_image_fnames)
        if len(self.corrupt_image_fnames) == 0:
            raise IOError('No corrupt image files found in the specified path')

        # self.clean_image_fnames = [fname for fname in self.clean_fnames if self._file_ext(fname) in '.png']
        # self.clean_image_fnames = natsorted(self.clean_image_fnames)
        # if len(self.clean_image_fnames) == 0:
        #     raise IOError('No clean image files found in the specified path')
        #
        self.seg_map_fnames = [fname for fname in self.seg_map_fnames if self._file_ext(fname) in '.tif']
        self.seg_map_fnames = natsorted(self.seg_map_fnames)
        if len(self.seg_map_fnames) == 0:
            raise IOError('No seg_map image files found in the specified path')

        self.transform = T.Compose([
            # T.RandomHorizontalFlip(p=0.5), # added since tissue is not symmetric, but removed since may not be consistent with tissue map
            # T.RandomVerticalFlip(p=0.5), # added since tissue is not symmetric, but removed since may not be consistent with tissue map
            T.ToTensor(),  # convert [0,255] to [0,1]
            T.Lambda(lambda t: (t * 2) - 1)  # convert [0,1] --> [-1, 1], since this is required in this training
        ])

        log.info(f"[Dataset] Built Imagenet dataset {self.corrupt_dir=}, size={len(self.corrupt_image_fnames)}!")
        # log.info(f"[Dataset] Built Imagenet dataset {self.clean_dir=}, size={len(self.clean_image_fnames)}!")
        log.info(f"[Dataset] Built Imagenet dataset {self.seg_map_dir=}, size={len(self.seg_map_fnames)}!")  # added for seg map

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _file_to_array(self, fname):
        return np.array(Image.open(os.path.join(self.dataset_dir, fname)))

    def __len__(self):
        return len(self.corrupt_image_fnames)

    def __getitem__(self, index):
        corrupt_fname = self.corrupt_image_fnames[index]
        # clean_fname = self.clean_image_fnames[index]
        seg_map_fname = self.seg_map_fnames[index]  # added
        # print(corrupt_fname)
        # print(clean_fname)
        # if corrupt_fname.split("image")[1] != clean_fname.split("image")[1]: # for batch_size = 1 only
        #     print("Please look at training data again, the images are not paired.")
        corrupt_img = self._file_to_array(os.path.join('IHC', corrupt_fname))  # IHC for IHC2HE, NS for NS2HE
        # clean_img = self._file_to_array(os.path.join('HE', clean_fname))  # HE for both.
        seg_img = self._file_to_array(os.path.join('HE_segmentation_map', seg_map_fname))
        corrupt_img = self.transform(corrupt_img)
        # clean_img = self.transform(clean_img)
        seg_map = self.transform(seg_img)  # need to conserve the uint8 nature of the mask
        return corrupt_img, seg_map
        # return clean_img, corrupt_img, seg_map  # added code
        # return clean_img, corrupt_img, clean_img # original code is this

def build_val_dataset(opt, log, corrupt_type):
    if "sr4x" in corrupt_type:
        val_dataset = imagenet.build_lmdb_dataset(opt, log, train=False) # full 50k val
    elif "inpaint" in corrupt_type:
        mask = corrupt_type.split("-")[1]
        val_dataset = imagenet.InpaintingVal10kSubset(opt, log, mask) # subset 10k val + mask
    elif corrupt_type == "mixture":
        from corruption.mixture import MixtureCorruptDatasetVal
        # val_dataset = imagenet.build_lmdb_dataset_val10k(opt, log)
        val_dataset = MyDataset(opt, log, train=False)
        val_dataset = MixtureCorruptDatasetVal(opt, val_dataset) # subset 10k val + mixture
    else:
        val_dataset = imagenet.build_lmdb_dataset_val10k(opt, log) # subset 10k val

    # build partition
    if opt.partition is not None:
        val_dataset = build_partition(opt, val_dataset, log)
    return val_dataset

def get_recon_imgs_fn(opt, nfe):
    sample_dir = RESULT_DIR / opt.ckpt / "samples_nfe{}{}".format(
        nfe, "_clip" if opt.clip_denoise else ""
    )
    os.makedirs(sample_dir, exist_ok=True)

    recon_imgs_fn = sample_dir / "recon{}.pt".format(
        "" if opt.partition is None else f"_{opt.partition}"
    )
    return recon_imgs_fn

def compute_batch(ckpt_opt, corrupt_type, corrupt_method, out):
    if "inpaint" in corrupt_type:
        clean_img, y, mask = out
        corrupt_img = clean_img * (1. - mask) + mask
        x1          = clean_img * (1. - mask) + mask * torch.randn_like(clean_img)
    elif corrupt_type == "mixture":
        corrupt_img, seg_map = out
        # clean_img, corrupt_img, seg_map = out
        mask = None
        y = None # added
        x1 = corrupt_img.to(opt.device) #added
    else:
        clean_img, y = out
        mask = None
        corrupt_img = corrupt_method(clean_img.to(opt.device))
        x1 = corrupt_img.to(opt.device)

    # cond = x1.detach() if ckpt_opt.cond_x1 else None #detach from gpu, move x1 to gpu to cpu for cond if cond_x1 = True
    if opt.corrupt == "mixture" and opt.cond_x1: # added
        cond = seg_map.detach().to(opt.device)
        print("Conditional I2SB mode")
    else:
        cond = None

    if ckpt_opt.add_x1_noise: # only for decolor
        x1 = x1 + torch.randn_like(x1)

    return corrupt_img, x1, mask, cond, y

@torch.no_grad()
def main(opt):
    log = Logger(opt.global_rank, ".log")

    # get (default) ckpt option
    ckpt_opt = ckpt_util.build_ckpt_option(opt, log, RESULT_DIR / opt.ckpt)
    corrupt_type = ckpt_opt.corrupt
    nfe = opt.nfe or ckpt_opt.interval-1

    # build corruption method
    corrupt_method = build_corruption(opt, log, corrupt_type=corrupt_type) #for image to image translation where corrupt_type = mixture, this does nothing.

    # build imagenet val dataset
    val_dataset = build_val_dataset(opt, log, corrupt_type)
    n_samples = len(val_dataset)

    # build dataset per gpu and loader
    subset_dataset = build_subset_per_gpu(opt, val_dataset, log)
    val_loader = DataLoader(subset_dataset,
        batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=1, drop_last=False,
    )

    # build runner
    runner = Runner(ckpt_opt, log, save_opt=False)

    # handle use_fp16 for ema
    if opt.use_fp16:
        runner.ema.copy_to() # copy weight from ema to net
        runner.net.diffusion_model.convert_to_fp16()
        runner.ema = ExponentialMovingAverage(runner.net.parameters(), decay=0.99) # re-init ema with fp16 weight

    # create save folder for reconstructed images
    recon_imgs_fn = get_recon_imgs_fn(opt, nfe)
    log.info(f"Recon images will be saved to {recon_imgs_fn}!")

    recon_imgs = []
    num = 0
    for loader_itr, out in enumerate(val_loader): #loader_itr = idx, and out = output of MixtureCorruptDatasetVal, which is three items: clean_img, corrupt_img, seg_map

        corrupt_img, x1, mask, cond, y = compute_batch(ckpt_opt, corrupt_type, corrupt_method, out) #

        xs, _ = runner.ddpm_sampling(
            ckpt_opt, x1, mask=mask, cond=cond, clip_denoise=opt.clip_denoise, nfe=nfe, verbose=opt.n_gpu_per_node==1
        )
        recon_img = xs[:, 0, ...].to(opt.device)

        assert recon_img.shape == corrupt_img.shape

        if loader_itr == 0 and opt.global_rank == 0: # debug
            os.makedirs(".debug", exist_ok=True)
            tu.save_image((corrupt_img+1)/2, ".debug/corrupt.png")
            tu.save_image((recon_img+1)/2, ".debug/recon.png")
            log.info("Saved debug images!")

        # [-1,1]
        gathered_recon_img = collect_all_subset(recon_img, log)
        recon_imgs.append(gathered_recon_img)


        if loader_itr % opt.save_every == 0:
            arr = torch.cat(recon_imgs, axis=0)[:n_samples]
            if opt.global_rank == 0:
                torch.save({"arr": arr}, recon_imgs_fn)

        num += len(gathered_recon_img)
        log.info(f"Collected {num} recon images!")
        dist.barrier()

    del runner

    arr = torch.cat(recon_imgs, axis=0)[:n_samples]

    if opt.global_rank == 0:
        torch.save({"arr": arr}, recon_imgs_fn)
        log.info(f"Save at {recon_imgs_fn}")
    dist.barrier()

    log.info(f"Sampling complete! Collect recon_imgs={arr.shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,  default=0)
    parser.add_argument("--n-gpu-per-node", type=int,  default=1,           help="number of gpu on each node")
    parser.add_argument("--master-address", type=str,  default='localhost', help="address for master")
    parser.add_argument("--node-rank",      type=int,  default=0,           help="the index of node")
    parser.add_argument("--num-proc-node",  type=int,  default=1,           help="The number of nodes in multi node env")

    # data
    parser.add_argument("--image-size",     type=int,  default=256)
    parser.add_argument("--dataset-dir",    type=Path, default="/dataset",  help="path to LMDB dataset")
    parser.add_argument("--partition",      type=str,  default=None,        help="e.g., '0_4' means the first 25% of the dataset")

    # sample
    parser.add_argument("--batch-size",     type=int,  default=32)
    parser.add_argument("--ckpt",           type=str,  default=None,        help="the checkpoint name from which we wish to sample")
    parser.add_argument("--nfe",            type=int,  default=None,        help="sampling steps")
    parser.add_argument("--clip-denoise",   action="store_true",            help="clamp predicted image to [-1,1] at each")
    parser.add_argument("--use-fp16",       action="store_true",            help="use fp16 network weight for faster sampling")
    # added to save recon.pt during sampling in case of crashing
    parser.add_argument("--save-every", type=int, default=100, help="save sampled recon.pt every x dataloader iterations")

    # added for conditional sampling
    parser.add_argument("--corrupt", type=str, default="mixture", help="restoration task")
    parser.add_argument("--cond-x1",        action="store_true",             help="conditional the network on degraded images")
    parser.add_argument("--image-type", type=str, default="png", help="image type, specify for jpg, else png + all other types")
    arg = parser.parse_args()

    opt = edict(
        distributed=(arg.n_gpu_per_node > 1),
        device="cuda",
    )
    opt.update(vars(arg))

    # one-time download: ADM checkpoint
    download_ckpt("data/")

    set_seed(opt.seed)

    if opt.distributed:
        size = opt.n_gpu_per_node

        processes = []
        for rank in range(size):
            opt = copy.deepcopy(opt)
            opt.local_rank = rank
            global_rank = rank + opt.node_rank * opt.n_gpu_per_node
            global_size = opt.num_proc_node * opt.n_gpu_per_node
            opt.global_rank = global_rank
            opt.global_size = global_size
            print('Node rank %d, local proc %d, global proc %d, global_size %d' % (opt.node_rank, rank, global_rank, global_size))
            p = Process(target=dist_util.init_processes, args=(global_rank, global_size, main, opt))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        torch.cuda.set_device(0)
        opt.global_rank = 0
        opt.local_rank = 0
        opt.global_size = 1
        dist_util.init_processes(0, opt.n_gpu_per_node, main, opt)
