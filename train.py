# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import random
import argparse

import copy
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.multiprocessing import Process

from logger import Logger
from distributed_util import init_processes
from corruption import build_corruption
from dataset import imagenet
from i2sb import Runner, download_ckpt

import colored_traceback.always
from ipdb import set_trace as debug

#added packages by me:
from torch.utils.data import Dataset
from torchvision import transforms as T
import torch.nn as nn
from PIL import Image
from functools import partial
from natsort import natsorted

RESULT_DIR = Path("results")

def set_seed(seed):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def create_training_options():
    # --------------- basic ---------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--name",           type=str,   default=None,        help="experiment ID")
    parser.add_argument("--ckpt",           type=str,   default=None,        help="resumed checkpoint name")
    parser.add_argument("--gpu",            type=int,   default=None,        help="set only if you wish to run on a particular device")
    parser.add_argument("--n-gpu-per-node", type=int,   default=1,           help="number of gpu on each node")
    parser.add_argument("--master-address", type=str,   default='localhost', help="address for master")
    parser.add_argument("--node-rank",      type=int,   default=0,           help="the index of node")
    parser.add_argument("--num-proc-node",  type=int,   default=1,           help="The number of nodes in multi node env")
    parser.add_argument("--save-pt-every",  type=int,   default= 5000,       help="Save latest model every x epochs") #added by kevin
    parser.add_argument("--val-every",      type=int,   default = 3000,      help="Run validation every x epochs") #added by kevin
    # parser.add_argument("--amp",            action="store_true")

    # --------------- SB model ---------------
    parser.add_argument("--image-size",     type=int,   default=256)
    parser.add_argument("--corrupt",        type=str,   default=None,        help="restoration task")
    parser.add_argument("--t0",             type=float, default=1e-4,        help="sigma start time in network parametrization")
    parser.add_argument("--T",              type=float, default=1.,          help="sigma end time in network parametrization")
    parser.add_argument("--interval",       type=int,   default=1000,        help="number of interval")
    parser.add_argument("--beta-max",       type=float, default=0.3,         help="max diffusion for the diffusion model")
    # parser.add_argument("--beta-min",       type=float, default=0.1)
    parser.add_argument("--ot-ode",         action="store_true",             help="use OT-ODE model")
    parser.add_argument("--clip-denoise",   action="store_true",             help="clamp predicted image to [-1,1] at each")

    # optional configs for conditional network
    parser.add_argument("--cond-x1",        action="store_true",             help="conditional the network on degraded images")
    parser.add_argument("--add-x1-noise",   action="store_true",             help="add noise to conditional network")

    # --------------- optimizer and loss ---------------
    parser.add_argument("--batch-size",     type=int,   default=256)
    parser.add_argument("--microbatch",     type=int,   default=2,           help="accumulate gradient over microbatch until full batch-size")
    parser.add_argument("--num-itr",        type=int,   default=1000000,     help="training iteration")
    parser.add_argument("--lr",             type=float, default=5e-5,        help="learning rate")
    parser.add_argument("--lr-gamma",       type=float, default=0.99,        help="learning rate decay ratio")
    parser.add_argument("--lr-step",        type=int,   default=1000,        help="learning rate decay step size")
    parser.add_argument("--l2-norm",        type=float, default=0.0)
    parser.add_argument("--ema",            type=float, default=0.99)

    # --------------- path and logging ---------------
    parser.add_argument("--dataset-dir",    type=Path,  default="/dataset",  help="path to LMDB dataset")
    parser.add_argument("--log-dir",        type=Path,  default=".log",      help="path to log std outputs and writer data")
    parser.add_argument("--log-writer",     type=str,   default=None,        help="log writer: can be tensorbard, wandb, or None")
    parser.add_argument("--wandb-api-key",  type=str,   default=None,        help="unique API key of your W&B account; see https://wandb.ai/authorize")
    parser.add_argument("--wandb-user",     type=str,   default=None,        help="user name of your W&B account")

    opt = parser.parse_args()

    # ========= auto setup =========
    opt.device='cuda' if opt.gpu is None else f'cuda:{opt.gpu}'
    if opt.name is None:
        opt.name = opt.corrupt
    opt.distributed = opt.n_gpu_per_node > 1
    opt.use_fp16 = False # disable fp16 for training

    # log ngc meta data
    if "NGC_JOB_ID" in os.environ.keys():
        opt.ngc_job_id = os.environ["NGC_JOB_ID"]

    # ========= path handle =========
    os.makedirs(opt.log_dir, exist_ok=True)
    opt.ckpt_path = RESULT_DIR / opt.name
    os.makedirs(opt.ckpt_path, exist_ok=True)

    if opt.ckpt is not None:
        ckpt_file = RESULT_DIR / opt.ckpt / "latest_old_5460.pt"
        assert ckpt_file.exists()
        opt.load = ckpt_file
    else:
        opt.load = None

    # ========= auto assert =========
    assert opt.batch_size % opt.microbatch == 0, f"{opt.batch_size=} is not dividable by {opt.microbatch}!"
    return opt

# this whole dataset class edited by kevin, as don't want to use lmdb. reference code that is edited from:  https://github.com/NVlabs/I2SB/issues/3
class MyDataset(Dataset):
    def __init__(self,opt,log,train):
        super().__init__()
        self.dataset_dir = opt.dataset_dir / ('train' if train else 'val')
        self.corrupt_dir = self.dataset_dir / 'IHC' # corrupt to clean -> IHC2HE
        self.clean_dir = self.dataset_dir / 'HE'
        self.image_size = opt.image_size

        if os.path.isdir(self.corrupt_dir):
            self.corrupt_fnames = [os.path.join(self.corrupt_dir, x) for x in os.listdir(self.corrupt_dir) if x.endswith(".png")]
        else:
            print(self.corrupt_dir)
            raise IOError('corrupt path must point to a valid directory')

        if os.path.isdir(self.clean_dir):
            self.clean_fnames = [os.path.join(self.clean_dir, x) for x in os.listdir(self.clean_dir) if x.endswith(".png")]
        else:
            raise IOError('clean path must point to a valid directory')

        self.corrupt_image_fnames = [fname for fname in self.corrupt_fnames if self._file_ext(fname) in '.png']
        self.corrupt_image_fnames = natsorted(self.corrupt_image_fnames)
        if len(self.corrupt_image_fnames) == 0:
            raise IOError('No corrupt image files found in the specified path')

        self.clean_image_fnames = [fname for fname in self.clean_fnames if self._file_ext(fname) in '.png']
        self.clean_image_fnames = natsorted(self.clean_image_fnames)

        if len(self.clean_image_fnames) == 0:
            raise IOError('No clean image files found in the specified path')

        self.transform = T.Compose([
            T.ToTensor(), # normalize [0,255] to [0,1]
            T.Lambda(lambda t: (t * 2) - 1)  # convert [0,1] --> [-1, 1], since this is required in this training
        ])

        log.info(f"[Dataset] Built Imagenet dataset {self.corrupt_dir=}, size={len(self.corrupt_image_fnames)}!")
        log.info(f"[Dataset] Built Imagenet dataset {self.clean_dir=}, size={len(self.clean_image_fnames)}!")

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _file_to_array(self, fname):
        return np.array(Image.open(os.path.join(self.dataset_dir,fname)))
    def __len__(self):
        return len(self.clean_image_fnames)

    def __getitem__(self, index):
        corrupt_fname = self.corrupt_image_fnames[index]
        clean_fname = self.clean_image_fnames[index]
        # print(corrupt_fname)
        # print(clean_fname)
        # if corrupt_fname.split("image")[1] != clean_fname.split("image")[1]: # for batch_size = 1 only
        #     print("Please look at training data again, the images are not paired.")
        corrupt_img = self._file_to_array(os.path.join('IHC',corrupt_fname))
        clean_img = self._file_to_array(os.path.join('HE',clean_fname))
        corrupt_img = self.transform(corrupt_img)
        clean_img = self.transform(clean_img)
        return clean_img, corrupt_img, clean_img #clean_img, corrupt_img, y is original

def main(opt):
    log = Logger(opt.global_rank, opt.log_dir)
    log.info("=======================================================")
    log.info("         Image-to-Image Schrodinger Bridge")
    log.info("=======================================================")
    log.info("Command used:\n{}".format(" ".join(sys.argv)))
    log.info(f"Experiment ID: {opt.name}")

    # set seed: make sure each gpu has differnet seed!
    if opt.seed is not None:
        set_seed(opt.seed + opt.global_rank)

    # build imagenet dataset (default original code)

    # train_dataset = imagenet.build_lmdb_dataset(opt, log, train=True)
    # val_dataset   = imagenet.build_lmdb_dataset(opt, log, train=False)
    train_dataset = MyDataset(opt, log, train=True)
    val_dataset = MyDataset(opt, log, train=False)

    # note: images should be normalized to [-1,1] for corruption methods to work properly

    if opt.corrupt == "mixture":
        import corruption.mixture as mix
        train_dataset = mix.MixtureCorruptDatasetTrain(opt, train_dataset)
        val_dataset = mix.MixtureCorruptDatasetVal(opt, val_dataset)

    # build corruption method
    corrupt_method = build_corruption(opt, log)

    run = Runner(opt, log)
    run.train(opt, train_dataset, val_dataset, corrupt_method)
    log.info("Finish!")

if __name__ == '__main__':
    opt = create_training_options()

    assert opt.corrupt is not None

    # one-time download: ADM checkpoint
    download_ckpt("data/")

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
            p = Process(target=init_processes, args=(global_rank, global_size, main, opt))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        torch.cuda.set_device(0)
        opt.global_rank = 0
        opt.local_rank = 0
        opt.global_size = 1
        init_processes(0, opt.n_gpu_per_node, main, opt)
