from torch import nn
import numpy as np
import torch as th

import torchvision.transforms as tf
from torch.utils import data as tdata
from torch.optim import SGD
from torch.autograd import Variable
from argparse import ArgumentParser
from fire import Fire
from tqdm import trange, tqdm
import visdom
import time

from models.spherical_unet import Final1, SconvLSTM
from sconv.module import SphericalConv, SphereMSE
from datasets.data import VRVideo, VRVideoLSTM

import cv2

import ref
from opts import opts

def train(
        test_mode=False
):
    opt = opts().parse()

    viz = visdom.Visdom(server=opt.plot_server, port=opt.plot_port, env=opt.exp_name)

    transform = tf.Compose([
        tf.Resize((64, 128)),
        tf.ToTensor()
    ])
    dataset = VRVideoLSTM(ref.dataDir, 64, 128, 80,sequence_len = opt.sequence_len, frame_interval=5, cache_gt=True, transform=transform, gaussian_sigma=np.pi/20, kernel_rad=np.pi/7)
    if opt.clear_cache:
        dataset.clear_cache()
    loader = tdata.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=16, pin_memory=True)

    print("\nbuilding network...\n")
    model = SconvLSTM()
    optimizer = SGD(model.parameters(), opt.lr, momentum=0.9, weight_decay=1e-5)
    pmodel = nn.DataParallel(model).cuda()
    criterion = SphereMSE(64, 128).float().cuda()

    start_epoch = opt.start_epoch

    if opt.resume:
        ckpt = th.load('ckpt-' + exp_name + '-latest.pth.tar')
        model.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt['epoch']

    log_file = open(opt.snapshot_fname_log + '/' + opt.snapshot_fname_dir +'.out', 'w+')
    for epoch in trange(start_epoch, opt.epochs, desc='epoch'):
        tic = time.time()
        for i, (img_batch, last_batch, target_batch) in tqdm(enumerate(loader), desc='batch', total=len(loader)):
            img_var = Variable(img_batch).cuda()
            last_var = Variable(last_batch * 10).cuda()
            t_var = Variable(target_batch * 10).cuda()
            data_time = time.time() - tic
            tic = time.time()

            out = pmodel(img_var, last_var)
            loss = 0
            for k in range(len(out)):
                gt_var = t_var[:,k,:,:,:]
                loss = loss + criterion(out[k], gt_var)
            fwd_time = time.time() - tic
            tic = time.time()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bkw_time = time.time() - tic

            msg = '[{:03d}|{:05d}/{:05d}] time: data={}, fwd={}, bkw={}, total={}\nloss: {:g}'.format(
                epoch, i, len(loader), data_time, fwd_time, bkw_time, data_time+fwd_time+bkw_time, loss.data[0]
            )
            viz.images(img_batch[:,-1,:,:,:].cpu().numpy() * 255, win='img')
            viz.images(target_batch[:,-1,:,:,:].cpu().numpy() * 20000, win='gt')
            viz.images(out[-1].data.cpu().numpy() * 1000 , win='out')
            viz.text(msg, win='log')
            print(msg, file=log_file, flush=True)
            print(msg, flush=True)

            tic = time.time()

        if (epoch ) % opt.save_interval == 0:
            state_dict = model.state_dict()
            ckpt = dict(epoch=epoch, iter=i, state_dict=state_dict)
            th.save(ckpt, opt.snapshot_fname_prefix + '_epoch_' + str(epoch) + '.pth.tar')


if __name__ == '__main__':
    Fire(train)
