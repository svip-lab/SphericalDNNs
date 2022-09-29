# Common
import numpy as np
from argparse import ArgumentParser
from fire import Fire
from tqdm import trange, tqdm
#import visdom
import time
import cv2

# Torch
import torch as th
from torch import nn
import torchvision.transforms as tf
from torch.utils import data as tdata
from torch.optim import SGD
from torch.autograd import Variable

# Spherical Op
from sconv.module import SphereMSE

# Parameter Setting
import ref
from opts import opts

# Network
from models.baseline_ablation_unet import SphericalUNet

# Dataset
from datasets.data import VRVideo, VRVideoLSTM
from datasets.data_retrain import VRVideo as VRVideo_retrain

def train(test_mode=False):
    opt = opts().parse()

    print("\nConfig Visdom.\n")
    #viz = visdom.Visdom(server=opt.plot_server, port=opt.plot_port, env=opt.exp_name)

    print("\nBuilding Dataset..\n")
    transform = tf.Compose([
        tf.Resize((64, 128)),
        tf.ToTensor()
    ])
    #dataset = VRVideo(ref.dataDir, 64, 128, 80, frame_interval=5, cache_gt=True, transform=transform, gaussian_sigma=np.pi/20, kernel_rad=np.pi/7)
    sal_path = '/p300/2018-ECCV-Journal/codes-project/saliency-dection-360-videos/Saliency-SConv-LSTM/exp/eval/baseline_spherical_sal_unet_1e1_b32_07_07_epoch30_train/'
    dataset = VRVideo_retrain(ref.dataDir, 64, 128, 80, frame_interval=5, cache_gt=True, transform=transform, train=True, sal_path = sal_path, gaussian_sigma=np.pi/20, kernel_rad=np.pi/7)
    if opt.clear_cache:
        dataset.clear_cache()
    loader = tdata.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=16, pin_memory=True)

    print("\nBuilding Network...\n")
    model = SphericalUNet()
    optimizer = SGD(model.parameters(), opt.lr, momentum=0.9, weight_decay=1e-5)
    pmodel = nn.DataParallel(model).cuda()

    print('opt.use_smse', opt.use_smse)
    criterion = SphereMSE(64, 128).float().cuda()

    print("\nLoad Trained or Pre-trained Model....\n")
    if opt.resume:
        ckpt = th.load('ckpt-' + exp_name + '-latest.pth.tar')
        model.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt['epoch']

    start_epoch = opt.start_epoch
    
    log_file = open(opt.snapshot_fname_log + '/' + opt.snapshot_fname_dir +'.out', 'w+')
    print("\nStart Training .....\n")
    for epoch in trange(start_epoch, opt.epochs, desc='epoch'):
        tic = time.time()
        for i, (img_batch, last_batch, target_batch) in tqdm(enumerate(loader), desc='batch', total=len(loader)):
            img_var = Variable(img_batch).cuda()
            last_var = Variable(last_batch * 10).cuda()
            t_var = Variable(target_batch * 10).cuda()
            data_time = time.time() - tic
            tic = time.time()

            out = pmodel(img_var, last_var)
            loss = criterion(out, t_var)
            fwd_time = time.time() - tic
            tic = time.time()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bkw_time = time.time() - tic

            msg = '[{:03d}|{:05d}/{:05d}] time: data={}, fwd={}, bkw={}, total={}\nloss: {:g}'.format(
                epoch, i, len(loader), data_time, fwd_time, bkw_time, data_time+fwd_time+bkw_time, loss.data[0]
            )
            #viz.images(img_batch.cpu().numpy() * 255, win='img')
            #viz.images(target_batch.cpu().numpy() * 20000, win='gt')
            #viz.images(out.data.cpu().numpy() * 1000 , win='out')
            #viz.text(msg, win='log')
            print(msg, file=log_file, flush=True)
            print(msg, flush=True)

            tic = time.time()

        if (epoch ) % opt.save_interval == 0:
            state_dict = model.state_dict()
            ckpt = dict(epoch=epoch, iter=i, state_dict=state_dict)
            th.save(ckpt, opt.snapshot_fname_prefix + '_epoch_' + str(epoch) + '.pth.tar')

if __name__ == '__main__':
    Fire(train)
