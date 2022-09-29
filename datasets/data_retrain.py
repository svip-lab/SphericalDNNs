import numpy as np
import torch as th
import torch
import torch.utils.data as data
from PIL import Image
import os
import pickle
from scipy import signal
from sconv.functional.sconv import spherical_conv
from tqdm import tqdm
import numbers
import cv2
from functools import lru_cache
from random import Random
import torch.nn as nn

import ref
from opts import opts
opt = opts().parse()

import utils.equirect_rotate as equirect_rotate
import torchvision.transforms as tf


class VRVideo(data.Dataset):
    def __init__(self, root, frame_h, frame_w, video_train, frame_interval=1, transform=None, train=True, sal_path = 'X',
                 gaussian_sigma=np.pi / 20, kernel_rad=np.pi/7, kernel_size=(30, 60), cache_gt=True, rnd_seed=367643):
        self.frame_interval = frame_interval
        self.transform = transform
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.gaussian_sigma = gaussian_sigma
        self.kernel_size = kernel_size
        self.kernel_rad = kernel_rad
        self.cache_gt = cache_gt
        self.train = train

        rnd = Random(rnd_seed)

        # load target
        self.vinfo = pickle.load(open(os.path.join(root, 'vinfo.pkl'), 'rb'))

        # load image paths
        vset = list()
        for vid in tqdm(os.listdir(root), desc='scanning dir'):
            if os.path.isdir(os.path.join(root, vid)):
                vset.append(vid)
        vset.sort()
        assert set(self.vinfo.keys()) == set(vset)
        print('{} videos found.'.format(len(vset)))
        if isinstance(video_train, numbers.Integral):
            vset_train = set(rnd.sample(vset, k=video_train))
            vset_val = set(vset) - vset_train
        else:
            raise NotImplementedError()
        print('{}:{} videos chosen for training:testing.'.format(len(vset_train), len(vset_val)))
        # print('test videos: {}'.format(vset_val))

        vset = vset_train if train else vset_val
        #if train:
        #    sal_path = '/p300/2018-ECCV-Journal/codes-project/saliency-dection-360-videos/Saliency-SConv-LSTM/exp/eval/baseline_spherical_sal_unet_1e1_b32_07_07_epoch30_train/'
        #else:
        #    sal_path = '/p300/2018-ECCV-Journal/codes-project/saliency-dection-360-videos/Saliency-SConv-LSTM/exp/eval/baseline_spherical_sal_unet_1e1_b32_07_07_epoch30/'


        self.data = []
        self.data_sal = []
        self.target = []
        self.i2v = {}
        self.v2i = {}
        for vid in vset:
            obj_path = os.path.join(root, vid)
            sal_video_path = os.path.join(sal_path, vid)
            # fcnt = 0
            frame_list = [frame for frame in os.listdir(obj_path) if frame.endswith('.jpg')]
            frame_list.sort()
            for frame in frame_list:
                fid = frame[:-4]

                #if int(fid) < 50:
                #    continue
                
                # fcnt += 1
                # if fcnt >= frame_interval:
                self.i2v[len(self.data)] = (vid, fid)
                self.v2i[(vid, fid)] = len(self.data)
                self.data.append(os.path.join(obj_path, frame))
                self.data_sal.append(os.path.join(sal_video_path, frame))

                self.target.append(self.vinfo[vid][fid])

                #import pdb; pdb.set_trace()
                    # fcnt = 0

        self.target.append([(0.5, 0.5)])

        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        if opt.use_equirect_rotate:
            self.rot_transform = tf.Compose([
                tf.Resize((64, 128))
            ])

            self.y_z_list = {}
            self.y_z_list[0] = (0, 60)
            self.y_z_list[1] = (60, 0)
            self.y_z_list[2] = (60, 60)

    def __getitem__(self, item):
        if opt.use_equirect_rotate:
            rot_x, isInverse = 0, False
            if self.train:
                idx = np.random.randint(0, 3)
                rot_y, rot_z = self.y_z_list[idx]
                #rot_y, rot_z = np.random.randint(-80, 80), np.random.randint(-180, 180)
            else:
                rot_y, rot_z = opt.rot_y, opt.rot_z


        img = Image.open(open(self.data[item], 'rb'))
        # img = img.resize((self.frame_w, self.frame_h))
        if self.transform:
            if opt.use_equirect_rotate:
                img = self.rot_transform(img)
                img = np.array(img)
                #rot_x, isInverse = 0, False
                #rot_y, rot_z = np.random.randint(-80,80), np.random.randint(-180,180)
                img = equirect_rotate.Equirect_Rotate(img, rot_x, rot_y, rot_z, isInverse)
                img = torch.from_numpy(img).float()
            else:
                img = self.transform(img)
        else:
            img = np.array(img)

        vid, fid = self.i2v[item]
        if int(fid) - self.frame_interval <= 0:
            last = self._get_pred_salency_map(-1)
        else:
            last = self._get_pred_salency_map(self.v2i[(vid, '%04d' % (int(fid) - self.frame_interval))])

        target = self._get_salency_map(item)

        #print(target.shape, last.shape)
        if opt.conv_type == 'sphereconv' or opt.conv_type == 'standard' or opt.conv_type == 'gafilters':
            last = (last - last.min() )/ (last.max() - last.min())
            target = (target - target.min()) / ( target.max() - target.min() )

        if opt.use_equirect_rotate:
            last, target = last.permute(1,2,0).numpy(), target.permute(1,2,0).numpy()

            #print(last.shape, img.shape)
            last = equirect_rotate.Equirect_Rotate(last, rot_x, rot_y, rot_z, isInverse)
            target = equirect_rotate.Equirect_Rotate(target, rot_x, rot_y, rot_z, isInverse)

            last = torch.from_numpy(last).float().permute(2,0,1)
            target = torch.from_numpy(target).float().permute(2,0,1)
            img = img.permute(2,0,1)


        if self.train:
            return img, last, target
        else:
            return img, self.data[item], last, target, vid, fid



    def __len__(self):
        return len(self.data)

    def _get_pred_salency_map(self, item, use_cuda=False):
        cfile = self.data_sal[item][:-4] + '.bin'

        if item >= 0:
            if os.path.isfile(cfile):
                size = os.path.getsize(cfile)
                target = np.fromfile(cfile, dtype=np.float32)
                #print(target.shape)
                target.shape = 1,64,128 #1,128,256#
                #print(target.shape)
                target = target/target.max()#*255
                #target_map = th.from_numpy(np.load(cfile)).float()
                #target_map = self.downsample(target_map)
                return torch.from_numpy(target).float()
        target = np.zeros((self.frame_h, self.frame_w))
        for x_norm, y_norm in self.target[item]:
            x, y = min(int(x_norm * self.frame_w + 0.5), self.frame_w - 1), min(int(y_norm * self.frame_h + 0.5), self.frame_h - 1)
            target[y, x] = 10
        kernel = self._gen_gaussian_kernel()
        # print(kernel.max())
        if use_cuda:
            target_map = spherical_conv(
                th.from_numpy(
                    target.reshape(1, 1, *target.shape)
                ).cuda(),
                th.from_numpy(kernel.reshape(1, 1, *kernel.shape)).cuda(),
                kernel_rad=self.kernel_rad,
                padding_mode=0
            ).view(1, self.frame_h, self.frame_w)
        else:
            target_map = spherical_conv(
                th.from_numpy(
                    target.reshape(1, 1, *target.shape)
                ),
                th.from_numpy(kernel.reshape(1, 1, *kernel.shape)),
                kernel_rad=self.kernel_rad,
                padding_mode=0
            ).view(1, self.frame_h, self.frame_w)
        if item >= 0 and self.cache_gt:
            np.save(cfile, target_map.data.cpu().numpy() / len(self.target[item]))

        return target_map.data.float() / len(self.target[item])

    def _get_salency_map(self, item, use_cuda=False):
        cfile = self.data[item][:-4] + '_gt.npy'
        if item >= 0:
            if self.cache_gt and os.path.isfile(cfile):
                target_map = th.from_numpy(np.load(cfile)).float()
                target_map = self.downsample(target_map)
                return target_map.float()
                
                assert target_map.size() == (1, self.frame_h, self.frame_w)
                return th.from_numpy(np.load(cfile)).float()
        target = np.zeros((self.frame_h, self.frame_w))
        for x_norm, y_norm in self.target[item]:
            x, y = min(int(x_norm * self.frame_w + 0.5), self.frame_w - 1), min(int(y_norm * self.frame_h + 0.5), self.frame_h - 1)
            target[y, x] = 10
        kernel = self._gen_gaussian_kernel()
        # print(kernel.max())
        if use_cuda:
            target_map = spherical_conv(
                th.from_numpy(
                    target.reshape(1, 1, *target.shape)
                ).cuda(),
                th.from_numpy(kernel.reshape(1, 1, *kernel.shape)).cuda(),
                kernel_rad=self.kernel_rad,
                padding_mode=0
            ).view(1, self.frame_h, self.frame_w)
        else:
            target_map = spherical_conv(
                th.from_numpy(
                    target.reshape(1, 1, *target.shape)
                ),
                th.from_numpy(kernel.reshape(1, 1, *kernel.shape)),
                kernel_rad=self.kernel_rad,
                padding_mode=0
            ).view(1, self.frame_h, self.frame_w)
        if item >= 0 and self.cache_gt:
            np.save(cfile, target_map.data.cpu().numpy() / len(self.target[item]))

        return target_map.data.float() / len(self.target[item])

    def _gen_gaussian_kernel(self):
        sigma = self.gaussian_sigma
        kernel = th.zeros(self.kernel_size)
        delta_theta = self.kernel_rad / (self.kernel_size[0] - 1)
        sigma_idx = sigma / delta_theta
        gauss1d = signal.gaussian(2 * kernel.shape[0], sigma_idx)
        gauss2d = np.outer(gauss1d, np.ones(kernel.shape[1]))

        return gauss2d[-kernel.shape[0]:, :]

    def clear_cache(self):
        from tqdm import trange
        for item in trange(len(self), desc='cleaning'):
            cfile = self.data[item][:-4] + '_gt.npy'
            if os.path.isfile(cfile):
                print('remove {}'.format(cfile))
                os.remove(cfile)

        return self

    def cache_map(self):
        from tqdm import trange
        cache_gt = self.cache_gt
        self.cache_gt = True
        for item in trange(len(self), desc='caching'):

            # pool.apply_async(self._get_salency_map, (item, True))
            self._get_salency_map(item, use_cuda=True)
        self.cache_gt = cache_gt

        return self


class VRVideoLSTM(data.Dataset):
    def __init__(self, root, frame_h, frame_w, video_train, sequence_len = 2, frame_interval=5, transform=None, train=True, sal_path = 'X',
                 gaussian_sigma=np.pi / 20, kernel_rad=np.pi/7, kernel_size=(30, 60), cache_gt=True, rnd_seed=367643):
        self.frame_interval = frame_interval
        self.transform = transform
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.gaussian_sigma = gaussian_sigma
        self.kernel_size = kernel_size
        self.kernel_rad = kernel_rad
        self.cache_gt = cache_gt
        self.train = train
        self.sequence_len = sequence_len

        rnd = Random(rnd_seed)

        # load target
        self.vinfo = pickle.load(open(os.path.join(root, 'vinfo.pkl'), 'rb'))

        # load image paths
        vset = list()
        for vid in tqdm(os.listdir(root), desc='scanning dir'):
            if os.path.isdir(os.path.join(root, vid)):
                vset.append(vid)
        vset.sort()
        assert set(self.vinfo.keys()) == set(vset)
        print('{} videos found.'.format(len(vset)))
        if isinstance(video_train, numbers.Integral):
            vset_train = set(rnd.sample(vset, k=video_train))
            vset_val = set(vset) - vset_train
        else:
            raise NotImplementedError()
        print('{}:{} videos chosen for training:testing.'.format(len(vset_train), len(vset_val)))
        # print('test videos: {}'.format(vset_val))

        vset = vset_train if train else vset_val
        self.data = []
        self.data_sal = []
        self.target = []
        self.i2v = {}
        self.v2i = {}
        for vid in vset:
            obj_path = os.path.join(root, vid)
            sal_video_path = os.path.join(sal_path, vid)
            # fcnt = 0
            frame_list = [frame for frame in os.listdir(obj_path) if frame.endswith('.jpg')]
            frame_list.sort()
            for frame in frame_list:
                fid = frame[:-4]
                # fcnt += 1
                # if fcnt >= frame_interval:
                self.i2v[len(self.data)] = (vid, fid)
                self.v2i[(vid, fid)] = len(self.data)
                self.data.append(os.path.join(obj_path, frame))
                self.target.append(self.vinfo[vid][fid])
                self.data_sal.append(os.path.join(sal_video_path, frame))
                    # fcnt = 0

        self.target.append([(0.5, 0.5)])

        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        if opt.use_equirect_rotate:
            self.rot_transform = tf.Compose([
                tf.Resize((64, 128))
            ])

            self.y_z_list = {}
            self.y_z_list[0] = (0, 60)
            self.y_z_list[1] = (60, 0)
            self.y_z_list[2] = (60, 60)

    def __getitem__(self, item):
        img_list = []
        last_list = []
        target_list = []

        if item-self.sequence_len+1<0:
            item = self.sequence_len - 1

        if opt.use_equirect_rotate:
            rot_x, isInverse = 0, False
            if self.train:
                idx = np.random.randint(0, 3)
                rot_y, rot_z = self.y_z_list[idx]
                #rot_y, rot_z = np.random.randint(-80, 80), np.random.randint(-180, 180)
            else:
                rot_y, rot_z = opt.rot_y, opt.rot_z

        for idx in range(item-self.sequence_len+1, item+1, 1):
            #print(idx, item)
            img = Image.open(open(self.data[idx], 'rb'))
            # img = img.resize((self.frame_w, self.frame_h))
            if self.transform:


                #img = self.transform(img)

                if opt.use_equirect_rotate:
                    img = self.rot_transform(img)
                    img = np.array(img)

                    img = equirect_rotate.Equirect_Rotate(img, rot_x, rot_y, rot_z, isInverse)
                    img = torch.from_numpy(img).float()
                    img = img.permute(2, 0, 1)
                else:
                    img = self.transform(img)

            else:
                img = np.array(img)

            img_list.append(img)

            vid, fid = self.i2v[idx]
            if int(fid) - self.frame_interval <= 0:
                last = self._get_pred_salency_map(-1)
            else:
                last = self._get_pred_salency_map(self.v2i[(vid, '%04d' % (int(fid) - self.frame_interval))])

            target = self._get_salency_map(idx)

            if opt.conv_type == 'sphereconv' or opt.conv_type == 'standard' or opt.conv_type == 'gafilters':
                last = (last - last.min() )/ (last.max() - last.min())
                target = (target - target.min()) / ( target.max() - target.min() )

            if opt.use_equirect_rotate:
                last, target = last.permute(1, 2, 0).numpy(), target.permute(1, 2, 0).numpy()

                # print(last.shape, img.shape)
                last = equirect_rotate.Equirect_Rotate(last, rot_x, rot_y, rot_z, isInverse)
                target = equirect_rotate.Equirect_Rotate(target, rot_x, rot_y, rot_z, isInverse)

                last = torch.from_numpy(last).float().permute(2, 0, 1)
                target = torch.from_numpy(target).float().permute(2, 0, 1)


            last_list.append(last)
            target_list.append(target)

        img_var = torch.stack(img_list, 0)
        last_var = torch.stack(last_list, 0)
        target_var = torch.stack(target_list, 0)
        #exit()
        if self.train:
            return img_var, last_var, target_var
        else:
            return img_var, self.data[item], last_var, target_var, vid, fid

    def __len__(self):
        return len(self.data)


    def _get_pred_salency_map(self, item, use_cuda=False):
        cfile = self.data_sal[item][:-4] + '.bin'

        if item >= 0:
            if os.path.isfile(cfile):
                size = os.path.getsize(cfile)
                target = np.fromfile(cfile, dtype=np.float32)
                #print(target.shape)
                target.shape = 1,64,128
                #print(target.shape)
                target = target/target.max()#*255
                #target_map = th.from_numpy(np.load(cfile)).float()
                #target_map = self.downsample(target_map)
                return torch.from_numpy(target).float()
        target = np.zeros((self.frame_h, self.frame_w))
        for x_norm, y_norm in self.target[item]:
            x, y = min(int(x_norm * self.frame_w + 0.5), self.frame_w - 1), min(int(y_norm * self.frame_h + 0.5), self.frame_h - 1)
            target[y, x] = 10
        kernel = self._gen_gaussian_kernel()
        # print(kernel.max())
        if use_cuda:
            target_map = spherical_conv(
                th.from_numpy(
                    target.reshape(1, 1, *target.shape)
                ).cuda(),
                th.from_numpy(kernel.reshape(1, 1, *kernel.shape)).cuda(),
                kernel_rad=self.kernel_rad,
                padding_mode=0
            ).view(1, self.frame_h, self.frame_w)
        else:
            target_map = spherical_conv(
                th.from_numpy(
                    target.reshape(1, 1, *target.shape)
                ),
                th.from_numpy(kernel.reshape(1, 1, *kernel.shape)),
                kernel_rad=self.kernel_rad,
                padding_mode=0
            ).view(1, self.frame_h, self.frame_w)
        if item >= 0 and self.cache_gt:
            np.save(cfile, target_map.data.cpu().numpy() / len(self.target[item]))

        return target_map.data.float() / len(self.target[item])

    def _get_salency_map(self, item, use_cuda=False):
        cfile = self.data[item][:-4] + '_gt.npy'
        if item >= 0:
            if self.cache_gt and os.path.isfile(cfile):
                target_map = th.from_numpy(np.load(cfile)).float()
                if target_map.size(1) == self.frame_h:
                    return th.from_numpy(np.load(cfile)).float()
                else:
                    target_map = self.downsample(target_map)
                    return target_map.float()
                assert target_map.size() == (1, self.frame_h, self.frame_w)
                return th.from_numpy(np.load(cfile)).float()
        target = np.zeros((self.frame_h, self.frame_w))
        for x_norm, y_norm in self.target[item]:
            x, y = min(int(x_norm * self.frame_w + 0.5), self.frame_w - 1), min(int(y_norm * self.frame_h + 0.5), self.frame_h - 1)
            target[y, x] = 10
        kernel = self._gen_gaussian_kernel()
        # print(kernel.max())
        if use_cuda:
            target_map = spherical_conv(
                th.from_numpy(
                    target.reshape(1, 1, *target.shape)
                ).cuda(),
                th.from_numpy(kernel.reshape(1, 1, *kernel.shape)).cuda(),
                kernel_rad=self.kernel_rad,
                padding_mode=0
            ).view(1, self.frame_h, self.frame_w)
        else:
            target_map = spherical_conv(
                th.from_numpy(
                    target.reshape(1, 1, *target.shape)
                ),
                th.from_numpy(kernel.reshape(1, 1, *kernel.shape)),
                kernel_rad=self.kernel_rad,
                padding_mode=0
            ).view(1, self.frame_h, self.frame_w)
        if item >= 0 and self.cache_gt:
            np.save(cfile, target_map.data.cpu().numpy() / len(self.target[item]))

        return target_map.data.float() / len(self.target[item])

    def _gen_gaussian_kernel(self):
        sigma = self.gaussian_sigma
        kernel = th.zeros(self.kernel_size)
        delta_theta = self.kernel_rad / (self.kernel_size[0] - 1)
        sigma_idx = sigma / delta_theta
        gauss1d = signal.gaussian(2 * kernel.shape[0], sigma_idx)
        gauss2d = np.outer(gauss1d, np.ones(kernel.shape[1]))

        return gauss2d[-kernel.shape[0]:, :]

    def clear_cache(self):
        from tqdm import trange
        for item in trange(len(self), desc='cleaning'):
            cfile = self.data[item][:-4] + '_gt.npy'
            if os.path.isfile(cfile):
                print('remove {}'.format(cfile))
                os.remove(cfile)

        return self

    def cache_map(self):
        from tqdm import trange
        cache_gt = self.cache_gt
        self.cache_gt = True
        for item in trange(len(self), desc='caching'):

            # pool.apply_async(self._get_salency_map, (item, True))
            self._get_salency_map(item, use_cuda=True)
        self.cache_gt = cache_gt

        return self

