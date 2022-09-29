import pdb
import numpy as np
from numpy import sin, cos, tan, pi, arcsin, arctan
from functools import lru_cache
import torch
from torch import nn
from torch.nn.parameter import Parameter


# Calculate kernels of 3*3 SphereCNN
@lru_cache(None)
def get_xy3by3(delta_phi, delta_theta):
    return np.array([
        [
            (-tan(delta_theta), 1/cos(delta_theta)*tan(delta_phi)),
            (0, tan(delta_phi)),
            (tan(delta_theta), 1/cos(delta_theta)*tan(delta_phi)),
        ],
        [
            (-tan(delta_theta), 0),
            (1, 1),
            (tan(delta_theta), 0),
        ],
        [
            (-tan(delta_theta), -1/cos(delta_theta)*tan(delta_phi)),
            (0, -tan(delta_phi)),
            (tan(delta_theta), -1/cos(delta_theta)*tan(delta_phi)),
        ]
    ])


# Calculate kernels of 5*5 SphereCNN
@lru_cache(None)
def get_xy5by5(delta_phi, delta_theta):
    return np.array([
        # row 1
        [
            (-tan(2*delta_theta), 1/cos(2*delta_theta)*tan(2*delta_phi)),
            (-tan(  delta_theta), 1/cos(  delta_theta)*tan(2*delta_phi)),
            (                  0,                      tan(2*delta_phi)),
            ( tan(  delta_theta), 1/cos(  delta_theta)*tan(2*delta_phi)),
            ( tan(2*delta_theta), 1/cos(2*delta_theta)*tan(2*delta_phi)),
        ],
        # row 2
        [
            (-tan(2*delta_theta), 1/cos(2*delta_theta)*tan(delta_phi)),
            (-tan(  delta_theta), 1/cos(  delta_theta)*tan(delta_phi)),
            (                  0,                      tan(delta_phi)),
            ( tan(  delta_theta), 1/cos(  delta_theta)*tan(delta_phi)),
            ( tan(2*delta_theta), 1/cos(2*delta_theta)*tan(delta_phi)),
        ],
        # row 3
        [
            (-tan(2*delta_theta), 0),
            (-tan(  delta_theta), 0),
            (1,1),
            ( tan(  delta_theta), 0),
            ( tan(2*delta_theta), 0),
        ],
        # row 4
        [
            (-tan(2*delta_theta), -1/cos(2*delta_theta)*tan(delta_phi)),
            (-tan(  delta_theta), -1/cos(  delta_theta)*tan(delta_phi)),
            (                  0,                      -tan(delta_phi)),
            ( tan(  delta_theta), -1/cos(  delta_theta)*tan(delta_phi)),
            ( tan(2*delta_theta), -1/cos(2*delta_theta)*tan(delta_phi)),
        ],
        # row 5
        [
            (-tan(2*delta_theta), -1/cos(2*delta_theta)*tan(2*delta_phi)),
            (-tan(  delta_theta), -1/cos(  delta_theta)*tan(2*delta_phi)),
            (                  0,                      -tan(2*delta_phi)),
            ( tan(  delta_theta), -1/cos(  delta_theta)*tan(2*delta_phi)),
            ( tan(2*delta_theta), -1/cos(2*delta_theta)*tan(2*delta_phi)),
        ],
    ])


# Calculate kernels of 7*7 SphereCNN
@lru_cache(None)
def get_xy7by7(delta_phi, delta_theta):
    return np.array([
        # row 1
        [
            (-tan(3*delta_theta), 1/cos(3*delta_theta)*tan(3*delta_phi)),
            (-tan(2*delta_theta), 1/cos(2*delta_theta)*tan(3*delta_phi)),
            (-tan(  delta_theta), 1/cos(  delta_theta)*tan(3*delta_phi)),
            (                  0,                      tan(3*delta_phi)),
            ( tan(  delta_theta), 1/cos(  delta_theta)*tan(3*delta_phi)),
            ( tan(2*delta_theta), 1/cos(2*delta_theta)*tan(3*delta_phi)),
            ( tan(3*delta_theta), 1/cos(3*delta_theta)*tan(3*delta_phi)),
        ],
        # row 2
        [
            (-tan(3*delta_theta), 1/cos(3*delta_theta)*tan(2*delta_phi)),
            (-tan(2*delta_theta), 1/cos(2*delta_theta)*tan(2*delta_phi)),
            (-tan(  delta_theta), 1/cos(  delta_theta)*tan(2*delta_phi)),
            (                  0,                      tan(2*delta_phi)),
            ( tan(  delta_theta), 1/cos(  delta_theta)*tan(2*delta_phi)),
            ( tan(2*delta_theta), 1/cos(2*delta_theta)*tan(2*delta_phi)),
            ( tan(3*delta_theta), 1/cos(3*delta_theta)*tan(2*delta_phi)),
        ],
        # row 3
        [
            (-tan(3*delta_theta), 1/cos(3*delta_theta)*tan(delta_phi)),
            (-tan(2*delta_theta), 1/cos(2*delta_theta)*tan(delta_phi)),
            (-tan(  delta_theta), 1/cos(  delta_theta)*tan(delta_phi)),
            (                  0,                      tan(delta_phi)),
            ( tan(  delta_theta), 1/cos(  delta_theta)*tan(delta_phi)),
            ( tan(2*delta_theta), 1/cos(2*delta_theta)*tan(delta_phi)),
            ( tan(3*delta_theta), 1/cos(3*delta_theta)*tan(delta_phi)),
        ],
        # row 4
        [
            (-tan(3*delta_theta), 0),
            (-tan(2*delta_theta), 0),
            (-tan(  delta_theta), 0),
            (1,1),
            ( tan(  delta_theta), 0),
            ( tan(2*delta_theta), 0),
            ( tan(3*delta_theta), 0),
        ],
        # row 5
        [
            (-tan(3*delta_theta), -1/cos(3*delta_theta)*tan(delta_phi)),
            (-tan(2*delta_theta), -1/cos(2*delta_theta)*tan(delta_phi)),
            (-tan(  delta_theta), -1/cos(  delta_theta)*tan(delta_phi)),
            (                  0,                      -tan(delta_phi)),
            ( tan(  delta_theta), -1/cos(  delta_theta)*tan(delta_phi)),
            ( tan(2*delta_theta), -1/cos(2*delta_theta)*tan(delta_phi)),
            ( tan(3*delta_theta), -1/cos(3*delta_theta)*tan(delta_phi)),
        ],
        # row 6
        [
            (-tan(3*delta_theta), -1/cos(3*delta_theta)*tan(2*delta_phi)),
            (-tan(2*delta_theta), -1/cos(2*delta_theta)*tan(2*delta_phi)),
            (-tan(  delta_theta), -1/cos(  delta_theta)*tan(2*delta_phi)),
            (                  0,                      -tan(2*delta_phi)),
            ( tan(  delta_theta), -1/cos(  delta_theta)*tan(2*delta_phi)),
            ( tan(2*delta_theta), -1/cos(2*delta_theta)*tan(2*delta_phi)),
            ( tan(3*delta_theta), -1/cos(3*delta_theta)*tan(2*delta_phi)),
        ],
        # row 7
        [
            (-tan(3*delta_theta), -1/cos(3*delta_theta)*tan(3*delta_phi)),
            (-tan(2*delta_theta), -1/cos(2*delta_theta)*tan(3*delta_phi)),
            (-tan(  delta_theta), -1/cos(  delta_theta)*tan(3*delta_phi)),
            (                  0,                      -tan(3*delta_phi)),
            ( tan(  delta_theta), -1/cos(  delta_theta)*tan(3*delta_phi)),
            ( tan(2*delta_theta), -1/cos(2*delta_theta)*tan(3*delta_phi)),
            ( tan(3*delta_theta), -1/cos(3*delta_theta)*tan(3*delta_phi)),
        ],
    ])


@lru_cache(None)
def cal_index(ks, h, w, img_r, img_c):
    '''
        Calculate Kernel Sampling Pattern
        only support 3x3 filter
        return 9 locations: (3, 3, 2)
    '''
    # pixel -> rad
    phi = -((img_r+0.5)/h*pi - pi/2)
    theta = (img_c+0.5)/w*2*pi-pi

    delta_phi = pi/h
    delta_theta = 2*pi/w

    #print(ks)

    if ks == 3:
        xys = get_xy3by3(delta_phi, delta_theta)
    elif ks == 5:
        xys = get_xy5by5(delta_phi, delta_theta) 
    elif ks == 7:
        xys = get_xy7by7(delta_phi, delta_theta)
    else:
        raise Exception('Wrong Kernel Shape')

    x = xys[..., 0]
    y = xys[..., 1]
    rho = np.sqrt(x**2+y**2)
    v = arctan(rho)
    new_phi= arcsin(cos(v)*sin(phi) + y*sin(v)*cos(phi)/rho)
    new_theta = theta + arctan(x*sin(v) / (rho*cos(phi)*cos(v) - y*sin(phi)*sin(v)))
    # rad -> pixel
    new_r = (-new_phi+pi/2)*h/pi - 0.5
    new_c = (new_theta+pi)*w/2/pi - 0.5

    # indexs out of image, equirectangular leftmost and rightmost pixel is adjacent
    new_c = (new_c + w) % w
    new_result = np.stack([new_r, new_c], axis=-1)
    
    if ks == 3:
        new_result[1, 1] = (img_r, img_c)
    elif ks == 5:
        new_result[2, 2] = (img_r, img_c)
    elif ks == 7:
        new_result[3, 3] = (img_r, img_c)

    return new_result


@lru_cache(None)
def _gen_filters_coordinates(ks, h, w, stride):
    co = np.array([[cal_index(ks, h, w, i, j) for j in range(0, w, stride)] for i in range(0, h, stride)])
    return np.ascontiguousarray(co.transpose([4, 0, 1, 2, 3]))


def gen_filters_coordinates(ks, h, w, stride=1):
    '''
    return np array of kernel lo (2, H/stride, W/stride, kernel_size, kernel_size)
    '''
    assert(isinstance(h, int) and isinstance(w, int) and isinstance(ks, int))
    return _gen_filters_coordinates(ks, h, w, stride).copy()


def gen_grid_coordinates(ks, h, w, stride=1):
    coordinates = gen_filters_coordinates(ks, h, w, stride).copy()
    coordinates[0] = (coordinates[0] * 2 / h) - 1
    coordinates[1] = (coordinates[1] * 2 / w) - 1  
    coordinates = coordinates[::-1] 
    coordinates = coordinates.transpose(1, 3, 2, 4, 0)
    sz = coordinates.shape
    coordinates = coordinates.reshape(1, sz[0]*sz[1], sz[2]*sz[3], sz[4])
    return coordinates.copy()


class SphereConv2d(nn.Module):
    '''  SphereConv2D
    Note that this layer only support 3x3 filter
    '''
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, bias=True, mode='bilinear'):
        super(SphereConv2d, self).__init__()
        # Note that padding has no effect here
        # just keep it for a similar interface
       
        self.kernel_size = 3#kernel_size
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.mode = mode
        self.weight = Parameter(torch.Tensor(out_c, in_c, kernel_size, kernel_size))
        self.bs = bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_c))
        else:
            self.register_parameter('bias', None)
        self.grid_shape = None
        self.grid = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bs: 
            self.bias.data.zero_()

    def forward(self, x):
        if self.grid_shape is None or self.grid_shape != tuple(x.shape[2:4]):
            self.grid_shape = tuple(x.shape[2:4])
            coordinates = gen_grid_coordinates(self.kernel_size, x.shape[2], x.shape[3], self.stride)
            with torch.no_grad():
                self.grid = torch.FloatTensor(coordinates).to(x.device)
                self.grid.requires_grad = True

        with torch.no_grad():
            grid = self.grid.repeat(x.shape[0], 1, 1, 1)

        x = nn.functional.grid_sample(x, grid, mode=self.mode)
        x = nn.functional.conv2d(x, self.weight, self.bias, stride=self.kernel_size)
        return x


class SphereMaxPool2D(nn.Module):
    '''  SphereMaxPool2D
    Note that this layer only support 3x3 filter
    '''
    def __init__(self, stride=1, mode='bilinear'):
        super(SphereMaxPool2D, self).__init__()
        self.stride = stride
        self.mode = mode
        self.grid_shape = None
        self.grid = None
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)

    def forward(self, x):
        if self.grid_shape is None or self.grid_shape != tuple(x.shape[2:4]):
            self.grid_shape = tuple(x.shape[2:4])
            coordinates = gen_grid_coordinates(3, x.shape[2], x.shape[3], self.stride)
            with torch.no_grad():
                self.grid = torch.FloatTensor(coordinates).to(x.device)
                self.grid.requires_grad = True

        with torch.no_grad():
            grid = self.grid.repeat(x.shape[0], 1, 1, 1)

        return self.pool(nn.functional.grid_sample(x, grid, mode=self.mode))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    # test cnn
    cnn = SphereConv2D(3, 5, 1)
    out = cnn(torch.randn(2, 3, 10, 10))
    print('SphereConv2D(3, 5, 1) output shape: ', out.size())
    # test pool
    # create sample image
    h, w = 100, 200
    img = np.ones([h, w, 3])
    for r in range(h):
        for c in range(w):
            img[r, c, 0] = img[r, c, 0] - r/h
            img[r, c, 1] = img[r, c, 1] - c/w
    plt.imsave('demo_original', img)
    img = img.transpose([2, 0, 1])
    img = np.expand_dims(img, 0)  # (B, C, H, W)
    # pool
    pool = SphereMaxPool2D(1)
    out = pool(torch.from_numpy(img).float())
    out = np.squeeze(out.numpy(), 0).transpose([1, 2, 0])
    plt.imsave('demo_pool_1.png', out)
    print('Save image after pooling with stride 1: demo_pool_1.png')
    # pool with tride 3
    pool = SphereMaxPool2D(3)
    out = pool(torch.from_numpy(img).float())
    out = np.squeeze(out.numpy(), 0).transpose([1, 2, 0])
    plt.imsave('demo_pool_3.png', out)
    print('Save image after pooling with stride 3: demo_pool_3.png')