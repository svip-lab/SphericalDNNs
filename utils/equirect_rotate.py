import sys
import numpy as np
import math


def getRotMatrix(rot_x, rot_y, rot_z, unit='degree'):
    if (unit == 'degree'):
        rot_x = np.deg2rad(rot_x)
        rot_y = np.deg2rad(rot_y)
        rot_z = np.deg2rad(rot_z)

    RotMat_X = np.array([[1, 0, 0],
                         [0, math.cos(rot_x), -math.sin(rot_x)],
                         [0, math.sin(rot_x), math.cos(rot_x)]])
    RotMat_Y = np.array([[math.cos(rot_y), 0, math.sin(rot_y)],
                         [0, 1, 0],
                         [-math.sin(rot_y), 0, math.cos(rot_y)]])
    RotMat_Z = np.array([[math.cos(rot_z), -math.sin(rot_z), 0],
                         [math.sin(rot_z), math.cos(rot_z), 0],
                         [0, 0, 1]])
    return np.matmul(np.matmul(RotMat_X, RotMat_Y), RotMat_Z)


def Pixel2LonLat(equirect):
    # LongLat - shape = (N, 2N, (Long, Lat))
    Lon = np.array([2 * (x / equirect.shape[1] - 0.5) * np.pi for x in range(equirect.shape[1])])
    Lat = np.array([(0.5 - y / equirect.shape[0]) * np.pi for y in range(equirect.shape[0])])

    Lon = np.tile(Lon, (equirect.shape[0], 1))
    Lat = np.tile(Lat.reshape(equirect.shape[0], 1), (equirect.shape[1]))

    return np.dstack((Lon, Lat))


def LonLat2Sphere(LonLat):
    x = np.cos(LonLat[:, :, 1]) * np.cos(LonLat[:, :, 0])
    y = np.cos(LonLat[:, :, 1]) * np.sin(LonLat[:, :, 0])
    z = np.sin(LonLat[:, :, 1])

    return np.dstack((x, y, z))


def Sphere2LonLat(xyz):
    Lon = np.arctan2(xyz[:, :, 1], xyz[:, :, 0])
    Lat = np.pi / 2 - np.arccos(xyz[:, :, 2])

    return np.dstack((Lon, Lat))


def LonLat2Pixel(LonLat):
    width = LonLat.shape[1]
    height = LonLat.shape[0]
    j = (width * (LonLat[:, :, 0] / (2 * np.pi) + 0.5)) % width
    i = (height * (0.5 - (LonLat[:, :, 1] / np.pi))) % height

    return np.dstack((i, j)).astype('int')


def isEquirect(height, width):
    if (height * 2 != width):
        print("Warning: Source Image is not an Equirectangular Image...")
        print("height is %d, width is %d" % (height, width))
        return False
    return True


def Equirect_Rotate(src_img, rot_x, rot_y, rot_z, isInverse=False, unit='degree'):
    height = src_img.shape[0]
    width = src_img.shape[1]
    if (not isEquirect(height, width)):
        print("End program...")
        return

    Rot_Matrix = getRotMatrix(rot_x, rot_y, rot_z, unit)
    if (isInverse):
        Rot_Matrix = np.transpose(Rot_Matrix)

    out_img = np.zeros_like(src_img)

    # mapping equirect coordinate into LonLat coordinate system
    out_LonLat = Pixel2LonLat(out_img)

    # mapping LonLat coordinate into xyz(sphere) coordinate system
    out_xyz = LonLat2Sphere(out_LonLat)

    src_xyz = np.zeros_like(out_xyz)
    Rt = np.transpose(Rot_Matrix)
    for i in range(height):
        for j in range(width):
            src_xyz[i][j] = np.matmul(Rt, out_xyz[i][j])

    # mapping xyz(sphere) coordinate into LonLat Coordinate system
    src_LonLat = Sphere2LonLat(src_xyz)

    # mapping LonLat coordinate into equirect coordinate system
    src_Pixel = LonLat2Pixel(src_LonLat)

    for i in range(height):
        for j in range(width):
            pixel = src_Pixel[i][j]
            out_img[i][j] = src_img[pixel[0]][pixel[1]]

    return out_img