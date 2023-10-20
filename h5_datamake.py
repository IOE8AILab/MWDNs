import h5py
import numpy as np
import scipy.io
import os, sys
import cv2.cv2
import matplotlib.pyplot as plt
from PIL import Image


def save_h5(num, str_):
    batchlen = 1
    imgsize1 = 320
    imgsize2 = 320
    if num == 0:
        h5f = h5py.File('/media/gym/HDD3/LY_Data/lensless_320/lensless_320.h5', 'w')
        gt = h5f.create_dataset(name="gt",
                                     shape=(batchlen, 3, imgsize1, imgsize2),
                                     maxshape=(None, 3, imgsize1, imgsize2),
                                     dtype='uint8')
        blur = h5f.create_dataset(name="blur",
                                shape=(batchlen, 3, imgsize1, imgsize2),
                                maxshape=(None, 3, imgsize1, imgsize2),
                                dtype='uint8')

    else:
        h5f = h5py.File('/media/gym/HDD3/LY_Data/lensless_320/lensless_320.h5', 'a')
        gt = h5f['gt']
        blur = h5f['blur']

    str_gt = str_ + 'gt/' + str(num) + '.png'
    str_blur = str_ + 'blur/' + str(num) + '.png'
    images = cv2.cv2.imread(str_gt, -1).astype(np.uint8).transpose(2, 0, 1)     # 3,137,137
    images = np.pad(images, ((0, 0), (94, 89), (94, 89)), 'constant')
    blurs = cv2.cv2.imread(str_blur, -1).astype(np.uint8).transpose(2, 0, 1)
    images = np.expand_dims(images, axis=0)
    blurs = np.expand_dims(blurs, axis=0)

    gt.resize([batchlen * num + batchlen, 3, imgsize1, imgsize2])
    gt[batchlen * num:batchlen * num + batchlen] = images
    blur.resize([batchlen * num + batchlen, 3, imgsize1, imgsize2])
    blur[batchlen * num:batchlen * num + batchlen] = blurs
    
    h5f.close()


if __name__ == '__main__':

    path = '/media/gym/HDD3/LY_Data/lensless_320/'

    for i in range(25000):
        if i % 200 == 0:
            print(i)
        save_h5(i, path)
