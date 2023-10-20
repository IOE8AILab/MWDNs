from torch.utils.data import Dataset, DataLoader
import h5py
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import os, torch
import matplotlib
import cv2.cv2


class H5Dataset(Dataset):

    def __init__(self, gt, blur, init_num, num):
        self.gt = gt
        self.blur = blur
        self.num = num
        self.init_num = init_num

    def __getitem__(self, index):
        data_gt = self.gt[self.init_num + index].astype(np.float32) / 255.
        data_blur = self.blur[self.init_num + index].astype(np.float32) / 255.
        return data_gt, data_blur

    def __len__(self):
        return self.num


def load_data(path, total_num, batch_size):
    f = h5py.File(path, 'r')
    gt = f['gt']
    blur = f['blur']
    test_set = H5Dataset(gt, blur, 0, int(total_num*0.1))
    val_set = H5Dataset(gt, blur, int(total_num*0.1), int(total_num*0.1))
    train_set = H5Dataset(gt, blur, int(total_num*0.2), int(total_num*0.8))
    train_data = DataLoader(dataset=train_set, num_workers=4, batch_size=batch_size, pin_memory=True, shuffle=True)
    val_data = DataLoader(dataset=val_set, num_workers=4, batch_size=1, pin_memory=True, shuffle=False)
    test_data = DataLoader(dataset=test_set, num_workers=4, batch_size=1, pin_memory=True, shuffle=False)
    return train_data, val_data, test_data


def show_img(img, mode='a'):
    img = img.data.cpu().numpy()
    img = np.flip(img, axis=1)
    fig, ax = plt.subplots(1, len(img), figsize=(15, 10))
    for i in range(len(img)):
        ax[i].imshow(img[i].transpose(1, 2, 0))
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.04)
    path = './images/' + mode + '.png'
    plt.savefig(path, bbox_inches='tight', dpi=300)

