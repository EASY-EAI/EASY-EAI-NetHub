import os
import sys

realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:-1]))
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('EASY_EAI_nethub')+1]))

# import cv2
# from PIL import Image
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
# import torchvision.transforms as transforms

from module_lib.image_process_tool.mnist_parser import load_mnist


class mnist_dataset(object):
    """docstring for mnist_dataset"""
    def __init__(self, img_set, label_set):
        super(mnist_dataset, self).__init__()
        self.imgs = img_set
        self.labels = label_set

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]

        img = self.transform_img(img)

        return img, label

    def transform_img(self, img):
        _img = np.reshape(img, (1, 28, 28))
        _img = torch.tensor(_img)
        return _img

    def transform_label(self, label):
        return torch.tensor(label).long()


def get_mnist_loader_from_path(train_image_file,
                               train_label_file,
                               test_image_file,
                               test_label_file,
                               batch_size):
    
    result = load_mnist(train_image_file, train_label_file, test_image_file, test_label_file, normalize=True, one_hot=False)

    train_dataset = mnist_dataset(result[0][0], result[0][1])
    val_dataset = mnist_dataset(result[1][0], result[1][1])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=0, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=0, drop_last=True)

    return train_loader, val_loader