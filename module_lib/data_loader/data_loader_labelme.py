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

from tools.pixel_label_tools import Pixellabel_container, get_color_class


def search_file_in_folder(path, file_format_list, container):
    file_list = os.listdir(path)
    for file in file_list:
        file_true_path = os.path.join(path, file)
        if os.path.isdir(file_true_path):
            search_file_in_folder(file_true_path, file_format_list, container)
        else:
            _format = file.split('.')[-1]
            if _format in file_format_list:
                container.append(file_true_path)


def scan_file_from_folder(path_list, file_format_list):
    path_container = []
    for path in path_list:
        search_file_in_folder(path, file_format_list, path_container)

    return path_container


def split_dataset(dataset,split_rate,shuffle=True, random_seed=1024):
    '''
        return index of train&val, 
    '''
    indices = list(range(len(dataset)))
    split = int(np.floor(split_rate*len(dataset)))

    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices,val_indices = indices[split:],indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    return train_sampler, val_sampler


class labelme_dataset(object):
    """docstring for labelme_dataset"""
    def __init__(self,
                 data_dir_path,
                 img_format='jpg',
                 down_sample=False,
                 random_transform=True,
                 mosaic_transform=True,
                 class_color=None,
                 resize_rate=(1/4, 1/4),
                 output_size=(216, 384)):

        self.img_file_list = []
        self.label_file_list = []
        self.down_sample = down_sample
        self.mosaic_transform = mosaic_transform
        assert class_color is not None, 'class_color should not be None, check it out'
        self.class_color = class_color
        self.resize_rate = resize_rate
        self.output_size = tuple(output_size)

        all_path = scan_file_from_folder([data_dir_path], ['png'])
        print(data_dir_path)

        for i in range(len(all_path)):
            path = all_path[i]
            path_element = path.split('\\')
            if path_element[-2] == 'SegmentationClassPNG':
                self.label_file_list.append(path)
                path_element[-2] = 'JPEGImages'
                path_element[-1] = path_element[-1].rstrip('png')+img_format
                self.img_file_list.append('\\'.join(path_element))

    def __getitem__(self, index):
        img_name = self.img_file_list[index]
        gt_name = self.label_file_list[index]

        init_container = Pixellabel_container(None, None, 'color_map', None, None, class_color=self.class_color)
        init_container._load_file(img_name, method='label_png', annotation_file_path=gt_name, total_class_number=len(self.class_color))

        init_container.resize((int(init_container.img.shape[0]*self.resize_rate[0]), int(init_container.img.shape[1]*self.resize_rate[1])))
        init_container.random_crop(self.output_size, padding=False, keep_iou_rate=0.6, keep_id_list=None, tolerant_step=10)
        init_container.random_transform()
        # init_container._show_img()

        image_copy = init_container.img.copy()

        if self.down_sample is True:
            init_container.resize((int(init_container.img.shape[0]/8), int(init_container.img.shape[1]/8)))
            # init_container._show_img()

        label = init_container.onehot_label_with_background().astype('uint8')
        # init_container._show_img()

        image = np.transpose(image_copy, (2, 0, 1))
        label = np.transpose(label, (2, 0, 1))

        image = image/255.0

        image = torch.tensor(image).float()
        label = torch.tensor(label).float()

        # print('image.shape', image.shape)
        # print(image.dtype)
        # print('label.shape', label.shape)
        # print(label.dtype)

        return image, label

    def __len__(self):
        return len(self.img_file_list)


def get_labelme_loader_from_path(data_dir_path,
                                 batch_size,
                                 img_format,
                                 down_sample,
                                 random_transform,
                                 mosaic_transform,
                                 class_color,
                                 resize_rate,
                                 output_size,
                                 split_rate):

    dataset = labelme_dataset(data_dir_path=data_dir_path,
                              img_format=img_format,
                              down_sample=down_sample,
                              random_transform=random_transform,
                              mosaic_transform=mosaic_transform,
                              class_color=class_color,
                              resize_rate=resize_rate,
                              output_size=output_size)

    if split_rate!=0:
        train_sampler, val_sampler = split_dataset(dataset, split_rate)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   shuffle=False,num_workers=0, drop_last=True,sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0, sampler=val_sampler)
    else:
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   shuffle=False, num_workers=0, drop_last=True)
        val_loader = None

    return train_loader, val_loader, len(dataset.class_color), dataset
