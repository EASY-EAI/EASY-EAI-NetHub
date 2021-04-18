import os
import sys

realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:-1]))
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('EASY_EAI_nethub')+1]))


import random
import cv2
# from PIL import Image
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
# import torchvision.transforms as transforms


def split_dataset(dataset, split_rate, shuffle=True, random_seed=1024):
    '''
        return index of train&val, 
    '''
    indices = list(range(len(dataset)))
    split = int(np.floor(split_rate*len(dataset)))

    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    return train_sampler, val_sampler


class image_folder_dataset(object):
    """docstring for labelme_dataset"""
    def __init__(self,
                 data_dir_path,
                 img_format_list=['jpg'],
                 output_size=None):

        self.img_file_list = []
        self.label_file_list = []
        self.output_size = output_size
        self.img_format_list = img_format_list

        subclass = os.listdir(data_dir_path)
        self.class_name = subclass
        self.class_name_num = [i for i in range(len(subclass))]

        for i in range(len(subclass)):
            path = os.path.join(data_dir_path, subclass[i])
            content = os.listdir(path)
            for j in range(len(content)):
                file_element = content[j].split('.')
                if file_element[-1] in img_format_list:
                    self.img_file_list.append(os.path.join(path, content[j]))
                    self.label_file_list.append(i)

    def add_item(self, target_path, class_type, depth_level=None, limit_sample=None, shuffle=True):
        for root, dirs, files in os.walk(target_path):
            _count = 0
            if depth_level is not None:
                # start next loop if depth not match
                if (len(root.split('\\')) - len(target_path.split('\\'))) != depth_level:
                    continue

            temp_file_list = []
            for name in files:
                if name.split('.')[-1] in self.img_format_list:
                    temp_file_list.append(os.path.join(root, name))

                    if limit_sample is not None and shuffle is False:
                        # limit and not shuffle, then its allowed to break early
                        if len(temp_file_list) >= limit_sample:
                            break

            if shuffle is True:
                random.shuffle(temp_file_list)
            if limit_sample is not None:
                temp_file_list = temp_file_list[0: limit_sample]

            if isinstance(class_type, int):
                temp_label_list = [class_type for i in range(len(temp_file_list))]
            elif isinstance(class_type, str):
                temp_label_list = [self.class_name.index(class_type) for i in range(len(temp_label_list))]

            self.img_file_list = self.img_file_list + temp_file_list
            self.label_file_list = self.label_file_list + temp_label_list


    def __getitem__(self, index):
        img_name = self.img_file_list[index]
        gt_label = self.label_file_list[index]

        img = cv2.imread(img_name)
        img = img/255.0
        if self.output_size is not None:
            img = cv2.resize(img, tuple(self.output_size))
        img = np.transpose(img, (2, 0, 1))

        image = torch.tensor(img).float()
        label = torch.tensor(gt_label).int()

        return image, label

    def __len__(self):
        return len(self.img_file_list)


def get_img_loader_from_path(data_dir_path,
                             batch_size,
                             img_format_list,
                             output_size,
                             split_rate,
                             addition_root=None,
                             addition_root_label=None):

    dataset = image_folder_dataset(data_dir_path=data_dir_path,
                                   img_format_list=img_format_list,
                                   output_size=output_size)

    if addition_root is not None:
        for i in range(len(addition_root)):
            dataset.add_item(addition_root[i], addition_root_label[i])

    if split_rate != 0:
        train_sampler, val_sampler = split_dataset(dataset, split_rate)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   shuffle=False, num_workers=0, drop_last=True,sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0, sampler=val_sampler)
    else:
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   shuffle=False, num_workers=0, drop_last=True)
        val_loader = None

    return train_loader, val_loader, dataset


class blank_image_folder_dataset(object):
    """docstring for labelme_dataset"""
    def __init__(self,
                 class_name_list,
                 img_format_list=['jpg'],
                 output_size=None):

        self.img_file_list = []
        self.label_file_list = []
        self.output_size = output_size
        self.img_format_list = img_format_list

        self.class_name = class_name_list
        self.class_name_num = [i for i in range(len(self.class_name))]


    # def add_item(self, target_path, class_type, depth_level=None, limit_sample=None):
    #     # elder, no shuffle
    #     for root, dirs, files in os.walk(target_path):
    #         _count = 0
    #         if depth_level is not None:
    #             if (len(root.split('\\')) - len(target_path.split('\\'))) != depth_level:
    #                 continue
    #         for name in files:
    #             if limit_sample != None:
    #                 if _count > limit_sample:
    #                     print('excceed limit_sample')
    #                     break
    #             if name.split('.')[-1] in self.img_format_list:
    #                 self.img_file_list.append(os.path.join(root, name))

    #                 if isinstance(class_type, int):
    #                     self.label_file_list.append(class_type)
    #                 elif isinstance(class_type, str):
    #                     self.label_file_list.append(self.class_name.index(class_type))
    #                 _count += 1

    #     print('*'*20, len(self.img_file_list))

    def add_item(self, target_path, class_type, depth_level=None, limit_sample=None, shuffle=True):
        start_len = len(self.img_file_list)
        for root, dirs, files in os.walk(target_path):
            if depth_level is not None:
                # start next loop if depth not match
                if (len(root.split('\\')) - len(target_path.split('\\'))) != depth_level:
                    continue

            temp_file_list = []
            for name in files:
                if name.split('.')[-1] in self.img_format_list:
                    temp_file_list.append(os.path.join(root, name))
                    if limit_sample is not None and shuffle is False:
                        # limit and not shuffle, then its allowed to break early
                        if len(temp_file_list) >= limit_sample:
                            break

            if shuffle is True:
                random.shuffle(temp_file_list)
            if limit_sample is not None:
                temp_file_list = temp_file_list[0: limit_sample]

            if isinstance(class_type, int):
                temp_label_list = [class_type for i in range(len(temp_file_list))]
            elif isinstance(class_type, str):
                temp_label_list = [self.class_name.index(class_type) for i in range(len(temp_file_list))]

            self.img_file_list = self.img_file_list + temp_file_list
            self.label_file_list = self.label_file_list + temp_label_list

        # print('target_path''*'*20, len(temp_file_list))
        print('from [{}] get *{}* [{}]file in format {}'.format(target_path, len(self.img_file_list)- start_len, class_type, self.img_format_list))

    def __getitem__(self, index):
        img_name = self.img_file_list[index]
        gt_label = self.label_file_list[index]

        img = cv2.imread(img_name)
        if self.output_size is not None:
            img = cv2.resize(img, tuple(self.output_size))
        img = np.transpose(img, (2, 0, 1))

        image = torch.tensor(img).float()
        label = torch.tensor(gt_label).int()

        return image, label

    def __len__(self):
        return len(self.img_file_list)

    def print_class_state(self):
        for i in range(len(self.class_name)):
            _count = (np.array(self.label_file_list)==self.class_name_num[i]).sum()
            print('class [{}] has *{}* file'.format(self.class_name[i], _count))


def get_init_loader(class_name_list,
                    batch_size,
                    img_format_list,
                    output_size,
                    split_rate,
                    addition_root,
                    addition_root_label,
                    addition_root_depth_level=None,
                    limit_sample=None,
                    shuffle=True):

    dataset = blank_image_folder_dataset(class_name_list=class_name_list,
                                         img_format_list=img_format_list,
                                         output_size=output_size)

    if addition_root is not None:
        for i in range(len(addition_root)):
            dataset.add_item(addition_root[i], addition_root_label[i], addition_root_depth_level[i], limit_sample[i], shuffle)

    dataset.print_class_state()

    if split_rate != 0:
        train_sampler, val_sampler = split_dataset(dataset, split_rate)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   shuffle=False, num_workers=0, drop_last=True,sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0, sampler=val_sampler)
    else:
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   shuffle=False, num_workers=0, drop_last=True)
        val_loader = None

    return train_loader, val_loader, dataset
