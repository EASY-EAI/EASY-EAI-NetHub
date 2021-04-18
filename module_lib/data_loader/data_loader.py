import os
import sys

realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:-1]))
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('EASY_EAI_nethub')+1]))


import cv2
from PIL import Image
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import official_folder


'''
class custom_data_set():
    def __init__():
        pass
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, index, data_reinforce):

        data = data_reinforce.placeholder(data)
        return target
'''


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


def get_Image_loader_from_path(data_dir_path, batch_size, split_rate=0.05):
    '''
        return 
    '''
    dataset = official_folder.ImageFolder(data_dir_path,
                                          transforms.Compose([
                                                             transforms.RandomHorizontalFlip(p=0.5),
                                                             transforms.ToTensor(),
                                                             ]))
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

    return train_loader, val_loader, len(dataset.classes), dataset


class lfw_pairs_dataset():
    def __init__(self, path_file, issame_file, data_transforms=transforms.Compose([
                                                             transforms.ToTensor(),])):
        '''
            special for lfw txt reader
        '''
        self.lfw_issame_file = issame_file
        self.lfw_path_file = path_file
        fpath = open(self.lfw_path_file, 'r')
        path_lines = fpath.readlines()
        fpath.close()

        fsame = open(self.lfw_issame_file, 'r')
        same_lines = fsame.readlines()
        fsame.close()

        self.img_list_1 =[]
        self.img_list_2 =[]
        self.same_list =[]

        print('len lfw same',len(same_lines))
        print('len flw path',len(path_lines))
        for i in range(len(same_lines)):
            self.img_list_1.append(path_lines[i*2].strip('\n'))
            self.img_list_2.append(path_lines[i*2+1].strip('\n'))


        for same in same_lines:
            if same.strip('\n') =='True' or same.strip('\n')=='1':
                self.same_list.append(True)
            elif same.strip('\n') =='False' or same.strip('\n')=='0':
                self.same_list.append(False)

        self.transform = data_transforms


    def __len__(self):
        return len(self.same_list)

    def __getitem__(self, index):
        img_1 = cv2.imread(self.img_list_1[index])
        img_2 = cv2.imread(self.img_list_2[index])
        # img_1 = cv2.resize(img_1, (224,224))
        # img_2 = cv2.resize(img_2, (224,224))
        img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
        img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
        # img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        # img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
        img_1 = Image.fromarray(img_1)
        img_2 = Image.fromarray(img_2)

        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        label = int(self.same_list[index])
        label = torch.tensor(label)
        return img_1, img_2, label


def lfw_pairs_loader(img_path_file, issame_file, batch_size):

    lfw_dataset = lfw_pairs_dataset(img_path_file, 
                                    issame_file, 
                                    data_transforms=transforms.Compose([
                                                    transforms.ToTensor(),]))
    # print('lfw_dataset',len(lfw_dataset.same_list))
    # print('lfw_dataset len',len(lfw_dataset))
    lfw_pairs_loader = torch.utils.data.DataLoader(
        lfw_dataset, batch_size=batch_size, shuffle=False,
        )

    return lfw_pairs_loader


class cfp_pairs_dataset():
    def __init__(self, img_path, pair_file, data_transforms=transforms.Compose([
                                                             transforms.ToTensor(),])):
        '''
            special for lfw txt reader
        '''
        self.img_path = img_path
        self.pair_file = pair_file

        fpath = open(self.pair_file, 'r')
        pairs_line = fpath.readlines()
        fpath.close()


        self.img_list_1 =[]
        self.img_list_2 =[]
        self.same_list =[]

        print('len cfp same',len(pairs_line))

        for i in range(len(pairs_line)):

            content = pairs_line[i].split(' ')

            self.img_list_1.append(os.path.join(img_path, content[0].strip('\n')))
            self.img_list_2.append(os.path.join(img_path, content[1].strip('\n')))

            if content[2].strip('\n') == '1':
                self.same_list.append(True)
            elif content[2].strip('\n') == '-1':
                self.same_list.append(False)

        # for same in same_lines:
        #     if same.strip('\n') =='True' or same.strip('\n')=='1':
        #         self.same_list.append(True)
        #     elif same.strip('\n') =='False' or same.strip('\n')=='0':
        #         self.same_list.append(False)

        self.transform = data_transforms


    def __len__(self):
        return len(self.same_list)

    def __getitem__(self, index):
        img_1 = cv2.imread(self.img_list_1[index])
        img_2 = cv2.imread(self.img_list_2[index])
        # img_1 = cv2.resize(img_1, (224,224))
        # img_2 = cv2.resize(img_2, (224,224))
        img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
        img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
        # img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        # img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
        img_1 = Image.fromarray(img_1)
        img_2 = Image.fromarray(img_2)

        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        label = int(self.same_list[index])
        label = torch.tensor(label)
        return img_1, img_2, label


def cfw_pairs_loader(img_path, pair_file, batch_size):

    lfw_dataset = cfp_pairs_dataset(img_path,
                                    pair_file,
                                    data_transforms=transforms.Compose([
                                                    transforms.ToTensor(),]))
    # print('lfw_dataset',len(lfw_dataset.same_list))
    # print('lfw_dataset len',len(lfw_dataset))
    lfw_pairs_loader = torch.utils.data.DataLoader(
        lfw_dataset, batch_size=batch_size, shuffle=False,
        )

    return lfw_pairs_loader