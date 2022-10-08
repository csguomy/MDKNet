from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch.nn.functional as functional
import torch.utils.data as data
import random
import time
import scipy.io as scio
import h5py
import math
import glob
import util.utils as utils
import os


class DatasetConstructor(data.Dataset):
    def __init__(self):
        self.datasets_com = ['SHA', 'SHB', 'QNRF', 'NWPU']
        return

    # return current dataset(SHA/SHB/QNRF) for current image
    def get_cur_dataset(self, img_name):
        check_list = [img_name.find(da) for da in self.datasets_com]
        check_list = np.array(check_list)
        cur_idx = np.where(check_list != -1)[0][0]
        return self.datasets_com[cur_idx]

    def resize(self, img, dataset_name, rand_scale_rate=0.0, perform_resize=True):
        height = img.size[1]
        width = img.size[0]
        resize_height = height
        resize_width = width

        if rand_scale_rate > 0.0:
            cur_rand = random.uniform(1-rand_scale_rate, 1+rand_scale_rate)
            resize_height = int(cur_rand * resize_height)
            resize_width = int(cur_rand * resize_width)

        if dataset_name == "SHA":
            sz = 416 # or 512
            if resize_height <= sz:
                tmp = resize_height
                resize_height = sz
                resize_width = (resize_height / tmp) * resize_width
            if resize_width <= sz:
                tmp = resize_width
                resize_width = sz
                resize_height = (resize_width / tmp) * resize_height
            resize_height = math.ceil(resize_height / 32) * 32
            resize_width = math.ceil(resize_width / 32) * 32
        elif dataset_name == "SHB":
            resize_height = height
            resize_width = width
        elif dataset_name.find("QNRF") != -1 or dataset_name.find("NWPU") != -1: # it is QNRF_large
            if resize_width >= 2048:
                tmp = resize_width
                resize_width = 2048
                resize_height = (resize_width / tmp) * resize_height

            if resize_height >= 2048:
                tmp = resize_height
                resize_height = 2048
                resize_width = (resize_height / tmp) * resize_width

            if resize_height <= 512:
                tmp = resize_height
                resize_height = 512
                resize_width = (resize_height / tmp) * resize_width
            if resize_width <= 512:
                tmp = resize_width
                resize_width = 512
                resize_height = (resize_width / tmp) * resize_height

            # other constraints
            if resize_height < resize_width:
                if resize_width / resize_height > 2048/512: # the original is 512 instead of 416
                    resize_width = 2048
                    resize_height = 512
            else:
                if resize_height / resize_width > 2048/512:
                    resize_height = 2048
                    resize_width = 512

            resize_height = math.ceil(resize_height / 32) * 32
            resize_width = math.ceil(resize_width / 32) * 32
        else:
            raise NameError("No such dataset, only support SHA, SHB, QNRF")
        if perform_resize:
            img = transforms.Resize([resize_height, resize_width])(img)
            ratio_H = resize_height / height
            ratio_W = resize_width / width
            return img, ratio_H, ratio_W
        else:
            return resize_height, resize_width


# For evalation, we also return img_path.
# This help get the paths of '.mat' recording the real num(not from density map).
#
#
class EvalDatasetConstructor(DatasetConstructor):
    def __init__(self,
                 validate_num,
                 data_dir_path,
                 gt_dir_path,
                 mode="crop",
                 dataset_name="JSTL",
                 device=None,
                 no_sort=False,
                 ):
        super(EvalDatasetConstructor, self).__init__()
        self.imgs = []
        self.data_root = data_dir_path
        self.gt_root = gt_dir_path
        self.mode = mode
        self.no_sort = no_sort # just for debug, usually
        self.device = device
        self.dataset_name = dataset_name
        self.kernel = torch.FloatTensor(torch.ones(1, 1, 2, 2))
        # they are mapped as pairs
        imgs = sorted(glob.glob(self.data_root+'/*'))
        dens = sorted(glob.glob(self.gt_root+'/*'))
        self.validate_num = len(imgs)
        print('Constructing testing dataset...')
        print(self.validate_num)

        self.extra_dataset = False
        if self.dataset_name == 'Unknown':
            self.extra_dataset = True

        for i in range(self.validate_num):
            img_tmp = imgs[i]
            if self.extra_dataset == False:
                den = os.path.join(self.gt_root, os.path.basename(img_tmp)[:-4] + ".npy")
                assert den in dens, "Automatically generating density map paths corrputed!"
                self.imgs.append([imgs[i], den])
            else:
                self.imgs.append(imgs[i])

        if self.extra_dataset == False and self.no_sort == False:
            self.imgs_new = []
            self.cal_load_list = torch.zeros(self.validate_num)
            print('Pre-reading the resized image size info, and sort... please wait for round 1 min')

            for i in range(self.validate_num):
                img_path, _ = self.imgs[i]
                img = Image.open(img_path).convert("RGB")
                cur_dataset = super(EvalDatasetConstructor, self).get_cur_dataset(img_path)
                # do not resize, just get the resized size to acceralate
                H, W = super(EvalDatasetConstructor, self).resize(img, cur_dataset, perform_resize=False)
                cal_load =  H*W
                self.cal_load_list[i] = cal_load

            # sort the img_path in a descending order acoorindg the cal_load
            new_load_list, indices = torch.sort(self.cal_load_list, descending=True)
            for i in range(self.validate_num):
                cur_index = indices[i]
                # select img_path-den_path pair from self.imgs to form a new imgs_new list sorted by cal_load
                self.imgs_new.append(self.imgs[cur_index])

            # finally, rename self.imgs_new to self.imgs
            self.imgs = self.imgs_new
            

    def __getitem__(self, index):
        if self.mode == 'crop':
            if self.extra_dataset:
                img_path = self.imgs[index]
            else:
                img_path, gt_map_path = self.imgs[index]

            # get domain
            class_id = 0
            if os.path.basename(img_path).find('SHA') != -1:
                class_id = 0
            elif os.path.basename(img_path).find('SHB') != -1:
                class_id = 1
            elif os.path.basename(img_path).find('QNRF') != -1:
                class_id = 2
            elif os.path.basename(img_path).find('NWPU') != -1:
                class_id = 3
            else:
                class_id = -1

            class_id = torch.tensor(class_id).long()

            img = Image.open(img_path).convert("RGB")

            if self.extra_dataset:
                cur_dataset = 'QNRF' # using QNRF is fine for unknown
            else:
                cur_dataset = super(EvalDatasetConstructor, self).get_cur_dataset(img_path)
            img, _, _ = super(EvalDatasetConstructor, self).resize(img, cur_dataset)
            img = transforms.ToTensor()(img)
            img_resized = img
            img_shape = img.shape
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            patch_height, patch_width = (img_shape[1]) // 2, (img_shape[2]) // 2
            imgs = []
            for i in range(3):
                for j in range(3):
                    start_h, start_w = (patch_height // 2) * i, (patch_width // 2) * j
                    imgs.append(img[:, start_h:start_h + patch_height, start_w:start_w + patch_width])

            imgs = torch.stack(imgs)

            if self.extra_dataset: # unknown dataset, return these things are enough
                return img_resized, img_path, imgs, np.array(img_shape) # return 'img_shape' for obtain 'fake' density map size when evaluating

            # ------       for density maps
            patch_h, patch_w = imgs.size(2), imgs.size(3)
            gt_map = Image.fromarray(np.squeeze(np.load(gt_map_path).astype(np.float32)))
            gt_map = transforms.ToTensor()(np.array(gt_map))
            gt_map_shape = gt_map.shape
            if img_shape[1] != gt_map_shape[1] or img_shape[2] != gt_map_shape[2]:
                print('img shape is not same as gt_map: ', img_shape, gt_map_shape)
                assert 1==2
                
            gt_map = functional.conv2d(gt_map.view(1, *(gt_map_shape)), self.kernel, bias=None, stride=2, padding=0)
            gt_H, gt_W = gt_map_shape[1]//2, gt_map_shape[2]//2
            return img_path, imgs, gt_map.view(1, gt_H, gt_W), class_id, torch.tensor(gt_H), torch.tensor(gt_W), torch.tensor(patch_h), torch.tensor(patch_w)

    def __len__(self):
        return self.validate_num



'''
It is totally deprecated. You can delete it anytime.
'''
class EvalDatasetConstructor_old(DatasetConstructor):
    def __init__(self,
                 validate_num,
                 data_dir_path,
                 gt_dir_path,
                 mode="crop",
                 dataset_name="JSTL",
                 device=None,
                 ):
        super(EvalDatasetConstructor_old, self).__init__()
        self.validate_num = validate_num
        self.imgs = []
        self.data_root = data_dir_path
        self.gt_root = gt_dir_path
        self.mode = mode
        self.device = device
        self.dataset_name = dataset_name
        self.kernel = torch.FloatTensor(torch.ones(1, 1, 2, 2))
        # they are mapped as pairs
        imgs = sorted(glob.glob(self.data_root+'/*'))
        dens = sorted(glob.glob(self.gt_root+'/*'))
        # SHB_IMG_73.jpg --> SHB_GT_IMG_73.npy
        # QNRF_img_0002.jpg --> QNRF_GT_IMG_2.npy (only QNRF has 'img'(not IMG) and have prefix 0s)
        print('Constructing testing dataset...')
        for i in range(self.validate_num):
            img_tmp = imgs[i]
            # construct den path
            img_str = 'img' if 'img' in img_tmp else 'IMG'
            den_tmp = img_tmp.replace(img_str, 'GT_IMG').replace('.jpg', '.npy')
            # remove prefix 0s
            den = '_'.join(den_tmp.split('_')[:-1] + [str(int(den_tmp.split('_')[-1][:-4])) + '.npy'])
            den = os.path.join(self.gt_root, den.split('/')[-1])
            assert den in dens, "Automatically generating density map paths corrputed!"
            self.imgs.append([imgs[i], den, i+1])

    def __getitem__(self, index):
        if self.mode == 'crop':
            img_path, gt_map_path, img_index = self.imgs[index]

            class_id = 0
            if os.path.basename(img_path).find('SHA') != -1:
                class_id = 0
            elif os.path.basename(img_path).find('SHB') != -1:
                class_id = 1
            elif os.path.basename(img_path).find('QNRF') != -1:
                class_id = 2
            elif os.path.basename(img_path).find('NWPU') != -1:
                class_id = 3
            else:
                assert 1==2
            class_id = torch.tensor(class_id).long()

            img = Image.open(img_path).convert("RGB")
            cur_dataset = super(EvalDatasetConstructor_old, self).get_cur_dataset(img_path)
            img = super(EvalDatasetConstructor_old, self).resize(img, cur_dataset)
            img = transforms.ToTensor()(img)
            gt_map = Image.fromarray(np.squeeze(np.load(gt_map_path).astype(np.float32)))
            gt_map = transforms.ToTensor()(np.array(gt_map))

            img_shape, gt_shape = img.shape, gt_map.shape  # C, H, W
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            patch_height, patch_width = (img_shape[1]) // 2, (img_shape[2]) // 2
            imgs = []
            for i in range(3):
                for j in range(3):
                    start_h, start_w = (patch_height // 2) * i, (patch_width // 2) * j
                    imgs.append(img[:, start_h:start_h + patch_height, start_w:start_w + patch_width])

            imgs = torch.stack(imgs)
            gt_map = functional.conv2d(gt_map.view(1, *(gt_shape)), self.kernel, bias=None, stride=2, padding=0)
            return img_path, img_index, imgs, gt_map.view(1, gt_shape[1] // 2, gt_shape[2] // 2), class_id

    def __len__(self):
        return self.validate_num
