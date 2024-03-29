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


class TrainDatasetConstructor(DatasetConstructor):
    def __init__(self,
                 train_num,
                 data_dir_path,
                 gt_dir_path,
                 mode='crop',
                 dataset_name="JSTL",
                 device=None,
                 is_random_hsi=False,
                 is_flip=False,
                 is_mask=False,
                 is_throw0 = 0,
                 fine_size = 400,
                 opt=None,
                 ):
        super(TrainDatasetConstructor, self).__init__()
        
        self.train_num = train_num
        self.opt = opt
        self.imgs = []
        self.fine_size = fine_size
        self.permulation = np.random.permutation(self.train_num)
        self.data_root, self.gt_root = data_dir_path, gt_dir_path
        self.mode = mode
        self.device = device
        self.is_random_hsi = is_random_hsi
        self.is_flip = is_flip
        self.is_mask = is_mask
        self.dataset_name = dataset_name
        self.kernel = torch.FloatTensor(torch.ones(1, 1, 2, 2))
        self.is_throw0 = is_throw0
        self.online_map = True if self.opt.rand_scale_rate > 0.0 else False
                
        # they are mapped as pairs
        imgs = sorted(glob.glob(self.data_root+'/*'))
        dens = sorted(glob.glob(self.gt_root+'/*'))
        self.train_num = len(imgs)
        # the whole gt label for all training images
        #self.gt_label_all = torch.zeros(self.train_num).long()
        self.gt_label_all = []
        print('Constructing training dataset...')
        
        for i in range(self.train_num):
            img_tmp = imgs[i]

#             # for 3 datasets
#             if os.path.basename(img_tmp).find('SHA') != -1:
#                 continue            

            den = os.path.join(self.gt_root, os.path.basename(img_tmp)[:-4] + ".npy")
            assert den in dens, "Automatically generating density map paths corrputed!"
            # add cls id to each img

            class_id = 0
            if os.path.basename(imgs[i]).find('SHA') != -1:
                class_id = 0
            elif os.path.basename(imgs[i]).find('SHB') != -1:
                class_id = 1
            elif os.path.basename(imgs[i]).find('QNRF') != -1:
                class_id = 2
            elif os.path.basename(imgs[i]).find('NWPU') != -1:
                class_id = 3
            else:
                assert 1==2

            self.imgs.append([imgs[i], den, i, class_id]) # also add additional index
            #self.gt_label_all[i] = class_id
            # for 3 datasets
            self.gt_label_all.append(class_id)
        # for 3 datasets
        self.gt_label_all = torch.tensor(self.gt_label_all).long()
        
        # for 3 datasets
        self.train_num= len(self.imgs)

    def __getitem__(self, index):
        if self.mode == 'crop':
            img_path, gt_map_path, cur_idx, class_id = self.imgs[index]

            # single IsBN method
            # class_id = cur_idx % sef.opt.cls_num

            class_id = torch.tensor(class_id).long()
            img = Image.open(img_path).convert("RGB")
            cur_dataset = super(TrainDatasetConstructor, self).get_cur_dataset(img_path)
            img, ratio_h, ratio_w = super(TrainDatasetConstructor, self).resize(img, cur_dataset, self.opt.rand_scale_rate)
            width, height = img.size
            gt_map = None
            if self.online_map:
                mat_name = img_path.replace('images', 'ground_truth')[:-4] + ".mat"
                points = scio.loadmat(mat_name)['annPoints']
                gt_map = utils.get_density_map_gaussian(height, width, ratio_h, ratio_w,  points, fixed_value=4)
                gt_map = Image.fromarray(np.squeeze(np.reshape(gt_map, [height, width])))  # transpose into w, h
            else:
                gt_map = Image.fromarray(np.squeeze(np.load(gt_map_path).astype(np.float32)))

            if self.is_random_hsi:      
                img = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)(img)
          
            if self.is_flip:
                flip_random = random.random()
                if flip_random > 0.5:
                    img = F.hflip(img)
                    gt_map = F.hflip(gt_map)

            img, gt_map = transforms.ToTensor()(np.array(img)), transforms.ToTensor()(np.array(gt_map))
                
            img_shape = img.shape  # C, H, W
            gt_map_shape = gt_map.shape
            # also scale gt_map
            if img_shape[1] != gt_map_shape[1] or img_shape[2] != gt_map_shape[2]:
                print('img shape is not same as gt_map: ', img_shape, gt_map_shape)
                assert 1==2

            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            
            # crop to fine_size
            rh, rw = random.randint(0, img_shape[1] - self.fine_size), random.randint(0, img_shape[2] - self.fine_size)
            p_h, p_w = self.fine_size, self.fine_size
            img = img[:, rh:rh + p_h, rw:rw + p_w]
            gt_map = gt_map[:, rh:rh + p_h, rw:rw + p_w]       
            
            gt_map = functional.conv2d(gt_map.view(1, 1, self.fine_size, self.fine_size), self.kernel, bias=None, stride=2, padding=0)
            return img.view(3, self.fine_size, self.fine_size), gt_map.view(1, self.fine_size//2, self.fine_size//2), class_id, os.path.basename(img_path), torch.tensor(cur_idx)

    def __len__(self):
        return self.train_num


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
