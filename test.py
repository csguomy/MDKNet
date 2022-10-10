# config
import sys
import warnings
import time
import numpy as np
import torch
from config import config
from eval.Estimator import Estimator
from net.networks import *
from options.test_options import TestOptions
from Dataset.DatasetConstructor import EvalDatasetConstructor

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batch_size = 1  # test code only supports batchSize = 1
opt.is_flip = 0  # no flip


test_model_name = 'output/MDKNet_models/mdknet.pth'

# Mainly get settings for specific datasets
setting = config(opt)
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

# Data loaders
eval_dataset = EvalDatasetConstructor(
    setting.eval_num,
    setting.eval_img_path,
    setting.eval_gt_map_path,
    mode=setting.mode,
    dataset_name=setting.dataset_name,
    device=setting.device,
    no_sort=True)


def my_collfn(batch):
    img_path = [item[0] for item in batch]
    imgs = [item[1] for item in batch]
    gt_map = [item[2] for item in batch]
    class_id = [item[3] for item in batch]
    gt_H = [item[4] for item in batch]
    gt_W = [item[5] for item in batch]
    pH = [item[6] for item in batch]
    pW = [item[7] for item in batch]

    bz = len(batch)

    gt_H = torch.stack(gt_H, 0)
    gt_W = torch.stack(gt_W, 0)
    pH = torch.stack(pH, 0)
    pW = torch.stack(pW, 0)
    gt_h_max = torch.max(gt_H)
    gt_w_max = torch.max(gt_W)

    ph_max = torch.max(pH)
    pw_max = torch.max(pW)

    imgs_new = torch.zeros(bz, 9, 3, ph_max, pw_max) # bz * 9 * c * gth_max * gtw_max
    gt_map_new = torch.zeros(bz, 1, 1, gt_h_max, gt_w_max)

    # put map
    for i in range(bz):
        imgs_new[i, :, :, :pH[i], :pW[i]] = imgs[i]
        # h, w
        gt_map_new[i, :, :, :gt_H[i], :gt_W[i]] = gt_map[i]

    class_id = torch.stack(class_id, 0)
    return img_path, imgs_new, gt_map_new, class_id, pH, pW, gt_H, gt_W

assert opt.eval_size_per_GPU == 1, "Using this is fast enough and for large size evaluation"
batch_eval_size = opt.eval_size_per_GPU * len(opt.gpu_ids)
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=batch_eval_size, collate_fn=my_collfn)

# model construct
net = define_net(opt)
net = init_net(net, gpu_ids=opt.gpu_ids)
net.module.load_state_dict(torch.load(test_model_name, map_location=str(setting.device)))

criterion = torch.nn.MSELoss(reduction='sum').to(setting.device)
estimator = Estimator(opt, setting, eval_loader, criterion=criterion)

# evaluation
validate_MAE, validate_RMSE, validate_loss, time_cost, pred_mae, pred_mse = estimator.evaluate(net, eval_dataset.__len__())

# print results
print('SHA_MAE: ', pred_mae[0])
print('SHB_MAE: ', pred_mae[1])
print('QNRF_MAE: ', pred_mae[2])
print('NWPU_MAE: ', pred_mae[3])
print('-------')
print('SHA_MSE: ', pred_mse[0])
print('SHB_MSE: ', pred_mse[1])
print('QNRF_MSE: ', pred_mse[2])
print('NWPU_MSE: ', pred_mse[3])
