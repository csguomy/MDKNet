# config
'''
Main difference:
1. No complex img_name -> mat_name -> npy_name changes.
   All the corresponding names are the same except the extension part.
2. Delete 'dataset.shuffle()', using official shuffle in Dataloader
3. Using float16 npy to save memory
'''

import sys
import time
import torch.nn.functional as F
import os
import numpy as np
import torch
from config import config
import net.networks as networks
from eval.Estimator import Estimator
from options.train_options import TrainOptions
from Dataset.DatasetConstructor import TrainDatasetConstructor,EvalDatasetConstructor

from ipdb import launch_ipdb_on_exception


opt = TrainOptions().parse()

# Mainly get settings for specific datasets
setting = config(opt)

log_file = os.path.join(setting.model_save_path, opt.dataset_name+'.log')
log_f = open(log_file, "w")

# Data loaders
train_dataset = TrainDatasetConstructor(
    setting.train_num,
    setting.train_img_path,
    setting.train_gt_map_path,
    mode=setting.mode,
    dataset_name=setting.dataset_name,
    device=setting.device,
    is_random_hsi=setting.is_random_hsi,
    is_flip=setting.is_flip,
    fine_size=opt.fine_size,
    opt=opt
    )
eval_dataset = EvalDatasetConstructor(
    setting.eval_num,
    setting.eval_img_path,
    setting.eval_gt_map_path,
    mode=setting.mode,
    dataset_name=setting.dataset_name,
    device=setting.device,
    no_sort=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=setting.batch_size, shuffle=True, num_workers=opt.nThreads, pin_memory=True, drop_last=True)

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
net = networks.define_net(opt)
net = networks.init_net(net, gpu_ids=opt.gpu_ids)

net_ema = None

if opt.model_ema:
    from timm.utils import get_state_dict, ModelEma
    print('~~')
    # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
    net_ema = ModelEma(
        net,
        decay=opt.model_ema_decay,
        device='cpu' if opt.model_ema_force_cpu else None,
)

criterion = torch.nn.MSELoss(reduction='sum').to(setting.device) # first device is ok
#crit_cls = torch.nn.NLLLoss().to(setting.device)
estimator = Estimator(opt, setting, eval_loader, criterion=criterion)

optimizer = networks.select_optim(net, opt)
scheduler = networks.get_scheduler(optimizer, opt)


def convert_to_one_hot(cls_num, label, border_cls_num):
    # convert something like [2, 3, 0] -> [[0,0,1,0], [0,0,0,1],[1,0,0,0]]
    one_hot = F.one_hot(label, cls_num)

    # additonal add border labels in
    one_hot = torch.cat([one_hot, torch.zeros(label.shape[0], border_cls_num).to(label)], dim=1)

    return one_hot

"""
inputs: logits
"""
def SoftCrossEntropy_forlt(logits, target, weight, reduction='average'):

    loss = - (weight * (target * torch.log_softmax(logits, dim=1)).sum(dim=1)).sum()

    return loss

#
def SoftCrossEntropy_ori(inputs, target, reduction='average'):

    log_likelihood = -F.log_softmax(inputs, dim=1)
    batch = inputs.shape[0]

    if reduction == 'average':
        loss = torch.sum(torch.mul(log_likelihood, target)) / batch
    else:
        loss = torch.sum(torch.mul(log_likelihood, target))
    return loss

# linear decay the alpha value
# alpha * pred
def cal_alpha(cur_ep, start_ep, final_alpha, final_ep):
    if cur_ep < start_ep:
        return 0.
    else:
        return (cur_ep - start_ep) * (1-final_alpha) / (final_ep - start_ep)

# index template for assigning confidents of real labels to border labels
def get_template(cls_num):
    if cls_num == 4:
        idx_ori_mat = [[1,2,3], [0,2,3], [0,1,3], [0,1,2]]
        idx_des_mat = [[4,5,6], [4,7,8], [5,7,9], [6,8,9]]

    elif cls_num == 3:
        idx_ori_mat = [[1,2], [0,2], [0,1]]
        idx_des_mat = [[3,4], [3,5], [4,5]]

    elif cls_num == 5:
        idx_ori_mat = [[1,2,3,4], [0,2,3,4], [0,1,3,4], [0,1,2,4], [0,1,2,3]]
        idx_des_mat = [[5,6,7,8], [5,9,10,11], [6,9,12,13], [7,10,12,14], [8,11,13,14]]

    else:
        raise ValueError('Currently, only 3/4/5 classes are supported')

    return torch.tensor(idx_ori_mat), torch.tensor(idx_des_mat)


if opt.pretrain:
    print('-----------')
    print('-----------')
    print('-----------')
    print('-----------')
    print('Loading prtrained model:', opt.pretrain_model)
    net.module.load_state_dict(torch.load(opt.pretrain_model, map_location=str(setting.device)))

step = 0
eval_loss, eval_mae, eval_rmse = [], [], []


base_mae_sha, base_mae_shb, base_mae_qnrf, base_mae_nwpu = opt.base_mae.split(',')
base_mae_sha = float(base_mae_sha)
base_mae_shb = float(base_mae_shb)
base_mae_qnrf = float(base_mae_qnrf)
base_mae_nwpu = float(base_mae_nwpu)


# should be N_imgs * (4+6) for 4 classes
cls_num = net.module.cls_num
out_cls_num = net.module.out_cls_num
border_tuple_labels = net.module.border_labels

print('net.module.cls_num: ', net.module.cls_num)
print('net.module.out_cls_num: ', net.module.out_cls_num)
print('net.module.border_labels: ', net.module.border_labels)
print('train_dataset.gt_label_all:', len(train_dataset.gt_label_all))


# get correction conf mats
idx_ori_mat, idx_des_mat = get_template(cls_num)
idx_ori_mat = idx_ori_mat.to(setting.device)
idx_des_mat = idx_des_mat.to(setting.device)

pred_cul_label = torch.zeros(train_dataset.__len__(), out_cls_num).to(setting.device)
gt_label_all = convert_to_one_hot(cls_num, train_dataset.gt_label_all, out_cls_num - cls_num)
gt_label_all = gt_label_all.to(setting.device)
my_gt_label = pred_cul_label.clone() # before 105 epoch, it is useless, then for epoch [106, max_epoch], use it


test_time_start = time.time()
with launch_ipdb_on_exception():
    for epoch_index in range(setting.epoch):
        # eval
        if epoch_index % opt.eval_per_epoch == 0 and epoch_index > opt.start_eval_epoch:
            print('Evaluating epoch:', str(epoch_index))
            torch.cuda.empty_cache()
            
            if opt.model_ema:
                net_eval = net_ema.ema
            else:
                net_eval = net

            # pred_mae and pred_mse are for seperate datasets
            # mention:  __len__() returns the `number of batches`, so the total validate num is *batch_size_eval
            validate_MAE, validate_RMSE, validate_loss, time_cost, pred_mae, pred_mse = estimator.evaluate(net_eval, eval_dataset.__len__())
            
            # validate the code of eval
            if opt.start_eval_epoch==-1 and opt.test_eval==1:
                print('Test over~')
                break        
            
            eval_loss.append(validate_loss)
            eval_mae.append(validate_MAE)
            eval_rmse.append(eval_rmse)
            log_f.write(
                'In step {}, epoch {}, loss = {}, eval_mae = {}, eval_rmse = {}, mae_SHA = {}, mae_SHB = {}, mae_QNRF = {}, mae_NWPU = {}, mse_SHA = {}, mse_SHB = {}, mse_QNRF = {}, mse_NWPU = {},, time cost eval = {}s\n'.format(step, epoch_index, validate_loss, validate_MAE, validate_RMSE, pred_mae[0], pred_mae[1], pred_mae[2], pred_mae[3],
                        pred_mse[0], pred_mse[1], pred_mse[2], pred_mse[3], time_cost))
            log_f.flush()
            # save model with epoch and MAE

            save_now = False

            
            # multi-4
            if pred_mae[0] < base_mae_sha and pred_mae[1] < base_mae_shb and pred_mae[2] < base_mae_qnrf and pred_mae[3] < base_mae_nwpu:
                save_now = True        
            
#             # multi-3
#             if pred_mae[0] < base_mae_sha and pred_mae[2] < base_mae_shb and pred_mae[3] < base_mae_nwpu:
#                 save_now = True

            if save_now:
                best_model_name = setting.model_save_path + "/MAE_" + str(round(validate_MAE, 2)) + \
                    "_MSE_" + str(round(validate_RMSE, 2)) + '_mae_' + str(round(pred_mae[0], 2)) + \
                    '_' + str(round(pred_mae[1], 2)) + '_' + str(round(pred_mae[2], 2)) + '_' + str(round(pred_mae[3], 2)) + '_mse_' + str(round(pred_mse[0], 2)) + \
                    '_' + str(round(pred_mse[1], 2)) + '_' + str(round(pred_mse[2], 2)) + '_' + str(round(pred_mse[3], 2)) + \
                    '_Ep_' + str(epoch_index) + '.pth'

                if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net_eval.module.cpu().state_dict(), best_model_name)
                    # yes, seems this line of code can be deleted
                    # net.cuda(opt.gpu_ids[0])
                    net_eval.cuda(opt.gpu_ids[0])
                else:
                    torch.save(net_eval.cpu().state_dict(), best_model_name)
        
        if epoch_index >=200:
            opt.conf_window=7
        if epoch_index >=400:
            opt.conf_window=5
            
        # update my_gt_label
        if epoch_index > opt.start_faith_epoch + opt.conf_window:
            #  from epoch 106, so epoch_index - 1 - opt.start_faith_epoch
            # meaning, before training epoch 106, we get my_gt_label, and select target_label from `my_gt_label`
            # Before epoch 106, we totally believe gt label, and use gt label (ie, class_id) as the target label.
            if (epoch_index - 1 - opt.start_faith_epoch) % opt.conf_window == 0:
            #   # correct pred_cul_label
            #   helper_idx = torch.arange(pred_cul_label.shape[0]).view(1, -1).repeat(1, cls_num-1)
            #   manual_label = gt_label_all[]
            #   pred_cul_label[helper_idx, idx_des_mat[cla

                if opt.avg_pred:
                    pred_cul_label = pred_cul_label / opt.conf_window

                # update my_gt_label
                cur_alpha = cal_alpha(epoch_index, opt.start_faith_epoch, opt.final_conf, opt.max_epochs)
                my_gt_label = cur_alpha * pred_cul_label + (1 - cur_alpha) * gt_label_all
                # and softmax with T temperature
                my_gt_label = torch.softmax(my_gt_label/opt.Temp, dim=1)
                    
                # clear out the pred_cul_label
                pred_cul_label.fill_(0)


        time_per_epoch = 0

        for train_img, train_gt, class_id, img_path, idx in train_loader:
            train_img = train_img.to(setting.device)
            train_gt = train_gt.to(setting.device)
            class_id = class_id.to(setting.device)


            gt_id = convert_to_one_hot(cls_num, class_id, out_cls_num-cls_num)
            idx = idx.to(setting.device)
            # convert class_id to one_hot version, from 106 epoch
            if epoch_index > opt.start_faith_epoch + opt.conf_window:

       
                # select target label from `my_gt_label`
                target_label = my_gt_label[idx]

    
            else: # directly use gt label as target label
                target_label = gt_id
        
            
            net.train()
            x, y = train_img, train_gt
            start = time.time()
            prediction, logitH, logitT, pred_cur_label = net(x)

            if epoch_index > opt.start_faith_epoch:
                # for epoch in the window, eg: 101, 102, 103, 104, 105
                pred_cul_label[idx.view(-1)] += pred_cur_label.detach()
                # for the final epoch in the window; eg. 105
                # correct the conf
          
                if (epoch_index - opt.start_faith_epoch) % opt.conf_window == 0:
                    # idx.view(-1, 1).repeat(1, cls_num-1):  [[img_id_i, img_id_i, img_id_i], [img_id_j, img_id_j, img_id_j], ...]
                    # then select true label accumlated conf from pred_cul_label with index: idx_ori_mat[class_id.view(-1)]
                    # and assign these confidents to the values in pred_cul_label with index: idx_des_mat[class_id.view(-1)]
                    pred_cul_label[idx.view(-1, 1).repeat(1, cls_num-1), idx_des_mat[class_id.view(-1)]] += pred_cul_label[idx.view(-1, 1).repeat(1, cls_num-1), idx_ori_mat[class_id.view(-1)]]
          
            
    
            if opt.weight_with_target:


                weightH = target_label.sum(dim=1)

                if opt.cls_num == 4:
                    weightT = target_label[:, [1, 4, 7, 8]].sum(dim=1)

                elif opt.cls_num == 3:
                    # SHB
                    weightT = target_label[:, [1, 3, 5]].sum(dim=1)
                
            # here, directly using `gt_id` of `shb` as the tail class
            else:
                weightH = gt_id.sum(dim=1)
                weightT = gt_id[:,1].sum(dim=1)

            
            loss_ice = (SoftCrossEntropy_forlt(logitH, target_label, weightH) + SoftCrossEntropy_forlt(logitT, target_label, weightT)) / (weightH.sum() + weightT.sum()).float()  
            logit = logitH + logitT
            loss_fce = SoftCrossEntropy_ori(logit, target_label)

            loss_cls = loss_ice * opt.reslt_beta + (1 - opt.reslt_beta) * loss_fce
            loss_cls = loss_cls * opt.cls_w             
                
            loss = criterion(prediction, y)
            optimizer.zero_grad()
            (loss + loss_cls).backward()
            
            # update ema
            if opt.model_ema:
                print('~~')
                net_ema.update(net)
            
            loss_item = loss.detach().item()
            loss_cls_item = loss_cls.item()
            optimizer.step()

            step += 1
            end = time.time()
            time_per_epoch += end - start

            if step % opt.print_step == 0:
                print("Step:{:d}\t, Epoch:{:d}\t, Loss:{:.4f}, Cls Loss:{:.4f}".format(step, epoch_index, loss_item, loss_cls_item))

        scheduler.step()

        test_time_end = time.time()
        lr = optimizer.param_groups[0]['lr']
        print('lr now: %.7f' % lr)
