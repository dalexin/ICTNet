# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import config
import torch.distributed as dist

from models.ICNet import ICNet
device = torch.device('cuda')



class FullModel(nn.Module):

    def __init__(self, model, sem_loss, bd_loss, temperature_ic=0.6, k_mse=10, k_kd=1, k_ic=10):
        super(FullModel, self).__init__()

        self.model = model
        self.sem_loss = sem_loss
        self.bd_loss = bd_loss

        self.temperature_ic = temperature_ic
        self.k_mse = k_mse
        self.k_kd = k_kd
        self.k_ic = k_ic

        self.icnet = ICNet()
        self.icnet.load_state_dict(torch.load('./models/checkpoint/icnet_ck.pth',map_location=torch.device('cpu')))
        self.icnet.eval()
        self.icnet.to(device)

        self.temperature_ic1 = 0.8
        self.temperature_ic2 = 0.7
        self.temperature_seg = 0.8

        self.celoss = nn.CrossEntropyLoss()

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc
  
    def ic_kd(self, pred,label):

        # B,C,H,W = label.size()

        # print(pred.size(),label.size())
        # ce_loss = F.cross_entropy(pred,label)
        mse_loss = F.mse_loss(pred,label)
        # ce_loss = self.celoss(pred,label)
        # mse_loss = mse_loss*(self.temperature_ic1 ** 2)

        label = label.flatten(2)
        pred = pred.flatten(2)

        label_prob = F.softmax(label/self.temperature_ic,dim=(2))
        pred_prob = F.log_softmax(pred/self.temperature_ic,dim=(2))
        kl_loss = F.kl_div(pred_prob, label_prob, reduction='batchmean')
        kl_loss = (self.temperature_ic ** 2) * kl_loss
        
        total_loss = self.k_mse * mse_loss + self.k_kd * kl_loss

        # print(mse_loss,kl_loss)

        
        return self.k_ic * total_loss
    

    def forward(self, inputs, labels, bd_gt, epoch, *args, **kwargs):
    
        outputs, pred_icmap = self.model(inputs, *args, **kwargs)

        if pred_icmap is not None:
            with torch.no_grad():
                cly_x = F.interpolate(inputs, (512, 512), mode = 'bilinear')
                ic_map = self.icnet(cly_x)
                ic_map = F.interpolate(ic_map, (inputs.size(-2), inputs.size(-1)), mode = 'bilinear')
            ic_loss = self.ic_kd(pred_icmap, ic_map)
            
        else:
            ic_loss = torch.zeros(1)
            ic_loss = ic_loss.to(device)

        
        h, w = labels.size(1), labels.size(2)
        
        for i in range(len(outputs)):
            ph, pw = outputs[i].size(2), outputs[i].size(3)
            if ph != h or pw != w:
                outputs[i] = F.interpolate(outputs[i], size=(
                    h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

        acc  = self.pixel_acc(outputs[-1], labels)
        loss_s = self.sem_loss(outputs, labels)
        # loss_b = self.bd_loss(outputs[-1], bd_gt)

        # filler = torch.ones_like(labels) * config.TRAIN.IGNORE_LABEL
        # bd_label = torch.where(F.sigmoid(outputs[-1][:,0,:,:])>0.8, labels, filler)
        # loss_sb = self.sem_loss(outputs[-2], bd_label)
        loss = loss_s

        return torch.unsqueeze(loss,0), outputs, acc, [loss_s, ic_loss]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
            (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix

# def adjust_learning_rate(optimizer, base_lr, max_iters, 
#         cur_iters, power=0.9, nbb_mult=10):
#     lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
#     optimizer.param_groups[0]['lr'] = lr
#     if len(optimizer.param_groups) == 2:
#         optimizer.param_groups[1]['lr'] = lr * nbb_mult
#     return lr

def adjust_learning_rate(optimizer, base_lr, max_iters, 
        cur_iters,warm_up_steps, power=0.9, nbb_mult=10):
    
    if cur_iters < warm_up_steps:  # 当小于1000时，
        lr = cur_iters*base_lr/(warm_up_steps*2)
    else:
        lr = base_lr*((1-float(cur_iters)/max_iters)**(power))

    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr

    