import numpy as np
import torch
from torch import nn
from medpy.metric.binary import assd,dc
from datetime import datetime
import scipy.io as scio
import os.path as osp
import torch.backends.cudnn as cudnn
import os
from skimage.exposure import equalize_hist
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm   


BATCHSIZE     = 32
data_size     = [256, 256, 1]
label_size    = [256, 256, 1]
NUMCLASS      = 5

def _compute_metric(pred,target):

    pred = pred.astype(int)
    target = target.astype(int)
    dice_list  = []
    assd_list  = []
    pred_each_class_number = []
    true_each_class_number = []

    for c in range(1, NUMCLASS):
        test_pred = pred.copy()
        test_pred[test_pred != c] = 0
        test_gt = target.copy()
        test_gt[test_gt != c] = 0

        dice = dc(test_pred, test_gt)
        try:
            assd_metric = assd(test_pred, test_gt)
        except:
            print('assd error')
            assd_metric = 1

        dice_list.append(dice)
        assd_list.append(assd_metric)

    return  np.array(dice_list),np.array(assd_list)



def evaluation_Cardiac(model,TARGET_MODALITY,results_file,i_iter):
    if TARGET_MODALITY == 'CT':
        test_list_pth = '/home/zhr/MPSCL/data/datalist/test_ct.txt'
    if TARGET_MODALITY == 'MR':
        test_list_pth = '/home/zhr/MPSCL/data/datalist/test_mr.txt'
    with open(test_list_pth) as fp:
        rows = fp.readlines()
    testfile_list = [row[:-1] for row in rows]
    
    interp = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)

    img_mean   = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    model.eval()
    model.cuda()
    cudnn.benchmark = True
    cudnn.enabled = True

    dice_list = []
    assd_list = []
    for idx_file, fid in enumerate(testfile_list):
        _npz_dict = np.load(fid)
        data      = _npz_dict['arr_0']
        label     = _npz_dict['arr_1']

        if True:
            data = np.flip(data, axis=0)
            data = np.flip(data, axis=1)
            label = np.flip(label, axis=0)
            label = np.flip(label, axis=1)

        tmp_pred = np.zeros(label.shape)
        frame_list = [kk for kk in range(data.shape[2])]
        # pred_start_time = datetime.now()

        for ii in range(int(np.floor(data.shape[2] // BATCHSIZE))):
            data_batch = np.zeros([BATCHSIZE, 3, 256, 256])
            for idx, jj in enumerate(frame_list[ii * BATCHSIZE: (ii + 1) * BATCHSIZE]):
                item_data = data[..., jj]

                if TARGET_MODALITY == 'CT':
                    item_data = np.subtract(
                        np.multiply(np.divide(np.subtract(item_data, -2.8), np.subtract(3.2, -2.8)), 2.0),
                        1)  # {-2.8, 3.2} need to be changed according to the metadata statistics
                elif TARGET_MODALITY == 'MR':
                    item_data = np.subtract(
                        np.multiply(np.divide(np.subtract(item_data, -1.8), np.subtract(4.4, -1.8)), 2.0),
                        1)  # {-1.8, 4.4} need to be changed according to the metadata statistics
                item_data = np.expand_dims(item_data, -1)
                item_data = np.tile(item_data, [1, 1, 3])
                item_data = (item_data + 1) * 127.5
                item_data = item_data[:, :, ::-1].copy()  # change to BGR
                item_data -= img_mean
                item_data = np.transpose(item_data, [2, 0, 1])
                data_batch[idx, ...] = item_data

            imgs = torch.from_numpy(data_batch).cuda().float()
            with torch.no_grad():
                cla_feas_src,pred_b_aux, pred_b_main = model(imgs)

                pred_b_main = interp(pred_b_main)
                pred_b_main = torch.argmax(pred_b_main, dim=1)
                pred_b_main = pred_b_main.cpu().data.numpy()
            for idx, jj in enumerate(frame_list[ii * BATCHSIZE: (ii + 1) * BATCHSIZE]):
                tmp_pred[..., jj] = pred_b_main[idx, ...].copy()

        label = label.astype(int)
        dice, assd             = _compute_metric(tmp_pred,label)

        dice_list.append(dice)
        assd_list.append(assd)

    dice_arr = np.vstack(dice_list) #N_CT * N_Class
    assd_arr = np.vstack(assd_list) #N_CT * N_Class

    dice_arr  = 100 * dice_arr.transpose()  #N_Class * N_CT
    dice_mean = np.mean(dice_arr, axis=1) #N_Class
    dice_std  = np.std(dice_arr, axis=1) #N_Class

    print('dice arr is {}'.format(dice_arr.shape))
    print('Dice:')
    print('AA :%.2f(%.2f)' % (dice_mean[3], dice_std[3]))
    print('LAC:%.2f(%.2f)' % (dice_mean[1], dice_std[1]))
    print('LVC:%.2f(%.2f)' % (dice_mean[2], dice_std[2]))
    print('Myo:%.2f(%.2f)' % (dice_mean[0], dice_std[0]))
    print('Mean:%.2f' % np.mean(dice_mean))

    assd_arr  = assd_arr.transpose() #N_Class * N_CT
    assd_mean = np.mean(assd_arr, axis=1)
    assd_std  = np.std(assd_arr, axis=1)

    print('ASSD:')
    print('AA :%.2f(%.2f)' % (assd_mean[3], assd_std[3]))
    print('LAC:%.2f(%.2f)' % (assd_mean[1], assd_std[1]))
    print('LVC:%.2f(%.2f)' % (assd_mean[2], assd_std[2]))
    print('Myo:%.2f(%.2f)' % (assd_mean[0], assd_std[0]))
    print('Mean:%.2f' % np.mean(assd_mean))

    with open(results_file, "a") as f:
        info = f"[i_iter: {i_iter}]\n" \
            f"Dice:\n" \
            f"AA : {dice_mean[3]:.2f}({dice_std[3]:.2f})\n" \
            f"LAC: {dice_mean[1]:.2f}({dice_std[1]:.2f})\n" \
            f"LVC: {dice_mean[2]:.2f}({dice_std[2]:.2f})\n" \
            f"Myo: {dice_mean[0]:.2f}({dice_std[0]:.2f})\n" \
            f"Dice_Mean: {np.mean(dice_mean):.2f}\n" \
            f"ASSD:\n" \
            f"AA : {assd_mean[3]:.2f}({assd_std[3]:.2f})\n" \
            f"LAC: {assd_mean[1]:.2f}({assd_std[1]:.2f})\n" \
            f"LVC: {assd_mean[2]:.2f}({assd_std[2]:.2f})\n" \
            f"Myo: {assd_mean[0]:.2f}({assd_std[0]:.2f})\n" \
            f"ASSD_Mean: {np.mean(assd_mean):.2f}\n" 
        f.write(info + "\n\n")

    return dice_mean,dice_std,assd_mean,assd_std


def evaluation_Cardiac_gram(model,TARGET_MODALITY,results_file,i_iter):
    if TARGET_MODALITY == 'CT':
        test_list_pth = '/home/zhr/MPSCL/data/datalist/test_ct.txt'
    if TARGET_MODALITY == 'MR':
        test_list_pth = '/home/zhr/MPSCL/data/datalist/test_mr.txt'
    with open(test_list_pth) as fp:
        rows = fp.readlines()
    testfile_list = [row[:-1] for row in rows]
    
    interp = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)

    img_mean   = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    model.eval()
    model.cuda()
    cudnn.benchmark = True
    cudnn.enabled = True

    dice_list = []
    assd_list = []
    for idx_file, fid in enumerate(testfile_list):
        _npz_dict = np.load(fid)
        data      = _npz_dict['arr_0']
        label     = _npz_dict['arr_1']

        if True:
            data = np.flip(data, axis=0)
            data = np.flip(data, axis=1)
            label = np.flip(label, axis=0)
            label = np.flip(label, axis=1)

        tmp_pred = np.zeros(label.shape)
        frame_list = [kk for kk in range(data.shape[2])]
        # pred_start_time = datetime.now()

        for ii in range(int(np.floor(data.shape[2] // BATCHSIZE))):
            data_batch = np.zeros([BATCHSIZE, 3, 256, 256])
            for idx, jj in enumerate(frame_list[ii * BATCHSIZE: (ii + 1) * BATCHSIZE]):
                item_data = data[..., jj]

                if TARGET_MODALITY == 'CT':
                    item_data = np.subtract(
                        np.multiply(np.divide(np.subtract(item_data, -2.8), np.subtract(3.2, -2.8)), 2.0),
                        1)  # {-2.8, 3.2} need to be changed according to the metadata statistics
                elif TARGET_MODALITY == 'MR':
                    item_data = np.subtract(
                        np.multiply(np.divide(np.subtract(item_data, -1.8), np.subtract(4.4, -1.8)), 2.0),
                        1)  # {-1.8, 4.4} need to be changed according to the metadata statistics
                item_data = np.expand_dims(item_data, -1)
                item_data = np.tile(item_data, [1, 1, 3])
                item_data = (item_data + 1) * 127.5
                item_data = item_data[:, :, ::-1].copy()  # change to BGR
                item_data -= img_mean
                item_data = np.transpose(item_data, [2, 0, 1])
                data_batch[idx, ...] = item_data

            imgs = torch.from_numpy(data_batch).cuda().float()
            with torch.no_grad():
                _,cla_feas_src,pred_b_aux, pred_b_main = model(imgs)

                pred_b_main = interp(pred_b_main)
                pred_b_main = torch.argmax(pred_b_main, dim=1)
                pred_b_main = pred_b_main.cpu().data.numpy()
            for idx, jj in enumerate(frame_list[ii * BATCHSIZE: (ii + 1) * BATCHSIZE]):
                tmp_pred[..., jj] = pred_b_main[idx, ...].copy()

        label = label.astype(int)
        dice, assd             = _compute_metric(tmp_pred,label)

        dice_list.append(dice)
        assd_list.append(assd)

    dice_arr = np.vstack(dice_list) #N_CT * N_Class
    assd_arr = np.vstack(assd_list) #N_CT * N_Class

    dice_arr  = 100 * dice_arr.transpose()  #N_Class * N_CT
    dice_mean = np.mean(dice_arr, axis=1) #N_Class
    dice_std  = np.std(dice_arr, axis=1) #N_Class

    print('dice arr is {}'.format(dice_arr.shape))
    print('Dice:')
    print('AA :%.2f(%.2f)' % (dice_mean[3], dice_std[3]))
    print('LAC:%.2f(%.2f)' % (dice_mean[1], dice_std[1]))
    print('LVC:%.2f(%.2f)' % (dice_mean[2], dice_std[2]))
    print('Myo:%.2f(%.2f)' % (dice_mean[0], dice_std[0]))
    print('Mean:%.2f' % np.mean(dice_mean))

    assd_arr  = assd_arr.transpose() #N_Class * N_CT
    assd_mean = np.mean(assd_arr, axis=1)
    assd_std  = np.std(assd_arr, axis=1)

    print('ASSD:')
    print('AA :%.2f(%.2f)' % (assd_mean[3], assd_std[3]))
    print('LAC:%.2f(%.2f)' % (assd_mean[1], assd_std[1]))
    print('LVC:%.2f(%.2f)' % (assd_mean[2], assd_std[2]))
    print('Myo:%.2f(%.2f)' % (assd_mean[0], assd_std[0]))
    print('Mean:%.2f' % np.mean(assd_mean))

    with open(results_file, "a") as f:
        info = f"[i_iter: {i_iter}]\n" \
            f"Dice:\n" \
            f"AA : {dice_mean[3]:.2f}({dice_std[3]:.2f})\n" \
            f"LAC: {dice_mean[1]:.2f}({dice_std[1]:.2f})\n" \
            f"LVC: {dice_mean[2]:.2f}({dice_std[2]:.2f})\n" \
            f"Myo: {dice_mean[0]:.2f}({dice_std[0]:.2f})\n" \
            f"Dice_Mean: {np.mean(dice_mean):.2f}\n" \
            f"ASSD:\n" \
            f"AA : {assd_mean[3]:.2f}({assd_std[3]:.2f})\n" \
            f"LAC: {assd_mean[1]:.2f}({assd_std[1]:.2f})\n" \
            f"LVC: {assd_mean[2]:.2f}({assd_std[2]:.2f})\n" \
            f"Myo: {assd_mean[0]:.2f}({assd_std[0]:.2f})\n" \
            f"ASSD_Mean: {np.mean(assd_mean):.2f}\n" 
        f.write(info + "\n\n")

    return dice_mean,dice_std,assd_mean,assd_std


def evaluation_Abdomen(model,TARGET_MODALITY,results_file,i_iter):
    if TARGET_MODALITY == 'CT':
        test_list_pth = '/home/data_backup/abdominalDATA/val_ct.txt'
    if TARGET_MODALITY == 'MR':
        test_list_pth = '/home/data_backup/abdominalDATA/val_mr.txt'
    with open(test_list_pth) as fp:
        rows = fp.readlines()
    testfile_list = [row[:-1] for row in rows]

    interp = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)

    img_mean   = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    model.eval()
    model.cuda()
    cudnn.benchmark = True
    cudnn.enabled = True

    dice_list = []
    assd_list = []
    num_slice = 100
    # print(len(testfile_list))
    n = int(len(testfile_list)/num_slice)
    # print(n)
    for ii in range(n):
        tmp_pred = np.zeros([256,256,num_slice])
        label = np.zeros([256,256,num_slice])

        for jj in range(num_slice):
            fid = testfile_list[ii*num_slice+jj]
            _npz_dict = np.load(fid)
            item_data      = _npz_dict['arr_0']
            item_label     = _npz_dict['arr_1']
            label[..., jj] = item_label.copy()

            if TARGET_MODALITY == 'CT':
                item_data = np.subtract(
                    np.multiply(np.divide(np.subtract(item_data, -1.4), np.subtract(3.6,-1.4)), 2.0),
                    1)  # {-2.8, 3.2} need to be changed according to the metadata statistics
            elif TARGET_MODALITY == 'MR':
                item_data = np.subtract(
                    np.multiply(np.divide(np.subtract(item_data, -1.3), np.subtract(4.3,-1.3)), 2.0),
                    1)  # {-1.8, 4.4} need to be changed according to the metadata statistics
            item_data = np.expand_dims(item_data, -1)
            item_data = np.tile(item_data, [1, 1, 3])
            item_data = (item_data + 1) * 127.5
            item_data = item_data[:, :, ::-1].copy()  # change to BGR
            item_data -= img_mean
            item_data = np.transpose(item_data, [2, 0, 1])
            item_data = item_data[None, ...]

            imgs = torch.from_numpy(item_data).cuda().float()
            with torch.no_grad():
                cla_feas_src, pred_b_aux, pred_b_main = model(imgs)

                pred_b_main = interp(pred_b_main)
                pred_b_main = torch.argmax(pred_b_main, dim=1)
                pred_b_main = pred_b_main.cpu().data.numpy()
                tmp_pred[..., jj] = pred_b_main[0, ...].copy()


        label = label.astype(int)

        dice, assd             = _compute_metric(tmp_pred,label)

        dice_list.append(dice)
        assd_list.append(assd)

    print(dice_list)
    dice_arr = np.vstack(dice_list) #N_CT * N_Class
    assd_arr = np.vstack(assd_list) #N_CT * N_Class

    dice_arr  = 100 * dice_arr.transpose()  #N_Class * N_CT
    dice_mean = np.mean(dice_arr, axis=1) #N_Class
    dice_std  = np.std(dice_arr, axis=1) #N_Class

    print('dice arr is {}'.format(dice_arr.shape))
    print('Dice:')
    print('Liver :%.2f(%.2f)' % (dice_mean[0], dice_std[0]))
    print('R.kidney:%.2f(%.2f)' % (dice_mean[1], dice_std[1]))
    print('L.kidney:%.2f(%.2f)' % (dice_mean[2], dice_std[2]))
    print('Spleen:%.2f(%.2f)' % (dice_mean[3], dice_std[3]))
    print('Mean:%.2f' % np.mean(dice_mean))

    assd_arr  = assd_arr.transpose() #N_Class * N_CT
    assd_mean = np.mean(assd_arr, axis=1)
    assd_std  = np.std(assd_arr, axis=1)

    print('ASSD:')
    print('Liver :%.2f(%.2f)' % (assd_mean[0], assd_std[0]))
    print('R.kidney:%.2f(%.2f)' % (assd_mean[1], assd_std[1]))
    print('L.kidney:%.2f(%.2f)' % (assd_mean[2], assd_std[2]))
    print('Spleen:%.2f(%.2f)' % (assd_mean[3], assd_std[3]))
    print('Mean:%.2f' % np.mean(assd_mean))

    with open(results_file, "a") as f:
        info = f"[i_iter: {i_iter}]\n" \
            f"Dice:\n" \
            f"Liver : {dice_mean[0]:.2f}({dice_std[0]:.2f})\n" \
            f"R.kidney: {dice_mean[1]:.2f}({dice_std[1]:.2f})\n" \
            f"L.kidney: {dice_mean[2]:.2f}({dice_std[2]:.2f})\n" \
            f"Spleen: {dice_mean[3]:.2f}({dice_std[3]:.2f})\n" \
            f"Dice_Mean: {np.mean(dice_mean):.2f}\n" \
            f"ASSD:\n" \
            f"Liver : {assd_mean[0]:.2f}({assd_std[0]:.2f})\n" \
            f"R.kidney: {assd_mean[1]:.2f}({assd_std[1]:.2f})\n" \
            f"L.kidney: {assd_mean[2]:.2f}({assd_std[2]:.2f})\n" \
            f"Spleen: {assd_mean[3]:.2f}({assd_std[3]:.2f})\n" \
            f"ASSD_Mean: {np.mean(assd_mean):.2f}\n" 
        f.write(info + "\n\n")

    return dice_mean,dice_std,assd_mean,assd_std
    



def evaluation_Cardiac_eh(model,TARGET_MODALITY,results_file,i_iter):
    if TARGET_MODALITY == 'CT':
        test_list_pth = '/home/zhr/MPSCL/data/datalist/test_ct.txt'
    if TARGET_MODALITY == 'MR':
        test_list_pth = '/home/zhr/MPSCL/data/datalist/test_mr.txt'
    with open(test_list_pth) as fp:
        rows = fp.readlines()
    testfile_list = [row[:-1] for row in rows]
    
    interp = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)

    img_mean   = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    model.eval()
    model.cuda()
    cudnn.benchmark = True
    cudnn.enabled = True

    dice_list = []
    assd_list = []
    for idx_file, fid in enumerate(testfile_list):
        _npz_dict = np.load(fid)
        data      = _npz_dict['arr_0']
        label     = _npz_dict['arr_1']

        if True:
            data = np.flip(data, axis=0)
            data = np.flip(data, axis=1)
            label = np.flip(label, axis=0)
            label = np.flip(label, axis=1)

        tmp_pred = np.zeros(label.shape)
        frame_list = [kk for kk in range(data.shape[2])]
        # pred_start_time = datetime.now()

        for ii in range(int(np.floor(data.shape[2] // BATCHSIZE))):
            data_batch = np.zeros([BATCHSIZE, 3, 256, 256])
            for idx, jj in enumerate(frame_list[ii * BATCHSIZE: (ii + 1) * BATCHSIZE]):
                item_data = data[..., jj]

                if TARGET_MODALITY == 'CT':
                    item_data = np.subtract(
                        np.multiply(np.divide(np.subtract(item_data, -2.8), np.subtract(3.2, -2.8)), 2.0),
                        1)  # {-2.8, 3.2} need to be changed according to the metadata statistics
                elif TARGET_MODALITY == 'MR':
                    item_data = np.subtract(
                        np.multiply(np.divide(np.subtract(item_data, -1.8), np.subtract(4.4, -1.8)), 2.0),
                        1)  # {-1.8, 4.4} need to be changed according to the metadata statistics
                item_data = np.expand_dims(item_data, -1)
                item_data = np.tile(item_data, [1, 1, 3])
                item_data = (item_data + 1) * 127.5
                item_data = item_data[:, :, ::-1].copy()  # change to BGR
                item_data -= img_mean
                item_data = np.transpose(item_data, [2, 0, 1])
                item_data = equalize_hist(np.array(item_data))
                data_batch[idx, ...] = item_data

            imgs = torch.from_numpy(data_batch).cuda().float()
            with torch.no_grad():
                cla_feas_src,pred_b_aux, pred_b_main = model(imgs)

                pred_b_main = interp(pred_b_main)
                pred_b_main = torch.argmax(pred_b_main, dim=1)
                pred_b_main = pred_b_main.cpu().data.numpy()
            for idx, jj in enumerate(frame_list[ii * BATCHSIZE: (ii + 1) * BATCHSIZE]):
                tmp_pred[..., jj] = pred_b_main[idx, ...].copy()

        label = label.astype(int)
        dice, assd             = _compute_metric(tmp_pred,label)

        dice_list.append(dice)
        assd_list.append(assd)

    dice_arr = np.vstack(dice_list) #N_CT * N_Class
    assd_arr = np.vstack(assd_list) #N_CT * N_Class

    dice_arr  = 100 * dice_arr.transpose()  #N_Class * N_CT
    dice_mean = np.mean(dice_arr, axis=1) #N_Class
    dice_std  = np.std(dice_arr, axis=1) #N_Class

    print('dice arr is {}'.format(dice_arr.shape))
    print('Dice:')
    print('AA :%.2f(%.2f)' % (dice_mean[3], dice_std[3]))
    print('LAC:%.2f(%.2f)' % (dice_mean[1], dice_std[1]))
    print('LVC:%.2f(%.2f)' % (dice_mean[2], dice_std[2]))
    print('Myo:%.2f(%.2f)' % (dice_mean[0], dice_std[0]))
    print('Mean:%.2f' % np.mean(dice_mean))

    assd_arr  = assd_arr.transpose() #N_Class * N_CT
    assd_mean = np.mean(assd_arr, axis=1)
    assd_std  = np.std(assd_arr, axis=1)

    print('ASSD:')
    print('AA :%.2f(%.2f)' % (assd_mean[3], assd_std[3]))
    print('LAC:%.2f(%.2f)' % (assd_mean[1], assd_std[1]))
    print('LVC:%.2f(%.2f)' % (assd_mean[2], assd_std[2]))
    print('Myo:%.2f(%.2f)' % (assd_mean[0], assd_std[0]))
    print('Mean:%.2f' % np.mean(assd_mean))

    with open(results_file, "a") as f:
        info = f"[i_iter: {i_iter}]\n" \
            f"Dice:\n" \
            f"AA : {dice_mean[3]:.2f}({dice_std[3]:.2f})\n" \
            f"LAC: {dice_mean[1]:.2f}({dice_std[1]:.2f})\n" \
            f"LVC: {dice_mean[2]:.2f}({dice_std[2]:.2f})\n" \
            f"Myo: {dice_mean[0]:.2f}({dice_std[0]:.2f})\n" \
            f"Dice_Mean: {np.mean(dice_mean):.2f}\n" \
            f"ASSD:\n" \
            f"AA : {assd_mean[3]:.2f}({assd_std[3]:.2f})\n" \
            f"LAC: {assd_mean[1]:.2f}({assd_std[1]:.2f})\n" \
            f"LVC: {assd_mean[2]:.2f}({assd_std[2]:.2f})\n" \
            f"Myo: {assd_mean[0]:.2f}({assd_std[0]:.2f})\n" \
            f"ASSD_Mean: {np.mean(assd_mean):.2f}\n" 
        f.write(info + "\n\n")

    return dice_mean,dice_std,assd_mean,assd_std



def evaluation_Abdomen_original(model,TARGET_MODALITY,results_file,i_iter):
    if TARGET_MODALITY == 'CT':
        test_list_pth = '/home/data_backup/abdominalDATA/test_ct'
    if TARGET_MODALITY == 'MR':
        test_list_pth = '/home/data_backup/abdominalDATA/test_mr'

    txtfile = sorted([os.path.join(test_list_pth, f) for f in os.listdir(test_list_pth)])
    
    interp = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)

    img_mean   = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    model.eval()
    model.cuda()
    cudnn.benchmark = True
    cudnn.enabled = True

    dice_list = []
    assd_list = []
    
    n = len(txtfile)
    for ii in range(n):

        test_list_pth = txtfile[ii]
        with open(test_list_pth) as fp:
            rows = fp.readlines()
        testfile_list = [row[:-1] for row in rows]
        num_slice = len(testfile_list)
        tmp_pred = np.zeros([256,256,num_slice])
        label = np.zeros([256,256,num_slice])

        for jj in range(num_slice):
            fid = testfile_list[jj]
            _npz_dict = np.load(fid)
            item_data      = _npz_dict['arr_0']
            item_label     = _npz_dict['arr_1']
            label[..., jj] = item_label.copy()

            if TARGET_MODALITY == 'CT':
                item_data = np.subtract(
                    np.multiply(np.divide(np.subtract(item_data, -1.4), np.subtract(3.6,-1.4)), 2.0),
                    1)  # {-2.8, 3.2} need to be changed according to the metadata statistics
            elif TARGET_MODALITY == 'MR':
                item_data = np.subtract(
                    np.multiply(np.divide(np.subtract(item_data, -1.3), np.subtract(4.3,-1.3)), 2.0),
                    1)  # {-1.8, 4.4} need to be changed according to the metadata statistics
            item_data = np.expand_dims(item_data, -1)
            item_data = np.tile(item_data, [1, 1, 3])
            item_data = (item_data + 1) * 127.5
            item_data = item_data[:, :, ::-1].copy()  # change to BGR
            item_data -= img_mean
            item_data = np.transpose(item_data, [2, 0, 1])
            item_data = item_data[None, ...]


            imgs = torch.from_numpy(item_data).cuda().float()
            with torch.no_grad():
                cla_feas_src, pred_b_aux, pred_b_main = model(imgs)

                pred_b_main = interp(pred_b_main)
                pred_b_main = torch.argmax(pred_b_main, dim=1)
                pred_b_main = pred_b_main.cpu().data.numpy()
                tmp_pred[..., jj] = pred_b_main[0, ...].copy()


        label = label.astype(int)

        dice, assd             = _compute_metric(tmp_pred,label)

        dice_list.append(dice)
        assd_list.append(assd)

    dice_arr = np.vstack(dice_list) #N_CT * N_Class
    assd_arr = np.vstack(assd_list) #N_CT * N_Class

    dice_arr  = 100 * dice_arr.transpose()  #N_Class * N_CT
    dice_mean = np.mean(dice_arr, axis=1) #N_Class
    dice_std  = np.std(dice_arr, axis=1) #N_Class

    print('dice arr is {}'.format(dice_arr.shape))
    print('Dice:')
    print('Liver :%.2f(%.2f)' % (dice_mean[0], dice_std[0]))
    print('R.kidney:%.2f(%.2f)' % (dice_mean[1], dice_std[1]))
    print('L.kidney:%.2f(%.2f)' % (dice_mean[2], dice_std[2]))
    print('Spleen:%.2f(%.2f)' % (dice_mean[3], dice_std[3]))
    print('Mean:%.2f' % np.mean(dice_mean))

    assd_arr  = assd_arr.transpose() #N_Class * N_CT
    assd_mean = np.mean(assd_arr, axis=1)
    assd_std  = np.std(assd_arr, axis=1)

    print('ASSD:')
    print('Liver :%.2f(%.2f)' % (assd_mean[0], assd_std[0]))
    print('R.kidney:%.2f(%.2f)' % (assd_mean[1], assd_std[1]))
    print('L.kidney:%.2f(%.2f)' % (assd_mean[2], assd_std[2]))
    print('Spleen:%.2f(%.2f)' % (assd_mean[3], assd_std[3]))
    print('Mean:%.2f' % np.mean(assd_mean))

    with open(results_file, "a") as f:
        info = f"[i_iter: {i_iter}]\n" \
            f"Dice:\n" \
            f"Liver : {dice_mean[0]:.2f}({dice_std[0]:.2f})\n" \
            f"R.kidney: {dice_mean[1]:.2f}({dice_std[1]:.2f})\n" \
            f"L.kidney: {dice_mean[2]:.2f}({dice_std[2]:.2f})\n" \
            f"Spleen: {dice_mean[3]:.2f}({dice_std[3]:.2f})\n" \
            f"Dice_Mean: {np.mean(dice_mean):.2f}\n" \
            f"ASSD:\n" \
            f"Liver : {assd_mean[0]:.2f}({assd_std[0]:.2f})\n" \
            f"R.kidney: {assd_mean[1]:.2f}({assd_std[1]:.2f})\n" \
            f"L.kidney: {assd_mean[2]:.2f}({assd_std[2]:.2f})\n" \
            f"Spleen: {assd_mean[3]:.2f}({assd_std[3]:.2f})\n" \
            f"ASSD_Mean: {np.mean(assd_mean):.2f}\n" 
        f.write(info + "\n\n")

    return dice_mean,dice_std,assd_mean,assd_std





def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, jc, hd95, asd


def test_single_volume(case, net, test_save_path, root_path):
    h5f = h5py.File(root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        input = torch.tile(input, (1,3,1,1))
        interp = nn.Upsample(size=(slice.shape[1],slice.shape[0]),mode='bilinear',
                         align_corners=True)
        net.eval()
        with torch.no_grad():
            _,pred_src_aux, out_main = net(input)
            # print(out_main.shape)
            out_main = interp(out_main)
            # print(out_main.shape, slice.shape)
            if len(out_main)>1:
                out_main=out_main[0]
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred
    if np.sum(prediction == 1)==0:
        first_metric = 0,0,0,0
    else:
        first_metric = calculate_metric_percase(prediction == 1, label == 1)

    if np.sum(prediction == 2)==0:
        second_metric = 0,0,0,0
    else:
        second_metric = calculate_metric_percase(prediction == 2, label == 2)

    if np.sum(prediction == 3)==0:
        third_metric = 0,0,0,0
    else:
        third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    # sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    # sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    # sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric



def ACDC_evaluation(model, i_iter, cfg):
    root_path = '/home/data_backup/ACDC'
    with open(root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    
    test_save_path = "/home/lyn/CVPR_TMI/scripts/log_ACDC/{}.txt".format(cfg.TRAIN.DA_METHOD)
    # if os.path.exists(test_save_path):
    #     shutil.rmtree(test_save_path)
    # os.makedirs(test_save_path)
    net = model
   
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(case, net, test_save_path, root_path)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    avg_metric = [first_total / len(image_list), second_total / len(image_list), third_total / len(image_list)]
    print(avg_metric)
    print((avg_metric[0]+avg_metric[1]+avg_metric[2])/3)
    with open(test_save_path, 'w') as f:
        f.writelines('metric is {} \n'.format(avg_metric))
        f.writelines('average metric is {}\n'.format((avg_metric[0]+avg_metric[1]+avg_metric[2])/3))

    return avg_metric, test_save_path
