import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__)) # 获取当前绝对路径C
sys.path.append(curPath)
rootPath = os.path.split(curPath)[0]				 # 上一级目录B
sys.path.append(rootPath)
sys.path.append(os.path.split(rootPath)[0])

import numpy as np
import torch
from torch import nn
from model.deeplabv2 import get_deeplab_v2
import torch.backends.cudnn as cudnn
from PIL import Image
import torch.nn.functional as F
import cv2
from medpy.metric.binary import assd,dc

NUMCLASS      = 5
BATCHSIZE     = 32

def get_heatmap(     mask: np.ndarray,
                      use_rgb: bool = True,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
# mask必须在[0,1]之间
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
# 因为cv2读取出来是BGR，要转为RGB
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return np.uint8(heatmap)

def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-7)) / np.log2(c)

def normalize_img( img):
    return (img - img.min())/(img.max() - img.min())

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

def make_dir(out_dir,TARGET_MODALITY):
    os.makedirs(os.path.join(out_dir, TARGET_MODALITY, 'image'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, TARGET_MODALITY, 'label'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, TARGET_MODALITY, 'pred'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, TARGET_MODALITY, 'entropy'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, TARGET_MODALITY, 'merge'), exist_ok=True)

if __name__ == '__main__':
    model_path = '/home/data_backup/zhr_savedmodel/MR2CT_bcutmix_ST_contrastive_boundary_final_10000_smcontra_dim256_queuelen500_temperature1_new505/model_35500.pth'
    out_dir ='/home/data_backup/cardiac_save_data/ICME_SOTA'
    TARGET_MODALITY = 'CT'
    make_dir(out_dir, TARGET_MODALITY)
    pallette = [0, 0, 0, 203, 95, 95, 116, 177, 86, 237, 186, 125, 107, 157, 198]
    if TARGET_MODALITY == 'CT':
        test_list_pth = '/home/zhr/MPSCL/data/datalist/test_ct.txt'
    if TARGET_MODALITY == 'MR':
        test_list_pth = '/home/zhr/MPSCL/data/datalist/test_mr.txt'


    model = get_deeplab_v2(num_classes=5, multi_level=True)
    saved_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(saved_state_dict)


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
          
        tmp_data = np.zeros(label.shape)
        tmp_label = label.astype(int)
        tmp_pred = np.zeros(label.shape)
        tmp_entropy = np.zeros(label.shape)
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
                tmp_data[..., jj] = (item_data + 1) * 127.5
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
                main_soft = F.softmax(pred_b_main, dim=1) #batch 5 256 256
                max_probs_main, pred_b_main  = torch.max(main_soft, dim=1)
                # pred_b_main = torch.argmax(pred_b_main, dim=1)
                entropy_label = prob_2_entropy(main_soft) #batch 5 256 256
                entropy_label = torch.sum(entropy_label, axis=1) #batch 256 256
                pred_b_main = pred_b_main.cpu().data.numpy()
                entropy_label = entropy_label.cpu().data.numpy()
            for idx, jj in enumerate(frame_list[ii * BATCHSIZE: (ii + 1) * BATCHSIZE]):
                tmp_pred[..., jj] = pred_b_main[idx, ...].copy()
                tmp_entropy[..., jj] = entropy_label[idx, ...].copy()
                # print(np.unique(tmp_entropy[..., jj]))
                # print(np.unique((255*tmp_entropy[..., jj]).astype(np.uint8)))
                # datajj = Image.fromarray(tmp_data[..., jj].astype(np.uint8))
                # datajj.save(os.path.join(out_dir, TARGET_MODALITY, 'image', str(idx_file) + '_' + str(jj) + '.png'))
                import matplotlib.pyplot as plt
                plt.imsave(os.path.join(out_dir, TARGET_MODALITY, 'image', str(idx_file) + '_' + str(jj) + '.png'), tmp_data[..., jj], cmap='gray')
                datajj = Image.open(os.path.join(out_dir, TARGET_MODALITY, 'image', str(idx_file) + '_' + str(jj) + '.png'))
        
                labeljj = Image.fromarray(tmp_label[..., jj].astype(np.uint8))
                labeljj.putpalette(pallette)
                labeljj.save(os.path.join(out_dir, TARGET_MODALITY, 'label', str(idx_file) + '_' + str(jj) + '.png'))
                
                prejj = Image.fromarray(tmp_pred[..., jj].astype(np.uint8))
                prejj.putpalette(pallette)
                prejj.save(os.path.join(out_dir, TARGET_MODALITY, 'pred', str(idx_file) + '_' + str(jj) + '.png'))
                
                entropyjj = Image.fromarray(get_heatmap(tmp_entropy[..., jj]))
                entropyjj.save(os.path.join(out_dir, TARGET_MODALITY, 'entropy', str(idx_file) + '_' + str(jj) + '.png'))
                
                # entropyjj = Image.fromarray((255*tmp_entropy[..., jj]).astype(np.uint8)).convert('RGB')
                # entropyjj.save(os.path.join(out_dir, TARGET_MODALITY, 'entropy', str(idx_file) + '_' + str(jj) + '.png'))
                
                # # 通过matplotlib画图
                # import matplotlib.pyplot as plt
                # plt.imsave(os.path.join(out_dir, TARGET_MODALITY, 'entropy', str(idx_file) + '_' + str(jj) + '.png'), tmp_entropy[..., jj])
                # entropyjj = Image.open(os.path.join(out_dir, TARGET_MODALITY, 'entropy', str(idx_file) + '_' + str(jj) + '.png'))
                
                w, h = datajj.size
                mergejj = Image.new('RGB', (4 * w, 1 * h))
                mergejj.paste(datajj, (0 * w, 0))
                mergejj.paste(labeljj, (1 * w, 0))
                mergejj.paste(prejj, (2 * w, 0))
                mergejj.paste(entropyjj, (3 * w, 0))
                mergejj.save(os.path.join(out_dir, TARGET_MODALITY, 'merge', str(idx_file) + '_' + str(jj) + '.png'))

        label = label.astype(int)
        dice, ssd             = _compute_metric(tmp_pred,label)

        dice_list.append(dice)
        assd_list.append(ssd)

    dice_arr = np.vstack(dice_list) #N_CT * N_Class
    assd_arr = np.vstack(assd_list) #N_CT * N_Class

    dice_arr  = 100 * dice_arr.transpose()  #N_Class * N_CT
    dice_mean = np.mean(dice_arr, axis=1) #N_Class
    dice_std  = np.std(dice_arr, axis=1) #N_Class

    print('model_name: {}'.format(os.path.basename(model_path)))
    print('TARGET_MODALITY: {}'.format(TARGET_MODALITY))
    
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

    results_file = os.path.join(out_dir, TARGET_MODALITY, 'result.txt')
    with open(results_file, "a") as f:
        info = f"[model_name: {os.path.basename(model_path)}]\n" \
            f"[TARGET_MODALITY: {TARGET_MODALITY}]\n" \
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




