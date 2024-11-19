import os
import sys
sys.path.append('..')
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
# from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm
from model.discriminator import get_discriminatord
from utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from utils.func import loss_calc, bce_loss
from utils.loss import entropy_loss, dice_loss
from utils.func import prob_2_entropy, FDA_source_to_target_np
from domain_adaptation.eval_UDA import evaluation_Cardiac, evaluation_Abdomen, evaluation_Abdomen_original, evaluation_Cardiac_eh, evaluation_Cardiac_gram
import matplotlib.pyplot as plt
plt.switch_backend("agg")
import datetime
from skimage.exposure import match_histograms, equalize_hist
from skimage.measure import label
from skimage.filters import gaussian
from scipy import ndimage
import math



class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth ) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss
    
    def _dice_mask_loss(self, score, target, mask):
        target = target.float()
        mask = mask.float()
        smooth = 1e-10
        intersect = torch.sum(score * target * mask)
        y_sum = torch.sum(target * target * mask)
        z_sum = torch.sum(score * score * mask)
        loss = (2 * intersect + smooth ) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, mask=None, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target.unsqueeze(1))
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        if mask is not None:
            # bug found by @CamillerFerros at github issue#25
            mask = mask.repeat(1, self.n_classes, 1, 1).type(torch.float32)
            for i in range(0, self.n_classes): 
                dice = self._dice_mask_loss(inputs[:, i], target[:, i], mask[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        else:
            for i in range(0, self.n_classes):
                dice = self._dice_loss(inputs[:, i], target[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        return loss / self.n_classes



def generate_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x*1/2), int(img_y*1/2)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w+patch_x, h:h+patch_y] = 0
    crop_img = img[:, :, w:w+patch_x, h:h+patch_y]
    return mask.long(), crop_img


def cut_generate_mask(img):
    channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2]
    mask = torch.zeros(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x*1/2), int(img_y*1/2)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w+patch_x, h:h+patch_y] = 1
    crop_img = img[:, w:w+patch_x, h:h+patch_y]
    return mask.long(), crop_img


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
    

def get_current_consistency_weight(epoch, max_iter):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 1*sigmoid_rampup(epoch, max_iter)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


class BoundaryLoss(object):
    def __init__(self):
        self.get_boundary = GetBoundary()
        self.mse_loss = nn.MSELoss()

    def __call__(self, mask, label):

        batch_pred_edges = torch.zeros_like(mask).cuda()
        for i in range(mask.shape[0]):
            boundary = self.get_boundary(mask[i].cpu().numpy())
            batch_pred_edges[i] = torch.from_numpy(boundary).cuda()

        return self.mse_loss(batch_pred_edges, label)
    

class GetBoundary(object):
    def __init__(self, width = 4, num_classes = 4):
        self.width = width
        self.num_classes = num_classes

    def __call__(self, mask):

        label = np.zeros((256, 256, self.num_classes), dtype=np.uint8)
        for i in range(self.num_classes):
            label[:, :, i] = (mask == (i+1)).astype(np.uint8)

        for i in range(self.num_classes):
            dila = ndimage.binary_dilation(label[:, :, i], iterations=self.width).astype(label.dtype)
            eros = ndimage.binary_erosion(label[:, :, i], iterations=self.width).astype(label.dtype)
            temp_mask = dila + eros
            temp_mask[temp_mask==2]=0
            
            if i == 0: 
                boundary = temp_mask
            else:
                boundary += temp_mask
                
        boundary = boundary > 0
        boundary= boundary.astype(np.uint8) * 255
        boundary = ndimage.gaussian_filter(boundary, sigma=3) / 255.0

        return boundary
    

class CrossEntropyLoss2dPixelWiseWeighted(nn.Module):
    def __init__(self, weight=None, ignore_index=250, reduction='none'):
        super(CrossEntropyLoss2dPixelWiseWeighted, self).__init__()
        self.CE =  nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, output, target, pixelWiseWeight):
        loss = self.CE(output, target)
        # print(loss.shape)
        # print(pixelWiseWeight.shape)
        loss = torch.mean(loss * pixelWiseWeight)
        return loss
    

def generate_class_mask(pred, classes):
    pred, classes = torch.broadcast_tensors(pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
    N = pred.eq(classes).sum(0)
    return N


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss


def log_losses_tensorboard(writer,current_losses,i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value),i_iter)


def print_losses(current_losses,i_iter):
    list_strings = []
    for loss_name,loss_value in current_losses.items():
        list_strings.append(f'{loss_name}={to_numpy(loss_value):.5f}')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter={i_iter} {full_string}')


def to_numpy(tensor):
    if isinstance(tensor,(int,float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()


def load_checkpoint(model, checkpoint,):
    saved_state_dict = torch.load(checkpoint,map_location='cpu')
    model.load_state_dict(saved_state_dict)


def mixup(images1, images2, labels1, labels2, alpha=2):
    # 随机生成MixUp的比例系数
    lam = np.random.beta(alpha, alpha)

    # 对图像和标签进行MixUp
    mixed_images = lam * images1 + (1 - lam) * images2

    labels1 = torch.nn.functional.one_hot(labels1, 5).permute(0, 3, 1, 2)
    labels2 = torch.nn.functional.one_hot(labels2, 5).permute(0, 3, 1, 2)

    mixed_labels = lam * labels1 + (1 - lam) * labels2

    return mixed_images, mixed_labels

class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, weight = 1):
        super(StyleLoss, self).__init__()
        # self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, source, target):
        source_style = self.gram(source)*self.weight
        target_style = self.gram(target)*self.weight
        # mix_style = self.gram(mix).detach()*self.weight
        # loss = self.criterion(source_style, mix_style) + self.criterion(target_style, mix_style)
        loss = self.criterion(source_style, target_style) 
        return loss

def label_onehot(inputs, num_segments):
    batch_size, im_h, im_w = inputs.shape
    outputs = torch.zeros((num_segments, batch_size, im_h, im_w)).cuda()

    inputs_temp = inputs.clone()
    inputs_temp[inputs == 255] = 0
    outputs.scatter_(0, inputs_temp.unsqueeze(0), 1.0)
    outputs[:, inputs == 255] = 0

    return outputs.permute(1, 0, 2, 3)

def get_boundary_negative(source_teacher_feas, source_labels, numclass, memobank, queue_len, class_center_feas,dilation_iterations):
    # 负样本是源域数据经过teacher模型得到的
    # get negative mask
    source_teacher_feas = source_teacher_feas.detach()
    batch,c,fea_h,fea_w = source_teacher_feas.size()
    source_labels = label_downsample(source_labels,fea_h,fea_w).cpu()#[batch,fea_h,fea_w]
    # print('source_labels',  source_labels[0].tolist())
    negative_mask = []
    for i in range(1,numclass):
        single_class_mask = torch.zeros_like(source_labels) #[batch,fea_h,fea_w]
        label_map = torch.where(source_labels == i, 1, 0) #[batch,fea_h,fea_w]
        # print(label_map)
        # print(i, label_map[0].tolist())
        for j in range(batch):
            labelmapj = label_map[j].numpy()
            singleclassmaskj = ndimage.binary_dilation(labelmapj, iterations=dilation_iterations).astype(labelmapj.dtype)-labelmapj
            single_class_mask[j] = torch.from_numpy(singleclassmaskj)
        negative_mask.append(single_class_mask) # [numclass-1,batch,fea_h,fea_w] 在这里注意下标0对应的是第一个类别的mask

    source_teacher_feas = source_teacher_feas.transpose(1,2).transpose(2,3).contiguous() #batch*c*h*w->batch*h*c*w->batch*h*w*c
    source_teacher_feas = torch.reshape(source_teacher_feas,[batch*fea_h*fea_w,c]) # [batch*h*w] * c

    for i in range(1,numclass):
        single_class_negative_mask = negative_mask[i-1].view(-1) #[batch*h*w,]
        selected_negative_idx = torch.nonzero(single_class_negative_mask).squeeze()
        negative_sample = source_teacher_feas[selected_negative_idx]
        if(len(negative_sample.shape) == 1):
            negative_sample = negative_sample.unsqueeze(0)
        memobank[i].extend(negative_sample)
        if len(memobank[i])>queue_len:
            memobank[i] = memobank[i][-queue_len:]
        # 在训练开始的时候可能会出现memobank为空的情况，若为空，那么把不同类别的中心加入到memobank
        if(memobank[i] == []):
            for j in range(0, numclass):
                if(i!=j):
                    memobank[i].append(class_center_feas[j])

    return memobank



def get_negative(source_teacher_feas, source_labels, numclass, memobank, queue_len, class_center_feas):
    # 负样本是源域数据经过teacher模型得到的
    # get negative mask
    source_teacher_feas = source_teacher_feas.detach()
    batch,c,fea_h,fea_w = source_teacher_feas.size()
    source_labels = label_downsample(source_labels,fea_h,fea_w)#[batch,fea_h,fea_w]
    # print('source_labels',  source_labels[0].tolist())
    negative_mask = []
    for i in range(1,numclass):
        single_class_mask = torch.zeros_like(source_labels) #[batch,fea_h,fea_w]
        label_map = torch.where(source_labels == i, 1, 0) #[batch,fea_h,fea_w]
        # print(i, label_map[0].tolist())

        for j in range(batch):
            if(torch.equal(label_map[j], torch.zeros(label_map[j].shape).cuda()) == False):
                temp = torch.nonzero(label_map[j])
                rows = temp[:,0]
                cols = temp[:,1]
                xmin = torch.min(cols)
                xmax = torch.max(cols)
                ymin = torch.min(rows)
                ymax = torch.max(rows)
                single_class_mask[j,ymin:ymax+1,xmin:xmax+1] = 1
        single_class_mask = single_class_mask - label_map#  [batch,fea_h,fea_w]对于第i个类别，在这个batch中，负样本的点为1，其他为0
        # print(i,single_class_mask[0].tolist())
        negative_mask.append(single_class_mask) # [numclass-1,batch,fea_h,fea_w] 在这里注意下标0对应的是第一个类别的mask

    source_teacher_feas = source_teacher_feas.transpose(1,2).transpose(2,3).contiguous() #batch*c*h*w->batch*h*c*w->batch*h*w*c
    source_teacher_feas = torch.reshape(source_teacher_feas,[batch*fea_h*fea_w,c]) # [batch*h*w] * c

    for i in range(1,numclass):
        single_class_negative_mask = negative_mask[i-1].view(-1) #[batch*h*w,]
        selected_negative_idx = torch.nonzero(single_class_negative_mask).squeeze()
        negative_sample = source_teacher_feas[selected_negative_idx]
        if(len(negative_sample.shape) == 1):
            negative_sample = negative_sample.unsqueeze(0)
        memobank[i].extend(negative_sample)
        if len(memobank[i])>queue_len:
            memobank[i] = memobank[i][-queue_len:]
        # 在训练开始的时候可能会出现memobank为空的情况，若为空，那么把不同类别的中心加入到memobank
        if(memobank[i] == []):
            for j in range(0, numclass):
                if(i!=j):
                    memobank[i].append(class_center_feas[j])

    return memobank


def regional_contrastive_cos(anchor_feas, labels, class_center_feas, memobank, numclass, temp, reliable_mask = None):
    #  anchor_feas可是是源域的特征，也可以是目标域的特征。但都是通过student模型，是有梯度的[batch*c*h*w]
    #  labels 是对应标签，大小可以为[batch, 256, 256]，[batch, 33, 33]，[batch*33*33,]
    #  class_center_feas为正样本，大小为[clasnum, n_fea]
    #  memobank中存储的是负样本
    #  temp 是对比学习损失里的温度
    # reliable_mask大小可以为[batch, 33, 33]，[batch*33*33,]

    n,c,fea_h,fea_w = anchor_feas.size()
    if((len(labels.size()) == 3) & (labels.size(-1) == 256)):
        labels = label_downsample(labels,fea_h,fea_w)
    labels  = labels.view(-1) #[batch*h*w, ]



    class_center_feas = torch.nn.functional.normalize(class_center_feas,p=2,dim=1)# [numclass,c]
    anchor_feas = torch.nn.functional.normalize(anchor_feas,p=2,dim=1)
    anchor_feas = anchor_feas.transpose(1,2).transpose(2,3).contiguous() #batch*c*h*w->batch*h*c*w->batch*h*w*c
    anchor_feas = torch.reshape(anchor_feas,[n*fea_h*fea_w,c]) # [batch*h*w] * c

    eps = 1e-8
    L_contrastive = 0
    anchor_num = 0
    for i in range(1,numclass):
        positive_sample = class_center_feas[i].unsqueeze(-1) #[n_fea,1]
        negtive_sample = torch.cat(memobank[i], 0).reshape(-1, c) # [neg_n, n_fea]
        negtive_sample = torch.nn.functional.normalize(negtive_sample,p=2,dim=1)# [neg_n, n_fea]
        negtive_sample = torch.transpose(negtive_sample, 0, 1)  # [n_fea, neg_n]
        anchor_mask = torch.where(labels == i, 1, 0) # [batch*h*w, ]
        pos = torch.matmul(anchor_feas, positive_sample) #  [batch*h*w, 1]
        pos = torch.div(pos,temp)
        neg = torch.matmul(anchor_feas, negtive_sample) # [batch*h*w, neg_n]
        neg = torch.div(neg,temp)
        logits_down = torch.cat([pos, neg], 1)  # [batch*h*w, 1+neg_n]
        logits_down_max, _ = torch.max(logits_down, dim=1, keepdim=True)
        logits_down = torch.exp(logits_down - logits_down_max.detach()).sum(-1)  # [batch*h*w, ]  分母
        logits = torch.exp(pos - logits_down_max.detach()).squeeze(-1) / (logits_down + eps)
        if reliable_mask is not None:
            reliable_mask     = reliable_mask.view(-1)
            anchor_mask = anchor_mask*reliable_mask
        L_single_class_contrastive = -torch.log(logits + eps) * anchor_mask # [batch*h*w, ]
        anchor_num = anchor_num  + anchor_mask.sum()
        L_contrastive = L_contrastive + L_single_class_contrastive.sum()

    L_contrastive = torch.div(L_contrastive,anchor_num + 1e-12)
    return L_contrastive





def boundary_contrastive_cos(feas, labels, class_center_feas, memobank, numclass, temp, reliable_mask = None):
    #  feas可是是源域的特征，也可以是目标域的特征。但都是通过student模型，是有梯度的[batch,c,h,w]
    #  labels 是对应标签，大小为[batch, 256, 256]
    #  class_center_feas为正样本，大小为[clasnum, n_fea]
    #  memobank中存储的是负样本
    #  temp 是对比学习损失里的温度
    # reliable_mask大小可以为[batch, 33, 33]
    temp = 0.5
    num_queries = 256
    num_negatives = 50

    n,c,fea_h,fea_w = feas.size()
    labels = label_downsample(labels,fea_h,fea_w) # [batch, 33, 33]
    labels = label_onehot(labels) # batch 5 33 33
    if reliable_mask is not None:
        reliable_mask = reliable_mask.unsqueeze(1)# batch 1 33 33
        valid_pixel = labels*reliable_mask# batch 5 33 33
    else:
        valid_pixel = labels
    feas = feas.permute(0, 2, 3, 1) # batch 33 33 256
    seg_feat_all_list = []
    for i in range(1,numclass):
        valid_pixel_seg = valid_pixel[:, i]# batch 33 33
        seg_feat_all_list.append(feas[valid_pixel_seg.bool()])

    
    
    reco_loss = torch.tensor(0.0).cuda()
    seg_proto = class_center_feas




    class_center_feas = torch.nn.functional.normalize(class_center_feas,p=2,dim=1)# [numclass,c]
    anchor_feas = torch.nn.functional.normalize(anchor_feas,p=2,dim=1)
    anchor_feas = anchor_feas.transpose(1,2).transpose(2,3).contiguous() #batch*c*h*w->batch*h*c*w->batch*h*w*c
    anchor_feas = torch.reshape(anchor_feas,[n*fea_h*fea_w,c]) # [batch*h*w] * c

    eps = 1e-8
    L_contrastive = 0
    anchor_num = 0
    for i in range(1,numclass):
        positive_sample = class_center_feas[i].unsqueeze(-1) #[n_fea,1]
        negtive_sample = torch.cat(memobank[i], 0).reshape(-1, c) # [neg_n, n_fea]
        negtive_sample = torch.nn.functional.normalize(negtive_sample,p=2,dim=1)# [neg_n, n_fea]
        negtive_sample = torch.transpose(negtive_sample, 0, 1)  # [n_fea, neg_n]
        anchor_mask = torch.where(labels == i, 1, 0) # [batch*h*w, ]
        pos = torch.matmul(anchor_feas, positive_sample) #  [batch*h*w, 1]
        pos = torch.div(pos,temp)
        neg = torch.matmul(anchor_feas, negtive_sample) # [batch*h*w, neg_n]
        neg = torch.div(neg,temp)
        logits_down = torch.cat([pos, neg], 1)  # [batch*h*w, 1+neg_n]
        logits_down_max, _ = torch.max(logits_down, dim=1, keepdim=True)
        logits_down = torch.exp(logits_down - logits_down_max.detach()).sum(-1)  # [batch*h*w, ]  分母
        logits = torch.exp(pos - logits_down_max.detach()).squeeze(-1) / (logits_down + eps)
        if reliable_mask is not None:
            reliable_mask     = reliable_mask.view(-1)
            anchor_mask = anchor_mask*reliable_mask
        L_single_class_contrastive = -torch.log(logits + eps) * anchor_mask # [batch*h*w, ]
        anchor_num = anchor_num  + anchor_mask.sum()
        L_contrastive = L_contrastive + L_single_class_contrastive.sum()

    L_contrastive = torch.div(L_contrastive,anchor_num + 1e-12)
    return L_contrastive


def boundary_contrastive(anchor_feas, labels, class_center_feas, memobank, numclass, temp, reliable_mask = None):
    #  feas可是是源域的特征，也可以是目标域的特征。但都是通过student模型，是有梯度的[batch,c,h,w]
    #  labels 是对应标签，大小可以为[batch, 256, 256]，[batch, 33, 33]，[batch*33*33,]
    #  class_center_feas为正样本，大小为[clasnum, n_fea]
    #  memobank中存储的是负样本
    #  temp 是对比学习损失里的温度
    # reliable_mask大小可以为[batch, 33, 33]，[batch*33*33,]


    n,num_feat,fea_h,fea_w = anchor_feas.size()
    num_queries = n*fea_h*fea_w
    num_negatives = 256
    if((len(labels.size()) == 3) & (labels.size(-1) == 256)):
        labels = label_downsample(labels,fea_h,fea_w)
    labels  = labels.view(-1) #[batch*h*w, ]

    L_contrastive = 0
    anchor_num = 0
    anchor_feas = anchor_feas.transpose(1,2).transpose(2,3).contiguous() #batch*c*h*w->batch*h*c*w->batch*h*w*c
    anchor_feas = torch.reshape(anchor_feas,[n*fea_h*fea_w,num_feat]) # [batch*h*w] * num_feat

    for i in range(1,numclass):
        with torch.no_grad():
            positive_sample = (
                        class_center_feas[i]#(n_fea)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .repeat(num_queries, 1, 1)
                        .cuda()
                    )  # (num_queries, 1, n_fea)
            

            negtive_sample = torch.cat(memobank[i], 0).reshape(-1, num_feat).cuda() # [len(memobank[i]), n_fea]
            # print(negtive_sample.shape)
            # print(len(negtive_sample))
            negtive_sample_idx = torch.randint(
                len(negtive_sample), size=(num_queries * num_negatives,)
            )
            negtive_sample = negtive_sample[negtive_sample_idx]
            negtive_sample = negtive_sample.reshape(
                num_queries, num_negatives, num_feat
            ) # (num_queries, num_negative, num_feat)


            all_sample = torch.cat(
                (positive_sample, negtive_sample), dim=1
            )  # (num_queries, 1 + num_negative, num_feat)
        logits = torch.cosine_similarity(
            anchor_feas.unsqueeze(1), all_sample, dim=2
        ) #(num_queries, 1 + num_negative)
        reco_loss = F.cross_entropy(
            logits / temp, torch.zeros(num_queries).long().cuda(), reduction = 'none'
        )#(num_queries,)

        anchor_mask = torch.where(labels == i, 1, 0) # [batch*h*w, ]
        if reliable_mask is not None:
            reliable_mask     = reliable_mask.view(-1)
            anchor_mask = anchor_mask*reliable_mask
        L_single_class_contrastive = reco_loss * anchor_mask # [batch*h*w, ]
        anchor_num = anchor_num  + anchor_mask.sum()
        L_contrastive = L_contrastive + L_single_class_contrastive.sum()
    L_contrastive = torch.div(L_contrastive,anchor_num + 1e-12)
    return L_contrastive





def mpcl_loss_calc(feas,labels,class_center_feas,loss_func,
                               pixel_sel_loc=None,tag='source'):

    '''
    feas:  batch*c*h*w
    label: batch*img_h*img_w
    class_center_feas: n_class*n_feas
    '''

    n,c,fea_h,fea_w = feas.size()
    if tag == 'source':
        labels      = labels.float()
        labels      = F.interpolate(labels, size=fea_w, mode='nearest')
        labels      = labels.permute(0,2,1).contiguous()
        labels      = F.interpolate(labels, size=fea_h, mode='nearest')
        labels      = labels.permute(0, 2, 1).contiguous()         # batch*fea_h*fea_w

    labels  = labels.cuda()
    labels  = labels.view(-1).long()

    feas = torch.nn.functional.normalize(feas,p=2,dim=1)
    feas = feas.transpose(1,2).transpose(2,3).contiguous() #batch*c*h*w->batch*h*c*w->batch*h*w*c
    feas = torch.reshape(feas,[n*fea_h*fea_w,c]) # [batch*h*w] * c
    feas = feas.unsqueeze(1) # [batch*h*w] 1 * c

    class_center_feas = torch.nn.functional.normalize(class_center_feas,p=2,dim=1)
    class_center_feas = torch.transpose(class_center_feas, 0, 1)  # n_fea*n_class

    loss =  loss_func(feas,labels,class_center_feas,
                                                    pixel_sel_loc=pixel_sel_loc)
    return loss

class MPCL(nn.Module):
    def __init__(self, num_class=5,temperature=0.07,m=0.5,
                 base_temperature=0.07,easy_margin=False):
        super(MPCL, self).__init__()
        self.num_class        = num_class
        self.temperature      = temperature
        self.base_temperature = base_temperature
        self.m                = m
        self.cos_m            = math.cos(m)
        self.sin_m            = math.sin(m)
        self.th               = math.cos(math.pi - m)
        self.mm               = math.sin(math.pi - m) * m
        self.easy_margin      = easy_margin

    def forward(self, features, labels,class_center_feas,
                pixel_sel_loc=None, mask=None):
        """

         features: [batch_size*fea_h*fea_w] * 1 *c  normalized
         labels:   batch_size*fea_h*fea_w
         class_center_feas:  n_fea*n_class  normalized

        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        # build mask
        num_samples = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(num_samples, dtype=torch.float32).cuda()
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1).long()  # n_sample*1
            class_center_labels = torch.range(0,self.num_class-1).long().cuda()
            # print(class_center_labels)
            class_center_labels = class_center_labels.contiguous().view(-1,1) # n_class*1
            if labels.shape[0] != num_samples:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels,torch.transpose(class_center_labels,0,1)).float().cuda() # broadcast n_sample*n_class
        else:
            mask = mask.float().cuda()
        # n_sample = batch_size * fea_h * fea_w
        # mask n_sample*n_class  the mask_ij represents whether the i-th sample has the same label with j-th class or not.
        # in our experiment, the n_view = 1, so the contrast_count = 1
        contrast_count   = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [n*h*w]*fea_s

        anchor_feature = contrast_feature
        anchor_count   = contrast_count


        # compute logits
        cosine = torch.matmul(anchor_feature, class_center_feas) # [n*h*w] * n_class
        logits = torch.div(cosine,self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits        = logits - logits_max.detach()




        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0.0001, 1.0))
        phi  = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # print(phi)
        phi_logits = torch.div(phi,self.temperature)

        phi_logits_max, _ = torch.max(phi_logits, dim=1, keepdim=True)
        phi_logits = phi_logits - phi_logits_max.detach()



        mask = mask.repeat(anchor_count, contrast_count)

        tag_1             = (1-mask)
        tag_2             = mask
        exp_logits        = torch.exp(logits*tag_1 + phi_logits * tag_2)
        phi_logits        = (logits*tag_1) + (phi_logits*tag_2)
        log_prob          = phi_logits - torch.log(exp_logits.sum(1, keepdim=True)+1e-4)


        if pixel_sel_loc is not None:

            pixel_sel_loc     = pixel_sel_loc.view(-1)

            mean_log_prob_pos =  (mask * log_prob).sum(1)
            mean_log_prob_pos = pixel_sel_loc * mean_log_prob_pos
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = torch.div(loss.sum(),pixel_sel_loc.sum()+1e-4)
        else:

            mean_log_prob_pos = (mask * log_prob).sum(1)
            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.view(anchor_count, num_samples).mean()

        return loss

def label_downsample(labels,fea_h,fea_w):

    '''
    labels: N*H*W
    '''
    labels = labels.float().cuda()
    labels = F.interpolate(labels, size=fea_w, mode='nearest')
    labels = labels.permute(0, 2, 1).contiguous()
    labels = F.interpolate(labels, size=fea_h, mode='nearest')
    labels = labels.permute(0, 2, 1).contiguous()  # n*fea_h*fea_w
    labels = labels.int()
    return labels

def pixel_selection(batch_pixel_cosine,th):
    one_tag = torch.ones([1]).float().cuda()
    zero_tag = torch.zeros([1]).float().cuda()

    batch_sort_cosine,_ = torch.sort(batch_pixel_cosine,dim=1)
    pixel_sub_cosine    = batch_sort_cosine[:,-1]-batch_sort_cosine[:,-2]
    pixel_mask          = torch.where(pixel_sub_cosine>th,one_tag,zero_tag)

    return pixel_mask

def update_class_center_iter(cla_src_feas,batch_src_labels,class_center_feas,m):

    '''
    batch_src_feas  : n*c*h*w
    barch_src_labels: n*h*w
    '''
    batch_src_feas     = cla_src_feas.detach()
    batch_src_labels   = batch_src_labels.cuda()
    n,c,fea_h,fea_w    = batch_src_feas.size()
    batch_y_downsample = label_downsample(batch_src_labels, fea_h, fea_w)  # n*fea_h*fea_w
    batch_y_downsample = batch_y_downsample.unsqueeze(1)  # n*1*fea_h*fea_w
    batch_class_center_fea_list = []
    for i in range(5):
        fea_mask        = torch.eq(batch_y_downsample,i).float().cuda()  #n*1*fea_h*fea_w
        class_feas      = batch_src_feas * fea_mask  # n*c*fea_h*fea_w
        class_fea_sum   = torch.sum(class_feas, [0, 2, 3])  # c
        class_num       = torch.sum(fea_mask, [0, 1, 2, 3])
        if class_num == 0:
            batch_class_center_fea = class_center_feas[i,:].detach()
        else:
            batch_class_center_fea = class_fea_sum/class_num
        batch_class_center_fea = batch_class_center_fea.unsqueeze(0) # 1 * c
        batch_class_center_fea_list.append(batch_class_center_fea)
    batch_class_center_feas = torch.cat(batch_class_center_fea_list,dim=0) # n_class * c
    class_center_feas = m * class_center_feas + (1-m) * batch_class_center_feas

    return class_center_feas


def category_center(model, Sloader,dim):
    model.eval()
    model.cuda()
    cudnn.benchmark = True
    cudnn.enabled   = True
    with torch.no_grad():
        class_center_feas =  torch.zeros(5, dim)
        class_num_all = torch.zeros(5)
        for l, batch in enumerate(tqdm(Sloader)):
            images_source, batch_labels, _ = batch

            cla_feas, pred_src_aux, pred_src_main = model(images_source.cuda()) #cla_feas [batch,2048,33,33]
            batch_feas     = cla_feas.detach()
            batch_labels   = batch_labels.cuda()
            n,c,fea_h,fea_w    = batch_feas.size()
            batch_y_downsample = label_downsample(batch_labels, fea_h, fea_w)  # n*fea_h*fea_w
            batch_y_downsample = batch_y_downsample.unsqueeze(1)  # n*1*fea_h*fea_w
            for i in range(5):
                fea_mask        = torch.eq(batch_y_downsample,i).float().cuda()  #n*1*fea_h*fea_w
                class_feas      = batch_feas * fea_mask  # n*c*fea_h*fea_w
                class_fea_sum   = torch.sum(class_feas, [0, 2, 3]).cpu()  # c
                class_num       = torch.sum(fea_mask, [0, 1, 2, 3]).float().cpu()
                # class_center_feas[i,:] = class_center_feas[i,:]/(class_num_all[i]+class_num)*class_num_all[i] + class_fea_sum/(class_num_all[i]+class_num)
                # class_num_all[i] = class_num_all[i] + class_num
                class_center_feas[i,:] = class_center_feas[i,:] + class_fea_sum
                class_num_all[i] = class_num_all[i] + class_num
        class_num_all = class_num_all.unsqueeze(1)
        class_center_feas = class_center_feas/class_num_all
    return class_center_feas

def category_center_gram(model, Sloader,dim):
    model.eval()
    model.cuda()
    cudnn.benchmark = True
    cudnn.enabled   = True
    with torch.no_grad():
        class_center_feas =  torch.zeros(5, dim)
        class_num_all = torch.zeros(5)
        for l, batch in enumerate(tqdm(Sloader)):
            images_source, batch_labels, _ = batch

            _,cla_feas, pred_src_aux, pred_src_main = model(images_source.cuda()) #cla_feas [batch,2048,33,33]
            batch_feas     = cla_feas.detach()
            batch_labels   = batch_labels.cuda()
            n,c,fea_h,fea_w    = batch_feas.size()
            batch_y_downsample = label_downsample(batch_labels, fea_h, fea_w)  # n*fea_h*fea_w
            batch_y_downsample = batch_y_downsample.unsqueeze(1)  # n*1*fea_h*fea_w
            for i in range(5):
                fea_mask        = torch.eq(batch_y_downsample,i).float().cuda()  #n*1*fea_h*fea_w
                class_feas      = batch_feas * fea_mask  # n*c*fea_h*fea_w
                class_fea_sum   = torch.sum(class_feas, [0, 2, 3]).cpu()  # c
                class_num       = torch.sum(fea_mask, [0, 1, 2, 3]).float().cpu()
                # class_center_feas[i,:] = class_center_feas[i,:]/(class_num_all[i]+class_num)*class_num_all[i] + class_fea_sum/(class_num_all[i]+class_num)
                # class_num_all[i] = class_num_all[i] + class_num
                class_center_feas[i,:] = class_center_feas[i,:] + class_fea_sum
                class_num_all[i] = class_num_all[i] + class_num
        class_num_all = class_num_all.unsqueeze(1)
        class_center_feas = class_center_feas/class_num_all
    return class_center_feas



def generate_pseudo_label(cla_feas_trg,class_centers,cfg):

    '''
    class_centers: C*N_fea
    cla_feas_trg: N*N_fea*H*W
    '''


    cla_feas_trg_de     = cla_feas_trg.detach()
    batch,N_fea,H,W     = cla_feas_trg_de.size()
    cla_feas_trg_de     = F.normalize(cla_feas_trg_de,p=2,dim=1)
    class_centers_norm  = F.normalize(class_centers,p=2,dim=1)
    cla_feas_trg_de     = cla_feas_trg_de.transpose(1,2).contiguous().transpose(2,3).contiguous() # N*H*W*N_fea
    cla_feas_trg_de     = torch.reshape(cla_feas_trg_de,[-1,N_fea])
    class_centers_norm  = class_centers_norm.transpose(0,1)  # N_fea*C
    batch_pixel_cosine  = torch.matmul(cla_feas_trg_de,class_centers_norm) #N*N_class
    threshold = 0.25
    pixel_mask          = pixel_selection(batch_pixel_cosine,threshold)
    hard_pixel_label    = torch.argmax(batch_pixel_cosine,dim=1)

    return hard_pixel_label,pixel_mask

def train_ST(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg):
    '''
    UDA training
    '''

    results_file = "/home/zhr/ICME/scripts/log/{0}2{1}_{2}.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)

    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    num_classes       = cfg.NUM_CLASSES


    unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
    # dice_loss = DiceLoss(n_classes=num_classes).cuda()

    model.train()
    ema_model.train()

    model.cuda()
    ema_model.cuda()
    cudnn.benchmark = True
    cudnn.enabled   = True

    #optimized
    if cfg.TRAIN.OPTIM_G == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.TRAIN.LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                              lr=cfg.TRAIN.LEARNING_RATE,
                              momentum=cfg.TRAIN.MOMENTUM,
                              weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # interpolate output segmaps
    interp        = nn.Upsample(size=(input_size_source[1],input_size_source[0]),mode='bilinear',
                         align_corners=True)

    # labels for adversarial learning


    strain_loader_iter          = enumerate(strain_loader)
    trgtrain_loader_iter        = enumerate(trgtrain_loader)
    sval_loader_iter            = enumerate(sval_loader)

    transforms = None
    img_mean   = cfg.TRAIN.IMG_MEAN
    best_model =  -1

    for i_iter in tqdm(range(cfg.TRAIN.MAX_ITERS+1)):

        model.train()
        ema_model.train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer,i_iter,cfg)


        #UDA training
        try:
            _, batch = strain_loader_iter.__next__()
        except StopIteration:
            strain_loader_iter = enumerate(strain_loader)
            _, batch = strain_loader_iter.__next__()
        images_source, labels_source,_ = batch

        try:
            _, batch = trgtrain_loader_iter.__next__()
        except StopIteration:
            trgtrain_loader_iter = enumerate(trgtrain_loader)
            _, batch             = trgtrain_loader_iter.__next__()
        images_target, labels_target,_ = batch


        noise = torch.clamp(torch.randn_like(
                images_target) * 0.1, -0.2, 0.2)
        ema_inputs = images_target + noise

        with torch.no_grad():
            _, ema_pred_trg_axu, ema_pred_trg_main = ema_model(ema_inputs.cuda())
            ema_pred_trg_axu     = interp(ema_pred_trg_axu)
            ema_pred_trg_main     = interp(ema_pred_trg_main)

            ema_output_main_soft = torch.softmax(ema_pred_trg_main, dim=1)
            ema_output_aux_soft = torch.softmax(ema_pred_trg_axu, dim=1)
            max_probs_main, pseudo_label  = torch.max(ema_output_main_soft, dim=1)
            
            unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
            unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
            pixelWiseWeight = unlabeledWeight
            # onesWeights = torch.ones((unlabeledWeight.shape)).cuda()

            # MixMask, loss_mask = generate_mask(images_target.cuda())
            # pixelWiseWeight = unlabeledWeight * MixMask + onesWeights * (1 - MixMask)
            # images_classmix = images_target.cuda() * MixMask + images_source.cuda() * (1 - MixMask)
            # labels_classmix = pseudo_label.cuda() * MixMask + labels_source.cuda() * (1 - MixMask)
    
        if cfg.TRAIN.CONSWITCH:
            consistency_weight = get_current_consistency_weight(i_iter, cfg.TRAIN.MAX_ITERS)
        else:
            consistency_weight = 1

        # unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
        # unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
        # onesWeights = torch.ones((unlabeledWeight.shape)).cuda()
        # pixelWiseWeight = unlabeledWeight


        _,pred_src_aux, pred_src_main = model(images_source.cuda())
        _,pred_trg_aux, pred_trg_main = model(images_target.cuda())
        # _,pred_mix_aux, pred_mix_main = model(images_classmix.cuda())

        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux     = interp(pred_src_aux)
            pred_trg_aux     = interp(pred_trg_aux)
            # pred_mix_aux     = interp(pred_mix_aux)

            loss_seg_src_aux = loss_calc(pred_src_aux,labels_source)
            loss_dice_aux    = dice_loss(pred_src_aux,labels_source.cuda())
        else:
            loss_seg_src_aux = 0
            loss_dice_aux    = 0

        pred_src_main     = interp(pred_src_main)
        pred_trg_main     = interp(pred_trg_main)
        # pred_mix_main     = interp(pred_mix_main)
        loss_seg_src_main = loss_calc(pred_src_main,labels_source)
        loss_dice_main    = dice_loss(pred_src_main,labels_source.cuda())


        L_u_aux =  consistency_weight * unlabeled_loss(pred_trg_aux, pseudo_label, pixelWiseWeight)
        L_u_main = consistency_weight * unlabeled_loss(pred_trg_main, pseudo_label, pixelWiseWeight)
        # L_dice_aux =  consistency_weight * dice_loss(pred_mix_aux, labels_classmix)
        # L_dice_main = consistency_weight * dice_loss(pred_mix_main, labels_classmix)

        # consistency_loss = torch.mean((torch.softmax(pred_trg_main, dim=1) - ema_output_main_soft)**2)
        # consistency_aux_loss = torch.mean((torch.softmax(pred_trg_aux, dim=1) - ema_output_aux_soft)**2)

        loss = (cfg.TRAIN.LAMBDA_SEG_SRC_MAIN    * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_SRC_AUX   * loss_seg_src_aux
                + cfg.TRAIN.LAMBDA_DICE_SRC_MAIN * loss_dice_main
                + cfg.TRAIN.LAMBDA_DICE_SRC_AUX  * loss_dice_aux
                # + consistency_weight * consistency_loss 
                # + 0.1 * consistency_weight * consistency_aux_loss
                + L_u_main
                + 0.1 * L_u_aux
                # + 0.1 * L_dice_aux
                # + L_dice_main
                )
        
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, 0.99, i_iter)

        current_losses = {'loss_seg_src_aux' :loss_seg_src_aux,
                          'loss_seg_src_main':loss_seg_src_main,
                        #   'loss_target_entp_aux':loss_target_entp_aux,
                        #   'loss_target_entp_main':loss_target_entp_aux,
                          'loss_dice_aux': loss_dice_aux,
                          'loss_dice_main'   :loss_dice_main}
        print_losses(current_losses,i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            if cfg.DATASETS == 'Cardiac':
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Cardiac(model, cfg.TARGET, results_file, i_iter)
            else:
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Abdomen(model, cfg.TARGET, results_file, i_iter)
            
            # saved_path = "/home/data_backup/zhr_savedmodel/{0}2{1}_{2}_ourdice_myweight".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
            # os.makedirs(saved_path, exist_ok=True)
            # torch.save(model.state_dict(),  saved_path+f'/model_{i_iter}.pth')
            # tmp_dice = np.mean(dice_mean)
                        
            # if best_model < tmp_dice:
            #     print('taking snapshot ...')
            #     print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            #     snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            #     torch.save(model.state_dict(),  snapshot_dir / '{0}_{1}2{2}_Best_CutMix_ourdice_model.pth'.format(cfg.DATASETS, cfg.SOURCE, cfg.TARGET))
            #     best_model = tmp_dice

        sys.stdout.flush()



def train_ST_gram(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg):
    '''
    UDA training
    '''

    results_file = "/home/zhr/ICME/scripts/log/{0}2{1}_{2}.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)

    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    num_classes       = cfg.NUM_CLASSES


    unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
    # dice_loss = DiceLoss(n_classes=num_classes).cuda()
    styleloss = StyleLoss(weight = 1000).cuda()

    model.train()
    ema_model.train()

    model.cuda()
    ema_model.cuda()
    cudnn.benchmark = True
    cudnn.enabled   = True

    #optimized
    if cfg.TRAIN.OPTIM_G == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.TRAIN.LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                              lr=cfg.TRAIN.LEARNING_RATE,
                              momentum=cfg.TRAIN.MOMENTUM,
                              weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # interpolate output segmaps
    interp        = nn.Upsample(size=(input_size_source[1],input_size_source[0]),mode='bilinear',
                         align_corners=True)

    # labels for adversarial learning


    strain_loader_iter          = enumerate(strain_loader)
    trgtrain_loader_iter        = enumerate(trgtrain_loader)
    sval_loader_iter            = enumerate(sval_loader)

    transforms = None
    img_mean   = cfg.TRAIN.IMG_MEAN
    best_model =  -1

    for i_iter in tqdm(range(cfg.TRAIN.MAX_ITERS+1)):

        model.train()
        ema_model.train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer,i_iter,cfg)


        #UDA training
        try:
            _, batch = strain_loader_iter.__next__()
        except StopIteration:
            strain_loader_iter = enumerate(strain_loader)
            _, batch = strain_loader_iter.__next__()
        images_source, labels_source,_ = batch

        try:
            _, batch = trgtrain_loader_iter.__next__()
        except StopIteration:
            trgtrain_loader_iter = enumerate(trgtrain_loader)
            _, batch             = trgtrain_loader_iter.__next__()
        images_target, labels_target,_ = batch


        noise = torch.clamp(torch.randn_like(
                images_target) * 0.1, -0.2, 0.2)
        ema_inputs = images_target + noise

        with torch.no_grad():
            _, ema_pred_trg_axu, ema_pred_trg_main = ema_model(ema_inputs.cuda())
            ema_pred_trg_axu     = interp(ema_pred_trg_axu)
            ema_pred_trg_main     = interp(ema_pred_trg_main)

            ema_output_main_soft = torch.softmax(ema_pred_trg_main, dim=1)
            ema_output_aux_soft = torch.softmax(ema_pred_trg_axu, dim=1)
            max_probs_main, pseudo_label  = torch.max(ema_output_main_soft, dim=1)
            
            unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
            unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
            pixelWiseWeight = unlabeledWeight
            # onesWeights = torch.ones((unlabeledWeight.shape)).cuda()

            # MixMask, loss_mask = generate_mask(images_target.cuda())
            # pixelWiseWeight = unlabeledWeight * MixMask + onesWeights * (1 - MixMask)
            # images_classmix = images_target.cuda() * MixMask + images_source.cuda() * (1 - MixMask)
            # labels_classmix = pseudo_label.cuda() * MixMask + labels_source.cuda() * (1 - MixMask)
    
        if cfg.TRAIN.CONSWITCH:
            consistency_weight = get_current_consistency_weight(i_iter, cfg.TRAIN.MAX_ITERS)
        else:
            consistency_weight = 1

        # unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
        # unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
        # onesWeights = torch.ones((unlabeledWeight.shape)).cuda()
        # pixelWiseWeight = unlabeledWeight


        source_f,pred_src_aux, pred_src_main = model(images_source.cuda())
        target_f,pred_trg_aux, pred_trg_main = model(images_target.cuda())
        # _,pred_mix_aux, pred_mix_main = model(images_classmix.cuda())

        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux     = interp(pred_src_aux)
            pred_trg_aux     = interp(pred_trg_aux)
            # pred_mix_aux     = interp(pred_mix_aux)

            loss_seg_src_aux = loss_calc(pred_src_aux,labels_source)
            loss_dice_aux    = dice_loss(pred_src_aux,labels_source.cuda())
        else:
            loss_seg_src_aux = 0
            loss_dice_aux    = 0

        pred_src_main     = interp(pred_src_main)
        pred_trg_main     = interp(pred_trg_main)
        # pred_mix_main     = interp(pred_mix_main)
        loss_seg_src_main = loss_calc(pred_src_main,labels_source)
        loss_dice_main    = dice_loss(pred_src_main,labels_source.cuda())


        L_u_aux =  consistency_weight * unlabeled_loss(pred_trg_aux, pseudo_label, pixelWiseWeight)
        L_u_main = consistency_weight * unlabeled_loss(pred_trg_main, pseudo_label, pixelWiseWeight)
        gram_loss = styleloss(source_f,target_f,1)
        # L_dice_aux =  consistency_weight * dice_loss(pred_mix_aux, labels_classmix)
        # L_dice_main = consistency_weight * dice_loss(pred_mix_main, labels_classmix)

        # consistency_loss = torch.mean((torch.softmax(pred_trg_main, dim=1) - ema_output_main_soft)**2)
        # consistency_aux_loss = torch.mean((torch.softmax(pred_trg_aux, dim=1) - ema_output_aux_soft)**2)

        loss = (cfg.TRAIN.LAMBDA_SEG_SRC_MAIN    * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_SRC_AUX   * loss_seg_src_aux
                + cfg.TRAIN.LAMBDA_DICE_SRC_MAIN * loss_dice_main
                + cfg.TRAIN.LAMBDA_DICE_SRC_AUX  * loss_dice_aux
                # + consistency_weight * consistency_loss 
                # + 0.1 * consistency_weight * consistency_aux_loss
                + L_u_main
                + 0.1 * L_u_aux
                + gram_loss
                # + 0.1 * L_dice_aux
                # + L_dice_main
                )
        
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, 0.99, i_iter)

        current_losses = {'loss_seg_src_aux' :loss_seg_src_aux,
                          'loss_seg_src_main':loss_seg_src_main,
                        #   'loss_target_entp_aux':loss_target_entp_aux,
                        #   'loss_target_entp_main':loss_target_entp_aux,
                          'loss_dice_aux': loss_dice_aux,
                          'loss_dice_main'   :loss_dice_main}
        print_losses(current_losses,i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            if cfg.DATASETS == 'Cardiac':
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Cardiac(model, cfg.TARGET, results_file, i_iter)
            else:
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Abdomen(model, cfg.TARGET, results_file, i_iter)
            
            # saved_path = "/home/data_backup/zhr_savedmodel/{0}2{1}_{2}_ourdice_myweight".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
            # os.makedirs(saved_path, exist_ok=True)
            # torch.save(model.state_dict(),  saved_path+f'/model_{i_iter}.pth')
            # tmp_dice = np.mean(dice_mean)
                        
            # if best_model < tmp_dice:
            #     print('taking snapshot ...')
            #     print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            #     snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            #     torch.save(model.state_dict(),  snapshot_dir / '{0}_{1}2{2}_Best_CutMix_ourdice_model.pth'.format(cfg.DATASETS, cfg.SOURCE, cfg.TARGET))
            #     best_model = tmp_dice

        sys.stdout.flush()



def train_cutmix_ST(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg):
    '''
    UDA training
    '''

    results_file = "/home/zhr/ICME/scripts/log/{0}2{1}_{2}_ourdice_myweight_23cut2.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)

    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    num_classes       = cfg.NUM_CLASSES


    unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
    # dice_loss = DiceLoss(n_classes=num_classes).cuda()

    model.train()
    ema_model.train()

    model.cuda()
    ema_model.cuda()
    cudnn.benchmark = True
    cudnn.enabled   = True

    #optimized
    if cfg.TRAIN.OPTIM_G == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.TRAIN.LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                              lr=cfg.TRAIN.LEARNING_RATE,
                              momentum=cfg.TRAIN.MOMENTUM,
                              weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # interpolate output segmaps
    interp        = nn.Upsample(size=(input_size_source[1],input_size_source[0]),mode='bilinear',
                         align_corners=True)

    # labels for adversarial learning


    strain_loader_iter          = enumerate(strain_loader)
    trgtrain_loader_iter        = enumerate(trgtrain_loader)
    sval_loader_iter            = enumerate(sval_loader)

    transforms = None
    img_mean   = cfg.TRAIN.IMG_MEAN
    best_model =  -1

    for i_iter in tqdm(range(cfg.TRAIN.MAX_ITERS+1)):

        model.train()
        ema_model.train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer,i_iter,cfg)


        #UDA training
        try:
            _, batch = strain_loader_iter.__next__()
        except StopIteration:
            strain_loader_iter = enumerate(strain_loader)
            _, batch = strain_loader_iter.__next__()
        images_source, labels_source,_ = batch

        try:
            _, batch = trgtrain_loader_iter.__next__()
        except StopIteration:
            trgtrain_loader_iter = enumerate(trgtrain_loader)
            _, batch             = trgtrain_loader_iter.__next__()
        images_target, labels_target,_ = batch

        # src_in_trg = []
        # for i in range(cfg.TRAIN.BATCH_SIZE):
        #     st = match_histograms(np.array(images_source[i]), np.array(images_target[i]), channel_axis=0)
        #     src_in_trg.append(st)
        # images_source = torch.tensor(src_in_trg, dtype=torch.float32)

        noise = torch.clamp(torch.randn_like(
                images_target) * 0.1, -0.2, 0.2)
        ema_inputs = images_target + noise

        with torch.no_grad():
            _, ema_pred_trg_axu, ema_pred_trg_main = ema_model(ema_inputs.cuda())
            ema_pred_trg_axu     = interp(ema_pred_trg_axu)
            ema_pred_trg_main     = interp(ema_pred_trg_main)

            ema_output_main_soft = torch.softmax(ema_pred_trg_main, dim=1)
            ema_output_aux_soft = torch.softmax(ema_pred_trg_axu, dim=1)
            max_probs_main, pseudo_label  = torch.max(ema_output_main_soft, dim=1)
            
            unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
            unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
            onesWeights = torch.ones((unlabeledWeight.shape)).cuda()

            MixMask, loss_mask = generate_mask(images_target.cuda())
            pixelWiseWeight = unlabeledWeight * MixMask + onesWeights * (1 - MixMask)
            images_classmix = images_target.cuda() * MixMask + images_source.cuda() * (1 - MixMask)
            labels_classmix = pseudo_label.cuda() * MixMask + labels_source.cuda() * (1 - MixMask)

            # how2mask = np.random.uniform(0, 1, 1)
            # if how2mask < 2:
            #     MixMask, loss_mask = generate_mask(images_target.cuda())

            #     images_classmix = images_target.cuda() * MixMask + images_source.cuda() * (1 - MixMask)
            #     labels_classmix = pseudo_label.cuda() * MixMask + labels_source.cuda() * (1 - MixMask)
            # else:
            #     MixMask, loss_mask = generate_mask(images_target.cuda())

            #     images_classmix = images_target.cuda() * (1 - MixMask) + images_source.cuda() * MixMask
            #     labels_classmix = pseudo_label.cuda() * (1 - MixMask) + labels_source.cuda() * MixMask


            # MixMask, loss_mask = generate_mask(images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda())

            # pixelWiseWeight_trg = unlabeledWeight[:cfg.TRAIN.BATCH_SIZE//2] * MixMask + onesWeights[:cfg.TRAIN.BATCH_SIZE//2] * (1 - MixMask)
            # images_trg_classmix = images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + images_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)
            # labels_trg_classmix = pseudo_label[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + labels_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)

            # MixMask, loss_mask = generate_mask(images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda())
            
            # pixelWiseWeight_src = unlabeledWeight[cfg.TRAIN.BATCH_SIZE//2:] * (1 - MixMask) + onesWeights[cfg.TRAIN.BATCH_SIZE//2:] * MixMask
            # images_src_classmix = images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + images_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            # labels_src_classmix = pseudo_label[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + labels_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            
            # pixelWiseWeight = torch.cat([pixelWiseWeight_trg, pixelWiseWeight_src], dim=0)
            # images_classmix = torch.cat([images_trg_classmix, images_src_classmix], dim=0)
            # labels_classmix = torch.cat([labels_trg_classmix, labels_src_classmix], dim=0)
    
        if cfg.TRAIN.CONSWITCH:
            consistency_weight = get_current_consistency_weight(i_iter, cfg.TRAIN.MAX_ITERS)
        else:
            consistency_weight = 1

        # unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
        # unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
        # onesWeights = torch.ones((unlabeledWeight.shape)).cuda()
        # pixelWiseWeight = unlabeledWeight


        _,pred_src_aux, pred_src_main = model(images_source.cuda())
        _,pred_trg_aux, pred_trg_main = model(images_target.cuda())
        _,pred_mix_aux, pred_mix_main = model(images_classmix.cuda())

        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux     = interp(pred_src_aux)
            pred_trg_aux     = interp(pred_trg_aux)
            pred_mix_aux     = interp(pred_mix_aux)

            loss_seg_src_aux = loss_calc(pred_src_aux,labels_source)
            loss_dice_aux    = dice_loss(pred_src_aux,labels_source.cuda())
        else:
            loss_seg_src_aux = 0
            loss_dice_aux    = 0

        pred_src_main     = interp(pred_src_main)
        pred_trg_main     = interp(pred_trg_main)
        pred_mix_main     = interp(pred_mix_main)
        loss_seg_src_main = loss_calc(pred_src_main,labels_source)
        loss_dice_main    = dice_loss(pred_src_main,labels_source.cuda())


        L_u_aux =  consistency_weight * unlabeled_loss(pred_mix_aux, labels_classmix, pixelWiseWeight)
        L_u_main = consistency_weight * unlabeled_loss(pred_mix_main, labels_classmix, pixelWiseWeight)
        # L_dice_aux =  consistency_weight * dice_loss(pred_mix_aux, labels_classmix)
        # L_dice_main = consistency_weight * dice_loss(pred_mix_main, labels_classmix)

        consistency_loss = torch.mean((torch.softmax(pred_trg_main, dim=1) - ema_output_main_soft)**2)
        consistency_aux_loss = torch.mean((torch.softmax(pred_trg_aux, dim=1) - ema_output_aux_soft)**2)

        loss = (cfg.TRAIN.LAMBDA_SEG_SRC_MAIN    * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_SRC_AUX   * loss_seg_src_aux
                + cfg.TRAIN.LAMBDA_DICE_SRC_MAIN * loss_dice_main
                + cfg.TRAIN.LAMBDA_DICE_SRC_AUX  * loss_dice_aux
                + consistency_weight * consistency_loss 
                + 0.1 * consistency_weight * consistency_aux_loss
                + L_u_main
                + 0.1 * L_u_aux
                # + 0.1 * L_dice_aux
                # + L_dice_main
                )
        
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, 0.99, i_iter)

        current_losses = {'loss_seg_src_aux' :loss_seg_src_aux,
                          'loss_seg_src_main':loss_seg_src_main,
                        #   'loss_target_entp_aux':loss_target_entp_aux,
                        #   'loss_target_entp_main':loss_target_entp_aux,
                          'loss_dice_aux': loss_dice_aux,
                          'loss_dice_main'   :loss_dice_main}
        print_losses(current_losses,i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            if cfg.DATASETS == 'Cardiac':
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Cardiac(model, cfg.TARGET, results_file, i_iter)
            else:
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Abdomen(model, cfg.TARGET, results_file, i_iter)
            
            # saved_path = "/home/data_backup/zhr_savedmodel/{0}2{1}_{2}_ourdice_myweight_23cut".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
            # os.makedirs(saved_path, exist_ok=True)
            # torch.save(model.state_dict(),  saved_path+f'/model_{i_iter}.pth')
            # tmp_dice = np.mean(dice_mean)
                        
            # if best_model < tmp_dice:
            #     print('taking snapshot ...')
            #     print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            #     snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            #     torch.save(model.state_dict(),  snapshot_dir / '{0}_{1}2{2}_Best_CutMix_ourdice_model.pth'.format(cfg.DATASETS, cfg.SOURCE, cfg.TARGET))
            #     best_model = tmp_dice

        sys.stdout.flush()


def train_classmix_ST(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg):
    '''
    UDA training
    '''

    results_file = "/home/zhr/ICME/scripts/log_ablation/{0}2{1}_{2}_01cutmix0.51.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)

    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    num_classes       = cfg.NUM_CLASSES


    unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
    # dice_loss = DiceLoss(n_classes=num_classes).cuda()

    model.train()
    ema_model.train()

    model.cuda()
    ema_model.cuda()
    cudnn.benchmark = True
    cudnn.enabled   = True

    #optimized
    if cfg.TRAIN.OPTIM_G == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.TRAIN.LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                              lr=cfg.TRAIN.LEARNING_RATE,
                              momentum=cfg.TRAIN.MOMENTUM,
                              weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # interpolate output segmaps
    interp        = nn.Upsample(size=(input_size_source[1],input_size_source[0]),mode='bilinear',
                         align_corners=True)

    # labels for adversarial learning


    strain_loader_iter          = enumerate(strain_loader)
    trgtrain_loader_iter        = enumerate(trgtrain_loader)
    sval_loader_iter            = enumerate(sval_loader)

    transforms = None
    img_mean   = cfg.TRAIN.IMG_MEAN
    best_model =  -1
    dice_all = []

    for i_iter in tqdm(range(cfg.TRAIN.MAX_ITERS+1)):

        model.train()
        ema_model.train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer,i_iter,cfg)


        #UDA training
        try:
            _, batch = strain_loader_iter.__next__()
        except StopIteration:
            strain_loader_iter = enumerate(strain_loader)
            _, batch = strain_loader_iter.__next__()
        images_source, labels_source,_ = batch

        try:
            _, batch = trgtrain_loader_iter.__next__()
        except StopIteration:
            trgtrain_loader_iter = enumerate(trgtrain_loader)
            _, batch             = trgtrain_loader_iter.__next__()
        images_target, labels_target,_ = batch

        # src_in_trg = []
        # for i in range(cfg.TRAIN.BATCH_SIZE):
        #     st = match_histograms(np.array(images_source[i]), np.array(images_target[i]), channel_axis=0)
        #     src_in_trg.append(st)
        # images_source = torch.tensor(src_in_trg, dtype=torch.float32)

        # noise = torch.clamp(torch.randn_like(
        #         images_target) * 0.1, -0.2, 0.2)
        # ema_inputs = images_target + noise
        ema_inputs = images_target

        with torch.no_grad():
            _, ema_pred_trg_axu, ema_pred_trg_main = ema_model(ema_inputs.cuda())
            ema_pred_trg_axu     = interp(ema_pred_trg_axu)
            ema_pred_trg_main     = interp(ema_pred_trg_main)

            ema_output_main_soft = torch.softmax(ema_pred_trg_main, dim=1)
            ema_output_aux_soft = torch.softmax(ema_pred_trg_axu, dim=1)
            max_probs_main, pseudo_label  = torch.max(ema_output_main_soft, dim=1)
            
            unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
            unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
            onesWeights = torch.ones((unlabeledWeight.shape)).cuda()

            # MixMask, loss_mask = generate_mask(images_target.cuda())
            # pixelWiseWeight = unlabeledWeight * MixMask + onesWeights * (1 - MixMask)
            # images_classmix = images_target.cuda() * MixMask + images_source.cuda() * (1 - MixMask)
            # labels_classmix = pseudo_label.cuda() * MixMask + labels_source.cuda() * (1 - MixMask)




            for image_i in range(cfg.TRAIN.BATCH_SIZE):
                classes = torch.unique(labels_source[image_i])
                classes = classes[classes != 0]  # 筛选出非背景类
                nclasses = classes.shape[0]
                if nclasses > 1:
                    # if nclasses <=2:
                    #     classes = (classes[torch.Tensor(np.random.choice(nclasses, round(nclasses), replace=False)).long()]).cuda()
                    #     MixMask = generate_class_mask(labels_source[image_i].cuda(), classes).unsqueeze(0).cuda()
                    # else:
                    #     # num = np.random.randint(2,min(nclasses,3))
                    classes = (classes[torch.Tensor(np.random.choice(nclasses, round(nclasses/2), replace=False)).long()]).cuda()
                    MixMask = generate_class_mask(labels_source[image_i].cuda(), classes).unsqueeze(0).cuda()
                else:
                    MixMask, _ = cut_generate_mask(images_source[image_i].cuda())
                    MixMask = MixMask.unsqueeze(0).cuda()

                if image_i == 0:
                    All_MixMask = MixMask
                else:
                    All_MixMask = torch.cat((All_MixMask, MixMask))

            All_MixMask = torch.unsqueeze(All_MixMask, 1).repeat((1,3,1,1))
            # print(MixMask.shape, images_target.shape, pseudo_label.shape)
            images_classmix = images_target.cuda() * (1 - All_MixMask) + images_source.cuda() * All_MixMask
            labels_classmix = pseudo_label.cuda() * (1 - All_MixMask[:,0,:,:]) + labels_source.cuda() * All_MixMask[:,0,:,:]
            pixelWiseWeight = unlabeledWeight* (1 - All_MixMask[:,0,:,:]) + onesWeights * All_MixMask[:,0,:,:]


            # for image_i in range(cfg.TRAIN.BATCH_SIZE):
            #     classes = torch.unique(labels_source[image_i])
            #     classes = classes[classes != 0]  # 筛选出非背景类
            #     nclasses = classes.shape[0]

            #     # if nclasses <=2:
            #     #     classes = (classes[torch.Tensor(np.random.choice(nclasses, round(nclasses), replace=False)).long()]).cuda()
            #     #     MixMask = generate_class_mask(labels_source[image_i].cuda(), classes).unsqueeze(0).cuda()
            #     # else:
            #     #     classes = (classes[torch.Tensor(np.random.choice(nclasses, round(2), replace=False)).long()]).cuda()
            #     #     MixMask = generate_class_mask(labels_source[image_i].cuda(), classes).unsqueeze(0).cuda() # 1,256 256
             
            #     classes = (classes[torch.Tensor(np.random.choice(nclasses, round(nclasses/2), replace=False)).long()]).cuda()
            #     MixMask = generate_class_mask(labels_source[image_i].cuda(), classes).unsqueeze(0).cuda()
            #     # for clas in classes.tolist():
            #     #     classes_dict[clas] += 1
                
            #     # print(image_i,MixMask.sum())
               
            #     if image_i == 0:
            #         All_MixMask = MixMask
            #     else:
            #         All_MixMask = torch.cat((All_MixMask, MixMask)) # 4, 256 256
            #         # All_MixMask = torch.cat((MixMask, All_MixMask)) # 4, 256 256
            # # print(All_MixMask[0].sum(), All_MixMask[1].sum(), All_MixMask[2].sum(),All_MixMask[3].sum())

            # # dice_loss = dice_loss_class
            # All_MixMask = torch.unsqueeze(All_MixMask, 1).repeat((1,3,1,1)) # 4 3 256 256
            # # print(MixMask.shape, images_target.shape, pseudo_label.shape)
            # images_classmix = images_target.cuda() * (1 - All_MixMask) + images_source.cuda() * All_MixMask
            # labels_classmix = pseudo_label.cuda() * (1 - All_MixMask[:,0,:,:]) + labels_source.cuda() * All_MixMask[:,0,:,:]
            # pixelWiseWeight = unlabeledWeight* (1 - All_MixMask[:,0,:,:]) + onesWeights * All_MixMask[:,0,:,:]

    
        if cfg.TRAIN.CONSWITCH:
            consistency_weight = get_current_consistency_weight(i_iter, cfg.TRAIN.MAX_ITERS)
        else:
            consistency_weight = 1
        # consistency_weight = 1
        # unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
        # unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
        # onesWeights = torch.ones((unlabeledWeight.shape)).cuda()
        # pixelWiseWeight = unlabeledWeight


        _,pred_src_aux, pred_src_main = model(images_source.cuda())
        _,pred_trg_aux, pred_trg_main = model(images_target.cuda())
        _,pred_mix_aux, pred_mix_main = model(images_classmix.cuda())

        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux     = interp(pred_src_aux)
            pred_trg_aux     = interp(pred_trg_aux)
            pred_mix_aux     = interp(pred_mix_aux)

            loss_seg_src_aux = loss_calc(pred_src_aux,labels_source)
            loss_dice_aux    = dice_loss(pred_src_aux,labels_source.cuda())
        else:
            loss_seg_src_aux = 0
            loss_dice_aux    = 0

        pred_src_main     = interp(pred_src_main)
        pred_trg_main     = interp(pred_trg_main)
        pred_mix_main     = interp(pred_mix_main)
        loss_seg_src_main = loss_calc(pred_src_main,labels_source)
        loss_dice_main    = dice_loss(pred_src_main,labels_source.cuda())


        L_u_aux =  consistency_weight * unlabeled_loss(pred_mix_aux, labels_classmix, pixelWiseWeight)
        L_u_main = consistency_weight * unlabeled_loss(pred_mix_main, labels_classmix, pixelWiseWeight)
        # L_dice_aux =  consistency_weight * dice_loss(pred_mix_aux, labels_classmix)
        # L_dice_main = consistency_weight * dice_loss(pred_mix_main, labels_classmix)

        # consistency_loss = torch.mean((torch.softmax(pred_trg_main, dim=1) - ema_output_main_soft)**2)
        # consistency_aux_loss = torch.mean((torch.softmax(pred_trg_aux, dim=1) - ema_output_aux_soft)**2)

        loss = (cfg.TRAIN.LAMBDA_SEG_SRC_MAIN    * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_SRC_AUX   * loss_seg_src_aux
                + cfg.TRAIN.LAMBDA_DICE_SRC_MAIN * loss_dice_main
                + cfg.TRAIN.LAMBDA_DICE_SRC_AUX  * loss_dice_aux
                # + consistency_weight * consistency_loss 
                # + 0.1 * consistency_weight * consistency_aux_loss
                + L_u_main
                + 0.1 * L_u_aux
                # + 0.1 * L_dice_aux
                # + L_dice_main
                )
        
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, 0.99, i_iter)

        current_losses = {'loss_seg_src_aux' :loss_seg_src_aux,
                          'loss_seg_src_main':loss_seg_src_main,
                        #   'loss_target_entp_aux':loss_target_entp_aux,
                        #   'loss_target_entp_main':loss_target_entp_aux,
                          'loss_dice_aux': loss_dice_aux,
                          'loss_dice_main'   :loss_dice_main}
        print_losses(current_losses,i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            if cfg.DATASETS == 'Cardiac':
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Cardiac(model, cfg.TARGET, results_file, i_iter)
            else:
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Abdomen(model, cfg.TARGET, results_file, i_iter)
            
            saved_path = "/home/data_backup/zhr_savedmodel/{0}2{1}_{2}_01cutmix0.51".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
            os.makedirs(saved_path, exist_ok=True)
            torch.save(model.state_dict(),  saved_path+f'/model_{i_iter}.pth')
            tmp_dice = np.mean(dice_mean)
            dice_all.append(tmp_dice)
            if(i_iter == cfg.TRAIN.MAX_ITERS):
                dice_all = np.array(dice_all)
                np.save('/home/data_backup/zhr_savedmodel/dice_01cutmix0.51_'+cfg.SOURCE+'2'+cfg.TARGET+'_'+cfg.TRAIN.DA_METHOD, dice_all)
                        
            # if best_model < tmp_dice:
            #     print('taking snapshot ...')
            #     print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            #     snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            #     torch.save(model.state_dict(),  snapshot_dir / '{0}_{1}2{2}_Best_CutMix_ourdice_model.pth'.format(cfg.DATASETS, cfg.SOURCE, cfg.TARGET))
            #     best_model = tmp_dice

        sys.stdout.flush()


def train_bclassmix_ST(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg):
    '''
    UDA training
    '''

    results_file = "/home/zhr/ICME/scripts/log_ablation/{0}2{1}_{2}_01cutmix2.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)

    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    num_classes       = cfg.NUM_CLASSES


    unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
    # dice_loss = DiceLoss(n_classes=num_classes).cuda()

    model.train()
    ema_model.train()

    model.cuda()
    ema_model.cuda()
    cudnn.benchmark = True
    cudnn.enabled   = True

    #optimized
    if cfg.TRAIN.OPTIM_G == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.TRAIN.LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                              lr=cfg.TRAIN.LEARNING_RATE,
                              momentum=cfg.TRAIN.MOMENTUM,
                              weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # interpolate output segmaps
    interp        = nn.Upsample(size=(input_size_source[1],input_size_source[0]),mode='bilinear',
                         align_corners=True)

    # labels for adversarial learning


    strain_loader_iter          = enumerate(strain_loader)
    trgtrain_loader_iter        = enumerate(trgtrain_loader)
    sval_loader_iter            = enumerate(sval_loader)

    transforms = None
    img_mean   = cfg.TRAIN.IMG_MEAN
    best_model =  -1
    dice_all = []

    for i_iter in tqdm(range(cfg.TRAIN.MAX_ITERS+1)):

        model.train()
        ema_model.train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer,i_iter,cfg)


        #UDA training
        try:
            _, batch = strain_loader_iter.__next__()
        except StopIteration:
            strain_loader_iter = enumerate(strain_loader)
            _, batch = strain_loader_iter.__next__()
        images_source, labels_source,_ = batch

        try:
            _, batch = trgtrain_loader_iter.__next__()
        except StopIteration:
            trgtrain_loader_iter = enumerate(trgtrain_loader)
            _, batch             = trgtrain_loader_iter.__next__()
        images_target, labels_target,_ = batch

        # src_in_trg = []
        # for i in range(cfg.TRAIN.BATCH_SIZE):
        #     st = match_histograms(np.array(images_source[i]), np.array(images_target[i]), channel_axis=0)
        #     src_in_trg.append(st)
        # images_source = torch.tensor(src_in_trg, dtype=torch.float32)

        noise = torch.clamp(torch.randn_like(
                images_target) * 0.1, -0.2, 0.2)
        ema_inputs = images_target + noise

        with torch.no_grad():
            _, ema_pred_trg_axu, ema_pred_trg_main = ema_model(ema_inputs.cuda())
            ema_pred_trg_axu     = interp(ema_pred_trg_axu)
            ema_pred_trg_main     = interp(ema_pred_trg_main)

            ema_output_main_soft = torch.softmax(ema_pred_trg_main, dim=1)
            ema_output_aux_soft = torch.softmax(ema_pred_trg_axu, dim=1)
            max_probs_main, pseudo_label  = torch.max(ema_output_main_soft, dim=1)
            
            unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
            tg_mask = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
            sc_mask = torch.ones((tg_mask.shape)).cuda()

            # MixMask, loss_mask = generate_mask(images_target.cuda())
            # pixelWiseWeight = unlabeledWeight * MixMask + onesWeights * (1 - MixMask)
            # images_classmix = images_target.cuda() * MixMask + images_source.cuda() * (1 - MixMask)
            # labels_classmix = pseudo_label.cuda() * MixMask + labels_source.cuda() * (1 - MixMask)


            # for image_i in range(cfg.TRAIN.BATCH_SIZE):
            #     classes = torch.unique(labels_source[image_i])
            #     classes = classes[classes != 0]  # 筛选出非背景类
            #     nclasses = classes.shape[0]

            #     if nclasses <=2:
            #         classes = (classes[torch.Tensor(np.random.choice(nclasses, round(nclasses), replace=False)).long()]).cuda()
            #         MixMask = generate_class_mask(labels_source[image_i].cuda(), classes).unsqueeze(0).cuda()
            #     else:
            #         classes = (classes[torch.Tensor(np.random.choice(nclasses, round(2), replace=False)).long()]).cuda()
            #         MixMask = generate_class_mask(labels_source[image_i].cuda(), classes).unsqueeze(0).cuda() # 1,256 256
             

            #     # for clas in classes.tolist():
            #     #     classes_dict[clas] += 1
                
            #     # print(image_i,MixMask.sum())
               
            #     if image_i == 0:
            #         All_MixMask = MixMask
            #     else:
            #         All_MixMask = torch.cat((All_MixMask, MixMask)) # 4, 256 256
            #         # All_MixMask = torch.cat((MixMask, All_MixMask)) # 4, 256 256
            # # print(All_MixMask[0].sum(), All_MixMask[1].sum(), All_MixMask[2].sum(),All_MixMask[3].sum())

            # # dice_loss = dice_loss_class
            # All_MixMask = torch.unsqueeze(All_MixMask, 1).repeat((1,3,1,1)) # 4 3 256 256
            # # print(MixMask.shape, images_target.shape, pseudo_label.shape)
            # images_classmix = images_target.cuda() * (1 - All_MixMask) + images_source.cuda() * All_MixMask
            # labels_classmix = pseudo_label.cuda() * (1 - All_MixMask[:,0,:,:]) + labels_source.cuda() * All_MixMask[:,0,:,:]
            # pixelWiseWeight = unlabeledWeight* (1 - All_MixMask[:,0,:,:]) + onesWeights * All_MixMask[:,0,:,:]


            # ## 源域贴在目标域
            # for image_i in range(cfg.TRAIN.BATCH_SIZE//2):
            #     classes = torch.unique(labels_source[image_i])
            #     classes = classes[classes != 0]  # 筛选出非背景类
            #     nclasses = classes.shape[0]

            #     if nclasses <=2:
            #         classes = (classes[torch.Tensor(np.random.choice(nclasses, round(nclasses), replace=False)).long()]).cuda()
            #         MixMask = generate_class_mask(labels_source[image_i].cuda(), classes).unsqueeze(0).cuda()
            #     else:
            #         classes = (classes[torch.Tensor(np.random.choice(nclasses, round(2), replace=False)).long()]).cuda()
            #         MixMask = generate_class_mask(labels_source[image_i].cuda(), classes).unsqueeze(0).cuda() # 1,256 256
             

            #     # for clas in classes.tolist():
            #     #     classes_dict[clas] += 1
                    
            #     if image_i == 0:
            #         All_MixMask = MixMask
            #     else:
            #         All_MixMask = torch.cat((All_MixMask, MixMask)) # 2, 256 256

            # # dice_loss = dice_loss_class
            # All_MixMask = torch.unsqueeze(All_MixMask, 1).repeat((1,3,1,1)) # 2 3 256 256
            # # print(MixMask.shape, images_target.shape, pseudo_label.shape)
            # images_trg_classmix = images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - All_MixMask) + images_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * All_MixMask
            # labels_trg_classmix = pseudo_label[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - All_MixMask[:,0,:,:]) + labels_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * All_MixMask[:,0,:,:]
            # mixmask_trg = tg_mask[:cfg.TRAIN.BATCH_SIZE//2]* (1 - All_MixMask[:,0,:,:]) + sc_mask[:cfg.TRAIN.BATCH_SIZE//2] * All_MixMask[:,0,:,:]
            # ## 目标域贴在源域
            # for image_i in range(cfg.TRAIN.BATCH_SIZE//2,cfg.TRAIN.BATCH_SIZE):
                
            #     classes = torch.unique(pseudo_label[image_i])
            #     classes = classes[classes != 0]  # 筛选出非背景类
            #     nclasses = classes.shape[0]

            #     if nclasses <=2:
            #         classes = (classes[torch.Tensor(np.random.choice(nclasses, round(nclasses), replace=False)).long()]).cuda()
            #         MixMask = generate_class_mask(pseudo_label[image_i].cuda(), classes).unsqueeze(0).cuda()
            #     else:
            #         classes = (classes[torch.Tensor(np.random.choice(nclasses, round(2), replace=False)).long()]).cuda()
            #         MixMask = generate_class_mask(pseudo_label[image_i].cuda(), classes).unsqueeze(0).cuda() # 1,256 256
                
             

            #     # for clas in classes.tolist():
            #     #     classes_dict[clas] += 1
                    
            #     if image_i == 2:
            #         All_MixMask = MixMask
            #     else:
            #         All_MixMask = torch.cat((All_MixMask, MixMask)) # 2, 256 256

            # # dice_loss = dice_loss_class
            # All_MixMask = torch.unsqueeze(All_MixMask, 1).repeat((1,3,1,1)) # 2 3 256 256
            # # print(MixMask.shape, images_target.shape, pseudo_label.shape)
            # images_src_classmix = images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda() * All_MixMask + images_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - All_MixMask)
            # labels_src_classmix = pseudo_label[cfg.TRAIN.BATCH_SIZE//2:].cuda() * All_MixMask[:,0,:,:] + labels_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() *  (1 - All_MixMask[:,0,:,:])
            # mixmask_src = tg_mask[cfg.TRAIN.BATCH_SIZE//2:]* All_MixMask[:,0,:,:] + sc_mask[cfg.TRAIN.BATCH_SIZE//2:] * (1 - All_MixMask[:,0,:,:])

            # mix_mask = torch.cat([mixmask_trg, mixmask_src], dim=0)
            # # mix_veryreliablemask = torch.cat([mixveryreliablemask_trg, mixveryreliablemask_src], dim=0)
            # images_classmix = torch.cat([images_trg_classmix, images_src_classmix], dim=0)
            # labels_classmix = torch.cat([labels_trg_classmix, labels_src_classmix], dim=0)



            ## 源域贴在目标域
            for image_i in range(cfg.TRAIN.BATCH_SIZE//2):
                classes = torch.unique(labels_source[image_i])
                classes = classes[classes != 0]  # 筛选出非背景类
                nclasses = classes.shape[0]

                if nclasses > 1:
                    if nclasses <=2:
                        classes = (classes[torch.Tensor(np.random.choice(nclasses, round(nclasses), replace=False)).long()]).cuda()
                        MixMask = generate_class_mask(labels_source[image_i].cuda(), classes).unsqueeze(0).cuda()
                    else:
                        # num = np.random.randint(2,min(nclasses,3))
                        classes = (classes[torch.Tensor(np.random.choice(nclasses, round(2), replace=False)).long()]).cuda()
                        MixMask = generate_class_mask(labels_source[image_i].cuda(), classes).unsqueeze(0).cuda()
                else:
                    MixMask, _ = cut_generate_mask(images_source[image_i].cuda())
                    MixMask = MixMask.unsqueeze(0).cuda()

                # for clas in classes.tolist():
                #     classes_dict[clas] += 1
                    
                if image_i == 0:
                    All_MixMask = MixMask
                else:
                    All_MixMask = torch.cat((All_MixMask, MixMask)) # 2, 256 256

            # dice_loss = dice_loss_class
            All_MixMask = torch.unsqueeze(All_MixMask, 1).repeat((1,3,1,1)) # 2 3 256 256
            # print(MixMask.shape, images_target.shape, pseudo_label.shape)
            images_trg_classmix = images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - All_MixMask) + images_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * All_MixMask
            labels_trg_classmix = pseudo_label[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - All_MixMask[:,0,:,:]) + labels_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * All_MixMask[:,0,:,:]
            mixmask_trg = tg_mask[:cfg.TRAIN.BATCH_SIZE//2]* (1 - All_MixMask[:,0,:,:]) + sc_mask[:cfg.TRAIN.BATCH_SIZE//2] * All_MixMask[:,0,:,:]
            ## 目标域贴在源域
            for image_i in range(cfg.TRAIN.BATCH_SIZE//2,cfg.TRAIN.BATCH_SIZE):
                
                classes = torch.unique(pseudo_label[image_i])
                classes = classes[classes != 0]  # 筛选出非背景类
                nclasses = classes.shape[0]

                if nclasses > 1:
                    if nclasses <=2:
                        classes = (classes[torch.Tensor(np.random.choice(nclasses, round(nclasses), replace=False)).long()]).cuda()
                        MixMask = generate_class_mask(labels_source[image_i].cuda(), classes).unsqueeze(0).cuda()
                    else:
                        # num = np.random.randint(2,min(nclasses,3))
                        classes = (classes[torch.Tensor(np.random.choice(nclasses, round(2), replace=False)).long()]).cuda()
                        MixMask = generate_class_mask(labels_source[image_i].cuda(), classes).unsqueeze(0).cuda()
                else:
                    MixMask, _ = cut_generate_mask(images_source[image_i].cuda())
                    MixMask = MixMask.unsqueeze(0).cuda()

                # for clas in classes.tolist():
                #     classes_dict[clas] += 1
                    
                if image_i == 2:
                    All_MixMask = MixMask
                else:
                    All_MixMask = torch.cat((All_MixMask, MixMask)) # 2, 256 256

            # dice_loss = dice_loss_class
            All_MixMask = torch.unsqueeze(All_MixMask, 1).repeat((1,3,1,1)) # 2 3 256 256
            # print(MixMask.shape, images_target.shape, pseudo_label.shape)
            images_src_classmix = images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda() * All_MixMask + images_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - All_MixMask)
            labels_src_classmix = pseudo_label[cfg.TRAIN.BATCH_SIZE//2:].cuda() * All_MixMask[:,0,:,:] + labels_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() *  (1 - All_MixMask[:,0,:,:])
            mixmask_src = tg_mask[cfg.TRAIN.BATCH_SIZE//2:]* All_MixMask[:,0,:,:] + sc_mask[cfg.TRAIN.BATCH_SIZE//2:] * (1 - All_MixMask[:,0,:,:])

            mix_mask = torch.cat([mixmask_trg, mixmask_src], dim=0)
            # mix_veryreliablemask = torch.cat([mixveryreliablemask_trg, mixveryreliablemask_src], dim=0)
            images_classmix = torch.cat([images_trg_classmix, images_src_classmix], dim=0)
            labels_classmix = torch.cat([labels_trg_classmix, labels_src_classmix], dim=0)

    
        if cfg.TRAIN.CONSWITCH:
            consistency_weight = get_current_consistency_weight(i_iter, cfg.TRAIN.MAX_ITERS)
        else:
            consistency_weight = 1

        # consistency_weight = 1

        # unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
        # unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
        # onesWeights = torch.ones((unlabeledWeight.shape)).cuda()
        # pixelWiseWeight = unlabeledWeight


        _,pred_src_aux, pred_src_main = model(images_source.cuda())
        _,pred_trg_aux, pred_trg_main = model(images_target.cuda())
        _,pred_mix_aux, pred_mix_main = model(images_classmix.cuda())

        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux     = interp(pred_src_aux)
            pred_trg_aux     = interp(pred_trg_aux)
            pred_mix_aux     = interp(pred_mix_aux)

            loss_seg_src_aux = loss_calc(pred_src_aux,labels_source)
            loss_dice_aux    = dice_loss(pred_src_aux,labels_source.cuda())
        else:
            loss_seg_src_aux = 0
            loss_dice_aux    = 0

        pred_src_main     = interp(pred_src_main)
        pred_trg_main     = interp(pred_trg_main)
        pred_mix_main     = interp(pred_mix_main)
        loss_seg_src_main = loss_calc(pred_src_main,labels_source)
        loss_dice_main    = dice_loss(pred_src_main,labels_source.cuda())


        L_u_aux =  consistency_weight * unlabeled_loss(pred_mix_aux, labels_classmix, mix_mask)
        L_u_main = consistency_weight * unlabeled_loss(pred_mix_main, labels_classmix, mix_mask)
        # L_dice_aux =  consistency_weight * dice_loss(pred_mix_aux, labels_classmix)
        # L_dice_main = consistency_weight * dice_loss(pred_mix_main, labels_classmix)

        # consistency_loss = torch.mean((torch.softmax(pred_trg_main, dim=1) - ema_output_main_soft)**2)
        # consistency_aux_loss = torch.mean((torch.softmax(pred_trg_aux, dim=1) - ema_output_aux_soft)**2)

        loss = (cfg.TRAIN.LAMBDA_SEG_SRC_MAIN    * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_SRC_AUX   * loss_seg_src_aux
                + cfg.TRAIN.LAMBDA_DICE_SRC_MAIN * loss_dice_main
                + cfg.TRAIN.LAMBDA_DICE_SRC_AUX  * loss_dice_aux
                # + consistency_weight * consistency_loss 
                # + 0.1 * consistency_weight * consistency_aux_loss
                + L_u_main
                + 0.1 * L_u_aux
                # + 0.1 * L_dice_aux
                # + L_dice_main
                )
        
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, 0.99, i_iter)

        current_losses = {'loss_seg_src_aux' :loss_seg_src_aux,
                          'loss_seg_src_main':loss_seg_src_main,
                        #   'loss_target_entp_aux':loss_target_entp_aux,
                        #   'loss_target_entp_main':loss_target_entp_aux,
                          'loss_dice_aux': loss_dice_aux,
                          'loss_dice_main'   :loss_dice_main}
        print_losses(current_losses,i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            if cfg.DATASETS == 'Cardiac':
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Cardiac(model, cfg.TARGET, results_file, i_iter)
            else:
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Abdomen(model, cfg.TARGET, results_file, i_iter)
            
            saved_path = "/home/data_backup/zhr_savedmodel/{0}2{1}_{2}_01cutmix2".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
            os.makedirs(saved_path, exist_ok=True)
            torch.save(model.state_dict(),  saved_path+f'/model_{i_iter}.pth')
            tmp_dice = np.mean(dice_mean)
            dice_all.append(tmp_dice)
            if(i_iter == cfg.TRAIN.MAX_ITERS):
                dice_all = np.array(dice_all)
                np.save('/home/data_backup/zhr_savedmodel/dice_01cutmix2_'+cfg.SOURCE+'2'+cfg.TARGET+'_'+cfg.TRAIN.DA_METHOD, dice_all)
                        
            # if best_model < tmp_dice:
            #     print('taking snapshot ...')
            #     print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            #     snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            #     torch.save(model.state_dict(),  snapshot_dir / '{0}_{1}2{2}_Best_CutMix_ourdice_model.pth'.format(cfg.DATASETS, cfg.SOURCE, cfg.TARGET))
            #     best_model = tmp_dice

        sys.stdout.flush()



def train_classmix_ST_new(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg):
    '''
    UDA training
    '''

    results_file = "/home/zhr/ICME/scripts/log_ablation/{0}2{1}_{2}consistency_weight1.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)

    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    num_classes       = cfg.NUM_CLASSES


    unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
    # dice_loss = DiceLoss(n_classes=num_classes).cuda()

    model.train()
    ema_model.train()

    model.cuda()
    ema_model.cuda()
    cudnn.benchmark = True
    cudnn.enabled   = True

    #optimized
    if cfg.TRAIN.OPTIM_G == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.TRAIN.LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                              lr=cfg.TRAIN.LEARNING_RATE,
                              momentum=cfg.TRAIN.MOMENTUM,
                              weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # interpolate output segmaps
    interp        = nn.Upsample(size=(input_size_source[1],input_size_source[0]),mode='bilinear',
                         align_corners=True)

    # labels for adversarial learning


    strain_loader_iter          = enumerate(strain_loader)
    trgtrain_loader_iter        = enumerate(trgtrain_loader)
    sval_loader_iter            = enumerate(sval_loader)

    transforms = None
    img_mean   = cfg.TRAIN.IMG_MEAN
    best_model =  -1
    dice_all = []

    for i_iter in tqdm(range(cfg.TRAIN.MAX_ITERS+1)):

        model.train()
        ema_model.train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer,i_iter,cfg)


        #UDA training
        try:
            _, batch = strain_loader_iter.__next__()
        except StopIteration:
            strain_loader_iter = enumerate(strain_loader)
            _, batch = strain_loader_iter.__next__()
        images_source, labels_source,_ = batch

        try:
            _, batch = trgtrain_loader_iter.__next__()
        except StopIteration:
            trgtrain_loader_iter = enumerate(trgtrain_loader)
            _, batch             = trgtrain_loader_iter.__next__()
        images_target, labels_target,_ = batch

        # src_in_trg = []
        # for i in range(cfg.TRAIN.BATCH_SIZE):
        #     st = match_histograms(np.array(images_source[i]), np.array(images_target[i]), channel_axis=0)
        #     src_in_trg.append(st)
        # images_source = torch.tensor(src_in_trg, dtype=torch.float32)

        noise = torch.clamp(torch.randn_like(
                images_target) * 0.1, -0.2, 0.2)
        ema_inputs = images_target + noise

        with torch.no_grad():
            _, ema_pred_trg_axu, ema_pred_trg_main = ema_model(ema_inputs.cuda())
            
            ema_pred_trg_main     = interp(ema_pred_trg_main)
            ema_output_main_soft = torch.softmax(ema_pred_trg_main, dim=1)
            max_probs_main, pseudo_label  = torch.max(ema_output_main_soft, dim=1)
            
            percent_20 = 20 * (1 - (i_iter)/ cfg.TRAIN.MAX_ITERS)
            up_percent_80 = 100 - percent_20
            percent_50 = 50 * (1 - (i_iter)/ cfg.TRAIN.MAX_ITERS)
            up_percent_50 = 100 - percent_50

            entropy = -torch.sum(ema_output_main_soft * torch.log(ema_output_main_soft + 1e-10), dim=1)
            up_thresh_80 = np.percentile(
                entropy.detach().cpu().numpy().flatten(), up_percent_80
            )
            up_thresh_50 = np.percentile(
                entropy.detach().cpu().numpy().flatten(), up_percent_50
            )
            thresh_veryreliable = np.percentile(
                entropy.detach().cpu().numpy().flatten(), percent_20
            ) 

            tg_mask_80 = entropy.le(up_thresh_80).long().cuda()
            tg_mask_50 = entropy.le(up_thresh_50).long().cuda()

            persent = up_percent_80/100.0
            tg_mask = (tg_mask_80-tg_mask_50)*persent + tg_mask_50
            # print(torch.unique(tg_mask))
            # percent = drop_percent/100.0
            # tg_mask = tg_mask*percent
            tg_veryreliablemask = entropy.le(thresh_veryreliable).long().cuda()
            sc_mask = torch.ones((tg_mask.shape)).cuda()



            # unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
            # unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
            # onesWeights = torch.ones((unlabeledWeight.shape)).cuda()

            # MixMask, loss_mask = generate_mask(images_target.cuda())
            # pixelWiseWeight = unlabeledWeight * MixMask + onesWeights * (1 - MixMask)
            # images_classmix = images_target.cuda() * MixMask + images_source.cuda() * (1 - MixMask)
            # labels_classmix = pseudo_label.cuda() * MixMask + labels_source.cuda() * (1 - MixMask)


            for image_i in range(cfg.TRAIN.BATCH_SIZE):
                classes = torch.unique(labels_source[image_i])
                classes = classes[classes != 0]  # 筛选出非背景类
                nclasses = classes.shape[0]

                if nclasses <=2:
                    classes = (classes[torch.Tensor(np.random.choice(nclasses, round(nclasses), replace=False)).long()]).cuda()
                    MixMask = generate_class_mask(labels_source[image_i].cuda(), classes).unsqueeze(0).cuda()
                else:
                    classes = (classes[torch.Tensor(np.random.choice(nclasses, round(2), replace=False)).long()]).cuda()
                    MixMask = generate_class_mask(labels_source[image_i].cuda(), classes).unsqueeze(0).cuda() # 1,256 256
             

                # for clas in classes.tolist():
                #     classes_dict[clas] += 1
                
                # print(image_i,MixMask.sum())
               
                if image_i == 0:
                    All_MixMask = MixMask
                else:
                    All_MixMask = torch.cat((All_MixMask, MixMask)) # 4, 256 256
                    # All_MixMask = torch.cat((MixMask, All_MixMask)) # 4, 256 256
            # print(All_MixMask[0].sum(), All_MixMask[1].sum(), All_MixMask[2].sum(),All_MixMask[3].sum())

            # dice_loss = dice_loss_class
            All_MixMask = torch.unsqueeze(All_MixMask, 1).repeat((1,3,1,1)) # 4 3 256 256
            # print(MixMask.shape, images_target.shape, pseudo_label.shape)
            images_classmix = images_target.cuda() * (1 - All_MixMask) + images_source.cuda() * All_MixMask
            labels_classmix = pseudo_label.cuda() * (1 - All_MixMask[:,0,:,:]) + labels_source.cuda() * All_MixMask[:,0,:,:]
            mix_mask = tg_mask* (1 - All_MixMask[:,0,:,:]) + sc_mask * All_MixMask[:,0,:,:]

    


        # unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
        # unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
        # onesWeights = torch.ones((unlabeledWeight.shape)).cuda()
        # pixelWiseWeight = unlabeledWeight


        _,pred_src_aux, pred_src_main = model(images_source.cuda())
        _,pred_trg_aux, pred_trg_main = model(images_target.cuda())
        _,pred_mix_aux, pred_mix_main = model(images_classmix.cuda())

        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux     = interp(pred_src_aux)
            pred_trg_aux     = interp(pred_trg_aux)
            pred_mix_aux     = interp(pred_mix_aux)

            loss_seg_src_aux = loss_calc(pred_src_aux,labels_source)
            loss_dice_aux    = dice_loss(pred_src_aux,labels_source.cuda())
        else:
            loss_seg_src_aux = 0
            loss_dice_aux    = 0

        pred_src_main     = interp(pred_src_main)
        pred_trg_main     = interp(pred_trg_main)
        pred_mix_main     = interp(pred_mix_main)
        loss_seg_src_main = loss_calc(pred_src_main,labels_source)
        loss_dice_main    = dice_loss(pred_src_main,labels_source.cuda())



        if cfg.TRAIN.CONSWITCH:
            consistency_weight = get_current_consistency_weight(i_iter, cfg.TRAIN.MAX_ITERS)
        else:
            consistency_weight = 1
        
        loss_mix_aux = consistency_weight * unlabeled_loss(pred_mix_aux, labels_classmix, mix_mask)
        loss_mix = consistency_weight * unlabeled_loss(pred_mix_main, labels_classmix, mix_mask)  

        # loss_mix_aux =  consistency_weight * unlabeled_loss(pred_mix_aux, labels_classmix, pixelWiseWeight)
        # loss_mix = consistency_weight * unlabeled_loss(pred_mix_main, labels_classmix, pixelWiseWeight)

        # weight = np.size(np.array(labels_classmix.cpu())) / torch.sum(mix_mask)
        # loss_mix_aux = weight * unlabeled_loss(pred_mix_aux, labels_classmix, mix_mask)
        # loss_mix = weight * unlabeled_loss(pred_mix_main, labels_classmix, mix_mask)  




        loss = (cfg.TRAIN.LAMBDA_SEG_SRC_MAIN    * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_SRC_AUX   * loss_seg_src_aux
                + cfg.TRAIN.LAMBDA_DICE_SRC_MAIN * loss_dice_main
                + cfg.TRAIN.LAMBDA_DICE_SRC_AUX  * loss_dice_aux
                # + consistency_weight * consistency_loss 
                # + 0.1 * consistency_weight * consistency_aux_loss
                + 0.1*loss_mix_aux
                + loss_mix
                # + 0.1 * L_dice_aux
                # + L_dice_main
                )
        
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, 0.99, i_iter)

        current_losses = {'loss_seg_src_aux' :loss_seg_src_aux,
                          'loss_seg_src_main':loss_seg_src_main,
                        #   'loss_target_entp_aux':loss_target_entp_aux,
                        #   'loss_target_entp_main':loss_target_entp_aux,
                          'loss_dice_aux': loss_dice_aux,
                          'loss_dice_main'   :loss_dice_main}
        print_losses(current_losses,i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            if cfg.DATASETS == 'Cardiac':
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Cardiac(model, cfg.TARGET, results_file, i_iter)
            else:
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Abdomen(model, cfg.TARGET, results_file, i_iter)
            
            saved_path = "/home/data_backup/zhr_savedmodel/{0}2{1}_{2}consistency_weight1".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
            os.makedirs(saved_path, exist_ok=True)
            torch.save(model.state_dict(),  saved_path+f'/model_{i_iter}.pth')
            tmp_dice = np.mean(dice_mean)
            dice_all.append(tmp_dice)
            if(i_iter == cfg.TRAIN.MAX_ITERS):
                dice_all = np.array(dice_all)
                np.save('/home/data_backup/zhr_savedmodel/dice_consistency_weight1'+cfg.SOURCE+'2'+cfg.TARGET+'_'+cfg.TRAIN.DA_METHOD, dice_all)
                        
            # if best_model < tmp_dice:
            #     print('taking snapshot ...')
            #     print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            #     snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            #     torch.save(model.state_dict(),  snapshot_dir / '{0}_{1}2{2}_Best_CutMix_ourdice_model.pth'.format(cfg.DATASETS, cfg.SOURCE, cfg.TARGET))
            #     best_model = tmp_dice

        sys.stdout.flush()



def train_bcutmix_ST(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg):
    '''
    UDA training
    '''

    results_file = "/home/zhr/ICME/scripts/log/{0}2{1}_{2}_ourdice_myweight_23cut2.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)

    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    num_classes       = cfg.NUM_CLASSES


    unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
    # dice_loss = DiceLoss(n_classes=num_classes).cuda()
    #SEGMENTATION
    model.train()
    ema_model.train()

    model.cuda()
    ema_model.cuda()
    cudnn.benchmark = True
    cudnn.enabled   = True



    #optimized
    if cfg.TRAIN.OPTIM_G == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.TRAIN.LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                              lr=cfg.TRAIN.LEARNING_RATE,
                              momentum=cfg.TRAIN.MOMENTUM,
                              weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # interpolate output segmaps
    interp        = nn.Upsample(size=(input_size_source[1],input_size_source[0]),mode='bilinear',
                         align_corners=True)

    # labels for adversarial learning


    strain_loader_iter          = enumerate(strain_loader)
    trgtrain_loader_iter        = enumerate(trgtrain_loader)
    sval_loader_iter            = enumerate(sval_loader)

    transforms = None
    img_mean   = cfg.TRAIN.IMG_MEAN
    best_model =  -1

    for i_iter in tqdm(range(cfg.TRAIN.MAX_ITERS+1)):

        model.train()
        ema_model.train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer,i_iter,cfg)


        #UDA training
        try:
            _, batch = strain_loader_iter.__next__()
        except StopIteration:
            strain_loader_iter = enumerate(strain_loader)
            _, batch = strain_loader_iter.__next__()
        images_source, labels_source,_ = batch

        try:
            _, batch = trgtrain_loader_iter.__next__()
        except StopIteration:
            trgtrain_loader_iter = enumerate(trgtrain_loader)
            _, batch             = trgtrain_loader_iter.__next__()
        images_target, labels_target,_ = batch


        noise = torch.clamp(torch.randn_like(
                images_target) * 0.1, -0.2, 0.2)
        ema_inputs = images_target + noise

        with torch.no_grad():
            _, ema_pred_trg_axu, ema_pred_trg_main = ema_model(ema_inputs.cuda())
            ema_pred_trg_axu     = interp(ema_pred_trg_axu)
            ema_pred_trg_main     = interp(ema_pred_trg_main)

            ema_output_main_soft = torch.softmax(ema_pred_trg_main, dim=1)
            ema_output_aux_soft = torch.softmax(ema_pred_trg_axu, dim=1)
            max_probs_main, pseudo_label  = torch.max(ema_output_main_soft, dim=1)
            
            unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
            unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
            onesWeights = torch.ones((unlabeledWeight.shape)).cuda()




            MixMask, loss_mask = generate_mask(images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda())
            ## 源域贴在目标域
            pixelWiseWeight_trg = unlabeledWeight[:cfg.TRAIN.BATCH_SIZE//2] * MixMask + onesWeights[:cfg.TRAIN.BATCH_SIZE//2] * (1 - MixMask)
            images_trg_classmix = images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + images_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)
            labels_trg_classmix = pseudo_label[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + labels_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)

            MixMask, loss_mask = generate_mask(images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda())
            ## 目标域贴在源域
            pixelWiseWeight_src = unlabeledWeight[cfg.TRAIN.BATCH_SIZE//2:] * (1 - MixMask) + onesWeights[cfg.TRAIN.BATCH_SIZE//2:] * MixMask
            images_src_classmix = images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + images_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            labels_src_classmix = pseudo_label[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + labels_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            
            pixelWiseWeight = torch.cat([pixelWiseWeight_trg, pixelWiseWeight_src], dim=0)
            images_classmix = torch.cat([images_trg_classmix, images_src_classmix], dim=0)
            labels_classmix = torch.cat([labels_trg_classmix, labels_src_classmix], dim=0)
    
        if cfg.TRAIN.CONSWITCH:
            consistency_weight = get_current_consistency_weight(i_iter, cfg.TRAIN.MAX_ITERS)
        else:
            consistency_weight = 1

        # unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
        # unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
        # onesWeights = torch.ones((unlabeledWeight.shape)).cuda()
        # pixelWiseWeight = unlabeledWeight


        _,pred_src_aux, pred_src_main = model(images_source.cuda())
        _,pred_trg_aux, pred_trg_main = model(images_target.cuda())
        _,pred_mix_aux, pred_mix_main = model(images_classmix.cuda())

        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux     = interp(pred_src_aux)
            pred_trg_aux     = interp(pred_trg_aux)
            pred_mix_aux     = interp(pred_mix_aux)

            loss_seg_src_aux = loss_calc(pred_src_aux,labels_source)
            loss_dice_aux    = dice_loss(pred_src_aux,labels_source.cuda())
        else:
            loss_seg_src_aux = 0
            loss_dice_aux    = 0

        pred_src_main     = interp(pred_src_main)
        pred_trg_main     = interp(pred_trg_main)
        pred_mix_main     = interp(pred_mix_main)
        loss_seg_src_main = loss_calc(pred_src_main,labels_source)
        loss_dice_main    = dice_loss(pred_src_main,labels_source.cuda())


        L_u_aux =  consistency_weight * unlabeled_loss(pred_mix_aux, labels_classmix, pixelWiseWeight)
        L_u_main = consistency_weight * unlabeled_loss(pred_mix_main, labels_classmix, pixelWiseWeight)
        # L_dice_aux =  consistency_weight * dice_loss(pred_mix_aux, labels_classmix)
        # L_dice_main = consistency_weight * dice_loss(pred_mix_main, labels_classmix)

        consistency_loss = torch.mean((torch.softmax(pred_trg_main, dim=1) - ema_output_main_soft)**2)
        consistency_aux_loss = torch.mean((torch.softmax(pred_trg_aux, dim=1) - ema_output_aux_soft)**2)

        loss = (cfg.TRAIN.LAMBDA_SEG_SRC_MAIN    * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_SRC_AUX   * loss_seg_src_aux
                + cfg.TRAIN.LAMBDA_DICE_SRC_MAIN * loss_dice_main
                + cfg.TRAIN.LAMBDA_DICE_SRC_AUX  * loss_dice_aux
                + consistency_weight * consistency_loss 
                + 0.1 * consistency_weight * consistency_aux_loss
                + L_u_main
                + 0.1 * L_u_aux
                # + 0.1 * L_dice_aux
                # + L_dice_main
                )
        
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, 0.99, i_iter)

        current_losses = {'loss_seg_src_aux' :loss_seg_src_aux,
                          'loss_seg_src_main':loss_seg_src_main,
                        #   'loss_target_entp_aux':loss_target_entp_aux,
                        #   'loss_target_entp_main':loss_target_entp_aux,
                          'loss_dice_aux': loss_dice_aux,
                          'loss_dice_main'   :loss_dice_main}
        print_losses(current_losses,i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            if cfg.DATASETS == 'Cardiac':
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Cardiac(model, cfg.TARGET, results_file, i_iter)
            else:
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Abdomen(model, cfg.TARGET, results_file, i_iter)
            
            # saved_path = "/home/data_backup/zhr_savedmodel/{0}2{1}_{2}_ourdice_myweight_23cut".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
            # os.makedirs(saved_path, exist_ok=True)
            # torch.save(model.state_dict(),  saved_path+f'/model_{i_iter}.pth')
            # tmp_dice = np.mean(dice_mean)
                        
            # if best_model < tmp_dice:
            #     print('taking snapshot ...')
            #     print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            #     snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            #     torch.save(model.state_dict(),  snapshot_dir / '{0}_{1}2{2}_Best_CutMix_ourdice_model.pth'.format(cfg.DATASETS, cfg.SOURCE, cfg.TARGET))
            #     best_model = tmp_dice

        sys.stdout.flush()


def train_bcutmix_ST_nonemse(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg):
    '''
    UDA training
    '''

    results_file = "/home/zhr/ICME/scripts/log_ablation_new/{0}2{1}_{2}8.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)

    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    num_classes       = cfg.NUM_CLASSES


    unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
    # dice_loss = DiceLoss(n_classes=num_classes).cuda()
    #SEGMENTATION
    model.train()
    ema_model.train()

    model.cuda()
    ema_model.cuda()
    cudnn.benchmark = True
    cudnn.enabled   = True



    #optimized
    if cfg.TRAIN.OPTIM_G == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.TRAIN.LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                              lr=cfg.TRAIN.LEARNING_RATE,
                              momentum=cfg.TRAIN.MOMENTUM,
                              weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # interpolate output segmaps
    interp        = nn.Upsample(size=(input_size_source[1],input_size_source[0]),mode='bilinear',
                         align_corners=True)

    # labels for adversarial learning


    strain_loader_iter          = enumerate(strain_loader)
    trgtrain_loader_iter        = enumerate(trgtrain_loader)
    sval_loader_iter            = enumerate(sval_loader)

    transforms = None
    img_mean   = cfg.TRAIN.IMG_MEAN
    best_model =  -1
    dice_all = []

    for i_iter in tqdm(range(cfg.TRAIN.MAX_ITERS+1)):

        model.train()
        ema_model.train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer,i_iter,cfg)


        #UDA training
        try:
            _, batch = strain_loader_iter.__next__()
        except StopIteration:
            strain_loader_iter = enumerate(strain_loader)
            _, batch = strain_loader_iter.__next__()
        images_source, labels_source,_ = batch

        try:
            _, batch = trgtrain_loader_iter.__next__()
        except StopIteration:
            trgtrain_loader_iter = enumerate(trgtrain_loader)
            _, batch             = trgtrain_loader_iter.__next__()
        images_target, labels_target,_ = batch


        noise = torch.clamp(torch.randn_like(
                images_target) * 0.1, -0.2, 0.2)
        ema_inputs = images_target + noise

        with torch.no_grad():
            _, ema_pred_trg_axu, ema_pred_trg_main = ema_model(ema_inputs.cuda())
            ema_pred_trg_axu     = interp(ema_pred_trg_axu)
            ema_pred_trg_main     = interp(ema_pred_trg_main)

            ema_output_main_soft = torch.softmax(ema_pred_trg_main, dim=1)
            ema_output_aux_soft = torch.softmax(ema_pred_trg_axu, dim=1)
            max_probs_main, pseudo_label  = torch.max(ema_output_main_soft, dim=1)
            
            unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
            unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
            onesWeights = torch.ones((unlabeledWeight.shape)).cuda()




            MixMask, loss_mask = generate_mask(images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda())
            ## 源域贴在目标域
            pixelWiseWeight_trg = unlabeledWeight[:cfg.TRAIN.BATCH_SIZE//2] * MixMask + onesWeights[:cfg.TRAIN.BATCH_SIZE//2] * (1 - MixMask)
            images_trg_classmix = images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + images_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)
            labels_trg_classmix = pseudo_label[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + labels_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)

            MixMask, loss_mask = generate_mask(images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda())
            ## 目标域贴在源域
            pixelWiseWeight_src = unlabeledWeight[cfg.TRAIN.BATCH_SIZE//2:] * (1 - MixMask) + onesWeights[cfg.TRAIN.BATCH_SIZE//2:] * MixMask
            images_src_classmix = images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + images_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            labels_src_classmix = pseudo_label[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + labels_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            
            pixelWiseWeight = torch.cat([pixelWiseWeight_trg, pixelWiseWeight_src], dim=0)
            images_classmix = torch.cat([images_trg_classmix, images_src_classmix], dim=0)
            labels_classmix = torch.cat([labels_trg_classmix, labels_src_classmix], dim=0)
    
        if cfg.TRAIN.CONSWITCH:
            consistency_weight = get_current_consistency_weight(i_iter, cfg.TRAIN.MAX_ITERS)
        else:
            consistency_weight = 1

        # unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
        # unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
        # onesWeights = torch.ones((unlabeledWeight.shape)).cuda()
        # pixelWiseWeight = unlabeledWeight


        _,pred_src_aux, pred_src_main = model(images_source.cuda())
        _,pred_trg_aux, pred_trg_main = model(images_target.cuda())
        _,pred_mix_aux, pred_mix_main = model(images_classmix.cuda())

        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux     = interp(pred_src_aux)
            pred_trg_aux     = interp(pred_trg_aux)
            pred_mix_aux     = interp(pred_mix_aux)

            loss_seg_src_aux = loss_calc(pred_src_aux,labels_source)
            loss_dice_aux    = dice_loss(pred_src_aux,labels_source.cuda())
        else:
            loss_seg_src_aux = 0
            loss_dice_aux    = 0

        pred_src_main     = interp(pred_src_main)
        pred_trg_main     = interp(pred_trg_main)
        pred_mix_main     = interp(pred_mix_main)
        loss_seg_src_main = loss_calc(pred_src_main,labels_source)
        loss_dice_main    = dice_loss(pred_src_main,labels_source.cuda())


        L_u_aux =  consistency_weight * unlabeled_loss(pred_mix_aux, labels_classmix, pixelWiseWeight)
        L_u_main = consistency_weight * unlabeled_loss(pred_mix_main, labels_classmix, pixelWiseWeight)
        # L_dice_aux =  consistency_weight * dice_loss(pred_mix_aux, labels_classmix)
        # L_dice_main = consistency_weight * dice_loss(pred_mix_main, labels_classmix)

        # consistency_loss = torch.mean((torch.softmax(pred_trg_main, dim=1) - ema_output_main_soft)**2)
        # consistency_aux_loss = torch.mean((torch.softmax(pred_trg_aux, dim=1) - ema_output_aux_soft)**2)

        loss = (cfg.TRAIN.LAMBDA_SEG_SRC_MAIN    * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_SRC_AUX   * loss_seg_src_aux
                + cfg.TRAIN.LAMBDA_DICE_SRC_MAIN * loss_dice_main
                + cfg.TRAIN.LAMBDA_DICE_SRC_AUX  * loss_dice_aux
                # + consistency_weight * consistency_loss 
                # + 0.1 * consistency_weight * consistency_aux_loss
                + L_u_main
                + 0.1 * L_u_aux
                # + 0.1 * L_dice_aux
                # + L_dice_main
                )
        
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, 0.99, i_iter)

        current_losses = {'loss_seg_src_aux' :loss_seg_src_aux,
                          'loss_seg_src_main':loss_seg_src_main,
                        #   'loss_target_entp_aux':loss_target_entp_aux,
                        #   'loss_target_entp_main':loss_target_entp_aux,
                          'loss_dice_aux': loss_dice_aux,
                          'loss_dice_main'   :loss_dice_main}
        print_losses(current_losses,i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            if cfg.DATASETS == 'Cardiac':
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Cardiac(model, cfg.TARGET, results_file, i_iter)
            else:
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Abdomen(model, cfg.TARGET, results_file, i_iter)
            
            # saved_path = "/home/data_backup/zhr_savedmodel/{0}2{1}_{2}_ourdice_myweight_23cut".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
            # os.makedirs(saved_path, exist_ok=True)
            # torch.save(model.state_dict(),  saved_path+f'/model_{i_iter}.pth')

            tmp_dice = np.mean(dice_mean)
            dice_all.append(tmp_dice)
            if(i_iter == cfg.TRAIN.MAX_ITERS):
                dice_all = np.array(dice_all)
                np.save('/home/data_backup/zhr_savedmodel/dice_8'+cfg.SOURCE+'2'+cfg.TARGET+'_'+cfg.TRAIN.DA_METHOD, dice_all)
                        
                        
            # if best_model < tmp_dice:
            #     print('taking snapshot ...')
            #     print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            #     snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            #     torch.save(model.state_dict(),  snapshot_dir / '{0}_{1}2{2}_Best_CutMix_ourdice_model.pth'.format(cfg.DATASETS, cfg.SOURCE, cfg.TARGET))
            #     best_model = tmp_dice

        sys.stdout.flush()


def train_bcutmix_ST_new(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg):
    '''
    UDA training
    '''

    results_file = "/home/zhr/ICME/scripts/log_ablation_new/{0}2{1}_{2}8.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)

    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    num_classes       = cfg.NUM_CLASSES


    unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
    # dice_loss = DiceLoss(n_classes=num_classes).cuda()
    #SEGMENTATION
    model.train()
    ema_model.train()

    model.cuda()
    ema_model.cuda()
    cudnn.benchmark = True
    cudnn.enabled   = True



    #optimized
    if cfg.TRAIN.OPTIM_G == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.TRAIN.LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                              lr=cfg.TRAIN.LEARNING_RATE,
                              momentum=cfg.TRAIN.MOMENTUM,
                              weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # interpolate output segmaps
    interp        = nn.Upsample(size=(input_size_source[1],input_size_source[0]),mode='bilinear',
                         align_corners=True)

    # labels for adversarial learning


    strain_loader_iter          = enumerate(strain_loader)
    trgtrain_loader_iter        = enumerate(trgtrain_loader)
    sval_loader_iter            = enumerate(sval_loader)

    transforms = None
    img_mean   = cfg.TRAIN.IMG_MEAN
    best_model =  -1
    dice_all = []

    for i_iter in tqdm(range(cfg.TRAIN.MAX_ITERS+1)):

        model.train()
        ema_model.train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer,i_iter,cfg)


        #UDA training
        try:
            _, batch = strain_loader_iter.__next__()
        except StopIteration:
            strain_loader_iter = enumerate(strain_loader)
            _, batch = strain_loader_iter.__next__()
        images_source, labels_source,_ = batch

        try:
            _, batch = trgtrain_loader_iter.__next__()
        except StopIteration:
            trgtrain_loader_iter = enumerate(trgtrain_loader)
            _, batch             = trgtrain_loader_iter.__next__()
        images_target, labels_target,_ = batch


        noise = torch.clamp(torch.randn_like(
                images_target) * 0.1, -0.2, 0.2)
        ema_inputs = images_target + noise

        with torch.no_grad():
            _, ema_pred_trg_axu, ema_pred_trg_main = ema_model(ema_inputs.cuda())
            # source_teacher_feas, ema_pred_src_axu, ema_pred_src_main = ema_model(images_source.cuda())
            # ema_output_main_soft_feas = torch.softmax(ema_pred_trg_main, dim=1)
            # max_probs_main_feas, pseudo_label_feas  = torch.max(ema_output_main_soft_feas, dim=1)

            ema_pred_trg_axu     = interp(ema_pred_trg_axu)
            ema_pred_trg_main     = interp(ema_pred_trg_main)

            ema_output_main_soft = torch.softmax(ema_pred_trg_main, dim=1)
            ema_output_aux_soft = torch.softmax(ema_pred_trg_axu, dim=1)
            max_probs_main, pseudo_label  = torch.max(ema_output_main_soft, dim=1)




            
            # drop_percent = 80
            # percent_unreliable = (100 - drop_percent) * (1 - (i_iter)/ cfg.TRAIN.MAX_ITERS)
            # drop_percent = 100 - percent_unreliable
            # entropy = -torch.sum(ema_output_main_soft * torch.log(ema_output_main_soft + 1e-10), dim=1)
            # thresh = np.percentile(
            #     entropy.detach().cpu().numpy().flatten(), drop_percent
            # )
            # tg_mask = entropy.le(thresh).long().cuda()
            # sc_mask = torch.ones((tg_mask.shape)).cuda()

            percent_20 = 20 * (1 - (i_iter)/ cfg.TRAIN.MAX_ITERS)
            up_percent_80 = 100 - percent_20
            percent_50 = 50 * (1 - (i_iter)/ cfg.TRAIN.MAX_ITERS)
            up_percent_50 = 100 - percent_50

            percent_veryreliable = 50 * (1 - (i_iter)/ cfg.TRAIN.MAX_ITERS)

            entropy = -torch.sum(ema_output_main_soft * torch.log(ema_output_main_soft + 1e-10), dim=1)
            up_thresh_80 = np.percentile(
                entropy.detach().cpu().numpy().flatten(), up_percent_80
            )
            up_thresh_50 = np.percentile(
                entropy.detach().cpu().numpy().flatten(), up_percent_50
            )
            thresh_veryreliable = np.percentile(
                entropy.detach().cpu().numpy().flatten(), percent_veryreliable
            ) 

            tg_mask_80 = entropy.le(up_thresh_80).long().cuda()
            tg_mask_50 = entropy.le(up_thresh_50).long().cuda()

            persent = up_percent_80/100.0
            tg_mask = (tg_mask_80-tg_mask_50)*persent + tg_mask_50
            # print(torch.unique(tg_mask))
            # percent = drop_percent/100.0
            # tg_mask = tg_mask*percent
            tg_veryreliablemask = entropy.le(thresh_veryreliable).long().cuda()
            sc_mask = torch.ones((tg_mask.shape)).cuda()


            
            # unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
            # unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
            # onesWeights = torch.ones((unlabeledWeight.shape)).cuda()




            MixMask, loss_mask = generate_mask(images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda())
            ## 源域贴在目标域
            mixmask_trg = tg_mask[:cfg.TRAIN.BATCH_SIZE//2] * MixMask + sc_mask[:cfg.TRAIN.BATCH_SIZE//2] * (1 - MixMask)
            images_trg_classmix = images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + images_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)
            labels_trg_classmix = pseudo_label[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + labels_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)

            MixMask, loss_mask = generate_mask(images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda())
            ## 目标域贴在源域
            mixmask_src = tg_mask[cfg.TRAIN.BATCH_SIZE//2:] * (1 - MixMask) + sc_mask[cfg.TRAIN.BATCH_SIZE//2:] * MixMask
            images_src_classmix = images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + images_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            labels_src_classmix = pseudo_label[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + labels_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            
            mix_mask = torch.cat([mixmask_trg, mixmask_src], dim=0)
            images_classmix = torch.cat([images_trg_classmix, images_src_classmix], dim=0)
            labels_classmix = torch.cat([labels_trg_classmix, labels_src_classmix], dim=0)
    
        if cfg.TRAIN.CONSWITCH:
            consistency_weight = get_current_consistency_weight(i_iter, cfg.TRAIN.MAX_ITERS)
        else:
            consistency_weight = 1

        # unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
        # unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
        # onesWeights = torch.ones((unlabeledWeight.shape)).cuda()
        # pixelWiseWeight = unlabeledWeight


        _,pred_src_aux, pred_src_main = model(images_source.cuda())
        _,pred_trg_aux, pred_trg_main = model(images_target.cuda())
        _,pred_mix_aux, pred_mix_main = model(images_classmix.cuda())

        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux     = interp(pred_src_aux)
            pred_trg_aux     = interp(pred_trg_aux)
            pred_mix_aux     = interp(pred_mix_aux)

            loss_seg_src_aux = loss_calc(pred_src_aux,labels_source)
            loss_dice_aux    = dice_loss(pred_src_aux,labels_source.cuda())
        else:
            loss_seg_src_aux = 0
            loss_dice_aux    = 0

        pred_src_main     = interp(pred_src_main)
        pred_trg_main     = interp(pred_trg_main)
        pred_mix_main     = interp(pred_mix_main)
        loss_seg_src_main = loss_calc(pred_src_main,labels_source)
        loss_dice_main    = dice_loss(pred_src_main,labels_source.cuda())

       
        weight = np.size(np.array(labels_classmix.cpu())) / torch.sum(mix_mask)

        loss_mix_aux = weight * unlabeled_loss(pred_mix_aux, labels_classmix, mix_mask)

        loss_mix = weight * unlabeled_loss(pred_mix_main, labels_classmix, mix_mask)  # [10, 321, 321]

        # L_u_aux =  consistency_weight * unlabeled_loss(pred_mix_aux, labels_classmix, mix_mask)
        # L_u_main = consistency_weight * unlabeled_loss(pred_mix_main, labels_classmix, pixelWiseWeight)
        # L_dice_aux =  consistency_weight * dice_loss(pred_mix_aux, labels_classmix)
        # L_dice_main = consistency_weight * dice_loss(pred_mix_main, labels_classmix)

        # consistency_loss = torch.mean((torch.softmax(pred_trg_main, dim=1) - ema_output_main_soft)**2)
        # consistency_aux_loss = torch.mean((torch.softmax(pred_trg_aux, dim=1) - ema_output_aux_soft)**2)

        loss = (cfg.TRAIN.LAMBDA_SEG_SRC_MAIN    * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_SRC_AUX   * loss_seg_src_aux
                + cfg.TRAIN.LAMBDA_DICE_SRC_MAIN * loss_dice_main
                + cfg.TRAIN.LAMBDA_DICE_SRC_AUX  * loss_dice_aux
                + 0.1*loss_mix_aux
                + loss_mix
                # + consistency_weight * consistency_loss 
                # + 0.1 * consistency_weight * consistency_aux_loss
                # + L_u_main
                # + 0.1 * L_u_aux
                # + 0.1 * L_dice_aux
                # + L_dice_main
                )
        
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, 0.99, i_iter)

        current_losses = {'loss_seg_src_aux' :loss_seg_src_aux,
                          'loss_seg_src_main':loss_seg_src_main,
                        #   'loss_target_entp_aux':loss_target_entp_aux,
                        #   'loss_target_entp_main':loss_target_entp_aux,
                          'loss_dice_aux': loss_dice_aux,
                          'loss_dice_main'   :loss_dice_main,
                          'loss_mix' : loss_mix}
        print_losses(current_losses,i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            if cfg.DATASETS == 'Cardiac':
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Cardiac(model, cfg.TARGET, results_file, i_iter)
            else:
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Abdomen(model, cfg.TARGET, results_file, i_iter)
            
            # saved_path = "/home/data_backup/zhr_savedmodel/{0}2{1}_{2}".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
            # os.makedirs(saved_path, exist_ok=True)
            # torch.save(model.state_dict(),  saved_path+f'/model_{i_iter}.pth')
            # tmp_dice = np.mean(dice_mean)
            tmp_dice = np.mean(dice_mean)
            dice_all.append(tmp_dice)
            if(i_iter == cfg.TRAIN.MAX_ITERS):
                dice_all = np.array(dice_all)
                np.save('/home/data_backup/zhr_savedmodel/dice_8'+cfg.SOURCE+'2'+cfg.TARGET+'_'+cfg.TRAIN.DA_METHOD, dice_all)
                        
                        
            # if best_model < tmp_dice:
            #     print('taking snapshot ...')
            #     print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            #     snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            #     torch.save(model.state_dict(),  snapshot_dir / '{0}_{1}2{2}_Best_CutMix_ourdice_model.pth'.format(cfg.DATASETS, cfg.SOURCE, cfg.TARGET))
            #     best_model = tmp_dice

        sys.stdout.flush()


def train_cutmix_ST_new(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg):
    '''
    UDA training
    '''

    results_file = "/home/zhr/ICME/scripts/log/{0}2{1}_{2}1.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)

    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    num_classes       = cfg.NUM_CLASSES


    unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
    # dice_loss = DiceLoss(n_classes=num_classes).cuda()
    #SEGMENTATION
    model.train()
    ema_model.train()

    model.cuda()
    ema_model.cuda()
    cudnn.benchmark = True
    cudnn.enabled   = True



    #optimized
    if cfg.TRAIN.OPTIM_G == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.TRAIN.LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                              lr=cfg.TRAIN.LEARNING_RATE,
                              momentum=cfg.TRAIN.MOMENTUM,
                              weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # interpolate output segmaps
    interp        = nn.Upsample(size=(input_size_source[1],input_size_source[0]),mode='bilinear',
                         align_corners=True)

    # labels for adversarial learning


    strain_loader_iter          = enumerate(strain_loader)
    trgtrain_loader_iter        = enumerate(trgtrain_loader)
    sval_loader_iter            = enumerate(sval_loader)

    transforms = None
    img_mean   = cfg.TRAIN.IMG_MEAN
    best_model =  -1

    for i_iter in tqdm(range(cfg.TRAIN.MAX_ITERS+1)):

        model.train()
        ema_model.train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer,i_iter,cfg)


        #UDA training
        try:
            _, batch = strain_loader_iter.__next__()
        except StopIteration:
            strain_loader_iter = enumerate(strain_loader)
            _, batch = strain_loader_iter.__next__()
        images_source, labels_source,_ = batch

        try:
            _, batch = trgtrain_loader_iter.__next__()
        except StopIteration:
            trgtrain_loader_iter = enumerate(trgtrain_loader)
            _, batch             = trgtrain_loader_iter.__next__()
        images_target, labels_target,_ = batch


        noise = torch.clamp(torch.randn_like(
                images_target) * 0.1, -0.2, 0.2)
        ema_inputs = images_target + noise

        with torch.no_grad():
            _, ema_pred_trg_axu, ema_pred_trg_main = ema_model(ema_inputs.cuda())
            # source_teacher_feas, ema_pred_src_axu, ema_pred_src_main = ema_model(images_source.cuda())
            # ema_output_main_soft_feas = torch.softmax(ema_pred_trg_main, dim=1)
            # max_probs_main_feas, pseudo_label_feas  = torch.max(ema_output_main_soft_feas, dim=1)

            ema_pred_trg_axu     = interp(ema_pred_trg_axu)
            ema_pred_trg_main     = interp(ema_pred_trg_main)

            ema_output_main_soft = torch.softmax(ema_pred_trg_main, dim=1)
            ema_output_aux_soft = torch.softmax(ema_pred_trg_axu, dim=1)
            max_probs_main, pseudo_label  = torch.max(ema_output_main_soft, dim=1)




            
            drop_percent = 80
            percent_unreliable = (100 - drop_percent) * (1 - (i_iter)/ cfg.TRAIN.MAX_ITERS)
            drop_percent = 100 - percent_unreliable
            entropy = -torch.sum(ema_output_main_soft * torch.log(ema_output_main_soft + 1e-10), dim=1)
            thresh = np.percentile(
                entropy.detach().cpu().numpy().flatten(), drop_percent
            )
            tg_mask = entropy.le(thresh).long().cuda()
            sc_mask = torch.ones((tg_mask.shape)).cuda()


            
            # unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
            # unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
            # onesWeights = torch.ones((unlabeledWeight.shape)).cuda()

            MixMask, loss_mask = generate_mask(images_target.cuda())
            mix_mask = tg_mask * MixMask + sc_mask * (1 - MixMask)
            images_classmix = images_target.cuda() * MixMask + images_source.cuda() * (1 - MixMask)
            labels_classmix = pseudo_label.cuda() * MixMask + labels_source.cuda() * (1 - MixMask)



            # MixMask, loss_mask = generate_mask(images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda())
            # ## 源域贴在目标域
            # mixmask_trg = tg_mask[:cfg.TRAIN.BATCH_SIZE//2] * MixMask + sc_mask[:cfg.TRAIN.BATCH_SIZE//2] * (1 - MixMask)
            # images_trg_classmix = images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + images_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)
            # labels_trg_classmix = pseudo_label[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + labels_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)

            # MixMask, loss_mask = generate_mask(images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda())
            # ## 目标域贴在源域
            # mixmask_src = tg_mask[cfg.TRAIN.BATCH_SIZE//2:] * (1 - MixMask) + sc_mask[cfg.TRAIN.BATCH_SIZE//2:] * MixMask
            # images_src_classmix = images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + images_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            # labels_src_classmix = pseudo_label[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + labels_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            
            # mix_mask = torch.cat([mixmask_trg, mixmask_src], dim=0)
            # images_classmix = torch.cat([images_trg_classmix, images_src_classmix], dim=0)
            # labels_classmix = torch.cat([labels_trg_classmix, labels_src_classmix], dim=0)
    
        if cfg.TRAIN.CONSWITCH:
            consistency_weight = get_current_consistency_weight(i_iter, cfg.TRAIN.MAX_ITERS)
        else:
            consistency_weight = 1

        # unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
        # unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
        # onesWeights = torch.ones((unlabeledWeight.shape)).cuda()
        # pixelWiseWeight = unlabeledWeight


        _,pred_src_aux, pred_src_main = model(images_source.cuda())
        _,pred_trg_aux, pred_trg_main = model(images_target.cuda())
        _,pred_mix_aux, pred_mix_main = model(images_classmix.cuda())

        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux     = interp(pred_src_aux)
            pred_trg_aux     = interp(pred_trg_aux)
            pred_mix_aux     = interp(pred_mix_aux)

            loss_seg_src_aux = loss_calc(pred_src_aux,labels_source)
            loss_dice_aux    = dice_loss(pred_src_aux,labels_source.cuda())
        else:
            loss_seg_src_aux = 0
            loss_dice_aux    = 0

        pred_src_main     = interp(pred_src_main)
        pred_trg_main     = interp(pred_trg_main)
        pred_mix_main     = interp(pred_mix_main)
        loss_seg_src_main = loss_calc(pred_src_main,labels_source)
        loss_dice_main    = dice_loss(pred_src_main,labels_source.cuda())

        # drop_percent = 80
        # percent_unreliable = (100 - drop_percent) * (1 - i_iter / cfg.TRAIN.MAX_ITERS+1)
        # drop_percent = 100 - percent_unreliable
        # drop pixels with high entropy
        # prob = torch.softmax(pred_teacher, dim=1)
        # entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

        # thresh = np.percentile(
        #     entropy[target != 255].detach().cpu().numpy().flatten(), percent
        # )
        # thresh_mask = entropy.ge(thresh).bool() * (target != 255).bool()

        # target[thresh_mask] = 255
        weight = np.size(np.array(labels_classmix.cpu())) / torch.sum(mix_mask)
        # print(np.size(np.array(labels_classmix.cpu())))
        # print(torch.sum(mix_mask))
        # print(weight)

        loss_mix_aux = weight * unlabeled_loss(pred_mix_aux, labels_classmix, mix_mask)

        loss_mix = weight * unlabeled_loss(pred_mix_main, labels_classmix, mix_mask)  # [10, 321, 321]

        # L_u_aux =  consistency_weight * unlabeled_loss(pred_mix_aux, labels_classmix, mix_mask)
        # L_u_main = consistency_weight * unlabeled_loss(pred_mix_main, labels_classmix, pixelWiseWeight)
        # L_dice_aux =  consistency_weight * dice_loss(pred_mix_aux, labels_classmix)
        # L_dice_main = consistency_weight * dice_loss(pred_mix_main, labels_classmix)

        # consistency_loss = torch.mean((torch.softmax(pred_trg_main, dim=1) - ema_output_main_soft)**2)
        # consistency_aux_loss = torch.mean((torch.softmax(pred_trg_aux, dim=1) - ema_output_aux_soft)**2)

        loss = (cfg.TRAIN.LAMBDA_SEG_SRC_MAIN    * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_SRC_AUX   * loss_seg_src_aux
                + cfg.TRAIN.LAMBDA_DICE_SRC_MAIN * loss_dice_main
                + cfg.TRAIN.LAMBDA_DICE_SRC_AUX  * loss_dice_aux
                + 0.1*loss_mix_aux
                + loss_mix
                # + consistency_weight * consistency_loss 
                # + 0.1 * consistency_weight * consistency_aux_loss
                # + L_u_main
                # + 0.1 * L_u_aux
                # + 0.1 * L_dice_aux
                # + L_dice_main
                )
        
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, 0.99, i_iter)

        current_losses = {'loss_seg_src_aux' :loss_seg_src_aux,
                          'loss_seg_src_main':loss_seg_src_main,
                        #   'loss_target_entp_aux':loss_target_entp_aux,
                        #   'loss_target_entp_main':loss_target_entp_aux,
                          'loss_dice_aux': loss_dice_aux,
                          'loss_dice_main'   :loss_dice_main,
                          'loss_mix' : loss_mix}
        print_losses(current_losses,i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            if cfg.DATASETS == 'Cardiac':
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Cardiac(model, cfg.TARGET, results_file, i_iter)
            else:
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Abdomen(model, cfg.TARGET, results_file, i_iter)
            
            # saved_path = "/home/data_backup/zhr_savedmodel/{0}2{1}_{2}".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
            # os.makedirs(saved_path, exist_ok=True)
            # torch.save(model.state_dict(),  saved_path+f'/model_{i_iter}.pth')
            # tmp_dice = np.mean(dice_mean)
                        
            # if best_model < tmp_dice:
            #     print('taking snapshot ...')
            #     print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            #     snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            #     torch.save(model.state_dict(),  snapshot_dir / '{0}_{1}2{2}_Best_CutMix_ourdice_model.pth'.format(cfg.DATASETS, cfg.SOURCE, cfg.TARGET))
            #     best_model = tmp_dice

        sys.stdout.flush()



def train_cutmix_ST_gram(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg):
    '''
    UDA training
    '''

    results_file = "/home/zhr/ICME/scripts/log/{0}2{1}_{2}_layer1_teacher1.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)

    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    num_classes       = cfg.NUM_CLASSES


    unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
    # dice_loss = DiceLoss(n_classes=num_classes).cuda()
    styleloss = StyleLoss(weight = 1000).cuda()

    model.train()
    ema_model.train()

    model.cuda()
    ema_model.cuda()
    cudnn.benchmark = True
    cudnn.enabled   = True

    #optimized
    if cfg.TRAIN.OPTIM_G == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.TRAIN.LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                              lr=cfg.TRAIN.LEARNING_RATE,
                              momentum=cfg.TRAIN.MOMENTUM,
                              weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # interpolate output segmaps
    interp        = nn.Upsample(size=(input_size_source[1],input_size_source[0]),mode='bilinear',
                         align_corners=True)

    # labels for adversarial learning


    strain_loader_iter          = enumerate(strain_loader)
    trgtrain_loader_iter        = enumerate(trgtrain_loader)
    sval_loader_iter            = enumerate(sval_loader)

    transforms = None
    img_mean   = cfg.TRAIN.IMG_MEAN
    best_model =  -1

    for i_iter in tqdm(range(cfg.TRAIN.MAX_ITERS+1)):

        model.train()
        ema_model.train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer,i_iter,cfg)


        #UDA training
        try:
            _, batch = strain_loader_iter.__next__()
        except StopIteration:
            strain_loader_iter = enumerate(strain_loader)
            _, batch = strain_loader_iter.__next__()
        images_source, labels_source,_ = batch

        try:
            _, batch = trgtrain_loader_iter.__next__()
        except StopIteration:
            trgtrain_loader_iter = enumerate(trgtrain_loader)
            _, batch             = trgtrain_loader_iter.__next__()
        images_target, labels_target,_ = batch

        # src_in_trg = []
        # for i in range(cfg.TRAIN.BATCH_SIZE):
        #     st = match_histograms(np.array(images_source[i]), np.array(images_target[i]), channel_axis=0)
        #     src_in_trg.append(st)
        # images_source = torch.tensor(src_in_trg, dtype=torch.float32)

        noise = torch.clamp(torch.randn_like(
                images_target) * 0.1, -0.2, 0.2)
        ema_inputs = images_target + noise

        with torch.no_grad():
            _, ema_pred_trg_axu, ema_pred_trg_main = ema_model(ema_inputs.cuda())
            source_teacher_f, ema_pred_src_axu, ema_pred_src_main = ema_model(images_source.cuda())
            ema_pred_trg_axu     = interp(ema_pred_trg_axu)
            ema_pred_trg_main     = interp(ema_pred_trg_main)

            ema_output_main_soft = torch.softmax(ema_pred_trg_main, dim=1)
            ema_output_aux_soft = torch.softmax(ema_pred_trg_axu, dim=1)
            max_probs_main, pseudo_label  = torch.max(ema_output_main_soft, dim=1)
            
            unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
            unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
            onesWeights = torch.ones((unlabeledWeight.shape)).cuda()

            MixMask, loss_mask = generate_mask(images_target.cuda())
            pixelWiseWeight = unlabeledWeight * MixMask + onesWeights * (1 - MixMask)
            images_classmix = images_target.cuda() * MixMask + images_source.cuda() * (1 - MixMask)
            labels_classmix = pseudo_label.cuda() * MixMask + labels_source.cuda() * (1 - MixMask)

            # how2mask = np.random.uniform(0, 1, 1)
            # if how2mask < 2:
            #     MixMask, loss_mask = generate_mask(images_target.cuda())

            #     images_classmix = images_target.cuda() * MixMask + images_source.cuda() * (1 - MixMask)
            #     labels_classmix = pseudo_label.cuda() * MixMask + labels_source.cuda() * (1 - MixMask)
            # else:
            #     MixMask, loss_mask = generate_mask(images_target.cuda())

            #     images_classmix = images_target.cuda() * (1 - MixMask) + images_source.cuda() * MixMask
            #     labels_classmix = pseudo_label.cuda() * (1 - MixMask) + labels_source.cuda() * MixMask


            # MixMask, loss_mask = generate_mask(images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda())

            # pixelWiseWeight_trg = unlabeledWeight[:cfg.TRAIN.BATCH_SIZE//2] * MixMask + onesWeights[:cfg.TRAIN.BATCH_SIZE//2] * (1 - MixMask)
            # images_trg_classmix = images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + images_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)
            # labels_trg_classmix = pseudo_label[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + labels_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)

            # MixMask, loss_mask = generate_mask(images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda())
            
            # pixelWiseWeight_src = unlabeledWeight[cfg.TRAIN.BATCH_SIZE//2:] * (1 - MixMask) + onesWeights[cfg.TRAIN.BATCH_SIZE//2:] * MixMask
            # images_src_classmix = images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + images_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            # labels_src_classmix = pseudo_label[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + labels_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            
            # pixelWiseWeight = torch.cat([pixelWiseWeight_trg, pixelWiseWeight_src], dim=0)
            # images_classmix = torch.cat([images_trg_classmix, images_src_classmix], dim=0)
            # labels_classmix = torch.cat([labels_trg_classmix, labels_src_classmix], dim=0)
    
        if cfg.TRAIN.CONSWITCH:
            consistency_weight = get_current_consistency_weight(i_iter, cfg.TRAIN.MAX_ITERS)
        else:
            consistency_weight = 1

        # unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
        # unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
        # onesWeights = torch.ones((unlabeledWeight.shape)).cuda()
        # pixelWiseWeight = unlabeledWeight


        source_f,pred_src_aux, pred_src_main = model(images_source.cuda())
        target_f,pred_trg_aux, pred_trg_main = model(images_target.cuda())
        mix_f,pred_mix_aux, pred_mix_main = model(images_classmix.cuda())

        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux     = interp(pred_src_aux)
            pred_trg_aux     = interp(pred_trg_aux)
            pred_mix_aux     = interp(pred_mix_aux)

            loss_seg_src_aux = loss_calc(pred_src_aux,labels_source)
            loss_dice_aux    = dice_loss(pred_src_aux,labels_source.cuda())
        else:
            loss_seg_src_aux = 0
            loss_dice_aux    = 0

        pred_src_main     = interp(pred_src_main)
        pred_trg_main     = interp(pred_trg_main)
        pred_mix_main     = interp(pred_mix_main)
        loss_seg_src_main = loss_calc(pred_src_main,labels_source)
        loss_dice_main    = dice_loss(pred_src_main,labels_source.cuda())


        L_u_aux =  consistency_weight * unlabeled_loss(pred_mix_aux, labels_classmix, pixelWiseWeight)
        L_u_main = consistency_weight * unlabeled_loss(pred_mix_main, labels_classmix, pixelWiseWeight)
        # L_dice_aux =  consistency_weight * dice_loss(pred_mix_aux, labels_classmix)
        # L_dice_main = consistency_weight * dice_loss(pred_mix_main, labels_classmix)

        consistency_loss = torch.mean((torch.softmax(pred_trg_main, dim=1) - ema_output_main_soft)**2)
        consistency_aux_loss = torch.mean((torch.softmax(pred_trg_aux, dim=1) - ema_output_aux_soft)**2)
        gram_loss = styleloss(source_teacher_f,target_f,mix_f)

        loss = (cfg.TRAIN.LAMBDA_SEG_SRC_MAIN    * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_SRC_AUX   * loss_seg_src_aux
                + cfg.TRAIN.LAMBDA_DICE_SRC_MAIN * loss_dice_main
                + cfg.TRAIN.LAMBDA_DICE_SRC_AUX  * loss_dice_aux
                + consistency_weight * consistency_loss 
                + 0.1 * consistency_weight * consistency_aux_loss
                + L_u_main
                + 0.1 * L_u_aux
                + gram_loss
                # + 0.1 * L_dice_aux
                # + L_dice_main
                )
        
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, 0.99, i_iter)

        current_losses = {'loss_seg_src_aux' :loss_seg_src_aux,
                          'loss_seg_src_main':loss_seg_src_main,
                        #   'loss_target_entp_aux':loss_target_entp_aux,
                        #   'loss_target_entp_main':loss_target_entp_aux,
                          'loss_dice_aux': loss_dice_aux,
                          'loss_dice_main'   :loss_dice_main,
                          'gram_loss'     :gram_loss}
        print_losses(current_losses,i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            if cfg.DATASETS == 'Cardiac':
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Cardiac(model, cfg.TARGET, results_file, i_iter)
            else:
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Abdomen(model, cfg.TARGET, results_file, i_iter)
            
            # saved_path = "/home/data_backup/zhr_savedmodel/{0}2{1}_{2}_ourdice_myweight".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
            # os.makedirs(saved_path, exist_ok=True)
            # torch.save(model.state_dict(),  saved_path+f'/model_{i_iter}.pth')


            # tmp_dice = np.mean(dice_mean)
                        
            # if best_model < tmp_dice:
            #     print('taking snapshot ...')
            #     print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            #     snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            #     torch.save(model.state_dict(),  snapshot_dir / '{0}_{1}2{2}_Best_CutMix_ourdice_model.pth'.format(cfg.DATASETS, cfg.SOURCE, cfg.TARGET))
            #     best_model = tmp_dice

        sys.stdout.flush()

def train_bcutmix_ST_gram(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg):
    '''
    UDA training
    '''

    results_file = "/home/zhr/ICME/scripts/log/{0}2{1}_{2}_layer1_mix1.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)

    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    num_classes       = cfg.NUM_CLASSES


    unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
    # dice_loss = DiceLoss(n_classes=num_classes).cuda()
    styleloss = StyleLoss(weight = 1000).cuda()
    #SEGMENTATION
    model.train()
    ema_model.train()

    model.cuda()
    ema_model.cuda()
    cudnn.benchmark = True
    cudnn.enabled   = True



    #optimized
    if cfg.TRAIN.OPTIM_G == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.TRAIN.LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                              lr=cfg.TRAIN.LEARNING_RATE,
                              momentum=cfg.TRAIN.MOMENTUM,
                              weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # interpolate output segmaps
    interp        = nn.Upsample(size=(input_size_source[1],input_size_source[0]),mode='bilinear',
                         align_corners=True)

    # labels for adversarial learning


    strain_loader_iter          = enumerate(strain_loader)
    trgtrain_loader_iter        = enumerate(trgtrain_loader)
    sval_loader_iter            = enumerate(sval_loader)

    transforms = None
    img_mean   = cfg.TRAIN.IMG_MEAN
    best_model =  -1

    for i_iter in tqdm(range(cfg.TRAIN.MAX_ITERS+1)):

        model.train()
        ema_model.train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer,i_iter,cfg)


        #UDA training
        try:
            _, batch = strain_loader_iter.__next__()
        except StopIteration:
            strain_loader_iter = enumerate(strain_loader)
            _, batch = strain_loader_iter.__next__()
        images_source, labels_source,_ = batch

        try:
            _, batch = trgtrain_loader_iter.__next__()
        except StopIteration:
            trgtrain_loader_iter = enumerate(trgtrain_loader)
            _, batch             = trgtrain_loader_iter.__next__()
        images_target, labels_target,_ = batch


        noise = torch.clamp(torch.randn_like(
                images_target) * 0.1, -0.2, 0.2)
        ema_inputs = images_target + noise

        with torch.no_grad():
            _, ema_pred_trg_axu, ema_pred_trg_main = ema_model(ema_inputs.cuda())
            source_teacher_f, ema_pred_src_axu, ema_pred_src_main = ema_model(images_source.cuda())
            ema_pred_trg_axu     = interp(ema_pred_trg_axu)
            ema_pred_trg_main     = interp(ema_pred_trg_main)

            ema_output_main_soft = torch.softmax(ema_pred_trg_main, dim=1)
            ema_output_aux_soft = torch.softmax(ema_pred_trg_axu, dim=1)
            max_probs_main, pseudo_label  = torch.max(ema_output_main_soft, dim=1)
            
            unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
            unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
            onesWeights = torch.ones((unlabeledWeight.shape)).cuda()




            MixMask, loss_mask = generate_mask(images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda())
            ## 源域贴在目标域
            pixelWiseWeight_trg = unlabeledWeight[:cfg.TRAIN.BATCH_SIZE//2] * MixMask + onesWeights[:cfg.TRAIN.BATCH_SIZE//2] * (1 - MixMask)
            images_trg_classmix = images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + images_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)
            labels_trg_classmix = pseudo_label[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + labels_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)

            MixMask, loss_mask = generate_mask(images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda())
            ## 目标域贴在源域
            pixelWiseWeight_src = unlabeledWeight[cfg.TRAIN.BATCH_SIZE//2:] * (1 - MixMask) + onesWeights[cfg.TRAIN.BATCH_SIZE//2:] * MixMask
            images_src_classmix = images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + images_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            labels_src_classmix = pseudo_label[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + labels_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            
            pixelWiseWeight = torch.cat([pixelWiseWeight_trg, pixelWiseWeight_src], dim=0)
            images_classmix = torch.cat([images_trg_classmix, images_src_classmix], dim=0)
            labels_classmix = torch.cat([labels_trg_classmix, labels_src_classmix], dim=0)
    
        if cfg.TRAIN.CONSWITCH:
            consistency_weight = get_current_consistency_weight(i_iter, cfg.TRAIN.MAX_ITERS)
        else:
            consistency_weight = 1

        # unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
        # unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
        # onesWeights = torch.ones((unlabeledWeight.shape)).cuda()
        # pixelWiseWeight = unlabeledWeight


        source_f,pred_src_aux, pred_src_main = model(images_source.cuda())
        target_f,pred_trg_aux, pred_trg_main = model(images_target.cuda())
        mix_f,pred_mix_aux, pred_mix_main = model(images_classmix.cuda())

        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux     = interp(pred_src_aux)
            pred_trg_aux     = interp(pred_trg_aux)
            pred_mix_aux     = interp(pred_mix_aux)

            loss_seg_src_aux = loss_calc(pred_src_aux,labels_source)
            loss_dice_aux    = dice_loss(pred_src_aux,labels_source.cuda())
        else:
            loss_seg_src_aux = 0
            loss_dice_aux    = 0

        pred_src_main     = interp(pred_src_main)
        pred_trg_main     = interp(pred_trg_main)
        pred_mix_main     = interp(pred_mix_main)
        loss_seg_src_main = loss_calc(pred_src_main,labels_source)
        loss_dice_main    = dice_loss(pred_src_main,labels_source.cuda())


        L_u_aux =  consistency_weight * unlabeled_loss(pred_mix_aux, labels_classmix, pixelWiseWeight)
        L_u_main = consistency_weight * unlabeled_loss(pred_mix_main, labels_classmix, pixelWiseWeight)
        # L_dice_aux =  consistency_weight * dice_loss(pred_mix_aux, labels_classmix)
        # L_dice_main = consistency_weight * dice_loss(pred_mix_main, labels_classmix)

        consistency_loss = torch.mean((torch.softmax(pred_trg_main, dim=1) - ema_output_main_soft)**2)
        consistency_aux_loss = torch.mean((torch.softmax(pred_trg_aux, dim=1) - ema_output_aux_soft)**2)
        gram_loss = styleloss(source_f,target_f,mix_f)

        loss = (cfg.TRAIN.LAMBDA_SEG_SRC_MAIN    * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_SRC_AUX   * loss_seg_src_aux
                + cfg.TRAIN.LAMBDA_DICE_SRC_MAIN * loss_dice_main
                + cfg.TRAIN.LAMBDA_DICE_SRC_AUX  * loss_dice_aux
                + consistency_weight * consistency_loss 
                + 0.1 * consistency_weight * consistency_aux_loss
                + L_u_main
                + 0.1 * L_u_aux
                + gram_loss
                # + 0.1 * L_dice_aux
                # + L_dice_main
                )
        
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, 0.99, i_iter)

        current_losses = {'loss_seg_src_aux' :loss_seg_src_aux,
                          'loss_seg_src_main':loss_seg_src_main,
                        #   'loss_target_entp_aux':loss_target_entp_aux,
                        #   'loss_target_entp_main':loss_target_entp_aux,
                          'loss_dice_aux': loss_dice_aux,
                          'loss_dice_main'   :loss_dice_main,
                          'gram_loss'    :gram_loss}
        print_losses(current_losses,i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            if cfg.DATASETS == 'Cardiac':
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Cardiac(model, cfg.TARGET, results_file, i_iter)
            else:
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Abdomen(model, cfg.TARGET, results_file, i_iter)
            
            # saved_path = "/home/data_backup/zhr_savedmodel/{0}2{1}_{2}_ourdice_myweight".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
            # os.makedirs(saved_path, exist_ok=True)
            # torch.save(model.state_dict(),  saved_path+f'/model_{i_iter}.pth')
            # tmp_dice = np.mean(dice_mean)
                        
            # if best_model < tmp_dice:
            #     print('taking snapshot ...')
            #     print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            #     snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            #     torch.save(model.state_dict(),  snapshot_dir / '{0}_{1}2{2}_Best_CutMix_ourdice_model.pth'.format(cfg.DATASETS, cfg.SOURCE, cfg.TARGET))
            #     best_model = tmp_dice

        sys.stdout.flush()



def train_cutmix_ST_MPSCL(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg):
    '''
    UDA training
    '''

    results_file = "/home/zhr/ICME/scripts/log/{0}2{1}_{2}_iter13000.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)

    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    num_classes       = cfg.NUM_CLASSES


    unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
    # dice_loss = DiceLoss(n_classes=num_classes).cuda()

    model.train()
    ema_model.train()

    model.cuda()
    ema_model.cuda()
    cudnn.benchmark = True
    cudnn.enabled   = True

    #optimized
    if cfg.TRAIN.OPTIM_G == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.TRAIN.LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                              lr=cfg.TRAIN.LEARNING_RATE,
                              momentum=cfg.TRAIN.MOMENTUM,
                              weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        
    class_center_feas = np.load(cfg.TRAIN.CLASS_CENTER_FEA_INIT).squeeze()
    class_center_feas = torch.from_numpy(class_center_feas).float().cuda()

    # interpolate output segmaps
    interp        = nn.Upsample(size=(input_size_source[1],input_size_source[0]),mode='bilinear',
                         align_corners=True)

    # labels for adversarial learning


    strain_loader_iter          = enumerate(strain_loader)
    trgtrain_loader_iter        = enumerate(trgtrain_loader)
    sval_loader_iter            = enumerate(sval_loader)

    transforms = None
    img_mean   = cfg.TRAIN.IMG_MEAN
    best_model =  -1

    mpcl_loss_src = MPCL(num_class=num_classes, temperature=1.0,
                                       base_temperature=1.0, m=0.4)

    mpcl_loss_trg = MPCL(num_class=num_classes, temperature=1.0,
                                       base_temperature=1.0, m=0.2)

    for i_iter in tqdm(range(13000, cfg.TRAIN.MAX_ITERS+1)):

        model.train()
        ema_model.train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer,i_iter,cfg)


        #UDA training
        try:
            _, batch = strain_loader_iter.__next__()
        except StopIteration:
            strain_loader_iter = enumerate(strain_loader)
            _, batch = strain_loader_iter.__next__()
        images_source, labels_source,_ = batch

        try:
            _, batch = trgtrain_loader_iter.__next__()
        except StopIteration:
            trgtrain_loader_iter = enumerate(trgtrain_loader)
            _, batch             = trgtrain_loader_iter.__next__()
        images_target, labels_target,_ = batch


        noise = torch.clamp(torch.randn_like(
                images_target) * 0.1, -0.2, 0.2)
        ema_inputs = images_target + noise

        with torch.no_grad():
            cla_feas_trg, ema_pred_trg_axu, ema_pred_trg_main = ema_model(ema_inputs.cuda())
            source_teacher_feas, ema_pred_src_axu, ema_pred_src_main = ema_model(images_source.cuda())
            ema_pred_trg_axu     = interp(ema_pred_trg_axu)
            ema_pred_trg_main     = interp(ema_pred_trg_main)

            ema_output_main_soft = torch.softmax(ema_pred_trg_main, dim=1)
            ema_output_aux_soft = torch.softmax(ema_pred_trg_axu, dim=1)
            max_probs_main, pseudo_label  = torch.max(ema_output_main_soft, dim=1)
            
            unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
            unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
            onesWeights = torch.ones((unlabeledWeight.shape)).cuda()

            MixMask, loss_mask = generate_mask(images_target.cuda())
            pixelWiseWeight = unlabeledWeight * MixMask + onesWeights * (1 - MixMask)
            images_classmix = images_target.cuda() * MixMask + images_source.cuda() * (1 - MixMask)
            labels_classmix = pseudo_label.cuda() * MixMask + labels_source.cuda() * (1 - MixMask)

    
        if cfg.TRAIN.CONSWITCH:
            consistency_weight = get_current_consistency_weight(i_iter, cfg.TRAIN.MAX_ITERS)
        else:
            consistency_weight = 1


        source_feas,pred_src_aux, pred_src_main = model(images_source.cuda())
        target_feas,pred_trg_aux, pred_trg_main = model(images_target.cuda())
        _,pred_mix_aux, pred_mix_main = model(images_classmix.cuda())

        class_center_feas = update_class_center_iter(source_feas, labels_source, class_center_feas,m=0.2)
        hard_pixel_label,pixel_mask = generate_pseudo_label(target_feas, class_center_feas, cfg)

        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux     = interp(pred_src_aux)
            pred_trg_aux     = interp(pred_trg_aux)
            pred_mix_aux     = interp(pred_mix_aux)

            loss_seg_src_aux = loss_calc(pred_src_aux,labels_source)
            loss_dice_aux    = dice_loss(pred_src_aux,labels_source.cuda())
        else:
            loss_seg_src_aux = 0
            loss_dice_aux    = 0

        pred_src_main     = interp(pred_src_main)
        pred_trg_main     = interp(pred_trg_main)
        pred_mix_main     = interp(pred_mix_main)
        loss_seg_src_main = loss_calc(pred_src_main,labels_source)
        loss_dice_main    = dice_loss(pred_src_main,labels_source.cuda())



        mpcl_loss_tr           = mpcl_loss_calc(feas=source_feas,labels=labels_source,
                                                            class_center_feas=class_center_feas,
                                                            loss_func=mpcl_loss_src,tag='source')

        mpcl_loss_tg = mpcl_loss_calc(feas=target_feas, labels=hard_pixel_label,
                                                 class_center_feas=class_center_feas,
                                                 loss_func=mpcl_loss_trg,
                                                 pixel_sel_loc=pixel_mask, tag='target')



        L_u_aux =  consistency_weight * unlabeled_loss(pred_mix_aux, labels_classmix, pixelWiseWeight)
        L_u_main = consistency_weight * unlabeled_loss(pred_mix_main, labels_classmix, pixelWiseWeight)
        # L_dice_aux =  consistency_weight * dice_loss(pred_mix_aux, labels_classmix)
        # L_dice_main = consistency_weight * dice_loss(pred_mix_main, labels_classmix)

        consistency_loss = torch.mean((torch.softmax(pred_trg_main, dim=1) - ema_output_main_soft)**2)
        consistency_aux_loss = torch.mean((torch.softmax(pred_trg_aux, dim=1) - ema_output_aux_soft)**2)

        loss = (cfg.TRAIN.LAMBDA_SEG_SRC_MAIN    * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_SRC_AUX   * loss_seg_src_aux
                + cfg.TRAIN.LAMBDA_DICE_SRC_MAIN * loss_dice_main
                + cfg.TRAIN.LAMBDA_DICE_SRC_AUX  * loss_dice_aux
                + consistency_weight * consistency_loss 
                + 0.1 * consistency_weight * consistency_aux_loss
                + L_u_main
                + 0.1 * L_u_aux
                + 1*mpcl_loss_tr
                + 0.1*mpcl_loss_tg)
                # + 0.1 * L_dice_aux
                # + L_dice_main
                
        
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, 0.99, i_iter)

        current_losses = {'loss_seg_src_aux' :loss_seg_src_aux,
                          'loss_seg_src_main':loss_seg_src_main,
                        #   'loss_target_entp_aux':loss_target_entp_aux,
                        #   'loss_target_entp_main':loss_target_entp_aux,
                          'loss_dice_aux': loss_dice_aux,
                          'loss_dice_main'   :loss_dice_main,
                           'mpcl_loss_tr'  :1*mpcl_loss_tr,
                            'mpcl_loss_tg'  :0.1*mpcl_loss_tg}
        print_losses(current_losses,i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            if cfg.DATASETS == 'Cardiac':
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Cardiac(model, cfg.TARGET, results_file, i_iter)
            else:
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Abdomen(model, cfg.TARGET, results_file, i_iter)
            
            # saved_path = "/home/data_backup/zhr_savedmodel/{0}2{1}_{2}_ourdice_myweight".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
            # os.makedirs(saved_path, exist_ok=True)
            # torch.save(model.state_dict(),  saved_path+f'/model_{i_iter}.pth')
            # tmp_dice = np.mean(dice_mean)
                        
            # if best_model < tmp_dice:
            #     print('taking snapshot ...')
            #     print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            #     snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            #     torch.save(model.state_dict(),  snapshot_dir / '{0}_{1}2{2}_Best_CutMix_ourdice_model.pth'.format(cfg.DATASETS, cfg.SOURCE, cfg.TARGET))
            #     best_model = tmp_dice

        sys.stdout.flush()


def train_bcutmix_ST_MPSCL(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg):
    '''
    UDA training
    '''

    results_file = "/home/zhr/ICME/scripts/log/{0}2{1}_{2}_iter100005.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)

    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    num_classes       = cfg.NUM_CLASSES


    unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
    # dice_loss = DiceLoss(n_classes=num_classes).cuda()

    model.train()
    ema_model.train()

    model.cuda()
    ema_model.cuda()
    cudnn.benchmark = True
    cudnn.enabled   = True

    #optimized
    if cfg.TRAIN.OPTIM_G == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.TRAIN.LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                              lr=cfg.TRAIN.LEARNING_RATE,
                              momentum=cfg.TRAIN.MOMENTUM,
                              weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        
    class_center_feas = np.load(cfg.TRAIN.CLASS_CENTER_FEA_INIT).squeeze()
    class_center_feas = torch.from_numpy(class_center_feas).float().cuda()

    # interpolate output segmaps
    interp        = nn.Upsample(size=(input_size_source[1],input_size_source[0]),mode='bilinear',
                         align_corners=True)

    # labels for adversarial learning


    strain_loader_iter          = enumerate(strain_loader)
    trgtrain_loader_iter        = enumerate(trgtrain_loader)
    sval_loader_iter            = enumerate(sval_loader)

    transforms = None
    img_mean   = cfg.TRAIN.IMG_MEAN
    best_model =  -1

    mpcl_loss_src = MPCL(num_class=num_classes, temperature=1.0,
                                       base_temperature=1.0, m=0.4)

    mpcl_loss_trg = MPCL(num_class=num_classes, temperature=1.0,
                                       base_temperature=1.0, m=0.2)

    for i_iter in tqdm(range(10001, cfg.TRAIN.MAX_ITERS+1)):

        model.train()
        ema_model.train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer,i_iter,cfg)


        #UDA training
        try:
            _, batch = strain_loader_iter.__next__()
        except StopIteration:
            strain_loader_iter = enumerate(strain_loader)
            _, batch = strain_loader_iter.__next__()
        images_source, labels_source,_ = batch

        try:
            _, batch = trgtrain_loader_iter.__next__()
        except StopIteration:
            trgtrain_loader_iter = enumerate(trgtrain_loader)
            _, batch             = trgtrain_loader_iter.__next__()
        images_target, labels_target,_ = batch


        noise = torch.clamp(torch.randn_like(
                images_target) * 0.1, -0.2, 0.2)
        ema_inputs = images_target + noise

        with torch.no_grad():
            cla_feas_trg, ema_pred_trg_axu, ema_pred_trg_main = ema_model(ema_inputs.cuda())
            source_teacher_feas, ema_pred_src_axu, ema_pred_src_main = ema_model(images_source.cuda())
            ema_pred_trg_axu     = interp(ema_pred_trg_axu)
            ema_pred_trg_main     = interp(ema_pred_trg_main)

            ema_output_main_soft = torch.softmax(ema_pred_trg_main, dim=1)
            ema_output_aux_soft = torch.softmax(ema_pred_trg_axu, dim=1)
            max_probs_main, pseudo_label  = torch.max(ema_output_main_soft, dim=1)
            
            unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
            unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
            onesWeights = torch.ones((unlabeledWeight.shape)).cuda()

            MixMask, loss_mask = generate_mask(images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda())
            ## 源域贴在目标域
            pixelWiseWeight_trg = unlabeledWeight[:cfg.TRAIN.BATCH_SIZE//2] * MixMask + onesWeights[:cfg.TRAIN.BATCH_SIZE//2] * (1 - MixMask)
            images_trg_classmix = images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + images_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)
            labels_trg_classmix = pseudo_label[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + labels_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)

            MixMask, loss_mask = generate_mask(images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda())
            ## 目标域贴在源域
            pixelWiseWeight_src = unlabeledWeight[cfg.TRAIN.BATCH_SIZE//2:] * (1 - MixMask) + onesWeights[cfg.TRAIN.BATCH_SIZE//2:] * MixMask
            images_src_classmix = images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + images_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            labels_src_classmix = pseudo_label[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + labels_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            
            pixelWiseWeight = torch.cat([pixelWiseWeight_trg, pixelWiseWeight_src], dim=0)
            images_classmix = torch.cat([images_trg_classmix, images_src_classmix], dim=0)
            labels_classmix = torch.cat([labels_trg_classmix, labels_src_classmix], dim=0)
    
    
        if cfg.TRAIN.CONSWITCH:
            consistency_weight = get_current_consistency_weight(i_iter, cfg.TRAIN.MAX_ITERS)
        else:
            consistency_weight = 1


        source_feas,pred_src_aux, pred_src_main = model(images_source.cuda())
        target_feas,pred_trg_aux, pred_trg_main = model(images_target.cuda())
        _,pred_mix_aux, pred_mix_main = model(images_classmix.cuda())

        class_center_feas = update_class_center_iter(source_feas, labels_source, class_center_feas,m=0.2)
        hard_pixel_label,pixel_mask = generate_pseudo_label(target_feas, class_center_feas, cfg)
        print(pixel_mask.sum())

        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux     = interp(pred_src_aux)
            pred_trg_aux     = interp(pred_trg_aux)
            pred_mix_aux     = interp(pred_mix_aux)

            loss_seg_src_aux = loss_calc(pred_src_aux,labels_source)
            loss_dice_aux    = dice_loss(pred_src_aux,labels_source.cuda())
        else:
            loss_seg_src_aux = 0
            loss_dice_aux    = 0

        pred_src_main     = interp(pred_src_main)
        pred_trg_main     = interp(pred_trg_main)
        pred_mix_main     = interp(pred_mix_main)
        loss_seg_src_main = loss_calc(pred_src_main,labels_source)
        loss_dice_main    = dice_loss(pred_src_main,labels_source.cuda())



        mpcl_loss_tr           = mpcl_loss_calc(feas=source_feas,labels=labels_source,
                                                            class_center_feas=class_center_feas,
                                                            loss_func=mpcl_loss_src,tag='source')

        mpcl_loss_tg = mpcl_loss_calc(feas=target_feas, labels=hard_pixel_label,
                                                 class_center_feas=class_center_feas,
                                                 loss_func=mpcl_loss_trg,
                                                 pixel_sel_loc=pixel_mask, tag='target')



        L_u_aux =  consistency_weight * unlabeled_loss(pred_mix_aux, labels_classmix, pixelWiseWeight)
        L_u_main = consistency_weight * unlabeled_loss(pred_mix_main, labels_classmix, pixelWiseWeight)
        # L_dice_aux =  consistency_weight * dice_loss(pred_mix_aux, labels_classmix)
        # L_dice_main = consistency_weight * dice_loss(pred_mix_main, labels_classmix)

        consistency_loss = torch.mean((torch.softmax(pred_trg_main, dim=1) - ema_output_main_soft)**2)
        consistency_aux_loss = torch.mean((torch.softmax(pred_trg_aux, dim=1) - ema_output_aux_soft)**2)

        loss = (cfg.TRAIN.LAMBDA_SEG_SRC_MAIN    * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_SRC_AUX   * loss_seg_src_aux
                + cfg.TRAIN.LAMBDA_DICE_SRC_MAIN * loss_dice_main
                + cfg.TRAIN.LAMBDA_DICE_SRC_AUX  * loss_dice_aux
                + consistency_weight * consistency_loss 
                + 0.1 * consistency_weight * consistency_aux_loss
                + L_u_main
                + 0.1 * L_u_aux
                + 1*mpcl_loss_tr
                + 0.1*mpcl_loss_tg)
                # + 0.1 * L_dice_aux
                # + L_dice_main
                
        
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, 0.99, i_iter)

        current_losses = {'loss_seg_src_aux' :loss_seg_src_aux,
                          'loss_seg_src_main':loss_seg_src_main,
                        #   'loss_target_entp_aux':loss_target_entp_aux,
                        #   'loss_target_entp_main':loss_target_entp_aux,
                          'loss_dice_aux': loss_dice_aux,
                          'loss_dice_main'   :loss_dice_main,
                           'mpcl_loss_tr'  :1*mpcl_loss_tr,
                            'mpcl_loss_tg'  :0.1*mpcl_loss_tg}
        print_losses(current_losses,i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            if cfg.DATASETS == 'Cardiac':
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Cardiac(model, cfg.TARGET, results_file, i_iter)
            else:
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Abdomen(model, cfg.TARGET, results_file, i_iter)
            
            # saved_path = "/home/data_backup/zhr_savedmodel/{0}2{1}_{2}_ourdice_myweight".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
            # os.makedirs(saved_path, exist_ok=True)
            # torch.save(model.state_dict(),  saved_path+f'/model_{i_iter}.pth')
            # tmp_dice = np.mean(dice_mean)
                        
            # if best_model < tmp_dice:
            #     print('taking snapshot ...')
            #     print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            #     snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            #     torch.save(model.state_dict(),  snapshot_dir / '{0}_{1}2{2}_Best_CutMix_ourdice_model.pth'.format(cfg.DATASETS, cfg.SOURCE, cfg.TARGET))
            #     best_model = tmp_dice

        sys.stdout.flush()


def train_bcutmix_ST_MPSCL_ourpseudolabel(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg):
    '''
    UDA training
    '''

    results_file = "/home/zhr/ICME/scripts/log/{0}2{1}_{2}_iter10000_entropy2.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)

    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    num_classes       = cfg.NUM_CLASSES


    unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
    # dice_loss = DiceLoss(n_classes=num_classes).cuda()

    model.train()
    ema_model.train()

    model.cuda()
    ema_model.cuda()
    cudnn.benchmark = True
    cudnn.enabled   = True

    #optimized
    if cfg.TRAIN.OPTIM_G == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.TRAIN.LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                              lr=cfg.TRAIN.LEARNING_RATE,
                              momentum=cfg.TRAIN.MOMENTUM,
                              weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        
    class_center_feas = np.load(cfg.TRAIN.CLASS_CENTER_FEA_INIT).squeeze()
    class_center_feas = torch.from_numpy(class_center_feas).float().cuda()

    # interpolate output segmaps
    interp        = nn.Upsample(size=(input_size_source[1],input_size_source[0]),mode='bilinear',
                         align_corners=True)

    # labels for adversarial learning


    strain_loader_iter          = enumerate(strain_loader)
    trgtrain_loader_iter        = enumerate(trgtrain_loader)
    sval_loader_iter            = enumerate(sval_loader)

    transforms = None
    img_mean   = cfg.TRAIN.IMG_MEAN
    best_model =  -1

    mpcl_loss_src = MPCL(num_class=num_classes, temperature=1.0,
                                       base_temperature=1.0, m=0.4)

    mpcl_loss_trg = MPCL(num_class=num_classes, temperature=1.0,
                                       base_temperature=1.0, m=0.2)

    for i_iter in tqdm(range(10001, cfg.TRAIN.MAX_ITERS+1)):

        model.train()
        ema_model.train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer,i_iter,cfg)


        #UDA training
        try:
            _, batch = strain_loader_iter.__next__()
        except StopIteration:
            strain_loader_iter = enumerate(strain_loader)
            _, batch = strain_loader_iter.__next__()
        images_source, labels_source,_ = batch

        try:
            _, batch = trgtrain_loader_iter.__next__()
        except StopIteration:
            trgtrain_loader_iter = enumerate(trgtrain_loader)
            _, batch             = trgtrain_loader_iter.__next__()
        images_target, labels_target,_ = batch


        noise = torch.clamp(torch.randn_like(
                images_target) * 0.1, -0.2, 0.2)
        ema_inputs = images_target + noise







        with torch.no_grad():
            cla_feas_trg, ema_pred_trg_axu, ema_pred_trg_main = ema_model(ema_inputs.cuda())
            source_teacher_feas, ema_pred_src_axu, ema_pred_src_main = ema_model(images_source.cuda())

            ema_output_main_soft_feas = torch.softmax(ema_pred_trg_main, dim=1)
            max_probs_main_feas, pseudo_label_feas  = torch.max(ema_output_main_soft_feas, dim=1)
            # tg_mask = max_probs_main_feas.ge(cfg.TRAIN.THRESHOLD).long() 
            # print(tg_mask.sum())
            # print(tg_mask.shape)
            # print(pseudo_label_feas.shape)



            drop_percent = 80
            percent_unreliable = (100 - drop_percent) * (1 - (i_iter -10000)/ cfg.TRAIN.MAX_ITERS)
            drop_percent = 100 - percent_unreliable
            entropy = -torch.sum(ema_output_main_soft_feas * torch.log(ema_output_main_soft_feas + 1e-10), dim=1)
            thresh = np.percentile(
                entropy.detach().cpu().numpy().flatten(), drop_percent
            )
            tg_mask = entropy.le(thresh).long() 
            # print(tg_mask.sum())
            # print(tg_mask.shape)


            class_center_feas = update_class_center_iter(source_teacher_feas, labels_source, class_center_feas,m=0.2)

            
            ema_pred_trg_axu     = interp(ema_pred_trg_axu)
            ema_pred_trg_main     = interp(ema_pred_trg_main)

            ema_output_main_soft = torch.softmax(ema_pred_trg_main, dim=1)
            ema_output_aux_soft = torch.softmax(ema_pred_trg_axu, dim=1)
            max_probs_main, pseudo_label  = torch.max(ema_output_main_soft, dim=1)
            
            unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
            unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
            onesWeights = torch.ones((unlabeledWeight.shape)).cuda()

            MixMask, loss_mask = generate_mask(images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda())
            ## 源域贴在目标域
            pixelWiseWeight_trg = unlabeledWeight[:cfg.TRAIN.BATCH_SIZE//2] * MixMask + onesWeights[:cfg.TRAIN.BATCH_SIZE//2] * (1 - MixMask)
            images_trg_classmix = images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + images_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)
            labels_trg_classmix = pseudo_label[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + labels_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)

            MixMask, loss_mask = generate_mask(images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda())
            ## 目标域贴在源域
            pixelWiseWeight_src = unlabeledWeight[cfg.TRAIN.BATCH_SIZE//2:] * (1 - MixMask) + onesWeights[cfg.TRAIN.BATCH_SIZE//2:] * MixMask
            images_src_classmix = images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + images_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            labels_src_classmix = pseudo_label[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + labels_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            
            pixelWiseWeight = torch.cat([pixelWiseWeight_trg, pixelWiseWeight_src], dim=0)
            images_classmix = torch.cat([images_trg_classmix, images_src_classmix], dim=0)
            labels_classmix = torch.cat([labels_trg_classmix, labels_src_classmix], dim=0)
    
    
        if cfg.TRAIN.CONSWITCH:
            consistency_weight = get_current_consistency_weight(i_iter, cfg.TRAIN.MAX_ITERS)
        else:
            consistency_weight = 1


        source_feas,pred_src_aux, pred_src_main = model(images_source.cuda())
        target_feas,pred_trg_aux, pred_trg_main = model(images_target.cuda())
        _,pred_mix_aux, pred_mix_main = model(images_classmix.cuda())

        # class_center_feas = update_class_center_iter(source_feas, labels_source, class_center_feas,m=0.2)
        # hard_pixel_label,pixel_mask = generate_pseudo_label(target_feas, class_center_feas, cfg)
        # print(pixel_mask.sum())

        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux     = interp(pred_src_aux)
            pred_trg_aux     = interp(pred_trg_aux)
            pred_mix_aux     = interp(pred_mix_aux)

            loss_seg_src_aux = loss_calc(pred_src_aux,labels_source)
            loss_dice_aux    = dice_loss(pred_src_aux,labels_source.cuda())
        else:
            loss_seg_src_aux = 0
            loss_dice_aux    = 0

        pred_src_main     = interp(pred_src_main)
        pred_trg_main     = interp(pred_trg_main)
        pred_mix_main     = interp(pred_mix_main)
        loss_seg_src_main = loss_calc(pred_src_main,labels_source)
        loss_dice_main    = dice_loss(pred_src_main,labels_source.cuda())



        mpcl_loss_tr           = mpcl_loss_calc(feas=source_feas,labels=labels_source,
                                                            class_center_feas=class_center_feas,
                                                            loss_func=mpcl_loss_src,tag='source')

        mpcl_loss_tg = mpcl_loss_calc(feas=target_feas, labels=pseudo_label_feas,
                                                 class_center_feas=class_center_feas,
                                                 loss_func=mpcl_loss_trg,
                                                 pixel_sel_loc=tg_mask, tag='target')



        L_u_aux =  consistency_weight * unlabeled_loss(pred_mix_aux, labels_classmix, pixelWiseWeight)
        L_u_main = consistency_weight * unlabeled_loss(pred_mix_main, labels_classmix, pixelWiseWeight)
        # L_dice_aux =  consistency_weight * dice_loss(pred_mix_aux, labels_classmix)
        # L_dice_main = consistency_weight * dice_loss(pred_mix_main, labels_classmix)

        consistency_loss = torch.mean((torch.softmax(pred_trg_main, dim=1) - ema_output_main_soft)**2)
        consistency_aux_loss = torch.mean((torch.softmax(pred_trg_aux, dim=1) - ema_output_aux_soft)**2)

        loss = (cfg.TRAIN.LAMBDA_SEG_SRC_MAIN    * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_SRC_AUX   * loss_seg_src_aux
                + cfg.TRAIN.LAMBDA_DICE_SRC_MAIN * loss_dice_main
                + cfg.TRAIN.LAMBDA_DICE_SRC_AUX  * loss_dice_aux
                + consistency_weight * consistency_loss 
                + 0.1 * consistency_weight * consistency_aux_loss
                + L_u_main
                + 0.1 * L_u_aux
                + 1*mpcl_loss_tr
                + 0.1*mpcl_loss_tg)
                # + 0.1 * L_dice_aux
                # + L_dice_main
                
        
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, 0.99, i_iter)

        current_losses = {'loss_seg_src_aux' :loss_seg_src_aux,
                          'loss_seg_src_main':loss_seg_src_main,
                        #   'loss_target_entp_aux':loss_target_entp_aux,
                        #   'loss_target_entp_main':loss_target_entp_aux,
                          'loss_dice_aux': loss_dice_aux,
                          'loss_dice_main'   :loss_dice_main,
                           'mpcl_loss_tr'  :1*mpcl_loss_tr,
                            'mpcl_loss_tg'  :0.1*mpcl_loss_tg}
        print_losses(current_losses,i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            if cfg.DATASETS == 'Cardiac':
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Cardiac(model, cfg.TARGET, results_file, i_iter)
            else:
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Abdomen(model, cfg.TARGET, results_file, i_iter)
            
            # saved_path = "/home/data_backup/zhr_savedmodel/{0}2{1}_{2}_ourdice_myweight".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
            # os.makedirs(saved_path, exist_ok=True)
            # torch.save(model.state_dict(),  saved_path+f'/model_{i_iter}.pth')
            # tmp_dice = np.mean(dice_mean)
                        
            # if best_model < tmp_dice:
            #     print('taking snapshot ...')
            #     print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            #     snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            #     torch.save(model.state_dict(),  snapshot_dir / '{0}_{1}2{2}_Best_CutMix_ourdice_model.pth'.format(cfg.DATASETS, cfg.SOURCE, cfg.TARGET))
            #     best_model = tmp_dice

        sys.stdout.flush()

def train_bcutmix_ST_MPSCL_ourpseudolabel_new(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg):
    '''
    UDA training
    '''

    results_file = "/home/zhr/ICME/scripts/log/{0}2{1}_{2}_smcontra.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)

    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    num_classes       = cfg.NUM_CLASSES


    unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
    # dice_loss = DiceLoss(n_classes=num_classes).cuda()

    model.train()
    ema_model.train()

    model.cuda()
    ema_model.cuda()
    cudnn.benchmark = True
    cudnn.enabled   = True

    #optimized
    if cfg.TRAIN.OPTIM_G == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.TRAIN.LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                              lr=cfg.TRAIN.LEARNING_RATE,
                              momentum=cfg.TRAIN.MOMENTUM,
                              weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        
    class_center_feas = np.load(cfg.TRAIN.CLASS_CENTER_FEA_INIT).squeeze()
    class_center_feas = torch.from_numpy(class_center_feas).float().cuda()

    # interpolate output segmaps
    interp        = nn.Upsample(size=(input_size_source[1],input_size_source[0]),mode='bilinear',
                         align_corners=True)

    # labels for adversarial learning


    strain_loader_iter          = enumerate(strain_loader)
    trgtrain_loader_iter        = enumerate(trgtrain_loader)
    sval_loader_iter            = enumerate(sval_loader)

    transforms = None
    img_mean   = cfg.TRAIN.IMG_MEAN
    best_model =  -1

    mpcl_loss_src = MPCL(num_class=num_classes, temperature=1.0,
                                       base_temperature=1.0, m=0.4)

    mpcl_loss_trg = MPCL(num_class=num_classes, temperature=1.0,
                                       base_temperature=1.0, m=0.2)

    for i_iter in tqdm(range(6001, cfg.TRAIN.MAX_ITERS+1)):

        model.train()
        ema_model.train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer,i_iter,cfg)


        #UDA training
        try:
            _, batch = strain_loader_iter.__next__()
        except StopIteration:
            strain_loader_iter = enumerate(strain_loader)
            _, batch = strain_loader_iter.__next__()
        images_source, labels_source,_ = batch

        try:
            _, batch = trgtrain_loader_iter.__next__()
        except StopIteration:
            trgtrain_loader_iter = enumerate(trgtrain_loader)
            _, batch             = trgtrain_loader_iter.__next__()
        images_target, labels_target,_ = batch


        noise = torch.clamp(torch.randn_like(
                images_target) * 0.1, -0.2, 0.2)
        ema_inputs = images_target + noise







        with torch.no_grad():
            cla_feas_trg, ema_pred_trg_axu, ema_pred_trg_main = ema_model(ema_inputs.cuda())
            source_teacher_feas, ema_pred_src_axu, ema_pred_src_main = ema_model(images_source.cuda())



            


            class_center_feas = update_class_center_iter(source_teacher_feas, labels_source, class_center_feas,m=0.2)

            
            ema_pred_trg_axu     = interp(ema_pred_trg_axu)
            ema_pred_trg_main     = interp(ema_pred_trg_main)

            ema_output_main_soft = torch.softmax(ema_pred_trg_main, dim=1)
            ema_output_aux_soft = torch.softmax(ema_pred_trg_axu, dim=1)
            max_probs_main, pseudo_label  = torch.max(ema_output_main_soft, dim=1)
            
            
            
            drop_percent = 80
            percent_unreliable = (100 - drop_percent) * (1 - (i_iter)/ cfg.TRAIN.MAX_ITERS)
            drop_percent = 100 - percent_unreliable
            entropy = -torch.sum(ema_output_main_soft * torch.log(ema_output_main_soft + 1e-10), dim=1)
            thresh = np.percentile(
                entropy.detach().cpu().numpy().flatten(), drop_percent
            )
            tg_mask = entropy.le(thresh).long().cuda()
            sc_mask = torch.ones((tg_mask.shape)).cuda()


            
            # unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
            # unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
            # onesWeights = torch.ones((unlabeledWeight.shape)).cuda()




            MixMask, loss_mask = generate_mask(images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda())
            ## 源域贴在目标域
            mixmask_trg = tg_mask[:cfg.TRAIN.BATCH_SIZE//2] * MixMask + sc_mask[:cfg.TRAIN.BATCH_SIZE//2] * (1 - MixMask)
            images_trg_classmix = images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + images_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)
            labels_trg_classmix = pseudo_label[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + labels_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)

            MixMask, loss_mask = generate_mask(images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda())
            ## 目标域贴在源域
            mixmask_src = tg_mask[cfg.TRAIN.BATCH_SIZE//2:] * (1 - MixMask) + sc_mask[cfg.TRAIN.BATCH_SIZE//2:] * MixMask
            images_src_classmix = images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + images_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            labels_src_classmix = pseudo_label[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + labels_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            
            mix_mask = torch.cat([mixmask_trg, mixmask_src], dim=0)
            images_classmix = torch.cat([images_trg_classmix, images_src_classmix], dim=0)
            labels_classmix = torch.cat([labels_trg_classmix, labels_src_classmix], dim=0)
    
    
        if cfg.TRAIN.CONSWITCH:
            consistency_weight = get_current_consistency_weight(i_iter, cfg.TRAIN.MAX_ITERS)
        else:
            consistency_weight = 1


        source_feas,pred_src_aux, pred_src_main = model(images_source.cuda())
        target_feas,pred_trg_aux, pred_trg_main = model(images_target.cuda())
        mix_feas,pred_mix_aux, pred_mix_main = model(images_classmix.cuda())

        # class_center_feas = update_class_center_iter(source_feas, labels_source, class_center_feas,m=0.2)
        # hard_pixel_label,pixel_mask = generate_pseudo_label(target_feas, class_center_feas, cfg)
        # print(pixel_mask.sum())

        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux     = interp(pred_src_aux)
            # pred_trg_aux     = interp(pred_trg_aux)
            pred_mix_aux     = interp(pred_mix_aux)

            loss_seg_src_aux = loss_calc(pred_src_aux,labels_source)
            loss_dice_aux    = dice_loss(pred_src_aux,labels_source.cuda())
        else:
            loss_seg_src_aux = 0
            loss_dice_aux    = 0

        pred_src_main     = interp(pred_src_main)
        # pred_trg_main     = interp(pred_trg_main)
        pred_mix_main     = interp(pred_mix_main)
        loss_seg_src_main = loss_calc(pred_src_main,labels_source)
        loss_dice_main    = dice_loss(pred_src_main,labels_source.cuda())



        mpcl_loss_tr           = mpcl_loss_calc(feas=source_feas,labels=labels_source,
                                                            class_center_feas=class_center_feas,
                                                            loss_func=mpcl_loss_src,tag='source')
        
        # print(mix_feas.shape[1])
        # print(mix_feas.shape[2])



        # pseudo_label_small = label_downsample(pseudo_label,mix_feas.shape[2],mix_feas.shape[3])
        # tg_mask_small = label_downsample(tg_mask,mix_feas.shape[2],mix_feas.shape[3])                                          
        # mpcl_loss_tg = mpcl_loss_calc(feas=target_feas, labels=pseudo_label_small,
        #                                          class_center_feas=class_center_feas,
        #                                          loss_func=mpcl_loss_trg,
        #                                          pixel_sel_loc=tg_mask_small, tag='target')


        labels_classmix_small = label_downsample(labels_classmix,mix_feas.shape[2],mix_feas.shape[3])
        mix_mask_small = label_downsample(mix_mask,mix_feas.shape[2],mix_feas.shape[3])
        mpcl_loss_mix = mpcl_loss_calc(feas=mix_feas, labels=labels_classmix_small,
                                                 class_center_feas=class_center_feas,
                                                 loss_func=mpcl_loss_trg,
                                                 pixel_sel_loc=mix_mask_small, tag='target')
        # print(pseudo_label_small.shape)
        # print(tg_mask_small.shape)
        # print(labels_classmix_small.shape)
        # print(mix_mask_small.shape)
        
       
  

        weight = np.size(np.array(labels_classmix.cpu())) / torch.sum(mix_mask)

        loss_mix_aux = weight * unlabeled_loss(pred_mix_aux, labels_classmix, mix_mask)

        loss_mix = weight * unlabeled_loss(pred_mix_main, labels_classmix, mix_mask)  # [10, 321, 321]





        loss = (cfg.TRAIN.LAMBDA_SEG_SRC_MAIN    * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_SRC_AUX   * loss_seg_src_aux
                + cfg.TRAIN.LAMBDA_DICE_SRC_MAIN * loss_dice_main
                + cfg.TRAIN.LAMBDA_DICE_SRC_AUX  * loss_dice_aux
                + 0.1*loss_mix_aux
                + loss_mix
                # + consistency_weight * consistency_loss 
                # + 0.1 * consistency_weight * consistency_aux_loss
                # + L_u_main
                # + 0.1 * L_u_aux
                + 1*mpcl_loss_tr
                + 0.1*mpcl_loss_mix)
                # + 0.1 * L_dice_aux
                # + L_dice_main
                
        
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, 0.99, i_iter)

        current_losses = {'loss_seg_src_aux' :loss_seg_src_aux,
                          'loss_seg_src_main':loss_seg_src_main,
                        #   'loss_target_entp_aux':loss_target_entp_aux,
                        #   'loss_target_entp_main':loss_target_entp_aux,
                          'loss_dice_aux': loss_dice_aux,
                          'loss_dice_main'   :loss_dice_main,
                           'mpcl_loss_tr'  :1*mpcl_loss_tr,
                            'mpcl_loss_mix'  :0.1*mpcl_loss_mix}
        print_losses(current_losses,i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            if cfg.DATASETS == 'Cardiac':
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Cardiac(model, cfg.TARGET, results_file, i_iter)
            else:
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Abdomen(model, cfg.TARGET, results_file, i_iter)
            
            # saved_path = "/home/data_backup/zhr_savedmodel/{0}2{1}_{2}_ourdice_myweight".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
            # os.makedirs(saved_path, exist_ok=True)
            # torch.save(model.state_dict(),  saved_path+f'/model_{i_iter}.pth')
            # tmp_dice = np.mean(dice_mean)
                        
            # if best_model < tmp_dice:
            #     print('taking snapshot ...')
            #     print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            #     snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            #     torch.save(model.state_dict(),  snapshot_dir / '{0}_{1}2{2}_Best_CutMix_ourdice_model.pth'.format(cfg.DATASETS, cfg.SOURCE, cfg.TARGET))
            #     best_model = tmp_dice

        sys.stdout.flush()



def train_bcutmix_ST_MPSCL_ourpseudolabel_final(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg, strain_loader_):
    '''
    UDA training
    '''

    results_file = "/home/zhr/ICME/scripts/log_final/{0}2{1}_{2}_6000_scontra.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)

    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    num_classes       = cfg.NUM_CLASSES


    unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
    # dice_loss = DiceLoss(n_classes=num_classes).cuda()

    model.train()
    ema_model.train()

    model.cuda()
    ema_model.cuda()
    cudnn.benchmark = True
    cudnn.enabled   = True

    #optimized
    if cfg.TRAIN.OPTIM_G == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.TRAIN.LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                              lr=cfg.TRAIN.LEARNING_RATE,
                              momentum=cfg.TRAIN.MOMENTUM,
                              weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        
    # class_center_feas = np.load(cfg.TRAIN.CLASS_CENTER_FEA_INIT).squeeze()
    # class_center_feas = torch.from_numpy(class_center_feas).float().cuda()

    # interpolate output segmaps
    interp        = nn.Upsample(size=(input_size_source[1],input_size_source[0]),mode='bilinear',
                         align_corners=True)

    # labels for adversarial learning


    strain_loader_iter          = enumerate(strain_loader)
    trgtrain_loader_iter        = enumerate(trgtrain_loader)
    sval_loader_iter            = enumerate(sval_loader)

    transforms = None
    img_mean   = cfg.TRAIN.IMG_MEAN
    best_model =  -1

    mpcl_loss_src = MPCL(num_class=num_classes, temperature=1.0,
                                       base_temperature=1.0, m=0.4)

    mpcl_loss_trg = MPCL(num_class=num_classes, temperature=1.0,
                                       base_temperature=1.0, m=0.2)
    contra_iter = 6000

    for i_iter in tqdm(range(cfg.TRAIN.MAX_ITERS+1)):

        model.train()
        ema_model.train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer,i_iter,cfg)


        #UDA training
        try:
            _, batch = strain_loader_iter.__next__()
        except StopIteration:
            strain_loader_iter = enumerate(strain_loader)
            _, batch = strain_loader_iter.__next__()
        images_source, labels_source,_ = batch

        try:
            _, batch = trgtrain_loader_iter.__next__()
        except StopIteration:
            trgtrain_loader_iter = enumerate(trgtrain_loader)
            _, batch             = trgtrain_loader_iter.__next__()
        images_target, labels_target,_ = batch


        noise = torch.clamp(torch.randn_like(
                images_target) * 0.1, -0.2, 0.2)
        ema_inputs = images_target + noise



        with torch.no_grad():
            cla_feas_trg, ema_pred_trg_axu, ema_pred_trg_main = ema_model(ema_inputs.cuda())
            source_teacher_feas, ema_pred_src_axu, ema_pred_src_main = ema_model(images_source.cuda())


            if(i_iter > (contra_iter+1)):
                class_center_feas = update_class_center_iter(source_teacher_feas, labels_source, class_center_feas,m=0.2)

            
            ema_pred_trg_main     = interp(ema_pred_trg_main)
            ema_output_main_soft = torch.softmax(ema_pred_trg_main, dim=1)
            max_probs_main, pseudo_label  = torch.max(ema_output_main_soft, dim=1)
            
            
            drop_percent = 80
            percent_unreliable = (100 - drop_percent) * (1 - (i_iter)/ cfg.TRAIN.MAX_ITERS)
            drop_percent = 100 - percent_unreliable
            entropy = -torch.sum(ema_output_main_soft * torch.log(ema_output_main_soft + 1e-10), dim=1)
            thresh = np.percentile(
                entropy.detach().cpu().numpy().flatten(), drop_percent
            )
            tg_mask = entropy.le(thresh).long().cuda()
            sc_mask = torch.ones((tg_mask.shape)).cuda()



            MixMask, loss_mask = generate_mask(images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda())
            ## 源域贴在目标域
            mixmask_trg = tg_mask[:cfg.TRAIN.BATCH_SIZE//2] * MixMask + sc_mask[:cfg.TRAIN.BATCH_SIZE//2] * (1 - MixMask)
            images_trg_classmix = images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + images_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)
            labels_trg_classmix = pseudo_label[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + labels_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)

            MixMask, loss_mask = generate_mask(images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda())
            ## 目标域贴在源域
            mixmask_src = tg_mask[cfg.TRAIN.BATCH_SIZE//2:] * (1 - MixMask) + sc_mask[cfg.TRAIN.BATCH_SIZE//2:] * MixMask
            images_src_classmix = images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + images_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            labels_src_classmix = pseudo_label[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + labels_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            
            mix_mask = torch.cat([mixmask_trg, mixmask_src], dim=0)
            images_classmix = torch.cat([images_trg_classmix, images_src_classmix], dim=0)
            labels_classmix = torch.cat([labels_trg_classmix, labels_src_classmix], dim=0)
    
    
        # if cfg.TRAIN.CONSWITCH:
        #     consistency_weight = get_current_consistency_weight(i_iter, cfg.TRAIN.MAX_ITERS)
        # else:
        #     consistency_weight = 1


        source_feas,pred_src_aux, pred_src_main = model(images_source.cuda())
        target_feas,pred_trg_aux, pred_trg_main = model(images_target.cuda())
        mix_feas,pred_mix_aux, pred_mix_main = model(images_classmix.cuda())


        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux     = interp(pred_src_aux)
            pred_mix_aux     = interp(pred_mix_aux)

            loss_seg_src_aux = loss_calc(pred_src_aux,labels_source)
            loss_dice_aux    = dice_loss(pred_src_aux,labels_source.cuda())
        else:
            loss_seg_src_aux = 0
            loss_dice_aux    = 0

        pred_src_main     = interp(pred_src_main)
        pred_mix_main     = interp(pred_mix_main)
        loss_seg_src_main = loss_calc(pred_src_main,labels_source)
        loss_dice_main    = dice_loss(pred_src_main,labels_source.cuda())



        weight = np.size(np.array(labels_classmix.cpu())) / torch.sum(mix_mask)

        loss_mix_aux = weight * unlabeled_loss(pred_mix_aux, labels_classmix, mix_mask)
        loss_mix = weight * unlabeled_loss(pred_mix_main, labels_classmix, mix_mask)  

        if(i_iter < (contra_iter+1)):
            contra_loss = 0
        else:
            if(i_iter == (contra_iter+1)):
                class_center_feas = category_center(ema_model, strain_loader_,source_feas.shape[1])
                class_center_feas = class_center_feas.float().cuda()
                ema_model.train()
            
            contra_loss_tr           = mpcl_loss_calc(feas=source_feas,labels=labels_source,
                                                                class_center_feas=class_center_feas,
                                                                loss_func=mpcl_loss_src,tag='source')
            

            # pseudo_label_small = label_downsample(pseudo_label,mix_feas.shape[2],mix_feas.shape[3])
            # tg_mask_small = label_downsample(tg_mask,mix_feas.shape[2],mix_feas.shape[3])                                          
            # contra_loss_tg = mpcl_loss_calc(feas=target_feas, labels=pseudo_label_small,
            #                                          class_center_feas=class_center_feas,
            #                                          loss_func=mpcl_loss_trg,
            #                                          pixel_sel_loc=tg_mask_small, tag='target')


            # labels_classmix_small = label_downsample(labels_classmix,mix_feas.shape[2],mix_feas.shape[3])
            # mix_mask_small = label_downsample(mix_mask,mix_feas.shape[2],mix_feas.shape[3])
            # contra_loss_mix = mpcl_loss_calc(feas=mix_feas, labels=labels_classmix_small,
            #                                         class_center_feas=class_center_feas,
            #                                         loss_func=mpcl_loss_trg,
            #                                         pixel_sel_loc=mix_mask_small, tag='target')

            # contra_loss = contra_loss_tr + 0.1*contra_loss_mix
            contra_loss = contra_loss_tr 
       



        loss = (cfg.TRAIN.LAMBDA_SEG_SRC_MAIN    * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_SRC_AUX   * loss_seg_src_aux
                + cfg.TRAIN.LAMBDA_DICE_SRC_MAIN * loss_dice_main
                + cfg.TRAIN.LAMBDA_DICE_SRC_AUX  * loss_dice_aux
                + 0.1*loss_mix_aux
                + loss_mix
                + contra_loss)
                
        
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, 0.99, i_iter)

        current_losses = {'loss_seg_src_aux' :loss_seg_src_aux,
                          'loss_seg_src_main':loss_seg_src_main,
                        #   'loss_target_entp_aux':loss_target_entp_aux,
                        #   'loss_target_entp_main':loss_target_entp_aux,
                          'loss_dice_aux': loss_dice_aux,
                          'loss_dice_main'   :loss_dice_main,
                          'loss_mix'  :loss_mix,
                           'contra_loss'  :contra_loss}
        print_losses(current_losses,i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            if cfg.DATASETS == 'Cardiac':
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Cardiac(model, cfg.TARGET, results_file, i_iter)
            else:
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Abdomen(model, cfg.TARGET, results_file, i_iter)
            
            # saved_path = "/home/data_backup/zhr_savedmodel/{0}2{1}_{2}_ourdice_myweight".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
            # os.makedirs(saved_path, exist_ok=True)
            # torch.save(model.state_dict(),  saved_path+f'/model_{i_iter}.pth')
            # tmp_dice = np.mean(dice_mean)
                        
            # if best_model < tmp_dice:
            #     print('taking snapshot ...')
            #     print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            #     snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            #     torch.save(model.state_dict(),  snapshot_dir / '{0}_{1}2{2}_Best_CutMix_ourdice_model.pth'.format(cfg.DATASETS, cfg.SOURCE, cfg.TARGET))
            #     best_model = tmp_dice

        sys.stdout.flush()


def train_bcutmix_ST_contrastive_final(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg, strain_loader_):
    '''
    UDA training
    '''

    results_file = "/home/zhr/ICME/scripts/log_final/{0}2{1}_{2}_50000_consistencyweight1.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)

    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    num_classes       = cfg.NUM_CLASSES


    unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
    # dice_loss = DiceLoss(n_classes=num_classes).cuda()

    model.train()
    ema_model.train()

    model.cuda()
    ema_model.cuda()
    cudnn.benchmark = True
    cudnn.enabled   = True

    #optimized
    if cfg.TRAIN.OPTIM_G == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.TRAIN.LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                              lr=cfg.TRAIN.LEARNING_RATE,
                              momentum=cfg.TRAIN.MOMENTUM,
                              weight_decay=cfg.TRAIN.WEIGHT_DECAY)

        
    # class_center_feas = np.load(cfg.TRAIN.CLASS_CENTER_FEA_INIT).squeeze()
    # class_center_feas = torch.from_numpy(class_center_feas).float().cuda()

    # interpolate output segmaps
    interp        = nn.Upsample(size=(input_size_source[1],input_size_source[0]),mode='bilinear',
                         align_corners=True)

    # labels for adversarial learning


    strain_loader_iter          = enumerate(strain_loader)
    trgtrain_loader_iter        = enumerate(trgtrain_loader)
    sval_loader_iter            = enumerate(sval_loader)

    transforms = None
    img_mean   = cfg.TRAIN.IMG_MEAN
    best_model =  -1

    # mpcl_loss_src = MPCL(num_class=num_classes, temperature=1.0,
    #                                    base_temperature=1.0, m=0.4)

    # mpcl_loss_trg = MPCL(num_class=num_classes, temperature=1.0,
    #                                    base_temperature=1.0, m=0.2)
    contra_iter = 50000


    memobank = [[] for i in range(num_classes)]
    queue_len = 256
    temp = 1

    for i_iter in tqdm(range(cfg.TRAIN.MAX_ITERS+1)):

        model.train()
        ema_model.train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer,i_iter,cfg)


        #UDA training
        try:
            _, batch = strain_loader_iter.__next__()
        except StopIteration:
            strain_loader_iter = enumerate(strain_loader)
            _, batch = strain_loader_iter.__next__()
        images_source, labels_source,_ = batch

        try:
            _, batch = trgtrain_loader_iter.__next__()
        except StopIteration:
            trgtrain_loader_iter = enumerate(trgtrain_loader)
            _, batch             = trgtrain_loader_iter.__next__()
        images_target, labels_target,_ = batch


        noise = torch.clamp(torch.randn_like(
                images_target) * 0.1, -0.2, 0.2)
        ema_inputs = images_target + noise



        with torch.no_grad():
            cla_feas_trg, ema_pred_trg_axu, ema_pred_trg_main = ema_model(ema_inputs.cuda())
            source_teacher_feas, ema_pred_src_axu, ema_pred_src_main = ema_model(images_source.cuda())


            if(i_iter > (contra_iter+1)):
                class_center_feas = update_class_center_iter(source_teacher_feas, labels_source, class_center_feas,m=0.2)
                
            
            ema_pred_trg_main     = interp(ema_pred_trg_main)
            ema_output_main_soft = torch.softmax(ema_pred_trg_main, dim=1)
            max_probs_main, pseudo_label  = torch.max(ema_output_main_soft, dim=1)
            
            
            drop_percent = 80
            percent_unreliable = (100 - drop_percent) * (1 - (i_iter)/ cfg.TRAIN.MAX_ITERS)
            drop_percent = 100 - percent_unreliable
            entropy = -torch.sum(ema_output_main_soft * torch.log(ema_output_main_soft + 1e-10), dim=1)
            thresh = np.percentile(
                entropy.detach().cpu().numpy().flatten(), drop_percent
            )
            tg_mask = entropy.le(thresh).long().cuda()
            sc_mask = torch.ones((tg_mask.shape)).cuda()



            MixMask, loss_mask = generate_mask(images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda())
            ## 源域贴在目标域
            mixmask_trg = tg_mask[:cfg.TRAIN.BATCH_SIZE//2] * MixMask + sc_mask[:cfg.TRAIN.BATCH_SIZE//2] * (1 - MixMask)
            images_trg_classmix = images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + images_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)
            labels_trg_classmix = pseudo_label[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + labels_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)

            MixMask, loss_mask = generate_mask(images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda())
            ## 目标域贴在源域
            mixmask_src = tg_mask[cfg.TRAIN.BATCH_SIZE//2:] * (1 - MixMask) + sc_mask[cfg.TRAIN.BATCH_SIZE//2:] * MixMask
            images_src_classmix = images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + images_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            labels_src_classmix = pseudo_label[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + labels_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            
            mix_mask = torch.cat([mixmask_trg, mixmask_src], dim=0)
            images_classmix = torch.cat([images_trg_classmix, images_src_classmix], dim=0)
            labels_classmix = torch.cat([labels_trg_classmix, labels_src_classmix], dim=0)
    
    
        if cfg.TRAIN.CONSWITCH:
            consistency_weight = get_current_consistency_weight(i_iter, cfg.TRAIN.MAX_ITERS)
        else:
            consistency_weight = 1


        source_feas,pred_src_aux, pred_src_main = model(images_source.cuda())
        target_feas,pred_trg_aux, pred_trg_main = model(images_target.cuda())
        mix_feas,pred_mix_aux, pred_mix_main = model(images_classmix.cuda())


        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux     = interp(pred_src_aux)
            pred_mix_aux     = interp(pred_mix_aux)

            loss_seg_src_aux = loss_calc(pred_src_aux,labels_source)
            loss_dice_aux    = dice_loss(pred_src_aux,labels_source.cuda())
        else:
            loss_seg_src_aux = 0
            loss_dice_aux    = 0

        pred_src_main     = interp(pred_src_main)
        pred_mix_main     = interp(pred_mix_main)
        loss_seg_src_main = loss_calc(pred_src_main,labels_source)
        loss_dice_main    = dice_loss(pred_src_main,labels_source.cuda())



        # weight = np.size(np.array(labels_classmix.cpu())) / torch.sum(mix_mask)

        # loss_mix_aux = weight * unlabeled_loss(pred_mix_aux, labels_classmix, mix_mask)
        # loss_mix = weight * unlabeled_loss(pred_mix_main, labels_classmix, mix_mask)  

        loss_mix_aux = consistency_weight * unlabeled_loss(pred_mix_aux, labels_classmix, mix_mask)
        loss_mix = consistency_weight * unlabeled_loss(pred_mix_main, labels_classmix, mix_mask)  

        if(i_iter < (contra_iter+1)):
            contra_loss = 0
        else:
            if(i_iter == (contra_iter+1)):
                class_center_feas = category_center(ema_model, strain_loader_,source_feas.shape[1])
                class_center_feas = class_center_feas.float().cuda()
                ema_model.train()
            memobank = get_negative(source_teacher_feas, labels_source, num_classes, memobank, queue_len, class_center_feas)
            regional_contrastive_s = regional_contrastive_cos(source_feas, labels_source, class_center_feas, memobank, num_classes, temp)
            # tg_mask_small = label_downsample(tg_mask,mix_feas.shape[2],mix_feas.shape[3]) 
            # regional_contrastive_r = regional_contrastive_cos(target_feas, pseudo_label, class_center_feas, memobank, num_classes, temp, tg_mask_small)
            # mix_mask_small = label_downsample(mix_mask,mix_feas.shape[2],mix_feas.shape[3])
            # regional_contrastive_m = regional_contrastive_cos(mix_feas, labels_classmix, class_center_feas, memobank, num_classes, temp, mix_mask_small)
            # contra_loss_tr           = mpcl_loss_calc(feas=source_feas,labels=labels_source,
            #                                                     class_center_feas=class_center_feas,
            #                                                     loss_func=mpcl_loss_src,tag='source')
            

            # pseudo_label_small = label_downsample(pseudo_label,mix_feas.shape[2],mix_feas.shape[3])
            # tg_mask_small = label_downsample(tg_mask,mix_feas.shape[2],mix_feas.shape[3])                                          
            # contra_loss_tg = mpcl_loss_calc(feas=target_feas, labels=pseudo_label_small,
            #                                          class_center_feas=class_center_feas,
            #                                          loss_func=mpcl_loss_trg,
            #                                          pixel_sel_loc=tg_mask_small, tag='target')


            # labels_classmix_small = label_downsample(labels_classmix,mix_feas.shape[2],mix_feas.shape[3])
            # mix_mask_small = label_downsample(mix_mask,mix_feas.shape[2],mix_feas.shape[3])
            # contra_loss_mix = mpcl_loss_calc(feas=mix_feas, labels=labels_classmix_small,
            #                                         class_center_feas=class_center_feas,
            #                                         loss_func=mpcl_loss_trg,
            #                                         pixel_sel_loc=mix_mask_small, tag='target')

            # contra_loss = contra_loss_tr + 0.1*contra_loss_mix
            contra_loss = 0.1*regional_contrastive_s 
       



        loss = (cfg.TRAIN.LAMBDA_SEG_SRC_MAIN    * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_SRC_AUX   * loss_seg_src_aux
                + cfg.TRAIN.LAMBDA_DICE_SRC_MAIN * loss_dice_main
                + cfg.TRAIN.LAMBDA_DICE_SRC_AUX  * loss_dice_aux
                + 0.1*loss_mix_aux
                + loss_mix
                + contra_loss)
                
        
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, 0.99, i_iter)

        current_losses = {'loss_seg_src_aux' :loss_seg_src_aux,
                          'loss_seg_src_main':loss_seg_src_main,
                        #   'loss_target_entp_aux':loss_target_entp_aux,
                        #   'loss_target_entp_main':loss_target_entp_aux,
                          'loss_dice_aux': loss_dice_aux,
                          'loss_dice_main'   :loss_dice_main,
                          'loss_mix'  :loss_mix,
                           'contra_loss'  :contra_loss}
        print_losses(current_losses,i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            if cfg.DATASETS == 'Cardiac':
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Cardiac(model, cfg.TARGET, results_file, i_iter)
            else:
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Abdomen(model, cfg.TARGET, results_file, i_iter)
            
            # saved_path = "/home/data_backup/zhr_savedmodel/{0}2{1}_{2}_ourdice_myweight".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
            # os.makedirs(saved_path, exist_ok=True)
            # torch.save(model.state_dict(),  saved_path+f'/model_{i_iter}.pth')
            # tmp_dice = np.mean(dice_mean)
                        
            # if best_model < tmp_dice:
            #     print('taking snapshot ...')
            #     print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            #     snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            #     torch.save(model.state_dict(),  snapshot_dir / '{0}_{1}2{2}_Best_CutMix_ourdice_model.pth'.format(cfg.DATASETS, cfg.SOURCE, cfg.TARGET))
            #     best_model = tmp_dice

        sys.stdout.flush()



def train_bcutmix_ST_contrastive_boundary(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg, strain_loader_):
    '''
    UDA training
    '''

    results_file = "/home/zhr/ICME/scripts/log_final/{0}2{1}_{2}_6000_scontra_temp1_dim256_droppercent90.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
    loss_file = "/home/zhr/ICME/scripts/loss_final/{0}2{1}_{2}_6000_scontra_temp1_dim256_droppercent90.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    num_classes       = cfg.NUM_CLASSES


    unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
    # dice_loss = DiceLoss(n_classes=num_classes).cuda()

    model.train()
    ema_model.train()

    model.cuda()
    ema_model.cuda()
    cudnn.benchmark = True
    cudnn.enabled   = True

    #optimized
    if cfg.TRAIN.OPTIM_G == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.TRAIN.LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                              lr=cfg.TRAIN.LEARNING_RATE,
                              momentum=cfg.TRAIN.MOMENTUM,
                              weight_decay=cfg.TRAIN.WEIGHT_DECAY)

        
    # class_center_feas = np.load(cfg.TRAIN.CLASS_CENTER_FEA_INIT).squeeze()
    # class_center_feas = torch.from_numpy(class_center_feas).float().cuda()

    # interpolate output segmaps
    interp        = nn.Upsample(size=(input_size_source[1],input_size_source[0]),mode='bilinear',
                         align_corners=True)

    # labels for adversarial learning


    strain_loader_iter          = enumerate(strain_loader)
    trgtrain_loader_iter        = enumerate(trgtrain_loader)
    sval_loader_iter            = enumerate(sval_loader)

    transforms = None
    img_mean   = cfg.TRAIN.IMG_MEAN
    best_model =  -1

    # mpcl_loss_src = MPCL(num_class=num_classes, temperature=1.0,
    #                                    base_temperature=1.0, m=0.4)

    # mpcl_loss_trg = MPCL(num_class=num_classes, temperature=1.0,
    #                                    base_temperature=1.0, m=0.2)
    contra_iter = 6000


    memobank = [[] for i in range(num_classes)]
    queue_len = 256
    temp = 1

    for i_iter in tqdm(range(cfg.TRAIN.MAX_ITERS+1)):

        model.train()
        ema_model.train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer,i_iter,cfg)


        #UDA training
        try:
            _, batch = strain_loader_iter.__next__()
        except StopIteration:
            strain_loader_iter = enumerate(strain_loader)
            _, batch = strain_loader_iter.__next__()
        images_source, labels_source,_ = batch

        try:
            _, batch = trgtrain_loader_iter.__next__()
        except StopIteration:
            trgtrain_loader_iter = enumerate(trgtrain_loader)
            _, batch             = trgtrain_loader_iter.__next__()
        images_target, labels_target,_ = batch


        noise = torch.clamp(torch.randn_like(
                images_target) * 0.1, -0.2, 0.2)
        ema_inputs = images_target + noise



        with torch.no_grad():
            cla_feas_trg, ema_pred_trg_axu, ema_pred_trg_main = ema_model(ema_inputs.cuda())
            source_teacher_feas, ema_pred_src_axu, ema_pred_src_main = ema_model(images_source.cuda())


            if(i_iter > (contra_iter+1)):
                class_center_feas = update_class_center_iter(source_teacher_feas, labels_source, class_center_feas,m=0.2)
                
            
            ema_pred_trg_main     = interp(ema_pred_trg_main)
            ema_output_main_soft = torch.softmax(ema_pred_trg_main, dim=1)
            max_probs_main, pseudo_label  = torch.max(ema_output_main_soft, dim=1)
            
            
            drop_percent = 90
            percent_unreliable = (100 - drop_percent) * (1 - (i_iter)/ cfg.TRAIN.MAX_ITERS)
            drop_percent = 100 - percent_unreliable
            entropy = -torch.sum(ema_output_main_soft * torch.log(ema_output_main_soft + 1e-10), dim=1)
            thresh = np.percentile(
                entropy.detach().cpu().numpy().flatten(), drop_percent
            )
            thresh_veryreliable = np.percentile(
                entropy.detach().cpu().numpy().flatten(), percent_unreliable
            ) 
            tg_mask = entropy.le(thresh).long().cuda()
            tg_veryreliablemask = entropy.le(thresh_veryreliable).long().cuda()
            sc_mask = torch.ones((tg_mask.shape)).cuda()



            MixMask, loss_mask = generate_mask(images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda())
            ## 源域贴在目标域
            mixmask_trg = tg_mask[:cfg.TRAIN.BATCH_SIZE//2] * MixMask + sc_mask[:cfg.TRAIN.BATCH_SIZE//2] * (1 - MixMask)
            mixveryreliablemask_trg = tg_veryreliablemask[:cfg.TRAIN.BATCH_SIZE//2] * MixMask + sc_mask[:cfg.TRAIN.BATCH_SIZE//2] * (1 - MixMask)
            images_trg_classmix = images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + images_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)
            labels_trg_classmix = pseudo_label[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + labels_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)

            MixMask, loss_mask = generate_mask(images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda())
            ## 目标域贴在源域
            mixmask_src = tg_mask[cfg.TRAIN.BATCH_SIZE//2:] * (1 - MixMask) + sc_mask[cfg.TRAIN.BATCH_SIZE//2:] * MixMask
            mixveryreliablemask_src = tg_veryreliablemask[cfg.TRAIN.BATCH_SIZE//2:] * (1 - MixMask) + sc_mask[cfg.TRAIN.BATCH_SIZE//2:] * MixMask
            images_src_classmix = images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + images_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            labels_src_classmix = pseudo_label[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + labels_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            
            mix_mask = torch.cat([mixmask_trg, mixmask_src], dim=0)
            mix_veryreliablemask = torch.cat([mixveryreliablemask_trg, mixveryreliablemask_src], dim=0)
            images_classmix = torch.cat([images_trg_classmix, images_src_classmix], dim=0)
            labels_classmix = torch.cat([labels_trg_classmix, labels_src_classmix], dim=0)
    
    
        # if cfg.TRAIN.CONSWITCH:
        #     consistency_weight = get_current_consistency_weight(i_iter, cfg.TRAIN.MAX_ITERS)
        # else:
        #     consistency_weight = 1


        source_feas,pred_src_aux, pred_src_main = model(images_source.cuda())
        target_feas,pred_trg_aux, pred_trg_main = model(images_target.cuda())
        mix_feas,pred_mix_aux, pred_mix_main = model(images_classmix.cuda())


        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux     = interp(pred_src_aux)
            pred_mix_aux     = interp(pred_mix_aux)

            loss_seg_src_aux = loss_calc(pred_src_aux,labels_source)
            loss_dice_aux    = dice_loss(pred_src_aux,labels_source.cuda())
        else:
            loss_seg_src_aux = 0
            loss_dice_aux    = 0

        pred_src_main     = interp(pred_src_main)
        pred_mix_main     = interp(pred_mix_main)
        loss_seg_src_main = loss_calc(pred_src_main,labels_source)
        loss_dice_main    = dice_loss(pred_src_main,labels_source.cuda())



        weight = np.size(np.array(labels_classmix.cpu())) / torch.sum(mix_mask)

        loss_mix_aux = weight * unlabeled_loss(pred_mix_aux, labels_classmix, mix_mask)
        loss_mix = weight * unlabeled_loss(pred_mix_main, labels_classmix, mix_mask)  

        if(i_iter < (contra_iter+1)):
            contra_loss = 0
        else:
            if(i_iter == (contra_iter+1)):
                class_center_feas = category_center(ema_model, strain_loader_,source_feas.shape[1])
                class_center_feas = class_center_feas.float().cuda()
                ema_model.train()
            memobank = get_boundary_negative(source_teacher_feas, labels_source, num_classes, memobank, queue_len, class_center_feas,3)
            regional_contrastive_s = regional_contrastive_cos(source_feas, labels_source, class_center_feas, memobank, num_classes, temp)
            # tg_mask_small = label_downsample(tg_mask,mix_feas.shape[2],mix_feas.shape[3]) 
            # regional_contrastive_r = regional_contrastive_cos(target_feas, pseudo_label, class_center_feas, memobank, num_classes, temp, tg_mask_small)
            # mix_mask_small = label_downsample(mix_mask,mix_feas.shape[2],mix_feas.shape[3])
            # mix_veryreliablemask_small = label_downsample(mix_veryreliablemask,mix_feas.shape[2],mix_feas.shape[3])
            # regional_contrastive_m = regional_contrastive_cos(mix_feas, labels_classmix, class_center_feas, memobank, num_classes, temp, mix_veryreliablemask_small)
            
            
            
            
            # contra_loss_tr           = mpcl_loss_calc(feas=source_feas,labels=labels_source,
            #                                                     class_center_feas=class_center_feas,
            #                                                     loss_func=mpcl_loss_src,tag='source')
            

            # pseudo_label_small = label_downsample(pseudo_label,mix_feas.shape[2],mix_feas.shape[3])
            # tg_mask_small = label_downsample(tg_mask,mix_feas.shape[2],mix_feas.shape[3])                                          
            # contra_loss_tg = mpcl_loss_calc(feas=target_feas, labels=pseudo_label_small,
            #                                          class_center_feas=class_center_feas,
            #                                          loss_func=mpcl_loss_trg,
            #                                          pixel_sel_loc=tg_mask_small, tag='target')


            # labels_classmix_small = label_downsample(labels_classmix,mix_feas.shape[2],mix_feas.shape[3])
            # mix_mask_small = label_downsample(mix_mask,mix_feas.shape[2],mix_feas.shape[3])
            # contra_loss_mix = mpcl_loss_calc(feas=mix_feas, labels=labels_classmix_small,
            #                                         class_center_feas=class_center_feas,
            #                                         loss_func=mpcl_loss_trg,
            #                                         pixel_sel_loc=mix_mask_small, tag='target')

            # contra_loss = contra_loss_tr + 0.1*contra_loss_mix
            contra_loss = 0.1*regional_contrastive_s
       



        loss = (cfg.TRAIN.LAMBDA_SEG_SRC_MAIN    * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_SRC_AUX   * loss_seg_src_aux
                + cfg.TRAIN.LAMBDA_DICE_SRC_MAIN * loss_dice_main
                + cfg.TRAIN.LAMBDA_DICE_SRC_AUX  * loss_dice_aux
                + 0.1*loss_mix_aux
                + loss_mix
                + contra_loss)
                
        
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, 0.99, i_iter)

        current_losses = {'loss_seg_src_aux' :loss_seg_src_aux,
                          'loss_seg_src_main':loss_seg_src_main,
                        #   'loss_target_entp_aux':loss_target_entp_aux,
                        #   'loss_target_entp_main':loss_target_entp_aux,
                          'loss_dice_aux': loss_dice_aux,
                          'loss_dice_main'   :loss_dice_main,
                          'loss_mix'  :loss_mix,
                           'contra_loss'  :contra_loss}
        print_losses(current_losses,i_iter)
        with open(loss_file, "a") as f:
            info = f"[i_iter: {i_iter}]\n" \
                f"loss_seg_src_aux : {loss_seg_src_aux:.5f}  loss_seg_src_aux : {loss_seg_src_aux:.5f}\n" \
                f"loss_dice_aux : {loss_dice_aux:.5f}  loss_dice_main : {loss_dice_main:.5f}\n" \
                f"loss_mix : {loss_mix:.5f}  contra_loss : {contra_loss:.5f}\n" 
                
            f.write(info + "\n")

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            if cfg.DATASETS == 'Cardiac':
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Cardiac(model, cfg.TARGET, results_file, i_iter)
            else:
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Abdomen(model, cfg.TARGET, results_file, i_iter)
            
            # saved_path = "/home/data_backup/zhr_savedmodel/{0}2{1}_{2}_ourdice_myweight".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
            # os.makedirs(saved_path, exist_ok=True)
            # torch.save(model.state_dict(),  saved_path+f'/model_{i_iter}.pth')
            # tmp_dice = np.mean(dice_mean)
                        
            # if best_model < tmp_dice:
            #     print('taking snapshot ...')
            #     print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            #     snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            #     torch.save(model.state_dict(),  snapshot_dir / '{0}_{1}2{2}_Best_CutMix_ourdice_model.pth'.format(cfg.DATASETS, cfg.SOURCE, cfg.TARGET))
            #     best_model = tmp_dice

        sys.stdout.flush()


def train_bclassmix_ST_contrastive_boundary(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg, strain_loader_):
    '''
    UDA training
    '''

    results_file = "/home/zhr/ICME/scripts/log_final/{0}2{1}_{2}_bclassmix_nonecontra_new1.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
    loss_file = "/home/zhr/ICME/scripts/loss_final/{0}2{1}_{2}_bclassmix_nonecontra_new1.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    num_classes       = cfg.NUM_CLASSES


    unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
    # dice_loss = DiceLoss(n_classes=num_classes).cuda()

    model.train()
    ema_model.train()

    model.cuda()
    ema_model.cuda()
    cudnn.benchmark = True
    cudnn.enabled   = True

    #optimized
    if cfg.TRAIN.OPTIM_G == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.TRAIN.LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                              lr=cfg.TRAIN.LEARNING_RATE,
                              momentum=cfg.TRAIN.MOMENTUM,
                              weight_decay=cfg.TRAIN.WEIGHT_DECAY)

        
    # class_center_feas = np.load(cfg.TRAIN.CLASS_CENTER_FEA_INIT).squeeze()
    # class_center_feas = torch.from_numpy(class_center_feas).float().cuda()

    # interpolate output segmaps
    interp        = nn.Upsample(size=(input_size_source[1],input_size_source[0]),mode='bilinear',
                         align_corners=True)

    # labels for adversarial learning


    strain_loader_iter          = enumerate(strain_loader)
    trgtrain_loader_iter        = enumerate(trgtrain_loader)
    sval_loader_iter            = enumerate(sval_loader)

    transforms = None
    img_mean   = cfg.TRAIN.IMG_MEAN
    best_model =  -1

    # mpcl_loss_src = MPCL(num_class=num_classes, temperature=1.0,
    #                                    base_temperature=1.0, m=0.4)

    # mpcl_loss_trg = MPCL(num_class=num_classes, temperature=1.0,
    #                                    base_temperature=1.0, m=0.2)
    contra_iter = 50000


    memobank = [[] for i in range(num_classes)]
    queue_len = 256
    temp = 1
    classes_dict = {1:1, 
                2:1, 
                3:1, 
                4:1}

    for i_iter in tqdm(range(cfg.TRAIN.MAX_ITERS+1)):

        model.train()
        ema_model.train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer,i_iter,cfg)


        #UDA training
        try:
            _, batch = strain_loader_iter.__next__()
        except StopIteration:
            strain_loader_iter = enumerate(strain_loader)
            _, batch = strain_loader_iter.__next__()
        images_source, labels_source,_ = batch

        try:
            _, batch = trgtrain_loader_iter.__next__()
        except StopIteration:
            trgtrain_loader_iter = enumerate(trgtrain_loader)
            _, batch             = trgtrain_loader_iter.__next__()
        images_target, labels_target,_ = batch


        noise = torch.clamp(torch.randn_like(
                images_target) * 0.1, -0.2, 0.2)
        ema_inputs = images_target + noise



        with torch.no_grad():
            cla_feas_trg, ema_pred_trg_axu, ema_pred_trg_main = ema_model(ema_inputs.cuda())
            source_teacher_feas, ema_pred_src_axu, ema_pred_src_main = ema_model(images_source.cuda())


            if(i_iter > (contra_iter+1)):
                class_center_feas = update_class_center_iter(source_teacher_feas, labels_source, class_center_feas,m=0.2)
                
            
            ema_pred_trg_main     = interp(ema_pred_trg_main)
            ema_output_main_soft = torch.softmax(ema_pred_trg_main, dim=1)
            max_probs_main, pseudo_label  = torch.max(ema_output_main_soft, dim=1)
            
            
            drop_percent = 80
            percent_unreliable = (100 - drop_percent) * (1 - (i_iter)/ cfg.TRAIN.MAX_ITERS)
            drop_percent = 100 - percent_unreliable
            entropy = -torch.sum(ema_output_main_soft * torch.log(ema_output_main_soft + 1e-10), dim=1)
            thresh = np.percentile(
                entropy.detach().cpu().numpy().flatten(), drop_percent
            )
            thresh_veryreliable = np.percentile(
                entropy.detach().cpu().numpy().flatten(), percent_unreliable
            ) 
            tg_mask = entropy.le(thresh).long().cuda()   #batch 256 256
            tg_veryreliablemask = entropy.le(thresh_veryreliable).long().cuda() #batch 256 256
            sc_mask = torch.ones((tg_mask.shape)).cuda() #batch 256 256




            # MixMask, loss_mask = generate_mask(images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda())
            # ## 源域贴在目标域
            # mixmask_trg = tg_mask[:cfg.TRAIN.BATCH_SIZE//2] * MixMask + sc_mask[:cfg.TRAIN.BATCH_SIZE//2] * (1 - MixMask)
            # mixveryreliablemask_trg = tg_veryreliablemask[:cfg.TRAIN.BATCH_SIZE//2] * MixMask + sc_mask[:cfg.TRAIN.BATCH_SIZE//2] * (1 - MixMask)
            # images_trg_classmix = images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + images_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)
            # labels_trg_classmix = pseudo_label[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + labels_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)

            # MixMask, loss_mask = generate_mask(images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda())
            # ## 目标域贴在源域
            # mixmask_src = tg_mask[cfg.TRAIN.BATCH_SIZE//2:] * (1 - MixMask) + sc_mask[cfg.TRAIN.BATCH_SIZE//2:] * MixMask
            # mixveryreliablemask_src = tg_veryreliablemask[cfg.TRAIN.BATCH_SIZE//2:] * (1 - MixMask) + sc_mask[cfg.TRAIN.BATCH_SIZE//2:] * MixMask
            # images_src_classmix = images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + images_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            # labels_src_classmix = pseudo_label[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + labels_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            
            # mix_mask = torch.cat([mixmask_trg, mixmask_src], dim=0)
            # mix_veryreliablemask = torch.cat([mixveryreliablemask_trg, mixveryreliablemask_src], dim=0)
            # images_classmix = torch.cat([images_trg_classmix, images_src_classmix], dim=0)
            # labels_classmix = torch.cat([labels_trg_classmix, labels_src_classmix], dim=0)


            # for image_i in range(cfg.TRAIN.BATCH_SIZE):
            #     classes = torch.unique(labels_source[image_i])
            #     classes = classes[classes != 0]  # 筛选出非背景类
            #     nclasses = classes.shape[0]

            #     if nclasses <=2:
            #         classes = (classes[torch.Tensor(np.random.choice(nclasses, round(nclasses), replace=False)).long()]).cuda()
            #         MixMask = generate_class_mask(labels_source[image_i].cuda(), classes).unsqueeze(0).cuda()
            #     else:
            #         classes = (classes[torch.Tensor(np.random.choice(nclasses, round(2), replace=False)).long()]).cuda()
            #         MixMask = generate_class_mask(labels_source[image_i].cuda(), classes).unsqueeze(0).cuda() # 1,256 256
             

            #     for clas in classes.tolist():
            #         classes_dict[clas] += 1
                
            #     # print(image_i,MixMask.sum())
               
            #     if image_i == 0:
            #         All_MixMask = MixMask
            #     else:
            #         All_MixMask = torch.cat((All_MixMask, MixMask)) # 4, 256 256
            # # print(All_MixMask[0].sum(), All_MixMask[1].sum(), All_MixMask[2].sum(),All_MixMask[3].sum())

            # # dice_loss = dice_loss_class
            # All_MixMask = torch.unsqueeze(All_MixMask, 1).repeat((1,3,1,1)) # 4 3 256 256
            # # print(MixMask.shape, images_target.shape, pseudo_label.shape)
            # images_classmix = images_target.cuda() * (1 - All_MixMask) + images_source.cuda() * All_MixMask
            # labels_classmix = pseudo_label.cuda() * (1 - All_MixMask[:,0,:,:]) + labels_source.cuda() * All_MixMask[:,0,:,:]
            # mix_mask = tg_mask* (1 - All_MixMask[:,0,:,:]) + sc_mask * All_MixMask[:,0,:,:]



            ## 源域贴在目标域
            for image_i in range(cfg.TRAIN.BATCH_SIZE//2):
                classes = torch.unique(labels_source[image_i])
                classes = classes[classes != 0]  # 筛选出非背景类
                nclasses = classes.shape[0]

                if nclasses <=2:
                    classes = (classes[torch.Tensor(np.random.choice(nclasses, round(nclasses), replace=False)).long()]).cuda()
                    MixMask = generate_class_mask(labels_source[image_i].cuda(), classes).unsqueeze(0).cuda()
                else:
                    classes = (classes[torch.Tensor(np.random.choice(nclasses, round(2), replace=False)).long()]).cuda()
                    MixMask = generate_class_mask(labels_source[image_i].cuda(), classes).unsqueeze(0).cuda() # 1,256 256
             

                for clas in classes.tolist():
                    classes_dict[clas] += 1
                    
                if image_i == 0:
                    All_MixMask = MixMask
                else:
                    All_MixMask = torch.cat((All_MixMask, MixMask)) # 2, 256 256

            # dice_loss = dice_loss_class
            All_MixMask = torch.unsqueeze(All_MixMask, 1).repeat((1,3,1,1)) # 2 3 256 256
            # print(MixMask.shape, images_target.shape, pseudo_label.shape)
            images_trg_classmix = images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - All_MixMask) + images_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * All_MixMask
            labels_trg_classmix = pseudo_label[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - All_MixMask[:,0,:,:]) + labels_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * All_MixMask[:,0,:,:]
            mixmask_trg = tg_mask[:cfg.TRAIN.BATCH_SIZE//2]* (1 - All_MixMask[:,0,:,:]) + sc_mask[:cfg.TRAIN.BATCH_SIZE//2] * All_MixMask[:,0,:,:]
            ## 目标域贴在源域
            for image_i in range(cfg.TRAIN.BATCH_SIZE//2,cfg.TRAIN.BATCH_SIZE):
                
                classes = torch.unique(pseudo_label[image_i])
                classes = classes[classes != 0]  # 筛选出非背景类
                nclasses = classes.shape[0]

                if nclasses <=2:
                    classes = (classes[torch.Tensor(np.random.choice(nclasses, round(nclasses), replace=False)).long()]).cuda()
                    MixMask = generate_class_mask(pseudo_label[image_i].cuda(), classes).unsqueeze(0).cuda()
                else:
                    classes = (classes[torch.Tensor(np.random.choice(nclasses, round(2), replace=False)).long()]).cuda()
                    MixMask = generate_class_mask(pseudo_label[image_i].cuda(), classes).unsqueeze(0).cuda() # 1,256 256
                
             

                for clas in classes.tolist():
                    classes_dict[clas] += 1
                    
                if image_i == 2:
                    All_MixMask = MixMask
                else:
                    All_MixMask = torch.cat((All_MixMask, MixMask)) # 2, 256 256

            # dice_loss = dice_loss_class
            All_MixMask = torch.unsqueeze(All_MixMask, 1).repeat((1,3,1,1)) # 2 3 256 256
            # print(MixMask.shape, images_target.shape, pseudo_label.shape)
            images_src_classmix = images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda() * All_MixMask + images_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - All_MixMask)
            labels_src_classmix = pseudo_label[cfg.TRAIN.BATCH_SIZE//2:].cuda() * All_MixMask[:,0,:,:] + labels_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() *  (1 - All_MixMask[:,0,:,:])
            mixmask_src = tg_mask[cfg.TRAIN.BATCH_SIZE//2:]* All_MixMask[:,0,:,:] + sc_mask[cfg.TRAIN.BATCH_SIZE//2:] * (1 - All_MixMask[:,0,:,:])

            mix_mask = torch.cat([mixmask_trg, mixmask_src], dim=0)
            # mix_veryreliablemask = torch.cat([mixveryreliablemask_trg, mixveryreliablemask_src], dim=0)
            images_classmix = torch.cat([images_trg_classmix, images_src_classmix], dim=0)
            labels_classmix = torch.cat([labels_trg_classmix, labels_src_classmix], dim=0)




        # if cfg.TRAIN.CONSWITCH:
        #     consistency_weight = get_current_consistency_weight(i_iter, cfg.TRAIN.MAX_ITERS)
        # else:
        #     consistency_weight = 1


        source_feas,pred_src_aux, pred_src_main = model(images_source.cuda())
        target_feas,pred_trg_aux, pred_trg_main = model(images_target.cuda())
        mix_feas,pred_mix_aux, pred_mix_main = model(images_classmix.cuda())


        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux     = interp(pred_src_aux)
            pred_mix_aux     = interp(pred_mix_aux)

            loss_seg_src_aux = loss_calc(pred_src_aux,labels_source)
            loss_dice_aux    = dice_loss(pred_src_aux,labels_source.cuda())
        else:
            loss_seg_src_aux = 0
            loss_dice_aux    = 0

        pred_src_main     = interp(pred_src_main)
        pred_mix_main     = interp(pred_mix_main)
        loss_seg_src_main = loss_calc(pred_src_main,labels_source)
        loss_dice_main    = dice_loss(pred_src_main,labels_source.cuda())



        weight = np.size(np.array(labels_classmix.cpu())) / torch.sum(mix_mask)

        loss_mix_aux = weight * unlabeled_loss(pred_mix_aux, labels_classmix, mix_mask)
        loss_mix = weight * unlabeled_loss(pred_mix_main, labels_classmix, mix_mask)  

        if(i_iter < (contra_iter+1)):
            contra_loss = 0
        else:
            if(i_iter == (contra_iter+1)):
                class_center_feas = category_center(ema_model, strain_loader_,source_feas.shape[1])
                class_center_feas = class_center_feas.float().cuda()
                ema_model.train()
            memobank = get_boundary_negative(source_teacher_feas, labels_source, num_classes, memobank, queue_len, class_center_feas,3)
            regional_contrastive_s = regional_contrastive_cos(source_feas, labels_source, class_center_feas, memobank, num_classes, temp)
            # tg_mask_small = label_downsample(tg_mask,mix_feas.shape[2],mix_feas.shape[3]) 
            # regional_contrastive_r = regional_contrastive_cos(target_feas, pseudo_label, class_center_feas, memobank, num_classes, temp, tg_mask_small)
            # mix_mask_small = label_downsample(mix_mask,mix_feas.shape[2],mix_feas.shape[3])
            # mix_veryreliablemask_small = label_downsample(mix_veryreliablemask,mix_feas.shape[2],mix_feas.shape[3])
            # regional_contrastive_m = regional_contrastive_cos(mix_feas, labels_classmix, class_center_feas, memobank, num_classes, temp, mix_veryreliablemask_small)
            
            
            
            
            # contra_loss_tr           = mpcl_loss_calc(feas=source_feas,labels=labels_source,
            #                                                     class_center_feas=class_center_feas,
            #                                                     loss_func=mpcl_loss_src,tag='source')
            

            # pseudo_label_small = label_downsample(pseudo_label,mix_feas.shape[2],mix_feas.shape[3])
            # tg_mask_small = label_downsample(tg_mask,mix_feas.shape[2],mix_feas.shape[3])                                          
            # contra_loss_tg = mpcl_loss_calc(feas=target_feas, labels=pseudo_label_small,
            #                                          class_center_feas=class_center_feas,
            #                                          loss_func=mpcl_loss_trg,
            #                                          pixel_sel_loc=tg_mask_small, tag='target')


            # labels_classmix_small = label_downsample(labels_classmix,mix_feas.shape[2],mix_feas.shape[3])
            # mix_mask_small = label_downsample(mix_mask,mix_feas.shape[2],mix_feas.shape[3])
            # contra_loss_mix = mpcl_loss_calc(feas=mix_feas, labels=labels_classmix_small,
            #                                         class_center_feas=class_center_feas,
            #                                         loss_func=mpcl_loss_trg,
            #                                         pixel_sel_loc=mix_mask_small, tag='target')

            # contra_loss = contra_loss_tr + 0.1*contra_loss_mix
            contra_loss = 0.1*regional_contrastive_s
       



        loss = (cfg.TRAIN.LAMBDA_SEG_SRC_MAIN    * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_SRC_AUX   * loss_seg_src_aux
                + cfg.TRAIN.LAMBDA_DICE_SRC_MAIN * loss_dice_main
                + cfg.TRAIN.LAMBDA_DICE_SRC_AUX  * loss_dice_aux
                + 0.1*loss_mix_aux
                + loss_mix
                + contra_loss)
                
        
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, 0.99, i_iter)

        current_losses = {'loss_seg_src_aux' :loss_seg_src_aux,
                          'loss_seg_src_main':loss_seg_src_main,
                        #   'loss_target_entp_aux':loss_target_entp_aux,
                        #   'loss_target_entp_main':loss_target_entp_aux,
                          'loss_dice_aux': loss_dice_aux,
                          'loss_dice_main'   :loss_dice_main,
                          'loss_mix'  :loss_mix,
                           'contra_loss'  :contra_loss,
                           'classes_1': classes_dict[1],
                          'classes_2': classes_dict[2],
                          'classes_3': classes_dict[3],
                          'classes_4': classes_dict[4]}
        print_losses(current_losses,i_iter)
        # with open(loss_file, "a") as f:
        #     info = f"[i_iter: {i_iter}]\n" \
        #         f"loss_seg_src_aux : {loss_seg_src_aux:.5f}  loss_seg_src_aux : {loss_seg_src_aux:.5f}\n" \
        #         f"loss_dice_aux : {loss_dice_aux:.5f}  loss_dice_main : {loss_dice_main:.5f}\n" \
        #         f"loss_mix : {loss_mix:.5f}  contra_loss : {contra_loss:.5f}\n" 
                
        #     f.write(info + "\n")

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            if cfg.DATASETS == 'Cardiac':
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Cardiac(model, cfg.TARGET, results_file, i_iter)
            else:
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Abdomen(model, cfg.TARGET, results_file, i_iter)
            
            # saved_path = "/home/data_backup/zhr_savedmodel/{0}2{1}_{2}_ourdice_myweight".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
            # os.makedirs(saved_path, exist_ok=True)
            # torch.save(model.state_dict(),  saved_path+f'/model_{i_iter}.pth')
            # tmp_dice = np.mean(dice_mean)
                        
            # if best_model < tmp_dice:
            #     print('taking snapshot ...')
            #     print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            #     snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            #     torch.save(model.state_dict(),  snapshot_dir / '{0}_{1}2{2}_Best_CutMix_ourdice_model.pth'.format(cfg.DATASETS, cfg.SOURCE, cfg.TARGET))
            #     best_model = tmp_dice

        sys.stdout.flush()




def train_bcutmix_ST_contrastive_boundary_final(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg, strain_loader_):
    '''
    UDA training
    '''

    results_file = "/home/zhr/ICME/scripts/log_ablation/{0}2{1}_{2}_10000_stcontra_dim256_queuelen500_temperature1_new50_0.012.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
    loss_file = "/home/zhr/ICME/scripts/loss_ablation/{0}2{1}_{2}_10000_stcontra_dim256_queuelen500_temperature1_new50_0.012.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)

    # results_file = "/home/zhr/ICME/scripts/log_ablation/{0}2{1}_{2}consistency_weight.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
    # loss_file = "/home/zhr/ICME/scripts/loss_ablation/{0}2{1}_{2}consistency_weight.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)




    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    num_classes       = cfg.NUM_CLASSES


    unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
    # dice_loss = DiceLoss(n_classes=num_classes).cuda()

    model.train()
    ema_model.train()

    model.cuda()
    ema_model.cuda()
    cudnn.benchmark = True
    cudnn.enabled   = True

    #optimized
    if cfg.TRAIN.OPTIM_G == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.TRAIN.LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                              lr=cfg.TRAIN.LEARNING_RATE,
                              momentum=cfg.TRAIN.MOMENTUM,
                              weight_decay=cfg.TRAIN.WEIGHT_DECAY)

        
    # class_center_feas = np.load(cfg.TRAIN.CLASS_CENTER_FEA_INIT).squeeze()
    # class_center_feas = torch.from_numpy(class_center_feas).float().cuda()

    # interpolate output segmaps
    interp        = nn.Upsample(size=(input_size_source[1],input_size_source[0]),mode='bilinear',
                         align_corners=True)

    # labels for adversarial learning


    strain_loader_iter          = enumerate(strain_loader)
    trgtrain_loader_iter        = enumerate(trgtrain_loader)
    sval_loader_iter            = enumerate(sval_loader)

    transforms = None
    img_mean   = cfg.TRAIN.IMG_MEAN
    best_model =  -1

    # mpcl_loss_src = MPCL(num_class=num_classes, temperature=1.0,
    #                                    base_temperature=1.0, m=0.4)

    # mpcl_loss_trg = MPCL(num_class=num_classes, temperature=1.0,
    #                                    base_temperature=1.0, m=0.2)
    contra_iter = 10000
    dilation_iterations = 3

    memobank = [[] for i in range(num_classes)]
    queue_len = 500
    temp = 1

    for i_iter in tqdm(range(cfg.TRAIN.MAX_ITERS+1)):

        model.train()
        ema_model.train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer,i_iter,cfg)


        #UDA training
        try:
            _, batch = strain_loader_iter.__next__()
        except StopIteration:
            strain_loader_iter = enumerate(strain_loader)
            _, batch = strain_loader_iter.__next__()
        images_source, labels_source,_ = batch

        try:
            _, batch = trgtrain_loader_iter.__next__()
        except StopIteration:
            trgtrain_loader_iter = enumerate(trgtrain_loader)
            _, batch             = trgtrain_loader_iter.__next__()
        images_target, labels_target,_ = batch


        noise = torch.clamp(torch.randn_like(
                images_target) * 0.1, -0.2, 0.2)
        ema_inputs = images_target + noise



        with torch.no_grad():
            cla_feas_trg, ema_pred_trg_axu, ema_pred_trg_main = ema_model(ema_inputs.cuda())
            source_teacher_feas, ema_pred_src_axu, ema_pred_src_main = ema_model(images_source.cuda())


            if(i_iter > (contra_iter+1)):
                class_center_feas = update_class_center_iter(source_teacher_feas, labels_source, class_center_feas,m=0.2)
                
            
            ema_pred_trg_main     = interp(ema_pred_trg_main)
            ema_output_main_soft = torch.softmax(ema_pred_trg_main, dim=1)
            max_probs_main, pseudo_label  = torch.max(ema_output_main_soft, dim=1)
            
            
            # drop_percent = 80
            # percent_unreliable = (100 - drop_percent) * (1 - (i_iter)/ cfg.TRAIN.MAX_ITERS)
            # drop_percent = 100 - percent_unreliable
            # entropy = -torch.sum(ema_output_main_soft * torch.log(ema_output_main_soft + 1e-10), dim=1)
            # thresh = np.percentile(
            #     entropy.detach().cpu().numpy().flatten(), drop_percent
            # )
            # thresh_veryreliable = np.percentile(
            #     entropy.detach().cpu().numpy().flatten(), percent_unreliable
            # ) 
            # tg_mask = entropy.le(thresh).long().cuda()
            # # percent = drop_percent/100.0
            # # tg_mask = tg_mask*percent
            # tg_veryreliablemask = entropy.le(thresh_veryreliable).long().cuda()
            # sc_mask = torch.ones((tg_mask.shape)).cuda()



            percent_20 = 20 * (1 - (i_iter)/ cfg.TRAIN.MAX_ITERS)
            up_percent_80 = 100 - percent_20
            percent_50 = 50 * (1 - (i_iter)/ cfg.TRAIN.MAX_ITERS)
            up_percent_50 = 100 - percent_50

            percent_veryreliable = 50 * (1 - (i_iter)/ cfg.TRAIN.MAX_ITERS)

            entropy = -torch.sum(ema_output_main_soft * torch.log(ema_output_main_soft + 1e-10), dim=1)
            up_thresh_80 = np.percentile(
                entropy.detach().cpu().numpy().flatten(), up_percent_80
            )
            up_thresh_50 = np.percentile(
                entropy.detach().cpu().numpy().flatten(), up_percent_50
            )
            thresh_veryreliable = np.percentile(
                entropy.detach().cpu().numpy().flatten(), percent_veryreliable
            ) 

            tg_mask_80 = entropy.le(up_thresh_80).long().cuda()
            tg_mask_50 = entropy.le(up_thresh_50).long().cuda()

            persent = up_percent_80/100.0
            tg_mask = (tg_mask_80-tg_mask_50)*persent + tg_mask_50
            # print(torch.unique(tg_mask))
            # percent = drop_percent/100.0
            # tg_mask = tg_mask*percent
            tg_veryreliablemask = entropy.le(thresh_veryreliable).long().cuda()
            sc_mask = torch.ones((tg_mask.shape)).cuda()

            # percent_40 = 40 * (1 - (i_iter)/ cfg.TRAIN.MAX_ITERS)
            # up_percent_60 = 100 - percent_40
            # percent_70 = 70 * (1 - (i_iter)/ cfg.TRAIN.MAX_ITERS)
            # up_percent_30 = 100 - percent_70

            # entropy = -torch.sum(ema_output_main_soft * torch.log(ema_output_main_soft + 1e-10), dim=1)
            # up_thresh_60 = np.percentile(
            #     entropy.detach().cpu().numpy().flatten(), up_percent_60
            # )
            # up_thresh_30 = np.percentile(
            #     entropy.detach().cpu().numpy().flatten(), up_percent_30
            # )
            # thresh_veryreliable = np.percentile(
            #     entropy.detach().cpu().numpy().flatten(), percent_40
            # ) 

            # tg_mask_60 = entropy.le(up_thresh_60).long().cuda()
            # tg_mask_30 = entropy.le(up_thresh_30).long().cuda()

            # persent = up_percent_60/100.0
            # tg_mask = (tg_mask_60-tg_mask_30)*persent + tg_mask_30
            # # print(torch.unique(tg_mask))
            # # percent = drop_percent/100.0
            # # tg_mask = tg_mask*percent
            # tg_veryreliablemask = entropy.le(thresh_veryreliable).long().cuda()
            # sc_mask = torch.ones((tg_mask.shape)).cuda()

            



            



            MixMask, loss_mask = generate_mask(images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda())
            ## 源域贴在目标域
            mixmask_trg = tg_mask[:cfg.TRAIN.BATCH_SIZE//2] * MixMask + sc_mask[:cfg.TRAIN.BATCH_SIZE//2] * (1 - MixMask)
            mixveryreliablemask_trg = tg_veryreliablemask[:cfg.TRAIN.BATCH_SIZE//2] * MixMask + sc_mask[:cfg.TRAIN.BATCH_SIZE//2] * (1 - MixMask)
            images_trg_classmix = images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + images_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)
            labels_trg_classmix = pseudo_label[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + labels_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)

            MixMask, loss_mask = generate_mask(images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda())
            ## 目标域贴在源域
            mixmask_src = tg_mask[cfg.TRAIN.BATCH_SIZE//2:] * (1 - MixMask) + sc_mask[cfg.TRAIN.BATCH_SIZE//2:] * MixMask
            mixveryreliablemask_src = tg_veryreliablemask[cfg.TRAIN.BATCH_SIZE//2:] * (1 - MixMask) + sc_mask[cfg.TRAIN.BATCH_SIZE//2:] * MixMask
            images_src_classmix = images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + images_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            labels_src_classmix = pseudo_label[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + labels_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            
            mix_mask = torch.cat([mixmask_trg, mixmask_src], dim=0)
            mix_veryreliablemask = torch.cat([mixveryreliablemask_trg, mixveryreliablemask_src], dim=0)
            images_classmix = torch.cat([images_trg_classmix, images_src_classmix], dim=0)
            labels_classmix = torch.cat([labels_trg_classmix, labels_src_classmix], dim=0)
    
    
        # if cfg.TRAIN.CONSWITCH:
        #     consistency_weight = get_current_consistency_weight(i_iter, cfg.TRAIN.MAX_ITERS)
        # else:
        #     consistency_weight = 1


        source_feas,pred_src_aux, pred_src_main = model(images_source.cuda())
        target_feas,pred_trg_aux, pred_trg_main = model(images_target.cuda())
        mix_feas,pred_mix_aux, pred_mix_main = model(images_classmix.cuda())


        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux     = interp(pred_src_aux)
            pred_mix_aux     = interp(pred_mix_aux)

            loss_seg_src_aux = loss_calc(pred_src_aux,labels_source)
            loss_dice_aux    = dice_loss(pred_src_aux,labels_source.cuda())
        else:
            loss_seg_src_aux = 0
            loss_dice_aux    = 0

        pred_src_main     = interp(pred_src_main)
        pred_mix_main     = interp(pred_mix_main)
        loss_seg_src_main = loss_calc(pred_src_main,labels_source)
        loss_dice_main    = dice_loss(pred_src_main,labels_source.cuda())

        # if cfg.TRAIN.CONSWITCH:
        #     consistency_weight = get_current_consistency_weight(i_iter, cfg.TRAIN.MAX_ITERS)
        # else:
        #     consistency_weight = 1

        # loss_mix_aux = consistency_weight * unlabeled_loss(pred_mix_aux, labels_classmix, mix_mask)
        # loss_mix = consistency_weight * unlabeled_loss(pred_mix_main, labels_classmix, mix_mask)  

        weight = np.size(np.array(labels_classmix.cpu())) / torch.sum(mix_mask)
        loss_mix_aux = weight * unlabeled_loss(pred_mix_aux, labels_classmix, mix_mask)
        loss_mix = weight * unlabeled_loss(pred_mix_main, labels_classmix, mix_mask)  

        if(i_iter < (contra_iter+1)):
            contra_loss = 0
        else:
            if(i_iter == (contra_iter+1)):
                class_center_feas = category_center(ema_model, strain_loader_,source_feas.shape[1])
                class_center_feas = class_center_feas.float().cuda()
                ema_model.train()
            memobank = get_boundary_negative(source_teacher_feas, labels_source, num_classes, memobank, queue_len, class_center_feas,dilation_iterations)
            # regional_contrastive_s = regional_contrastive_cos(source_feas, labels_source, class_center_feas, memobank, num_classes, temp)
            boundary_contrastive_s = boundary_contrastive(source_feas, labels_source, class_center_feas, memobank, num_classes, temp)
            
            tg_veryreliablemask_small = label_downsample(tg_veryreliablemask,target_feas.shape[2],target_feas.shape[3])
            boundary_contrastive_t = boundary_contrastive(target_feas, pseudo_label, class_center_feas, memobank, num_classes, temp,tg_veryreliablemask_small)
            
            # mix_veryreliablemask_small = label_downsample(mix_veryreliablemask,mix_feas.shape[2],mix_feas.shape[3])
            # boundary_contrastive_m = boundary_contrastive(mix_feas, labels_classmix, class_center_feas, memobank, num_classes, temp,mix_veryreliablemask_small)

            # tg_mask_small = label_downsample(tg_mask,mix_feas.shape[2],mix_feas.shape[3]) 
            # regional_contrastive_r = regional_contrastive_cos(target_feas, pseudo_label, class_center_feas, memobank, num_classes, temp, tg_mask_small)
            # mix_mask_small = label_downsample(mix_mask,mix_feas.shape[2],mix_feas.shape[3])
            
            # regional_contrastive_m = regional_contrastive_cos(mix_feas, labels_classmix, class_center_feas, memobank, num_classes, temp, mix_veryreliablemask_small)
           
            
            
            
            # contra_loss_tr           = mpcl_loss_calc(feas=source_feas,labels=labels_source,
            #                                                     class_center_feas=class_center_feas,
            #                                                     loss_func=mpcl_loss_src,tag='source')
            

            # pseudo_label_small = label_downsample(pseudo_label,mix_feas.shape[2],mix_feas.shape[3])
            # tg_mask_small = label_downsample(tg_mask,mix_feas.shape[2],mix_feas.shape[3])                                          
            # contra_loss_tg = mpcl_loss_calc(feas=target_feas, labels=pseudo_label_small,
            #                                          class_center_feas=class_center_feas,
            #                                          loss_func=mpcl_loss_trg,
            #                                          pixel_sel_loc=tg_mask_small, tag='target')


            # labels_classmix_small = label_downsample(labels_classmix,mix_feas.shape[2],mix_feas.shape[3])
            # mix_mask_small = label_downsample(mix_mask,mix_feas.shape[2],mix_feas.shape[3])
            # contra_loss_mix = mpcl_loss_calc(feas=mix_feas, labels=labels_classmix_small,
            #                                         class_center_feas=class_center_feas,
            #                                         loss_func=mpcl_loss_trg,
            #                                         pixel_sel_loc=mix_mask_small, tag='target')

            # contra_loss = contra_loss_tr + 0.1*contra_loss_mix
            # contra_loss = (0.1*boundary_contrastive_s + 0.1*boundary_contrastive_m)/2
            # contra_loss = 0.5*boundary_contrastive_m
            # contra_loss = (0.1*boundary_contrastive_s + 0.1*boundary_contrastive_t)/2
            contra_loss = 0.1*boundary_contrastive_s + 0.01*boundary_contrastive_t
       



        loss = (cfg.TRAIN.LAMBDA_SEG_SRC_MAIN    * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_SRC_AUX   * loss_seg_src_aux
                + cfg.TRAIN.LAMBDA_DICE_SRC_MAIN * loss_dice_main
                + cfg.TRAIN.LAMBDA_DICE_SRC_AUX  * loss_dice_aux
                + 0.1*loss_mix_aux
                + loss_mix
                + contra_loss)
                
        
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, 0.99, i_iter)

        current_losses = {'loss_seg_src_aux' :loss_seg_src_aux,
                          'loss_seg_src_main':loss_seg_src_main,
                        #   'loss_target_entp_aux':loss_target_entp_aux,
                        #   'loss_target_entp_main':loss_target_entp_aux,
                          'loss_dice_aux': loss_dice_aux,
                          'loss_dice_main'   :loss_dice_main,
                          'loss_mix'  :loss_mix,
                           'contra_loss'  :contra_loss}
        print_losses(current_losses,i_iter)
        with open(loss_file, "a") as f:
            info = f"[i_iter: {i_iter}]\n" \
                f"loss_seg_src_aux : {loss_seg_src_aux:.5f}  loss_seg_src_aux : {loss_seg_src_aux:.5f}\n" \
                f"loss_dice_aux : {loss_dice_aux:.5f}  loss_dice_main : {loss_dice_main:.5f}\n" \
                f"loss_mix : {loss_mix:.5f}  contra_loss : {contra_loss:.5f} weight : {weight}\n" 
                
            f.write(info + "\n")

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            if cfg.DATASETS == 'Cardiac':
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Cardiac(model, cfg.TARGET, results_file, i_iter)
            else:
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Abdomen(model, cfg.TARGET, results_file, i_iter)
            
            saved_path = "/home/data_backup/zhr_savedmodel/{0}2{1}_{2}_10000_stcontra_dim256_queuelen500_temperature1_new50_0.012".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
            # saved_path = "/home/data_backup/zhr_savedmodel/{0}2{1}_{2}consistency_weight".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
            os.makedirs(saved_path, exist_ok=True)
            torch.save(model.state_dict(),  saved_path+f'/model_{i_iter}.pth')
            tmp_dice = np.mean(dice_mean)
                        
            # if best_model < tmp_dice:
            #     print('taking snapshot ...')
            #     print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            #     snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            #     torch.save(model.state_dict(),  snapshot_dir / '{0}_{1}2{2}_Best_CutMix_ourdice_model.pth'.format(cfg.DATASETS, cfg.SOURCE, cfg.TARGET))
            #     best_model = tmp_dice

        sys.stdout.flush()




def train_contrast_final_10000(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg, strain_loader_):
    '''
    UDA training
    '''

    results_file = "/home/zhr/ICME/scripts/log_ablation_new/{0}2{1}_{2}_smcontra6.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
    loss_file = "/home/zhr/ICME/scripts/loss_ablation_new/{0}2{1}_{2}_smcontra6.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)

    # results_file = "/home/zhr/ICME/scripts/log_ablation/{0}2{1}_{2}consistency_weight.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
    # loss_file = "/home/zhr/ICME/scripts/loss_ablation/{0}2{1}_{2}consistency_weight.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)




    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    num_classes       = cfg.NUM_CLASSES


    unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
    # dice_loss = DiceLoss(n_classes=num_classes).cuda()

    model.train()
    ema_model.train()

    model.cuda()
    ema_model.cuda()
    cudnn.benchmark = True
    cudnn.enabled   = True

    #optimized
    if cfg.TRAIN.OPTIM_G == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.TRAIN.LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                              lr=cfg.TRAIN.LEARNING_RATE,
                              momentum=cfg.TRAIN.MOMENTUM,
                              weight_decay=cfg.TRAIN.WEIGHT_DECAY)

        
    # class_center_feas = np.load(cfg.TRAIN.CLASS_CENTER_FEA_INIT).squeeze()
    # class_center_feas = torch.from_numpy(class_center_feas).float().cuda()

    # interpolate output segmaps
    interp        = nn.Upsample(size=(input_size_source[1],input_size_source[0]),mode='bilinear',
                         align_corners=True)

    # labels for adversarial learning


    strain_loader_iter          = enumerate(strain_loader)
    trgtrain_loader_iter        = enumerate(trgtrain_loader)
    sval_loader_iter            = enumerate(sval_loader)

    transforms = None
    img_mean   = cfg.TRAIN.IMG_MEAN
    best_model =  -1

    # mpcl_loss_src = MPCL(num_class=num_classes, temperature=1.0,
    #                                    base_temperature=1.0, m=0.4)

    # mpcl_loss_trg = MPCL(num_class=num_classes, temperature=1.0,
    #                                    base_temperature=1.0, m=0.2)
    contra_iter = 10000
    dilation_iterations = 3

    memobank = [[] for i in range(num_classes)]
    queue_len = 500
    temp = 1

    for i_iter in tqdm(range(10001, cfg.TRAIN.MAX_ITERS+1)):

        model.train()
        ema_model.train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer,i_iter,cfg)


        #UDA training
        try:
            _, batch = strain_loader_iter.__next__()
        except StopIteration:
            strain_loader_iter = enumerate(strain_loader)
            _, batch = strain_loader_iter.__next__()
        images_source, labels_source,_ = batch

        try:
            _, batch = trgtrain_loader_iter.__next__()
        except StopIteration:
            trgtrain_loader_iter = enumerate(trgtrain_loader)
            _, batch             = trgtrain_loader_iter.__next__()
        images_target, labels_target,_ = batch


        noise = torch.clamp(torch.randn_like(
                images_target) * 0.1, -0.2, 0.2)
        ema_inputs = images_target + noise



        with torch.no_grad():
            cla_feas_trg, ema_pred_trg_axu, ema_pred_trg_main = ema_model(ema_inputs.cuda())
            source_teacher_feas, ema_pred_src_axu, ema_pred_src_main = ema_model(images_source.cuda())


            if(i_iter > (contra_iter+1)):
                class_center_feas = update_class_center_iter(source_teacher_feas, labels_source, class_center_feas,m=0.2)
                
            
            ema_pred_trg_main     = interp(ema_pred_trg_main)
            ema_output_main_soft = torch.softmax(ema_pred_trg_main, dim=1)
            max_probs_main, pseudo_label  = torch.max(ema_output_main_soft, dim=1)
            
            
            # drop_percent = 80
            # percent_unreliable = (100 - drop_percent) * (1 - (i_iter)/ cfg.TRAIN.MAX_ITERS)
            # drop_percent = 100 - percent_unreliable
            # entropy = -torch.sum(ema_output_main_soft * torch.log(ema_output_main_soft + 1e-10), dim=1)
            # thresh = np.percentile(
            #     entropy.detach().cpu().numpy().flatten(), drop_percent
            # )
            # thresh_veryreliable = np.percentile(
            #     entropy.detach().cpu().numpy().flatten(), percent_unreliable
            # ) 
            # tg_mask = entropy.le(thresh).long().cuda()
            # # percent = drop_percent/100.0
            # # tg_mask = tg_mask*percent
            # tg_veryreliablemask = entropy.le(thresh_veryreliable).long().cuda()
            # sc_mask = torch.ones((tg_mask.shape)).cuda()



            percent_20 = 20 * (1 - (i_iter)/ cfg.TRAIN.MAX_ITERS)
            up_percent_80 = 100 - percent_20
            percent_50 = 50 * (1 - (i_iter)/ cfg.TRAIN.MAX_ITERS)
            up_percent_50 = 100 - percent_50

            percent_veryreliable = 50 * (1 - (i_iter)/ cfg.TRAIN.MAX_ITERS)

            entropy = -torch.sum(ema_output_main_soft * torch.log(ema_output_main_soft + 1e-10), dim=1)
            up_thresh_80 = np.percentile(
                entropy.detach().cpu().numpy().flatten(), up_percent_80
            )
            up_thresh_50 = np.percentile(
                entropy.detach().cpu().numpy().flatten(), up_percent_50
            )
            thresh_veryreliable = np.percentile(
                entropy.detach().cpu().numpy().flatten(), percent_veryreliable
            ) 

            tg_mask_80 = entropy.le(up_thresh_80).long().cuda()
            tg_mask_50 = entropy.le(up_thresh_50).long().cuda()

            persent = up_percent_80/100.0
            tg_mask = (tg_mask_80-tg_mask_50)*persent + tg_mask_50
            # print(torch.unique(tg_mask))
            # percent = drop_percent/100.0
            # tg_mask = tg_mask*percent
            tg_veryreliablemask = entropy.le(thresh_veryreliable).long().cuda()
            sc_mask = torch.ones((tg_mask.shape)).cuda()

            # percent_40 = 40 * (1 - (i_iter)/ cfg.TRAIN.MAX_ITERS)
            # up_percent_60 = 100 - percent_40
            # percent_70 = 70 * (1 - (i_iter)/ cfg.TRAIN.MAX_ITERS)
            # up_percent_30 = 100 - percent_70

            # entropy = -torch.sum(ema_output_main_soft * torch.log(ema_output_main_soft + 1e-10), dim=1)
            # up_thresh_60 = np.percentile(
            #     entropy.detach().cpu().numpy().flatten(), up_percent_60
            # )
            # up_thresh_30 = np.percentile(
            #     entropy.detach().cpu().numpy().flatten(), up_percent_30
            # )
            # thresh_veryreliable = np.percentile(
            #     entropy.detach().cpu().numpy().flatten(), percent_40
            # ) 

            # tg_mask_60 = entropy.le(up_thresh_60).long().cuda()
            # tg_mask_30 = entropy.le(up_thresh_30).long().cuda()

            # persent = up_percent_60/100.0
            # tg_mask = (tg_mask_60-tg_mask_30)*persent + tg_mask_30
            # # print(torch.unique(tg_mask))
            # # percent = drop_percent/100.0
            # # tg_mask = tg_mask*percent
            # tg_veryreliablemask = entropy.le(thresh_veryreliable).long().cuda()
            # sc_mask = torch.ones((tg_mask.shape)).cuda()

            



            



            MixMask, loss_mask = generate_mask(images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda())
            ## 源域贴在目标域
            mixmask_trg = tg_mask[:cfg.TRAIN.BATCH_SIZE//2] * MixMask + sc_mask[:cfg.TRAIN.BATCH_SIZE//2] * (1 - MixMask)
            mixveryreliablemask_trg = tg_veryreliablemask[:cfg.TRAIN.BATCH_SIZE//2] * MixMask + sc_mask[:cfg.TRAIN.BATCH_SIZE//2] * (1 - MixMask)
            images_trg_classmix = images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + images_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)
            labels_trg_classmix = pseudo_label[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + labels_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)

            MixMask, loss_mask = generate_mask(images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda())
            ## 目标域贴在源域
            mixmask_src = tg_mask[cfg.TRAIN.BATCH_SIZE//2:] * (1 - MixMask) + sc_mask[cfg.TRAIN.BATCH_SIZE//2:] * MixMask
            mixveryreliablemask_src = tg_veryreliablemask[cfg.TRAIN.BATCH_SIZE//2:] * (1 - MixMask) + sc_mask[cfg.TRAIN.BATCH_SIZE//2:] * MixMask
            images_src_classmix = images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + images_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            labels_src_classmix = pseudo_label[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + labels_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            
            mix_mask = torch.cat([mixmask_trg, mixmask_src], dim=0)
            mix_veryreliablemask = torch.cat([mixveryreliablemask_trg, mixveryreliablemask_src], dim=0)
            images_classmix = torch.cat([images_trg_classmix, images_src_classmix], dim=0)
            labels_classmix = torch.cat([labels_trg_classmix, labels_src_classmix], dim=0)
    
    
        # if cfg.TRAIN.CONSWITCH:
        #     consistency_weight = get_current_consistency_weight(i_iter, cfg.TRAIN.MAX_ITERS)
        # else:
        #     consistency_weight = 1


        source_feas,pred_src_aux, pred_src_main = model(images_source.cuda())
        target_feas,pred_trg_aux, pred_trg_main = model(images_target.cuda())
        mix_feas,pred_mix_aux, pred_mix_main = model(images_classmix.cuda())


        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux     = interp(pred_src_aux)
            pred_mix_aux     = interp(pred_mix_aux)

            loss_seg_src_aux = loss_calc(pred_src_aux,labels_source)
            loss_dice_aux    = dice_loss(pred_src_aux,labels_source.cuda())
        else:
            loss_seg_src_aux = 0
            loss_dice_aux    = 0

        pred_src_main     = interp(pred_src_main)
        pred_mix_main     = interp(pred_mix_main)
        loss_seg_src_main = loss_calc(pred_src_main,labels_source)
        loss_dice_main    = dice_loss(pred_src_main,labels_source.cuda())

        # if cfg.TRAIN.CONSWITCH:
        #     consistency_weight = get_current_consistency_weight(i_iter, cfg.TRAIN.MAX_ITERS)
        # else:
        #     consistency_weight = 1

        # loss_mix_aux = consistency_weight * unlabeled_loss(pred_mix_aux, labels_classmix, mix_mask)
        # loss_mix = consistency_weight * unlabeled_loss(pred_mix_main, labels_classmix, mix_mask)  

        weight = np.size(np.array(labels_classmix.cpu())) / torch.sum(mix_mask)
        loss_mix_aux = weight * unlabeled_loss(pred_mix_aux, labels_classmix, mix_mask)
        loss_mix = weight * unlabeled_loss(pred_mix_main, labels_classmix, mix_mask)  

        if(i_iter < (contra_iter+1)):
            contra_loss = 0
        else:
            if(i_iter == (contra_iter+1)):
                class_center_feas = category_center(ema_model, strain_loader_,source_feas.shape[1])
                class_center_feas = class_center_feas.float().cuda()
                ema_model.train()
            memobank = get_boundary_negative(source_teacher_feas, labels_source, num_classes, memobank, queue_len, class_center_feas,dilation_iterations)
            # regional_contrastive_s = regional_contrastive_cos(source_feas, labels_source, class_center_feas, memobank, num_classes, temp)
            boundary_contrastive_s = boundary_contrastive(source_feas, labels_source, class_center_feas, memobank, num_classes, temp)
            
            # tg_veryreliablemask_small = label_downsample(tg_veryreliablemask,target_feas.shape[2],target_feas.shape[3])
            # boundary_contrastive_t = boundary_contrastive(target_feas, pseudo_label, class_center_feas, memobank, num_classes, temp,tg_veryreliablemask_small)
            
            mix_veryreliablemask_small = label_downsample(mix_veryreliablemask,mix_feas.shape[2],mix_feas.shape[3])
            boundary_contrastive_m = boundary_contrastive(mix_feas, labels_classmix, class_center_feas, memobank, num_classes, temp,mix_veryreliablemask_small)

            # tg_mask_small = label_downsample(tg_mask,mix_feas.shape[2],mix_feas.shape[3]) 
            # regional_contrastive_r = regional_contrastive_cos(target_feas, pseudo_label, class_center_feas, memobank, num_classes, temp, tg_mask_small)
            # mix_mask_small = label_downsample(mix_mask,mix_feas.shape[2],mix_feas.shape[3])
            
            # regional_contrastive_m = regional_contrastive_cos(mix_feas, labels_classmix, class_center_feas, memobank, num_classes, temp, mix_veryreliablemask_small)
           
            
            
            
            # contra_loss_tr           = mpcl_loss_calc(feas=source_feas,labels=labels_source,
            #                                                     class_center_feas=class_center_feas,
            #                                                     loss_func=mpcl_loss_src,tag='source')
            

            # pseudo_label_small = label_downsample(pseudo_label,mix_feas.shape[2],mix_feas.shape[3])
            # tg_mask_small = label_downsample(tg_mask,mix_feas.shape[2],mix_feas.shape[3])                                          
            # contra_loss_tg = mpcl_loss_calc(feas=target_feas, labels=pseudo_label_small,
            #                                          class_center_feas=class_center_feas,
            #                                          loss_func=mpcl_loss_trg,
            #                                          pixel_sel_loc=tg_mask_small, tag='target')


            # labels_classmix_small = label_downsample(labels_classmix,mix_feas.shape[2],mix_feas.shape[3])
            # mix_mask_small = label_downsample(mix_mask,mix_feas.shape[2],mix_feas.shape[3])
            # contra_loss_mix = mpcl_loss_calc(feas=mix_feas, labels=labels_classmix_small,
            #                                         class_center_feas=class_center_feas,
            #                                         loss_func=mpcl_loss_trg,
            #                                         pixel_sel_loc=mix_mask_small, tag='target')

            # contra_loss = contra_loss_tr + 0.1*contra_loss_mix
            # contra_loss = (0.1*boundary_contrastive_s + 0.1*boundary_contrastive_m)/2
            # contra_loss = 0.5*boundary_contrastive_m
            # contra_loss = (0.1*boundary_contrastive_s + 0.1*boundary_contrastive_t)/2
            contra_loss = (0.1*boundary_contrastive_s + 0.1*boundary_contrastive_m)/2
       



        loss = (cfg.TRAIN.LAMBDA_SEG_SRC_MAIN    * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_SRC_AUX   * loss_seg_src_aux
                + cfg.TRAIN.LAMBDA_DICE_SRC_MAIN * loss_dice_main
                + cfg.TRAIN.LAMBDA_DICE_SRC_AUX  * loss_dice_aux
                + 0.1*loss_mix_aux
                + loss_mix
                + contra_loss)
                
        
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, 0.99, i_iter)

        current_losses = {'loss_seg_src_aux' :loss_seg_src_aux,
                          'loss_seg_src_main':loss_seg_src_main,
                        #   'loss_target_entp_aux':loss_target_entp_aux,
                        #   'loss_target_entp_main':loss_target_entp_aux,
                          'loss_dice_aux': loss_dice_aux,
                          'loss_dice_main'   :loss_dice_main,
                          'loss_mix'  :loss_mix,
                           'contra_loss'  :contra_loss}
        print_losses(current_losses,i_iter)
        with open(loss_file, "a") as f:
            info = f"[i_iter: {i_iter}]\n" \
                f"loss_seg_src_aux : {loss_seg_src_aux:.5f}  loss_seg_src_aux : {loss_seg_src_aux:.5f}\n" \
                f"loss_dice_aux : {loss_dice_aux:.5f}  loss_dice_main : {loss_dice_main:.5f}\n" \
                f"loss_mix : {loss_mix:.5f}  contra_loss : {contra_loss:.5f} weight : {weight}\n" 
                
            f.write(info + "\n")

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            if cfg.DATASETS == 'Cardiac':
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Cardiac(model, cfg.TARGET, results_file, i_iter)
            else:
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Abdomen(model, cfg.TARGET, results_file, i_iter)
            
            saved_path = "/home/data_backup/zhr_savedmodel/{0}2{1}_{2}_sm6".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
            # saved_path = "/home/data_backup/zhr_savedmodel/{0}2{1}_{2}consistency_weight".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
            os.makedirs(saved_path, exist_ok=True)
            torch.save(model.state_dict(),  saved_path+f'/model_{i_iter}.pth')
            tmp_dice = np.mean(dice_mean)
                        
            # if best_model < tmp_dice:
            #     print('taking snapshot ...')
            #     print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            #     snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            #     torch.save(model.state_dict(),  snapshot_dir / '{0}_{1}2{2}_Best_CutMix_ourdice_model.pth'.format(cfg.DATASETS, cfg.SOURCE, cfg.TARGET))
            #     best_model = tmp_dice

        sys.stdout.flush()









def train_bcutmix_ST_contrastive_boundary_gram_final(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg, strain_loader_):
    '''
    UDA training
    '''

    results_file = "/home/zhr/ICME/scripts/log_ablation/{0}2{1}_{2}_10000_scontra_dim256_queuelen500_temperature1_mmgram_new50.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
    loss_file = "/home/zhr/ICME/scripts/loss_ablation/{0}2{1}_{2}_10000_scontra_dim256_queuelen500_temperature1_mmgram_new50.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)

    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    num_classes       = cfg.NUM_CLASSES


    unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
    # dice_loss = DiceLoss(n_classes=num_classes).cuda()
    styleloss = StyleLoss(weight = 1000).cuda()
    model.train()
    ema_model.train()

    model.cuda()
    ema_model.cuda()
    cudnn.benchmark = True
    cudnn.enabled   = True

    #optimized
    if cfg.TRAIN.OPTIM_G == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.TRAIN.LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                              lr=cfg.TRAIN.LEARNING_RATE,
                              momentum=cfg.TRAIN.MOMENTUM,
                              weight_decay=cfg.TRAIN.WEIGHT_DECAY)

        
    # class_center_feas = np.load(cfg.TRAIN.CLASS_CENTER_FEA_INIT).squeeze()
    # class_center_feas = torch.from_numpy(class_center_feas).float().cuda()

    # interpolate output segmaps
    interp        = nn.Upsample(size=(input_size_source[1],input_size_source[0]),mode='bilinear',
                         align_corners=True)

    # labels for adversarial learning


    strain_loader_iter          = enumerate(strain_loader)
    trgtrain_loader_iter        = enumerate(trgtrain_loader)
    sval_loader_iter            = enumerate(sval_loader)

    transforms = None
    img_mean   = cfg.TRAIN.IMG_MEAN
    best_model =  -1

    # mpcl_loss_src = MPCL(num_class=num_classes, temperature=1.0,
    #                                    base_temperature=1.0, m=0.4)

    # mpcl_loss_trg = MPCL(num_class=num_classes, temperature=1.0,
    #                                    base_temperature=1.0, m=0.2)
    contra_iter = 10000
    dilation_iterations = 3

    memobank = [[] for i in range(num_classes)]
    queue_len = 500
    temp = 1

    for i_iter in tqdm(range(cfg.TRAIN.MAX_ITERS+1)):

        model.train()
        ema_model.train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer,i_iter,cfg)


        #UDA training
        try:
            _, batch = strain_loader_iter.__next__()
        except StopIteration:
            strain_loader_iter = enumerate(strain_loader)
            _, batch = strain_loader_iter.__next__()
        images_source, labels_source,_ = batch

        try:
            _, batch = trgtrain_loader_iter.__next__()
        except StopIteration:
            trgtrain_loader_iter = enumerate(trgtrain_loader)
            _, batch             = trgtrain_loader_iter.__next__()
        images_target, labels_target,_ = batch


        noise = torch.clamp(torch.randn_like(
                images_target) * 0.1, -0.2, 0.2)
        ema_inputs = images_target + noise



        with torch.no_grad():
            _,cla_feas_trg, ema_pred_trg_axu, ema_pred_trg_main = ema_model(ema_inputs.cuda())
            source_teacher_layer1feas,source_teacher_feas, ema_pred_src_axu, ema_pred_src_main = ema_model(images_source.cuda())


            if(i_iter > (contra_iter+1)):
                class_center_feas = update_class_center_iter(source_teacher_feas, labels_source, class_center_feas,m=0.2)
                
            
            ema_pred_trg_main     = interp(ema_pred_trg_main)
            ema_output_main_soft = torch.softmax(ema_pred_trg_main, dim=1)
            max_probs_main, pseudo_label  = torch.max(ema_output_main_soft, dim=1)
            
        

            percent_20 = 20 * (1 - (i_iter)/ cfg.TRAIN.MAX_ITERS)
            up_percent_80 = 100 - percent_20
            percent_50 = 50 * (1 - (i_iter)/ cfg.TRAIN.MAX_ITERS)
            up_percent_50 = 100 - percent_50
            entropy = -torch.sum(ema_output_main_soft * torch.log(ema_output_main_soft + 1e-10), dim=1)
            up_thresh_80 = np.percentile(
                entropy.detach().cpu().numpy().flatten(), up_percent_80
            )
            up_thresh_50 = np.percentile(
                entropy.detach().cpu().numpy().flatten(), up_percent_50
            )
            thresh_veryreliable = np.percentile(
                entropy.detach().cpu().numpy().flatten(), percent_20
            ) 

            tg_mask_80 = entropy.le(up_thresh_80).long().cuda()
            tg_mask_50 = entropy.le(up_thresh_50).long().cuda()

            persent = up_percent_80/100.0
            tg_mask = (tg_mask_80-tg_mask_50)*persent + tg_mask_50
            # print(torch.unique(tg_mask))
            # percent = drop_percent/100.0
            # tg_mask = tg_mask*percent
            tg_veryreliablemask = entropy.le(thresh_veryreliable).long().cuda()
            sc_mask = torch.ones((tg_mask.shape)).cuda()



            MixMask, loss_mask = generate_mask(images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda())
            ## 源域贴在目标域
            mixmask_trg = tg_mask[:cfg.TRAIN.BATCH_SIZE//2] * MixMask + sc_mask[:cfg.TRAIN.BATCH_SIZE//2] * (1 - MixMask)
            mixveryreliablemask_trg = tg_veryreliablemask[:cfg.TRAIN.BATCH_SIZE//2] * MixMask + sc_mask[:cfg.TRAIN.BATCH_SIZE//2] * (1 - MixMask)
            images_trg_classmix = images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + images_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)
            labels_trg_classmix = pseudo_label[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + labels_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)

            MixMask, loss_mask = generate_mask(images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda())
            ## 目标域贴在源域
            mixmask_src = tg_mask[cfg.TRAIN.BATCH_SIZE//2:] * (1 - MixMask) + sc_mask[cfg.TRAIN.BATCH_SIZE//2:] * MixMask
            mixveryreliablemask_src = tg_veryreliablemask[cfg.TRAIN.BATCH_SIZE//2:] * (1 - MixMask) + sc_mask[cfg.TRAIN.BATCH_SIZE//2:] * MixMask
            images_src_classmix = images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + images_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            labels_src_classmix = pseudo_label[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + labels_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            
            mix_mask = torch.cat([mixmask_trg, mixmask_src], dim=0)
            mix_veryreliablemask = torch.cat([mixveryreliablemask_trg, mixveryreliablemask_src], dim=0)
            images_classmix = torch.cat([images_trg_classmix, images_src_classmix], dim=0)
            labels_classmix = torch.cat([labels_trg_classmix, labels_src_classmix], dim=0)
    
    
        # if cfg.TRAIN.CONSWITCH:
        #     consistency_weight = get_current_consistency_weight(i_iter, cfg.TRAIN.MAX_ITERS)
        # else:
        #     consistency_weight = 1


        _,source_feas,pred_src_aux, pred_src_main = model(images_source.cuda())
        target_layer1feas,target_feas,pred_trg_aux, pred_trg_main = model(images_target.cuda())
        mix_layer1feas,mix_feas,pred_mix_aux, pred_mix_main = model(images_classmix.cuda())


        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux     = interp(pred_src_aux)
            pred_mix_aux     = interp(pred_mix_aux)

            loss_seg_src_aux = loss_calc(pred_src_aux,labels_source)
            loss_dice_aux    = dice_loss(pred_src_aux,labels_source.cuda())
        else:
            loss_seg_src_aux = 0
            loss_dice_aux    = 0

        pred_src_main     = interp(pred_src_main)
        pred_mix_main     = interp(pred_mix_main)
        loss_seg_src_main = loss_calc(pred_src_main,labels_source)
        loss_dice_main    = dice_loss(pred_src_main,labels_source.cuda())



        weight = np.size(np.array(labels_classmix.cpu())) / torch.sum(mix_mask)

        loss_mix_aux = weight * unlabeled_loss(pred_mix_aux, labels_classmix, mix_mask)
        loss_mix = weight * unlabeled_loss(pred_mix_main, labels_classmix, mix_mask)  
        # print(mix_layer1feas.shape)
        # print(mix_layer1feas[:cfg.TRAIN.BATCH_SIZE//2].shape)
        # print(mix_layer1feas[cfg.TRAIN.BATCH_SIZE//2:].shape)
        gram_loss = styleloss(mix_layer1feas[:cfg.TRAIN.BATCH_SIZE//2],mix_layer1feas[cfg.TRAIN.BATCH_SIZE//2:])
        # gram_loss = 0
        if(i_iter < (contra_iter+1)):
            contra_loss = 0
        else:
            if(i_iter == (contra_iter+1)):
                class_center_feas = category_center_gram(ema_model, strain_loader_,source_feas.shape[1])
                class_center_feas = class_center_feas.float().cuda()
                ema_model.train()
            memobank = get_boundary_negative(source_teacher_feas, labels_source, num_classes, memobank, queue_len, class_center_feas,dilation_iterations)
            # regional_contrastive_s = regional_contrastive_cos(source_feas, labels_source, class_center_feas, memobank, num_classes, temp)
            boundary_contrastive_s = boundary_contrastive(source_feas, labels_source, class_center_feas, memobank, num_classes, temp)
            


            
            # tg_mask_small = label_downsample(tg_mask,mix_feas.shape[2],mix_feas.shape[3]) 
            # regional_contrastive_r = regional_contrastive_cos(target_feas, pseudo_label, class_center_feas, memobank, num_classes, temp, tg_mask_small)
            # mix_mask_small = label_downsample(mix_mask,mix_feas.shape[2],mix_feas.shape[3])
            # mix_veryreliablemask_small = label_downsample(mix_veryreliablemask,mix_feas.shape[2],mix_feas.shape[3])
            # regional_contrastive_m = regional_contrastive_cos(mix_feas, labels_classmix, class_center_feas, memobank, num_classes, temp, mix_veryreliablemask_small)
            # boundary_contrastive_m = boundary_contrastive(mix_feas, labels_classmix, class_center_feas, memobank, num_classes, temp,mix_veryreliablemask_small)
            
            
            
            # contra_loss_tr           = mpcl_loss_calc(feas=source_feas,labels=labels_source,
            #                                                     class_center_feas=class_center_feas,
            #                                                     loss_func=mpcl_loss_src,tag='source')
            

            # pseudo_label_small = label_downsample(pseudo_label,mix_feas.shape[2],mix_feas.shape[3])
            # tg_mask_small = label_downsample(tg_mask,mix_feas.shape[2],mix_feas.shape[3])                                          
            # contra_loss_tg = mpcl_loss_calc(feas=target_feas, labels=pseudo_label_small,
            #                                          class_center_feas=class_center_feas,
            #                                          loss_func=mpcl_loss_trg,
            #                                          pixel_sel_loc=tg_mask_small, tag='target')


            # labels_classmix_small = label_downsample(labels_classmix,mix_feas.shape[2],mix_feas.shape[3])
            # mix_mask_small = label_downsample(mix_mask,mix_feas.shape[2],mix_feas.shape[3])
            # contra_loss_mix = mpcl_loss_calc(feas=mix_feas, labels=labels_classmix_small,
            #                                         class_center_feas=class_center_feas,
            #                                         loss_func=mpcl_loss_trg,
            #                                         pixel_sel_loc=mix_mask_small, tag='target')

            # contra_loss = contra_loss_tr + 0.1*contra_loss_mix
            # contra_loss = (0.1*boundary_contrastive_s+ 0.1*boundary_contrastive_m)/2
            contra_loss = 0.1*boundary_contrastive_s
       



        loss = (cfg.TRAIN.LAMBDA_SEG_SRC_MAIN    * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_SRC_AUX   * loss_seg_src_aux
                + cfg.TRAIN.LAMBDA_DICE_SRC_MAIN * loss_dice_main
                + cfg.TRAIN.LAMBDA_DICE_SRC_AUX  * loss_dice_aux
                + 0.1*loss_mix_aux
                + loss_mix
                + contra_loss
                + gram_loss)
                
        
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, 0.99, i_iter)

        current_losses = {'loss_seg_src_aux' :loss_seg_src_aux,
                          'loss_seg_src_main':loss_seg_src_main,
                        #   'loss_target_entp_aux':loss_target_entp_aux,
                        #   'loss_target_entp_main':loss_target_entp_aux,
                          'loss_dice_aux': loss_dice_aux,
                          'loss_dice_main'   :loss_dice_main,
                          'loss_mix'  :loss_mix,
                           'contra_loss'  :contra_loss,
                           'gram_loss'  :gram_loss}
        print_losses(current_losses,i_iter)
        with open(loss_file, "a") as f:
            info = f"[i_iter: {i_iter}]\n" \
                f"loss_seg_src_aux : {loss_seg_src_aux:.5f}  loss_seg_src_aux : {loss_seg_src_aux:.5f}\n" \
                f"loss_dice_aux : {loss_dice_aux:.5f}  loss_dice_main : {loss_dice_main:.5f}\n" \
                f"loss_mix : {loss_mix:.5f}  contra_loss : {contra_loss:.5f}  gram_loss : {gram_loss:.5f}\n" 
                
            f.write(info + "\n")

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
        # if i_iter ==0:
            if cfg.DATASETS == 'Cardiac':
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Cardiac_gram(model, cfg.TARGET, results_file, i_iter)
            else:
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Abdomen(model, cfg.TARGET, results_file, i_iter)
            
            # saved_path = "/home/data_backup/zhr_savedmodel/{0}2{1}_{2}_ourdice_myweight".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
            # os.makedirs(saved_path, exist_ok=True)
            # torch.save(model.state_dict(),  saved_path+f'/model_{i_iter}.pth')
            # tmp_dice = np.mean(dice_mean)
                        
            # if best_model < tmp_dice:
            #     print('taking snapshot ...')
            #     print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            #     snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            #     torch.save(model.state_dict(),  snapshot_dir / '{0}_{1}2{2}_Best_CutMix_ourdice_model.pth'.format(cfg.DATASETS, cfg.SOURCE, cfg.TARGET))
            #     best_model = tmp_dice

        sys.stdout.flush()







def train_cutmix_ST_MPSCL_ourpseudolabel(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg):
    '''
    UDA training
    '''

    results_file = "/home/zhr/ICME/scripts/log/{0}2{1}_{2}1.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)

    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    num_classes       = cfg.NUM_CLASSES


    unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
    # dice_loss = DiceLoss(n_classes=num_classes).cuda()

    model.train()
    ema_model.train()

    model.cuda()
    ema_model.cuda()
    cudnn.benchmark = True
    cudnn.enabled   = True

    #optimized
    if cfg.TRAIN.OPTIM_G == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.TRAIN.LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                              lr=cfg.TRAIN.LEARNING_RATE,
                              momentum=cfg.TRAIN.MOMENTUM,
                              weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        
    class_center_feas = np.load(cfg.TRAIN.CLASS_CENTER_FEA_INIT).squeeze()
    class_center_feas = torch.from_numpy(class_center_feas).float().cuda()

    # interpolate output segmaps
    interp        = nn.Upsample(size=(input_size_source[1],input_size_source[0]),mode='bilinear',
                         align_corners=True)

    # labels for adversarial learning


    strain_loader_iter          = enumerate(strain_loader)
    trgtrain_loader_iter        = enumerate(trgtrain_loader)
    sval_loader_iter            = enumerate(sval_loader)

    transforms = None
    img_mean   = cfg.TRAIN.IMG_MEAN
    best_model =  -1

    mpcl_loss_src = MPCL(num_class=num_classes, temperature=1.0,
                                       base_temperature=1.0, m=0.4)

    mpcl_loss_trg = MPCL(num_class=num_classes, temperature=1.0,
                                       base_temperature=1.0, m=0.2)

    for i_iter in tqdm(range(5000, cfg.TRAIN.MAX_ITERS+1)):

        model.train()
        ema_model.train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer,i_iter,cfg)


        #UDA training
        try:
            _, batch = strain_loader_iter.__next__()
        except StopIteration:
            strain_loader_iter = enumerate(strain_loader)
            _, batch = strain_loader_iter.__next__()
        images_source, labels_source,_ = batch

        try:
            _, batch = trgtrain_loader_iter.__next__()
        except StopIteration:
            trgtrain_loader_iter = enumerate(trgtrain_loader)
            _, batch             = trgtrain_loader_iter.__next__()
        images_target, labels_target,_ = batch


        noise = torch.clamp(torch.randn_like(
                images_target) * 0.1, -0.2, 0.2)
        ema_inputs = images_target + noise

        with torch.no_grad():
            cla_feas_trg, ema_pred_trg_axu, ema_pred_trg_main = ema_model(ema_inputs.cuda())
            source_teacher_feas, ema_pred_src_axu, ema_pred_src_main = ema_model(images_source.cuda())
            
            ema_output_main_soft_feas = torch.softmax(ema_pred_trg_main, dim=1)
            max_probs_main_feas, pseudo_label_feas  = torch.max(ema_output_main_soft_feas, dim=1)
            tg_mask = max_probs_main_feas.ge(cfg.TRAIN.THRESHOLD).long() 
            class_center_feas = update_class_center_iter(source_teacher_feas, labels_source, class_center_feas,m=0.2)

            ema_pred_trg_axu     = interp(ema_pred_trg_axu)
            ema_pred_trg_main     = interp(ema_pred_trg_main)

            ema_output_main_soft = torch.softmax(ema_pred_trg_main, dim=1)
            ema_output_aux_soft = torch.softmax(ema_pred_trg_axu, dim=1)
            max_probs_main, pseudo_label  = torch.max(ema_output_main_soft, dim=1)
            
            unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
            unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
            onesWeights = torch.ones((unlabeledWeight.shape)).cuda()

            MixMask, loss_mask = generate_mask(images_target.cuda())
            pixelWiseWeight = unlabeledWeight * MixMask + onesWeights * (1 - MixMask)
            images_classmix = images_target.cuda() * MixMask + images_source.cuda() * (1 - MixMask)
            labels_classmix = pseudo_label.cuda() * MixMask + labels_source.cuda() * (1 - MixMask)

    
        if cfg.TRAIN.CONSWITCH:
            consistency_weight = get_current_consistency_weight(i_iter, cfg.TRAIN.MAX_ITERS)
        else:
            consistency_weight = 1


        source_feas,pred_src_aux, pred_src_main = model(images_source.cuda())
        target_feas,pred_trg_aux, pred_trg_main = model(images_target.cuda())
        _,pred_mix_aux, pred_mix_main = model(images_classmix.cuda())

        # class_center_feas = update_class_center_iter(source_feas, labels_source, class_center_feas,m=0.2)
        # hard_pixel_label,pixel_mask = generate_pseudo_label(target_feas, class_center_feas, cfg)

        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux     = interp(pred_src_aux)
            pred_trg_aux     = interp(pred_trg_aux)
            pred_mix_aux     = interp(pred_mix_aux)

            loss_seg_src_aux = loss_calc(pred_src_aux,labels_source)
            loss_dice_aux    = dice_loss(pred_src_aux,labels_source.cuda())
        else:
            loss_seg_src_aux = 0
            loss_dice_aux    = 0

        pred_src_main     = interp(pred_src_main)
        pred_trg_main     = interp(pred_trg_main)
        pred_mix_main     = interp(pred_mix_main)
        loss_seg_src_main = loss_calc(pred_src_main,labels_source)
        loss_dice_main    = dice_loss(pred_src_main,labels_source.cuda())



        mpcl_loss_tr           = mpcl_loss_calc(feas=source_feas,labels=labels_source,
                                                            class_center_feas=class_center_feas,
                                                            loss_func=mpcl_loss_src,tag='source')

        mpcl_loss_tg = mpcl_loss_calc(feas=target_feas, labels=pseudo_label_feas,
                                                 class_center_feas=class_center_feas,
                                                 loss_func=mpcl_loss_trg,
                                                 pixel_sel_loc=tg_mask, tag='target')



        L_u_aux =  consistency_weight * unlabeled_loss(pred_mix_aux, labels_classmix, pixelWiseWeight)
        L_u_main = consistency_weight * unlabeled_loss(pred_mix_main, labels_classmix, pixelWiseWeight)
        # L_dice_aux =  consistency_weight * dice_loss(pred_mix_aux, labels_classmix)
        # L_dice_main = consistency_weight * dice_loss(pred_mix_main, labels_classmix)

        consistency_loss = torch.mean((torch.softmax(pred_trg_main, dim=1) - ema_output_main_soft)**2)
        consistency_aux_loss = torch.mean((torch.softmax(pred_trg_aux, dim=1) - ema_output_aux_soft)**2)

        loss = (cfg.TRAIN.LAMBDA_SEG_SRC_MAIN    * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_SRC_AUX   * loss_seg_src_aux
                + cfg.TRAIN.LAMBDA_DICE_SRC_MAIN * loss_dice_main
                + cfg.TRAIN.LAMBDA_DICE_SRC_AUX  * loss_dice_aux
                + consistency_weight * consistency_loss 
                + 0.1 * consistency_weight * consistency_aux_loss
                + L_u_main
                + 0.1 * L_u_aux
                + mpcl_loss_tr
                + 0.1*mpcl_loss_tg)
                # + 0.1 * L_dice_aux
                # + L_dice_main
                
        
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, 0.99, i_iter)

        current_losses = {'loss_seg_src_aux' :loss_seg_src_aux,
                          'loss_seg_src_main':loss_seg_src_main,
                        #   'loss_target_entp_aux':loss_target_entp_aux,
                        #   'loss_target_entp_main':loss_target_entp_aux,
                          'loss_dice_aux': loss_dice_aux,
                          'loss_dice_main'   :loss_dice_main}
        print_losses(current_losses,i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            if cfg.DATASETS == 'Cardiac':
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Cardiac(model, cfg.TARGET, results_file, i_iter)
            else:
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Abdomen(model, cfg.TARGET, results_file, i_iter)
            
            # saved_path = "/home/data_backup/zhr_savedmodel/{0}2{1}_{2}_ourdice_myweight".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
            # os.makedirs(saved_path, exist_ok=True)
            # torch.save(model.state_dict(),  saved_path+f'/model_{i_iter}.pth')
            # tmp_dice = np.mean(dice_mean)
                        
            # if best_model < tmp_dice:
            #     print('taking snapshot ...')
            #     print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            #     snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            #     torch.save(model.state_dict(),  snapshot_dir / '{0}_{1}2{2}_Best_CutMix_ourdice_model.pth'.format(cfg.DATASETS, cfg.SOURCE, cfg.TARGET))
            #     best_model = tmp_dice

        sys.stdout.flush()


def train_cutmix_ST_contrastive(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg):
    '''
    UDA training
    '''

    results_file = "/home/zhr/ICME/scripts/log/{0}2{1}_{2}_mpl_0.3.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)

    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    num_classes       = cfg.NUM_CLASSES


    unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
    # dice_loss = DiceLoss(n_classes=num_classes).cuda()

    model.train()
    ema_model.train()

    model.cuda()
    ema_model.cuda()
    cudnn.benchmark = True
    cudnn.enabled   = True

    #optimized
    if cfg.TRAIN.OPTIM_G == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.TRAIN.LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                              lr=cfg.TRAIN.LEARNING_RATE,
                              momentum=cfg.TRAIN.MOMENTUM,
                              weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    
    class_center_feas = np.load(cfg.TRAIN.CLASS_CENTER_FEA_INIT).squeeze()
    class_center_feas = torch.from_numpy(class_center_feas).float().cuda()
    
    # interpolate output segmaps
    interp        = nn.Upsample(size=(input_size_source[1],input_size_source[0]),mode='bilinear',
                         align_corners=True)

    # labels for adversarial learning


    strain_loader_iter          = enumerate(strain_loader)
    trgtrain_loader_iter        = enumerate(trgtrain_loader)
    sval_loader_iter            = enumerate(sval_loader)

    transforms = None
    img_mean   = cfg.TRAIN.IMG_MEAN
    best_model =  -1

    memobank = [[] for i in range(num_classes)]
    queue_len = 256
    temp = 1

    for i_iter in tqdm(range(cfg.TRAIN.MAX_ITERS+1)):

        model.train()
        ema_model.train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer,i_iter,cfg)


        #UDA training
        try:
            _, batch = strain_loader_iter.__next__()
        except StopIteration:
            strain_loader_iter = enumerate(strain_loader)
            _, batch = strain_loader_iter.__next__()
        images_source, labels_source,_ = batch

        try:
            _, batch = trgtrain_loader_iter.__next__()
        except StopIteration:
            trgtrain_loader_iter = enumerate(trgtrain_loader)
            _, batch             = trgtrain_loader_iter.__next__()
        images_target, labels_target,_ = batch

        # src_in_trg = []
        # for i in range(cfg.TRAIN.BATCH_SIZE):
        #     st = match_histograms(np.array(images_source[i]), np.array(images_target[i]), channel_axis=0)
        #     src_in_trg.append(st)
        # images_source = torch.tensor(src_in_trg, dtype=torch.float32)

        noise = torch.clamp(torch.randn_like(
                images_target) * 0.1, -0.2, 0.2)
        ema_inputs = images_target + noise

        with torch.no_grad():
            target_teacher_feas, ema_pred_trg_axu, ema_pred_trg_main = ema_model(ema_inputs.cuda())
            source_teacher_feas, ema_pred_src_axu, ema_pred_src_main = ema_model(images_source.cuda())
            ema_output_main_soft_feas = torch.softmax(ema_pred_trg_main, dim=1)
            max_probs_main_feas, pseudo_label_feas  = torch.max(ema_output_main_soft_feas, dim=1)
            tg_mask = max_probs_main_feas.ge(0.7).long() 
            class_center_feas = update_class_center_iter(source_teacher_feas, labels_source, class_center_feas,m=0.2)
            memobank = get_negative(source_teacher_feas, labels_source, num_classes, memobank, queue_len, class_center_feas)
            hard_pixel_label,pixel_mask = generate_pseudo_label(target_teacher_feas, class_center_feas, cfg)

            ema_pred_trg_axu     = interp(ema_pred_trg_axu)
            ema_pred_trg_main     = interp(ema_pred_trg_main)

            ema_output_main_soft = torch.softmax(ema_pred_trg_main, dim=1)
            ema_output_aux_soft = torch.softmax(ema_pred_trg_axu, dim=1)
            max_probs_main, pseudo_label  = torch.max(ema_output_main_soft, dim=1)
            
            unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
            unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
            onesWeights = torch.ones((unlabeledWeight.shape)).cuda()

            MixMask, loss_mask = generate_mask(images_target.cuda())
            pixelWiseWeight = unlabeledWeight * MixMask + onesWeights * (1 - MixMask)
            images_classmix = images_target.cuda() * MixMask + images_source.cuda() * (1 - MixMask)
            labels_classmix = pseudo_label.cuda() * MixMask + labels_source.cuda() * (1 - MixMask)

            # how2mask = np.random.uniform(0, 1, 1)
            # if how2mask < 2:
            #     MixMask, loss_mask = generate_mask(images_target.cuda())

            #     images_classmix = images_target.cuda() * MixMask + images_source.cuda() * (1 - MixMask)
            #     labels_classmix = pseudo_label.cuda() * MixMask + labels_source.cuda() * (1 - MixMask)
            # else:
            #     MixMask, loss_mask = generate_mask(images_target.cuda())

            #     images_classmix = images_target.cuda() * (1 - MixMask) + images_source.cuda() * MixMask
            #     labels_classmix = pseudo_label.cuda() * (1 - MixMask) + labels_source.cuda() * MixMask


            # MixMask, loss_mask = generate_mask(images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda())

            # pixelWiseWeight_trg = unlabeledWeight[:cfg.TRAIN.BATCH_SIZE//2] * MixMask + onesWeights[:cfg.TRAIN.BATCH_SIZE//2] * (1 - MixMask)
            # images_trg_classmix = images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + images_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)
            # labels_trg_classmix = pseudo_label[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + labels_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)

            # MixMask, loss_mask = generate_mask(images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda())
            
            # pixelWiseWeight_src = unlabeledWeight[cfg.TRAIN.BATCH_SIZE//2:] * (1 - MixMask) + onesWeights[cfg.TRAIN.BATCH_SIZE//2:] * MixMask
            # images_src_classmix = images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + images_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            # labels_src_classmix = pseudo_label[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + labels_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            
            # pixelWiseWeight = torch.cat([pixelWiseWeight_trg, pixelWiseWeight_src], dim=0)
            # images_classmix = torch.cat([images_trg_classmix, images_src_classmix], dim=0)
            # labels_classmix = torch.cat([labels_trg_classmix, labels_src_classmix], dim=0)
    
        if cfg.TRAIN.CONSWITCH:
            consistency_weight = get_current_consistency_weight(i_iter, cfg.TRAIN.MAX_ITERS)
        else:
            consistency_weight = 1

        # unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
        # unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
        # onesWeights = torch.ones((unlabeledWeight.shape)).cuda()
        # pixelWiseWeight = unlabeledWeight


        cla_feas_src,pred_src_aux, pred_src_main = model(images_source.cuda())
        cla_feas_trg,pred_trg_aux, pred_trg_main = model(images_target.cuda())
        _,pred_mix_aux, pred_mix_main = model(images_classmix.cuda())




        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux     = interp(pred_src_aux)
            pred_trg_aux     = interp(pred_trg_aux)
            pred_mix_aux     = interp(pred_mix_aux)

            loss_seg_src_aux = loss_calc(pred_src_aux,labels_source)
            loss_dice_aux    = dice_loss(pred_src_aux,labels_source.cuda())
        else:
            loss_seg_src_aux = 0
            loss_dice_aux    = 0

        pred_src_main     = interp(pred_src_main)
        pred_trg_main     = interp(pred_trg_main)
        pred_mix_main     = interp(pred_mix_main)
        loss_seg_src_main = loss_calc(pred_src_main,labels_source)
        loss_dice_main    = dice_loss(pred_src_main,labels_source.cuda())


        L_u_aux =  consistency_weight * unlabeled_loss(pred_mix_aux, labels_classmix, pixelWiseWeight)
        L_u_main = consistency_weight * unlabeled_loss(pred_mix_main, labels_classmix, pixelWiseWeight)
        # L_dice_aux =  consistency_weight * dice_loss(pred_mix_aux, labels_classmix)
        # L_dice_main = consistency_weight * dice_loss(pred_mix_main, labels_classmix)

        consistency_loss = torch.mean((torch.softmax(pred_trg_main, dim=1) - ema_output_main_soft)**2)
        consistency_aux_loss = torch.mean((torch.softmax(pred_trg_aux, dim=1) - ema_output_aux_soft)**2)

        regional_contrastive_s = regional_contrastive_cos(cla_feas_src, labels_source, class_center_feas, memobank, num_classes, temp)
        regional_contrastive_r = regional_contrastive_cos(cla_feas_trg, hard_pixel_label, class_center_feas, memobank, num_classes, temp, pixel_mask)

        # regional_contrastive_r = regional_contrastive_cos(cla_feas_trg, pseudo_label_feas, class_center_feas, memobank, num_classes, temp, tg_mask)

        loss = (cfg.TRAIN.LAMBDA_SEG_SRC_MAIN    * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_SRC_AUX   * loss_seg_src_aux
                + cfg.TRAIN.LAMBDA_DICE_SRC_MAIN * loss_dice_main
                + cfg.TRAIN.LAMBDA_DICE_SRC_AUX  * loss_dice_aux
                + consistency_weight * consistency_loss 
                + 0.1 * consistency_weight * consistency_aux_loss
                + L_u_main
                + 0.1 * L_u_aux
                + 0.3*regional_contrastive_s
                + 0.03*regional_contrastive_r)
        
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, 0.99, i_iter)

        current_losses = {'loss_seg_src_aux' :loss_seg_src_aux,
                          'loss_seg_src_main':loss_seg_src_main,
                        #   'loss_target_entp_aux':loss_target_entp_aux,
                        #   'loss_target_entp_main':loss_target_entp_aux,
                          'loss_dice_aux': loss_dice_aux,
                          'loss_dice_main'   :loss_dice_main,
                          'regional_contrastive_s' :0.3*regional_contrastive_s,
                          'regional_contrastive_r' :0.03*regional_contrastive_r}
        print_losses(current_losses,i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            if cfg.DATASETS == 'Cardiac':
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Cardiac(model, cfg.TARGET, results_file, i_iter)
            else:
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Abdomen(model, cfg.TARGET, results_file, i_iter)
            
            # saved_path = "/home/data_backup/zhr_savedmodel/{0}2{1}_{2}_ourdice_myweight".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
            # os.makedirs(saved_path, exist_ok=True)
            # torch.save(model.state_dict(),  saved_path+f'/model_{i_iter}.pth')
            # tmp_dice = np.mean(dice_mean)
                        
            # if best_model < tmp_dice:
            #     print('taking snapshot ...')
            #     print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            #     snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            #     torch.save(model.state_dict(),  snapshot_dir / '{0}_{1}2{2}_Best_CutMix_ourdice_model.pth'.format(cfg.DATASETS, cfg.SOURCE, cfg.TARGET))
            #     best_model = tmp_dice

        sys.stdout.flush()


def train_bcutmix_ST_contrastive(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg):
    '''
    UDA training
    '''

    results_file = "/home/zhr/ICME/scripts/log/{0}2{1}_{2}_0.11.txt".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)

    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    num_classes       = cfg.NUM_CLASSES


    unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
    # dice_loss = DiceLoss(n_classes=num_classes).cuda()

    model.train()
    ema_model.train()

    model.cuda()
    ema_model.cuda()
    cudnn.benchmark = True
    cudnn.enabled   = True

    #optimized
    if cfg.TRAIN.OPTIM_G == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg.TRAIN.LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                              lr=cfg.TRAIN.LEARNING_RATE,
                              momentum=cfg.TRAIN.MOMENTUM,
                              weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    
    class_center_feas = np.load(cfg.TRAIN.CLASS_CENTER_FEA_INIT).squeeze()
    class_center_feas = torch.from_numpy(class_center_feas).float().cuda()
    
    # interpolate output segmaps
    interp        = nn.Upsample(size=(input_size_source[1],input_size_source[0]),mode='bilinear',
                         align_corners=True)

    # labels for adversarial learning


    strain_loader_iter          = enumerate(strain_loader)
    trgtrain_loader_iter        = enumerate(trgtrain_loader)
    sval_loader_iter            = enumerate(sval_loader)

    transforms = None
    img_mean   = cfg.TRAIN.IMG_MEAN
    best_model =  -1

    memobank = [[] for i in range(num_classes)]
    queue_len = 256
    temp = 1

    for i_iter in tqdm(range(10001, cfg.TRAIN.MAX_ITERS+1)):

        model.train()
        ema_model.train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer,i_iter,cfg)


        #UDA training
        try:
            _, batch = strain_loader_iter.__next__()
        except StopIteration:
            strain_loader_iter = enumerate(strain_loader)
            _, batch = strain_loader_iter.__next__()
        images_source, labels_source,_ = batch

        try:
            _, batch = trgtrain_loader_iter.__next__()
        except StopIteration:
            trgtrain_loader_iter = enumerate(trgtrain_loader)
            _, batch             = trgtrain_loader_iter.__next__()
        images_target, labels_target,_ = batch

        # src_in_trg = []
        # for i in range(cfg.TRAIN.BATCH_SIZE):
        #     st = match_histograms(np.array(images_source[i]), np.array(images_target[i]), channel_axis=0)
        #     src_in_trg.append(st)
        # images_source = torch.tensor(src_in_trg, dtype=torch.float32)

        noise = torch.clamp(torch.randn_like(
                images_target) * 0.1, -0.2, 0.2)
        ema_inputs = images_target + noise

        with torch.no_grad():
            target_teacher_feas, ema_pred_trg_axu, ema_pred_trg_main = ema_model(ema_inputs.cuda())
            source_teacher_feas, ema_pred_src_axu, ema_pred_src_main = ema_model(images_source.cuda())
            ema_output_main_soft_feas = torch.softmax(ema_pred_trg_main, dim=1)
            max_probs_main_feas, pseudo_label_feas  = torch.max(ema_output_main_soft_feas, dim=1)
            
            drop_percent = 80
            percent_unreliable = (100 - drop_percent) * (1 - (i_iter -10000)/ cfg.TRAIN.MAX_ITERS)
            drop_percent = 100 - percent_unreliable
            entropy = -torch.sum(ema_output_main_soft_feas * torch.log(ema_output_main_soft_feas + 1e-10), dim=1)
            thresh = np.percentile(
                entropy.detach().cpu().numpy().flatten(), drop_percent
            )
            tg_mask = entropy.le(thresh).long() 
            # print(tg_mask.sum())
            # print(tg_mask.shape)

            class_center_feas = update_class_center_iter(source_teacher_feas, labels_source, class_center_feas,m=0.2)
            memobank = get_negative(source_teacher_feas, labels_source, num_classes, memobank, queue_len, class_center_feas)
            # hard_pixel_label,pixel_mask = generate_pseudo_label(target_teacher_feas, class_center_feas, cfg)

            ema_pred_trg_axu     = interp(ema_pred_trg_axu)
            ema_pred_trg_main     = interp(ema_pred_trg_main)

            ema_output_main_soft = torch.softmax(ema_pred_trg_main, dim=1)
            ema_output_aux_soft = torch.softmax(ema_pred_trg_axu, dim=1)
            max_probs_main, pseudo_label  = torch.max(ema_output_main_soft, dim=1)
            
            unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
            unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
            onesWeights = torch.ones((unlabeledWeight.shape)).cuda()

            MixMask, loss_mask = generate_mask(images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda())
            ## 源域贴在目标域
            pixelWiseWeight_trg = unlabeledWeight[:cfg.TRAIN.BATCH_SIZE//2] * MixMask + onesWeights[:cfg.TRAIN.BATCH_SIZE//2] * (1 - MixMask)
            images_trg_classmix = images_target[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + images_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)
            labels_trg_classmix = pseudo_label[:cfg.TRAIN.BATCH_SIZE//2].cuda() * MixMask + labels_source[:cfg.TRAIN.BATCH_SIZE//2].cuda() * (1 - MixMask)

            MixMask, loss_mask = generate_mask(images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda())
            ## 目标域贴在源域
            pixelWiseWeight_src = unlabeledWeight[cfg.TRAIN.BATCH_SIZE//2:] * (1 - MixMask) + onesWeights[cfg.TRAIN.BATCH_SIZE//2:] * MixMask
            images_src_classmix = images_target[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + images_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            labels_src_classmix = pseudo_label[cfg.TRAIN.BATCH_SIZE//2:].cuda() * (1 - MixMask) + labels_source[cfg.TRAIN.BATCH_SIZE//2:].cuda() * MixMask
            
            pixelWiseWeight = torch.cat([pixelWiseWeight_trg, pixelWiseWeight_src], dim=0)
            images_classmix = torch.cat([images_trg_classmix, images_src_classmix], dim=0)
            labels_classmix = torch.cat([labels_trg_classmix, labels_src_classmix], dim=0)
    
        if cfg.TRAIN.CONSWITCH:
            consistency_weight = get_current_consistency_weight(i_iter, cfg.TRAIN.MAX_ITERS)
        else:
            consistency_weight = 1

        # unlabeled_weight = torch.sum(max_probs_main.ge(cfg.TRAIN.THRESHOLD).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
        # unlabeledWeight = unlabeled_weight * torch.ones(max_probs_main.shape).cuda()
        # onesWeights = torch.ones((unlabeledWeight.shape)).cuda()
        # pixelWiseWeight = unlabeledWeight


        cla_feas_src,pred_src_aux, pred_src_main = model(images_source.cuda())
        cla_feas_trg,pred_trg_aux, pred_trg_main = model(images_target.cuda())
        _,pred_mix_aux, pred_mix_main = model(images_classmix.cuda())




        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux     = interp(pred_src_aux)
            pred_trg_aux     = interp(pred_trg_aux)
            pred_mix_aux     = interp(pred_mix_aux)

            loss_seg_src_aux = loss_calc(pred_src_aux,labels_source)
            loss_dice_aux    = dice_loss(pred_src_aux,labels_source.cuda())
        else:
            loss_seg_src_aux = 0
            loss_dice_aux    = 0

        pred_src_main     = interp(pred_src_main)
        pred_trg_main     = interp(pred_trg_main)
        pred_mix_main     = interp(pred_mix_main)
        loss_seg_src_main = loss_calc(pred_src_main,labels_source)
        loss_dice_main    = dice_loss(pred_src_main,labels_source.cuda())


        L_u_aux =  consistency_weight * unlabeled_loss(pred_mix_aux, labels_classmix, pixelWiseWeight)
        L_u_main = consistency_weight * unlabeled_loss(pred_mix_main, labels_classmix, pixelWiseWeight)
        # L_dice_aux =  consistency_weight * dice_loss(pred_mix_aux, labels_classmix)
        # L_dice_main = consistency_weight * dice_loss(pred_mix_main, labels_classmix)

        consistency_loss = torch.mean((torch.softmax(pred_trg_main, dim=1) - ema_output_main_soft)**2)
        consistency_aux_loss = torch.mean((torch.softmax(pred_trg_aux, dim=1) - ema_output_aux_soft)**2)

        regional_contrastive_s = regional_contrastive_cos(cla_feas_src, labels_source, class_center_feas, memobank, num_classes, temp)
        # regional_contrastive_r = regional_contrastive_cos(cla_feas_trg, hard_pixel_label, class_center_feas, memobank, num_classes, temp, pixel_mask)

        regional_contrastive_r = regional_contrastive_cos(cla_feas_trg, pseudo_label_feas, class_center_feas, memobank, num_classes, temp, tg_mask)

        loss = (cfg.TRAIN.LAMBDA_SEG_SRC_MAIN    * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_SRC_AUX   * loss_seg_src_aux
                + cfg.TRAIN.LAMBDA_DICE_SRC_MAIN * loss_dice_main
                + cfg.TRAIN.LAMBDA_DICE_SRC_AUX  * loss_dice_aux
                + consistency_weight * consistency_loss 
                + 0.1 * consistency_weight * consistency_aux_loss
                + L_u_main
                + 0.1 * L_u_aux
                + 0.1*regional_contrastive_s
                + 0.01*regional_contrastive_r)
        
        loss.backward()
        optimizer.step()
        update_ema_variables(model, ema_model, 0.99, i_iter)

        current_losses = {'loss_seg_src_aux' :loss_seg_src_aux,
                          'loss_seg_src_main':loss_seg_src_main,
                        #   'loss_target_entp_aux':loss_target_entp_aux,
                        #   'loss_target_entp_main':loss_target_entp_aux,
                          'loss_dice_aux': loss_dice_aux,
                          'loss_dice_main'   :loss_dice_main,
                          'regional_contrastive_s' :0.1*regional_contrastive_s,
                          'regional_contrastive_r' :0.01*regional_contrastive_r}
        print_losses(current_losses,i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            if cfg.DATASETS == 'Cardiac':
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Cardiac(model, cfg.TARGET, results_file, i_iter)
            else:
                dice_mean,dice_std,assd_mean,assd_std = evaluation_Abdomen(model, cfg.TARGET, results_file, i_iter)
            
            # saved_path = "/home/data_backup/zhr_savedmodel/{0}2{1}_{2}_ourdice_myweight".format(cfg.SOURCE, cfg.TARGET, cfg.TRAIN.DA_METHOD)
            # os.makedirs(saved_path, exist_ok=True)
            # torch.save(model.state_dict(),  saved_path+f'/model_{i_iter}.pth')
            # tmp_dice = np.mean(dice_mean)
                        
            # if best_model < tmp_dice:
            #     print('taking snapshot ...')
            #     print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            #     snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            #     torch.save(model.state_dict(),  snapshot_dir / '{0}_{1}2{2}_Best_CutMix_ourdice_model.pth'.format(cfg.DATASETS, cfg.SOURCE, cfg.TARGET))
            #     best_model = tmp_dice

        sys.stdout.flush()






def train_domain_adaptation(model, ema_model, strain_loader, trgtrain_loader, sval_loader, cfg, strain_loader_ = None):

    if cfg.TRAIN.DA_METHOD == 'cutmix_ST':
        train_cutmix_ST(model, ema_model, strain_loader, sval_loader,trgtrain_loader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'classmix_ST':
        train_classmix_ST(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'classmix_ST_new':
        train_classmix_ST_new(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'bclassmix_ST':
        train_bclassmix_ST(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'bcutmix_ST':
        train_bcutmix_ST(model, ema_model, strain_loader, sval_loader,trgtrain_loader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'cutmix_ST_gram':
        train_cutmix_ST_gram(model, ema_model, strain_loader, sval_loader,trgtrain_loader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'ST':
        train_ST(model, ema_model, strain_loader, sval_loader,trgtrain_loader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'ST_gram':
        train_ST_gram(model, ema_model, strain_loader, sval_loader,trgtrain_loader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'bcutmix_ST_gram':
        train_bcutmix_ST_gram(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'cutmix_ST_MPSCL':
        train_cutmix_ST_MPSCL(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'cutmix_ST_MPSCL_ourpseudolabel':
        train_cutmix_ST_MPSCL_ourpseudolabel(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'cutmix_ST_contrastive':
        train_cutmix_ST_contrastive(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'bcutmix_ST_MPSCL':
        train_bcutmix_ST_MPSCL(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'bcutmix_ST_MPSCL_ourpseudolabel':
        train_bcutmix_ST_MPSCL_ourpseudolabel(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'bcutmix_ST_contrastive':
        train_bcutmix_ST_contrastive(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'bcutmix_ST_nonemse':
        train_bcutmix_ST_nonemse(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'bcutmix_ST_new':
        train_bcutmix_ST_new(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'cutmix_ST_new':
        train_cutmix_ST_new(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'bcutmix_ST_MPSCL_ourpseudolabel_new':
        train_bcutmix_ST_MPSCL_ourpseudolabel_new(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'bcutmix_ST_MPSCL_ourpseudolabel_final':
        train_bcutmix_ST_MPSCL_ourpseudolabel_final(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg, strain_loader_)
    elif cfg.TRAIN.DA_METHOD == 'bcutmix_ST_contrastive_final':
        train_bcutmix_ST_contrastive_final(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg, strain_loader_)
    elif cfg.TRAIN.DA_METHOD == 'bcutmix_ST_contrastive_boundary':
        train_bcutmix_ST_contrastive_boundary(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg, strain_loader_)
    elif cfg.TRAIN.DA_METHOD == 'bcutmix_ST_contrastive_boundary_final':
        train_bcutmix_ST_contrastive_boundary_final(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg, strain_loader_)
    elif cfg.TRAIN.DA_METHOD == 'bclassmix_ST_contrastive_boundary':
        train_bclassmix_ST_contrastive_boundary(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg, strain_loader_)
    elif cfg.TRAIN.DA_METHOD == 'bcutmix_ST_contrastive_boundary_gram_final':      
        train_bcutmix_ST_contrastive_boundary_gram_final(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg, strain_loader_)
    elif cfg.TRAIN.DA_METHOD == 'contrast_final_10000':  
        train_contrast_final_10000(model, ema_model, strain_loader,sval_loader, trgtrain_loader, cfg, strain_loader_)