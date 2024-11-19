from torch.utils.data import Dataset
import numpy as np
from scipy import ndimage


class CTDataset(Dataset):
    def __init__(self,data_pth,gt_pth,img_mean,transform=None,PL_Tag=False):

        with open(data_pth, 'r') as fp:
            self.ct_image_list = fp.readlines()

        if PL_Tag:
            self.ct_gt_list = None
        else:
            with open(gt_pth, 'r') as fp:
                self.ct_gt_list = fp.readlines()
        self.transform      = transform
        self.img_mean       = img_mean
        self.gen_pl         = PL_Tag   # generate pseudo label or not
        self.get_boundary = GetBoundary()

    def __getitem__(self, index):

        if self.gen_pl:
            img_pth = self.ct_image_list[index][:-1]
            img     = self.gl_load_data(img_pth)
            gt      = np.zeros([img.shape[1],img.shape[2]],dtype=int)
        else:
            img_pth = self.ct_image_list[index][:-1]
            gt_pth  = self.ct_gt_list[index][:-1]
            img,gt  = self.load_data(img_pth,gt_pth)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = np.transpose(img, (2, 0, 1))  # 3*h*w

        gt = gt.astype(int)
        gt_boundary = self.get_boundary(gt) * 255
        gt_boundary = ndimage.gaussian_filter(gt_boundary, sigma=3) / 255.0

        return img, gt, gt_boundary

    def __len__(self):
        return len(self.ct_image_list)
    def load_data(self,img_pth, gt_pth):
        img = np.load(img_pth) # h*w*1
        gt  = np.load(gt_pth)  # h*w

        img = np.expand_dims(img,-1)
        img = np.tile(img,[1,1,3])  #h*w*3
        img = (img + 1) * 127.5
        img = img[:, :, ::-1].copy()  # change to BGR
        img -= self.img_mean
        return img, gt

    def gl_load_data(self,img_pth):
        img = np.load(img_pth)  # h*w*1
        img = np.expand_dims(img,-1)
        img = np.tile(img, [1, 1, 3])  # h*w*3
        img = (img + 1) * 127.5
        img = img[:, :, ::-1].copy()  # change to BGR
        img -= self.img_mean
        return img


class MRDataset(Dataset):
    def __init__(self, data_pth, gt_pth,img_mean, transform=None,PL_Tag=False):
        with open(data_pth, 'r') as fp:
            self.mr_image_list = fp.readlines()
        if PL_Tag:
            self.mr_gt_list = None

        else:
            with open(gt_pth, 'r') as fp:
                self.mr_gt_list = fp.readlines()
        self.transform      = transform
        self.img_mean       = img_mean
        self.gen_pl         = PL_Tag
        self.get_boundary = GetBoundary()

    def __getitem__(self, index):
        if self.gen_pl:
            img_pth = self.mr_image_list[index][:-1] 
            img = self.gl_load_data(img_pth)
            gt = np.zeros([img.shape[1], img.shape[2]], dtype=int)
        else:
            img_pth = self.mr_image_list[index][:-1] ## 最后一个字符是换行符，不要最后一个字符
            gt_pth = self.mr_gt_list[index][:-1]
            img, gt = self.load_data(img_pth, gt_pth)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = np.transpose(img, (2, 0, 1))  # 3*h*w

        gt = gt.astype(int)
        gt_boundary = self.get_boundary(gt) * 255
        gt_boundary = ndimage.gaussian_filter(gt_boundary, sigma=3) / 255.0

        return img, gt, gt_boundary

    def __len__(self):
        return len(self.mr_image_list)

    def load_data(self, img_pth, gt_pth):
        img = np.load(img_pth) # h*w*1
        gt  = np.load(gt_pth)
        img = np.expand_dims(img,-1)
        img = np.tile(img,[1,1,3]) # h*w*3
        img = (img + 1) * 127.5
        img = img[:, :, ::-1].copy()  # change to BGR
        img -= self.img_mean
        return img, gt

    def gl_load_data(self,img_pth):
        img = np.load(img_pth)  # h*w
        img = np.expand_dims(img,-1)# h*w*1
        img = np.tile(img, [1, 1, 3])  # h*w*3
        img = (img + 1) * 127.5
        img = img[:, :, ::-1].copy()  # change to BGR
        img -= self.img_mean
        return img



class GetBoundary(object):
    def __init__(self, width = 5):
        self.width = width
    def __call__(self, mask):

        # 将标签图像转换为one-hot编码
        num_classes = 4
        label = np.zeros((256, 256, num_classes), dtype=np.uint8)
        for i in range(num_classes):
            label[:, :, i] = (mask == (i+1)).astype(np.uint8)

        for i in range(num_classes):
            dila = ndimage.binary_dilation(label[:, :, i], iterations=self.width).astype(label.dtype)
            eros = ndimage.binary_erosion(label[:, :, i], iterations=self.width).astype(label.dtype)
            temp_mask = dila + eros
            temp_mask[temp_mask==2]=0
            
            if i == 0: 
                boundary = temp_mask
            else:
                boundary += temp_mask
                
        boundary = boundary > 0
        return boundary.astype(np.uint8)
    


class CTAbdomenDataset(Dataset):
    def __init__(self,data_pth,img_mean,transform=None):

        with open(data_pth, 'r') as fp:
            self.ct_list = fp.readlines()

        self.transform      = transform
        self.img_mean       = img_mean
        self.get_boundary = GetBoundary()

    def __getitem__(self, index):

        pth = self.ct_list[index][:-1]
        img,gt  = self.load_data(pth)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = np.transpose(img, (2, 0, 1))  # 3*h*w
        gt = gt.astype(int)
        gt_boundary = self.get_boundary(gt) * 255
        gt_boundary = ndimage.gaussian_filter(gt_boundary, sigma=3) / 255.0

        return img, gt, gt_boundary

    def __len__(self):
        return len(self.ct_list)
    def load_data(self,pth):
        _npz_dict = np.load(pth)
        img = _npz_dict['arr_0']# h*w
        gt = _npz_dict['arr_1']# h*w
        img = np.subtract(
            np.multiply(np.divide(np.subtract(img, -1.4), np.subtract(3.6,-1.4)), 2.0),
            1)  # {-2.8, 3.2} need to be changed according to the metadata statistics
        img = np.expand_dims(img,-1)
        img = np.tile(img,[1,1,3])  #h*w*3
        img = (img + 1) * 127.5
        img = img[:, :, ::-1].copy()  # change to BGR
        img -= self.img_mean
        return img, gt


class MRAbdomenDataset(Dataset):
    def __init__(self,data_pth,img_mean,transform=None):

        with open(data_pth, 'r') as fp:
            self.mr_list = fp.readlines()

        self.transform      = transform
        self.img_mean       = img_mean
        self.get_boundary = GetBoundary()

    def __getitem__(self, index):

        pth = self.mr_list[index][:-1]
        img,gt  = self.load_data(pth)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = np.transpose(img, (2, 0, 1))  # 3*h*w

        gt = gt.astype(int)
        gt_boundary = self.get_boundary(gt) * 255
        gt_boundary = ndimage.gaussian_filter(gt_boundary, sigma=3) / 255.0

        return img, gt, gt_boundary

    def __len__(self):
        return len(self.mr_list)
    def load_data(self,pth):
        _npz_dict = np.load(pth)
        img = _npz_dict['arr_0']# h*w
        gt = _npz_dict['arr_1']# h*w
        img = np.subtract(
            np.multiply(np.divide(np.subtract(img, -1.3), np.subtract(4.3,-1.3)), 2.0),
            1)  # {-1.8, 4.4} need to be changed according to the metadata statistics
        img = np.expand_dims(img,-1)
        img = np.tile(img,[1,1,3])  #h*w*3
        img = (img + 1) * 127.5
        img = img[:, :, ::-1].copy()  # change to BGR
        img -= self.img_mean
        return img, gt
