# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 11:39:54 2018

@author: carri
"""
# for testing case
from __future__ import division

import os
import numpy as np
import cv2
import pydicom

import scipy.misc 
import random
# from dataloaders.helpers import *
from torch.utils.data import Dataset

def flip(I,flip_p):
    if flip_p>0.5:
        return np.fliplr(I)
    else:
        return I

def scale_im(img_temp,scale):
    new_dims = (  int(img_temp.shape[0]*scale),  int(img_temp.shape[1]*scale)   )
    return cv2.resize(img_temp,new_dims).astype(float)

def scale_gt(img_temp,scale):
    new_dims = (  int(img_temp.shape[0]*scale),  int(img_temp.shape[1]*scale)   )
    return cv2.resize(img_temp,new_dims,interpolation = cv2.INTER_NEAREST).astype(float)

def my_crop(img,gt):
    H = int(0.9 * img.shape[0])
    W = int(0.9 * img.shape[1])
    H_offset = random.choice(range(img.shape[0] - H))
    W_offset = random.choice(range(img.shape[1] - W))
    H_slice = slice(H_offset, H_offset + H)
    W_slice = slice(W_offset, W_offset + W)
    img = img[H_slice, W_slice, :]
    gt = gt[H_slice, W_slice]
    
    return img, gt
class custom_db(Dataset):
    def __init__(self, train=True,
                 inputRes=None,
                 db_root_dir='/DAVIS-2016',
                 transform=None,
                 meanval=(104.00699, 116.66877, 122.67892),
                 seq_name=None, sample_range=10, scales = None):
        """Loads image to label pairs for tool pose estimation
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        """
        self.train = train
        self.range = sample_range
        self.inputRes = inputRes
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.meanval = meanval
        self.seq_name = seq_name
        self.scales = scales
        
        names_img = np.sort(os.listdir(os.path.join(db_root_dir)))
        img_list = list(map(lambda x: os.path.join(db_root_dir, x), names_img))
        self.img_list = img_list
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
        target, sequence_name = self.make_img_gt_pair(idx) #Testing time to split the frame
        target_id = idx
        seq_name1 = self.img_list[target_id].split('/')[-2] #Get video name
        sample = {'target': target, 'seq_name': sequence_name, 'search_0': None}
        if self.range>=1:
            my_index = self.Index[seq_name1]
            search_num = list(range(my_index[0], my_index[1]))  
            search_ids = random.sample(search_num, self.range)#min(len(self.img_list)-1, target_id+np.random.randint(1,self.range+1))
        
            for i in range(0,self.range):
                search_id = search_ids[i]
                search, sequence_name = self.make_img_gt_pair(search_id)
                if sample['search_0'] is None:
                    sample['search_0'] = search
                else:
                    sample['search'+'_'+str(i)] = search
            #np.save('search1.npy',search)
            #np.save('search_gt.npy',search_gt)
            if self.seq_name is not None:
                fname = os.path.join(self.seq_name, "%05d" % idx)
                sample['fname'] = fname
       
        else:
            img, gt = self.make_img_gt_pair(idx)
            sample = img
            

        return sample  #This is the last output
    def make_img_gt_pair(self, idx): #The meaning of this function is to serve the getitem function
        """
        Make the image-ground-truth pair
        """
        img = cv2.imread('./dataset/t2.jpg', cv2.IMREAD_COLOR)
        
        imgs = []
         ## The image and the corresponding ground truth have been read for data augmentation.
        for scale in self.scales:

            if self.inputRes is not None:
                input_res = (int(self.inputRes[0]*scale),int(self.inputRes[0]*scale))
                img1 = cv2.resize(img.copy(), input_res)

            img1 = np.array(img1, dtype=np.float32)
            #img = img[:, :, ::-1]
            img1 = np.subtract(img1, np.array(self.meanval, dtype=np.float32))
            img1 = img1.transpose((2, 0, 1))  # NHWC -> NCHW
            imgs.append(img1)

                #gt = gt/np.max([gt.max(), 1e-8])
        #np.save('gt.npy')
        sequence_name = self.img_list[idx].split('/')[2]
        return imgs, sequence_name

       
class PairwiseImg(Dataset):
    """DAVIS 2016 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, train=True,
                 inputRes=None,
                 db_root_dir='/DAVIS-2016',
                 transform=None,
                 meanval=(104.00699, 116.66877, 122.67892),
                 seq_name=None, sample_range=10, scales = None):
        """Loads image to label pairs for tool pose estimation
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        """
        self.train = train
        self.range = sample_range
        self.inputRes = inputRes
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.meanval = meanval
        self.seq_name = seq_name
        self.scales = scales
        if self.train:
            fname = 'train_seqs'
        else:
            fname = 'val_seqs'

        if self.seq_name is None: #All datasets are involved in training
            with open(os.path.join(db_root_dir, fname + '.txt')) as f:
                seqs = f.readlines()
                img_list = []
                labels = []
                Index = {}
                for seq in seqs:                    
                    images = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', seq.strip('\n'))))
                    images_path = list(map(lambda x: os.path.join('JPEGImages/480p/', seq.strip(), x).replace('\\', '/'), images))
                    start_num = len(img_list)
                    img_list.extend(images_path)
                    end_num = len(img_list)
                    Index[seq.strip('\n')]= np.array([start_num, end_num])
                    lab = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', seq.strip('\n'))))
                    lab_path = list(map(lambda x: os.path.join('Annotations/480p/', seq.strip(), x), lab))
                    labels.extend(lab_path)
        elif seq_name=='my_db':
                     # Initialize the per sequence images for online training
            names_img = np.sort(os.listdir(os.path.join(db_root_dir)))
            img_list = list(map(lambda x: os.path.join(db_root_dir, x), names_img))
            #name_label = np.sort(os.listdir(os.path.join(db_root_dir,  str(seq_name))))
            labels = [os.path.join( (str(seq_name)+'/saliencymaps'), names_img[0])]
            labels.extend([None]*(len(names_img)-1)) #Add the element None after the labels list
            # if self.train:
            #     img_list = [img_list[0]]
            #     labels = [labels[0]]    
        else: #for all training samples， img_list :The path where the image is stored

            # Initialize the per sequence images for online training
            names_img = np.sort(os.listdir(os.path.join(db_root_dir, str(seq_name))))
            img_list = list(map(lambda x: os.path.join(( str(seq_name)), x), names_img))
            #name_label = np.sort(os.listdir(os.path.join(db_root_dir,  str(seq_name))))
            labels = [os.path.join( (str(seq_name)+'/saliencymaps'), names_img[0])]
            labels.extend([None]*(len(names_img)-1)) #Add the element None after the labels list
            if self.train:
                img_list = [img_list[0]]
                labels = [labels[0]]

        #assert (len(labels) == len(img_list))

        self.img_list = img_list
        self.labels = labels
        self.Index = Index
        #img_files = open('all_im.txt','w+')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        target, sequence_name = self.make_img_gt_pair(idx) #Testing time to split the frame
        target_id = idx
        seq_name1 = self.img_list[target_id].split('/')[-2] #获取视频名称
        sample = {'target': target, 'seq_name': sequence_name, 'search_0': None}
        if self.range>=1:
            my_index = self.Index[seq_name1]
            search_num = list(range(my_index[0], my_index[1]))  
            search_ids = random.sample(search_num, self.range)#min(len(self.img_list)-1, target_id+np.random.randint(1,self.range+1))
        
            for i in range(0,self.range):
                search_id = search_ids[i]
                search, sequence_name = self.make_img_gt_pair(search_id)
                if sample['search_0'] is None:
                    sample['search_0'] = search
                else:
                    sample['search'+'_'+str(i)] = search
            #np.save('search1.npy',search)
            #np.save('search_gt.npy',search_gt)
            if self.seq_name is not None:
                fname = os.path.join(self.seq_name, "%05d" % idx)
                sample['fname'] = fname
       
        else:
            img, gt = self.make_img_gt_pair(idx)
            sample = {'image': img, 'gt': gt}
            if self.seq_name is not None:
                fname = os.path.join(self.seq_name, "%05d" % idx)
                sample['fname'] = fname

        return sample  #这个类最后的输出

    def make_img_gt_pair(self, idx): #这个函数存在的意义是为了getitem函数服务的
        """
        Make the image-ground-truth pair
        """
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[idx]), cv2.IMREAD_COLOR)

        imgs = []
         ## Already read the image and the corresponding ground truth, data augmentation can be performed
        for scale in self.scales:

            if self.inputRes is not None:
                input_res = (int(self.inputRes[0]*scale),int(self.inputRes[0]*scale))
                img1 = cv2.resize(img.copy(), input_res)

            img1 = np.array(img1, dtype=np.float32)
            #img = img[:, :, ::-1]
            img1 = np.subtract(img1, np.array(self.meanval, dtype=np.float32))
            img1 = img1.transpose((2, 0, 1))  # NHWC -> NCHW
            imgs.append(img1)

                #gt = gt/np.max([gt.max(), 1e-8])
        #np.save('gt.npy')
        sequence_name = self.img_list[idx].split('/')[2]
        return imgs, sequence_name

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[0]))
        
        return list(img.shape[:2])


class PairwiseImgDicom(Dataset):
    """DAVIS 2016 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, train=True,
                 inputRes=None,
                 db_root_dir='./dataset/NSCLC-Radiomics/',
                 img_root_dir=None,
                 transform=None,
                 meanval=(114.45156),
                 seq_name=None, sample_range=10, scales = None):
        """Loads image to label pairs for tool pose estimation
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        """
        self.train = train
        self.range = sample_range
        self.inputRes = inputRes
        self.img_root_dir = img_root_dir
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.meanval = meanval
        self.seq_name = seq_name
        self.scales=scales
        lab_root = './dataset/NSCLC-binary-mask/2D'
        if self.train:
            fname = 'train_seqs'
        else:
            fname = 'val_seqs'
        patients = os.listdir(db_root_dir)
        patients=['LUNG1-001']
        image_list = []
        Index = {}
        labels = []
        for item in patients:

            item_in1 = os.listdir(db_root_dir + item)

            item_in2 = os.listdir(db_root_dir + item + '/' + item_in1[0])
            segment_addr = ''
            for i in range(0, len(item_in2)):
                state = item_in2[i].find('NA')
                if (state != -1):
                    data_folder = i
                    break
            raw_addr = db_root_dir + item + '/' + item_in1[0] + '/' + item_in2[data_folder]
            dicom_images = os.listdir(raw_addr)
            images_path = list(map(lambda x: raw_addr + '/' + x, dicom_images))
            start_num = len(image_list)
            image_list.extend(images_path)
            end_num = len(image_list)
            Index[item.strip('\n')] = np.array([start_num, end_num])
            lab = np.sort(os.listdir(os.path.join(lab_root, item)))
            lab_path = list(map(lambda x: lab_root + '/' + item + '/' + x, lab))
            labels.extend(lab_path)

            assert (len(labels) == len(image_list))

        self.img_list = image_list
        self.labels = labels
        self.Index = Index
        # img_files = open('all_im.txt','w+')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        target, target_name = self.make_img_gt_pair(idx)  # Testing time to split the frame
        target_id = idx
        seq_name1 = self.img_list[target_id].split('/')[3]  # get video name
        sample = {'target': target,'target_name':target_name, 'seq_name': seq_name1, 'search_0': None}
        if self.range >= 1:
            my_index = self.Index[seq_name1]
            search_num = list(range(my_index[0], my_index[1]))
            search_ids = random.sample(search_num,
                                       self.range)  # min(len(self.img_list)-1, target_id+np.random.randint(1,self.range+1))

            for i in range(0, self.range):
                search_id = search_ids[i]
                search, search_name = self.make_img_gt_pair(search_id)
                if sample['search_0'] is None:
                    sample['search_0'] = search
                    sample['search_0_name'] = search_name
                else:
                    sample['search' + '_' + str(i)] = search
                    sample['search' + '_' + str(i)+'_name'] = search_name
            # np.save('search1.npy',search)
            # np.save('search_gt.npy',search_gt)
            if self.seq_name is not None:
                fname = os.path.join(self.seq_name, "%05d" % idx)
                sample['fname'] = fname

        else:
            img, gt = self.make_img_gt_pair(idx)
            sample = {'image': img, 'gt': gt}
            if self.seq_name is not None:
                fname = os.path.join(self.seq_name, "%05d" % idx)
                sample['fname'] = fname

        return sample  # The final output of this class

    def get_pixels_hu(self, slice):
        image = slice.pixel_array
        # Convert to int16 (from sometimes int16),
        # should be possible as values should always be low enough (<32k)
        image = image.astype(np.float32)

        # Set outside-of-scan pixels to 0
        # The intercept is usually -1024, so air is approximately 0
        image[image == -2000] = 0

        # Convert to Hounsfield units (HU)
        intercept = slice.RescaleIntercept
        slope = slice.RescaleSlope

        if slope != 1:
            image = slope * image.astype(np.float32)

        image += np.float32(intercept)

        return np.array(image, dtype=np.float32)

    def slice_windowing(self, slice, level=-600, window=1500):
        """
        Function to display an image slice
        Input is a numpy 2D array
        """
        max = level + window / 2
        min = level - window / 2
        slice = slice.clip(min, max)
        # plt.figure()
        # plt.imshow(slice.T, cmap="gray", origin="lower")
        # plt.savefig('L'+str(level)+'W'+str(window))
        return slice

    def make_img_gt_pair(self, idx):  # The meaning of this function is to serve the getitem function
        """
        Make the image-ground-truth pair
        """
        img = pydicom.dcmread(self.img_list[idx])
        name,ext=os.path.splitext(os.path.basename(self.img_list[idx]))
        # cv2.imshow('',img)
        # cv2.waitKey(0)
        img = self.get_pixels_hu(img)
        img = self.slice_windowing(img)

        imgs = []
        ## Already read the image and the corresponding ground truth, data augmentation can be performed
        for scale in self.scales:

            if self.inputRes is not None:
                input_res = (int(self.inputRes[0] * scale), int(self.inputRes[0] * scale))
                img1 = cv2.resize(img.copy(), input_res)

            img1 = np.array(img1, dtype=np.float32)
            # img = img[:, :, ::-1]
            img1 = np.subtract(img1, np.array(self.meanval, dtype=np.float32))
            img1=img1[np.newaxis,:,:]
            imgs.append(img1)

            # gt = gt/np.max([gt.max(), 1e-8])
        # np.save('gt.npy')
        return imgs, name

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[0]))

        return list(img.shape[:2])


if __name__ == '__main__':
    import custom_transforms as tr
    import torch
    from torchvision import transforms
    from matplotlib import pyplot as plt

    transforms = transforms.Compose([tr.RandomHorizontalFlip(), tr.Resize(scales=[0.5, 0.8, 1]), tr.ToTensor()])

    #dataset = DAVIS2016(db_root_dir='/media/eec/external/Databases/Segmentation/DAVIS-2016',
                       # train=True, transform=transforms)
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
#
#    for i, data in enumerate(dataloader):
#        plt.figure()
#        plt.imshow(overlay_mask(im_normalize(tens2image(data['image'])), tens2image(data['gt'])))
#        if i == 10:
#            break
#
#    plt.show(block=True)
