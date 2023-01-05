# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 11:39:54 2018

@author: carri
"""

from __future__ import division
from ast import Add
import matplotlib.pyplot as plt
import os
from pickletools import uint8
import numpy as np
import cv2

import scipy.misc
import random

# from dataloaders.helpers import *
from torch.utils.data import Dataset
from PIL import  Image
import torch
import pydicom

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
    img_temp=img_temp.astype(float)
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

def crop_dicom(img,gt):

    H = int(0.9 * img.shape[0])
    W = int(0.9 * img.shape[1])
    H_offset = random.choice(range(img.shape[0] - H))
    W_offset = random.choice(range(img.shape[1] - W))
    H_slice = slice(H_offset, H_offset + H)
    W_slice = slice(W_offset, W_offset + W)
    img = img[H_slice, W_slice]
    gt = gt[H_slice, W_slice]
    
    return img, gt



def crop(img):
    H = int(0.9 * img.shape[0])
    W = int(0.9 * img.shape[1])
    H_offset = random.choice(range(img.shape[0] - H))
    W_offset = random.choice(range(img.shape[1] - W))
    H_slice = slice(H_offset, H_offset + H)
    W_slice = slice(W_offset, W_offset + W)
    if(len(img.shape)==2):
        return img[H_slice, W_slice]
    return img[H_slice,W_slice,:]



class PairwiseImgFeatureResnet(Dataset):
    """DAVIS 2016 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, train=True,
                 inputRes=None,
                 db_root_dir='./features/features_video',
                 img_root_dir='./features/features_img',
                 transform=None,
                 meanval=(104.00699, 116.66877, 122.67892),
                 seq_name=None, sample_range=10):
        """Loads image to label pairs for tool pose estimation
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        """
        self.train = train
        self.range = sample_range
        self.inputRes = inputRes
        self.img_root_dir = './features/features_img'
        self.db_root_dir = './features/features_video'
        self.img_lab_dir='./images'
        self.transform = transform
        self.meanval = meanval
        self.seq_name = seq_name

        if self.train:
            fname = 'train_seqs'
        else:
            fname = 'val_seqs'
        if self.seq_name is None:  # All datasets are involved in training
            with open(os.path.join(db_root_dir, fname + '.txt')) as f:
                seqs = f.readlines()
                video_list = []
                labels = []
                Index = {}
                db_lab_dir='./dataset/DAVIS-2016'
                image_list = []
                im_label = []
                num_video_seq = 0;
                for seq in seqs:
                    images = np.sort(
                        os.listdir(os.path.join(db_root_dir, seq.strip('\n'),'fea').replace('\\', '/')))
                    images_path = list(
                        map(lambda x: os.path.join('./features/features_video/', seq.strip(),'fea', x).replace('\\', '/'), images))
                    start_num = len(video_list)
                    video_list.extend(images_path)
                    end_num = len(video_list)
                    Index[seq.strip('\n')] = np.array([start_num, end_num])
                    lab = np.sort(
                        os.listdir(os.path.join(db_lab_dir, 'Annotations/480p/', seq.strip('\n')).replace('\\', '/')))
                    lab_path = list(
                        map(lambda x: os.path.join(db_lab_dir,'Annotations/480p/', seq.strip(), x).replace('\\', '/'), lab))
                    labels.extend(lab_path)
                    num_video_seq = num_video_seq + len(images)
                print("====>num video seq==", num_video_seq)
                my_imgs=open('img_names.txt','w')
                cnt=0;
                with open('./images/saliency_data.txt') as f:
                    seqs = f.readlines()
                    # data_list = np.sort(os.listdir(db_root_dir))
                    for seq in seqs:  # All datasets
                        seq = seq.strip('\n')
                        images = np.sort(os.listdir(
                            os.path.join(img_root_dir, seq.strip(),'fea').replace('\\','/')))  # For a dataset, such as DUT
                        # Initialize the original DAVIS splits for training the parent network
                        images_path = list(map(lambda x: 
                                                os.path.join(img_root_dir,seq,'fea' , x).replace('\\', '/'), images))

                        image_list.extend(images_path)
                        
                        for i in images_path:
                            cnt+=1
                            my_imgs.write(str(cnt)+' '+ i+'\n')
                        lab = np.sort(
                            os.listdir(os.path.join(self.img_lab_dir, seq.strip()).replace('\\', '/') + '/saliencymaps'))
                        lab_path = list(map(lambda x: os.path.join((seq + '/saliencymaps'), x).replace('\\', '/'), lab))
                        im_label.extend(lab_path)
                        print("num of img in ", seq, " is", len(images_path), '\n num of corresponding labels is ',
                              len(lab_path))

        else:  # For all training samples, video_list stores the path of the image

            # Initialize the per sequence images for online training
            names_img = np.sort(os.listdir(os.path.join(db_root_dir, str(seq_name))))
            video_list = list(map(lambda x: os.path.join((str(seq_name)), x), names_img))
            # name_label = np.sort(os.listdir(os.path.join(db_root_dir,  str(seq_name))))
            labels = [os.path.join((str(seq_name) + '/saliencymaps'), names_img[0])]
            labels.extend([None] * (len(names_img) - 1))  # 在labels这个列表后面添加元素None
            if self.train:
                video_list = [video_list[0]]
                labels = [labels[0]]

        assert (len(labels) == len(video_list))

        self.video_list = video_list
        self.labels = labels
        self.image_list = image_list
        self.img_labels = im_label
        self.Index = Index
        # img_files = open('all_im.txt','w+')

        assert (len(labels) == len(video_list))

        self.video_list = video_list
        self.labels = labels
        self.image_list = image_list
        self.img_labels = im_label
        self.Index = Index
        # img_files = open('all_im.txt','w+')

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        target_fea,target_mask, target_gt = self.make_video_gt_pair(idx)
        img_idx = np.random.randint(1, len(self.image_list) - 1)
        img_idx1 = np.random.randint(1, len(self.image_list) - 1)
        # print('index', img_idx, img_idx1)
        seq_name1 = self.video_list[idx].split('/')[-3]  # get video name
        if self.train:
            my_index = self.Index[seq_name1]
            search_id = np.random.randint(my_index[0], my_index[
                1])  # min(len(self.video_list)-1, target_id+np.random.randint(1,self.range+1))
            search_id1 = np.random.randint(my_index[0], my_index[1])
            search_fea,search_mask, search_gt = self.make_video_gt_pair(search_id)
            search1_fea,search1_mask, search_gt1 = self.make_video_gt_pair(search_id1)
            img_fea,img_mask, img_gt = self.make_img_gt_pair(img_idx)
            img1_fea,img1_mask, img_gt1 = self.make_img_gt_pair(img_idx1)
            sample = {'target_fea_0': target_fea,'target_mask_0':target_mask, 'target_gt_0': target_gt, 
                      'target_fea_1': search_fea,'target_mask_1':search_mask, 'target_gt_1': search_gt,
                      'target_fea_2': search1_fea,'target_mask_2':search1_mask, 'target_gt_2': search_gt1,
                       'img_fea': img_fea,'img_mask':img_mask, 'img_gt': img_gt, 
                       'img1_fea': img1_fea,'img1_mask':img1_mask,'img_gt1': img_gt1}
            # np.save('search1.npy',search)
            # np.save('search_gt.npy',search_gt)
            if self.seq_name is not None:
                fname = os.path.join(self.seq_name, "%05d" % idx)
                sample['fname'] = fname

            if self.transform is not None:
                sample = self.transform(sample)

        else:
            img, gt = self.make_video_gt_pair(idx)
            sample = {'image': img, 'gt': gt}
            if self.seq_name is not None:
                fname = os.path.join(self.seq_name, "%05d" % idx)
                sample['fname'] = fname

        return sample  # 这个类最后的输出

    def make_video_gt_pair(self, idx):  # The purpose of this function is to serve the getitem function
        """
              Make the image-ground-truth pair
              """
        add_separated=self.video_list[idx].split('/')
        add_separated[-2]='mask'
        mask_add='/'.join(add_separated)
        img_fea = torch.load(self.video_list[idx].replace('\\', '/'))
        img_mask=torch.load(mask_add)
        if self.labels[idx] is not None and self.train:
            label = cv2.imread(self.labels[idx],cv2.IMREAD_GRAYSCALE)
            # print(os.path.join(self.db_root_dir, self.labels[idx]))
        else:
            print('finall here!!')
            pass
            # gt = np.zeros(img.shape[:-1], dtype=np.uint8)

        ## The image and the corresponding ground truth have been read for data augmentation.
        if self.train:  # scaling, cropping and flipping
            label = crop(label)
            scale = random.uniform(0.7, 1.3)
            flip_p = random.uniform(0, 1)
            gt_temp = scale_gt(label, scale)
            gt_temp = flip(gt_temp, flip_p)

            label = gt_temp

        if self.inputRes is not None:
            # print('ok1')
            # scipy.misc.imsave('label.png',label)
            # scipy.misc.imsave('img.png',img)
            if self.labels[idx] is not None and self.train:
                label = cv2.resize(label, self.inputRes, interpolation=cv2.INTER_NEAREST)

       
        if self.labels[idx] is not None and self.train:
            gt = np.array(label, dtype=np.int32)
            gt[gt != 0] = 1
            # gt = gt/np.max([gt.max(), 1e-8])
        # np.save('gt.npy')

        return img_fea,img_mask, gt

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.db_root_dir, self.video_list[0]))

        return list(img.shape[:2])

    def make_img_gt_pair(self, idx):  # The meaning of this function is to serve the getitem function
        """
        Make the image-ground-truth pair
        """

        
        add_separated=self.image_list[idx].split('/')
        add_separated[-2]='mask'
        mask_add='/'.join(add_separated)
        img_fea = torch.load(self.image_list[idx].replace('\\', '/'))
        img_mask=torch.load(mask_add)
        # print(os.path.join(self.db_root_dir, self.img_list[idx]))
        if self.img_labels[idx] is not None and self.train:
            label = cv2.imread(os.path.join(self.img_lab_dir, self.img_labels[idx]), cv2.IMREAD_GRAYSCALE)
            # print(os.path.join(self.db_root_dir, self.labels[idx]))
        else:
            pass
            # gt = np.zeros(img.shape[:-1], dtype=np.uint8)

        if self.inputRes is not None:
            if self.img_labels[idx] is not None and self.train:
                label = cv2.resize(label, self.inputRes, interpolation=cv2.INTER_NEAREST)

        

        if self.img_labels[idx] is not None and self.train:
            gt = np.array(label, dtype=np.int32)
            gt[gt != 0] = 1
            # gt = gt/np.max([gt.max(), 1e-8])
        # np.save('gt.npy')
        return img_fea,img_mask, gt


class PairwiseImg(Dataset):
    """DAVIS 2016 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, train=True,
                 inputRes=None,
                 db_root_dir='DAVIS-2016',
                 img_root_dir = None,
                 transform=None,
                 meanval=(104.00699, 116.66877, 122.67892),
                 seq_name=None, sample_range=10):
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

        if self.train:
            fname = 'train_seqs'
        else:
            fname = 'val_seqs'
        if self.seq_name is None:  #All datasets are involved in training
            with open(os.path.join(db_root_dir, fname + '.txt')) as f:
                seqs = f.readlines()
                video_list = []
                labels = []
                Index = {}

                image_list = []
                im_label = []
                num_video_seq=0;
                for seq in seqs:
                    images = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', seq.strip('\n')).replace('\\','/')))
                    images_path = list(map(lambda x: os.path.join('JPEGImages/480p/', seq.strip(), x).replace('\\','/'), images))
                    start_num = len(video_list)
                    video_list.extend(images_path)
                    end_num = len(video_list)
                    Index[seq.strip('\n')] = np.array([start_num, end_num])
                    lab = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', seq.strip('\n')).replace('\\','/')))
                    lab_path = list(map(lambda x: os.path.join('Annotations/480p/', seq.strip(), x).replace('\\','/'), lab))
                    labels.extend(lab_path)
                    num_video_seq=num_video_seq+len(images)
                with open('./images/saliency_data.txt') as f:
                    seqs = f.readlines()
                    # data_list = np.sort(os.listdir(db_root_dir))
                    for seq in seqs:  # All datasets
                        seq = seq.strip('\n')
                        images = np.sort(
                            os.listdir(os.path.join(img_root_dir, seq.strip()).replace('\\','/') + '/images/'))  # For a dataset, such as DUT
                        # Initialize the original DAVIS splits for training the parent network
                        images_path = list(map(lambda x: os.path.join((seq + '/images'), x).replace('\\','/'), images))

                        image_list.extend(images_path)
                        lab = np.sort(os.listdir(os.path.join(img_root_dir, seq.strip()).replace('\\','/') + '/saliencymaps'))
                        lab_path = list(map(lambda x: os.path.join((seq + '/saliencymaps'), x).replace('\\','/'), lab))
                        im_label.extend(lab_path)
                        print("num of img in ",seq," is",len(images_path),'\n num of corresponding labels is ',len(lab_path))

        else:  # For all training samples, video_list stores the path of the image

            # Initialize the per sequence images for online training
            names_img = np.sort(os.listdir(os.path.join(db_root_dir, str(seq_name))))
            video_list = list(map(lambda x: os.path.join((str(seq_name)), x), names_img))
            # name_label = np.sort(os.listdir(os.path.join(db_root_dir,  str(seq_name))))
            labels = [os.path.join((str(seq_name) + '/saliencymaps'), names_img[0])]
            labels.extend([None] * (len(names_img) - 1))  # 在labels这个列表后面添加元素None
            if self.train:
                video_list = [video_list[0]]
                labels = [labels[0]]

        assert (len(labels) == len(video_list))

        self.video_list = video_list
        self.labels = labels
        self.image_list = image_list
        self.img_labels = im_label
        self.Index = Index
        # img_files = open('all_im.txt','w+')

        

    def __len__(self):
        return len(self.video_list)
    
    def __getitem__(self, idx):
        target, target_gt = self.make_video_gt_pair(idx)
        img_idx = np.random.randint(1,len(self.image_list)-1)
        img_idx1 = np.random.randint(1, len(self.image_list) - 1)
        # print('index', img_idx, img_idx1)
        seq_name1 = self.video_list[idx].split('/')[-2] #get video name
        if self.train:
            my_index = self.Index[seq_name1]
            search_id = np.random.randint(my_index[0], my_index[1])#min(len(self.video_list)-1, target_id+np.random.randint(1,self.range+1))
            search_id1 = np.random.randint(my_index[0], my_index[1])
            search, search_gt = self.make_video_gt_pair(search_id)
            search1, search_gt1 = self.make_video_gt_pair(search_id1)
            img, img_gt = self.make_img_gt_pair(img_idx)
            img1, img_gt1 = self.make_img_gt_pair(img_idx1)
            sample = {'target_0': target, 'target_gt_0': target_gt, 'target_1': search, 'target_gt_1': search_gt,
                      'target_2': search1, 'target_gt_2': search_gt1, 'img': img, 'img_gt': img_gt, 'img1': img1, 'img_gt1': img_gt1}
            #np.save('search1.npy',search)
            #np.save('search_gt.npy',search_gt)
            if self.seq_name is not None:
                fname = os.path.join(self.seq_name, "%05d" % idx)
                sample['fname'] = fname

            if self.transform is not None:
                sample = self.transform(sample)
       
        else:
            img, gt = self.make_video_gt_pair(idx)
            sample = {'image': img, 'gt': gt}
            if self.seq_name is not None:
                fname = os.path.join(self.seq_name, "%05d" % idx)
                sample['fname'] = fname
        
        
        
        return sample  #这个类最后的输出

    def make_video_gt_pair(self, idx): #这个函数存在的意义是为了getitem函数服务的
        """
        Make the image-ground-truth pair
        """
        img = cv2.imread(os.path.join(self.db_root_dir, self.video_list[idx]), cv2.IMREAD_COLOR)
        # cv2.imshow('',img)
        # cv2.waitKey(0)
        if self.labels[idx] is not None and self.train:
            label = cv2.imread(os.path.join(self.db_root_dir, self.labels[idx]), cv2.IMREAD_GRAYSCALE)
            #print(os.path.join(self.db_root_dir, self.labels[idx]))
        else:
            gt = np.zeros(img.shape[:-1], dtype=np.uint8)
            
         ## 已经读取了image以及对应的ground truth可以进行data augmentation了
        if self.train:  #scaling, cropping and flipping
             img, label = my_crop(img,label)
             scale = random.uniform(0.7, 1.3)
             flip_p = random.uniform(0, 1)
             img_temp = scale_im(img,scale)
             img_temp = flip(img_temp,flip_p)
             gt_temp = scale_gt(label,scale)
             gt_temp = flip(gt_temp,flip_p)
             
             img = img_temp
             label = gt_temp
             
        if self.inputRes is not None:
            img = cv2.resize(img, self.inputRes)
            #print('ok1')
            #scipy.misc.imsave('label.png',label)
            #scipy.misc.imsave('img.png',img)
            if self.labels[idx] is not None and self.train:
                label = cv2.resize(label, self.inputRes, interpolation = cv2.INTER_NEAREST)

        img = np.array(img, dtype=np.float32)
        #img = img[:, :, ::-1]
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))        
        img = img.transpose((2, 0, 1))  # NHWC -> NCHW
        
        if self.labels[idx] is not None and self.train:
                gt = np.array(label, dtype=np.int32)
                gt[gt!=0]=1
                #gt = gt/np.max([gt.max(), 1e-8])
        #np.save('gt.npy')
        return img, gt


    def get_img_size(self):
        img = cv2.imread(os.path.join(self.db_root_dir, self.video_list[0]))
        
        return list(img.shape[:2])

    def make_img_gt_pair(self, idx): #这个函数存在的意义是为了getitem函数服务的
        """
        Make the image-ground-truth pair
        """
        img = cv2.imread(os.path.join(self.img_root_dir, self.image_list[idx]),cv2.IMREAD_COLOR)
        #print(os.path.join(self.db_root_dir, self.img_list[idx]))
        if self.img_labels[idx] is not None and self.train:
            label = cv2.imread(os.path.join(self.img_root_dir, self.img_labels[idx]),cv2.IMREAD_GRAYSCALE)
            #print(os.path.join(self.db_root_dir, self.labels[idx]))
        else:
            gt = np.zeros(img.shape[:-1], dtype=np.uint8)
            
        if self.inputRes is not None:            
            img = cv2.resize(img, self.inputRes)
            if self.img_labels[idx] is not None and self.train:
                label = cv2.resize(label, self.inputRes, interpolation = cv2.INTER_NEAREST)

        img = np.array(img, dtype=np.float32)
        #img = img[:, :, ::-1]
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))        
        img = img.transpose((2, 0, 1))  # NHWC -> NCHW
        
        if self.img_labels[idx] is not None and self.train:
                gt = np.array(label, dtype=np.int32)
                gt[gt!=0]=1
                #gt = gt/np.max([gt.max(), 1e-8])
        #np.save('gt.npy')
        return img, gt


class PairwiseDicomImg(Dataset):
    """DAVIS 2016 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, train=True,
                 inputRes=None,
                 db_root_dir='./dataset/NSCLC-Radiomics/',
                 img_root_dir = None,
                 transform=None,
                 meanval=(114.45156),
                 seq_name=None, sample_range=10):
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
        lab_root='./dataset/NSCLC-binary-mask/2D'
        if self.train:
            fname = 'train_seqs'
        else:
            fname = 'val_seqs'
        patients=os.listdir(db_root_dir)  
        image_list=[]  
        Index = {}
        labels = []
        for item in patients:
        
            item_in1=os.listdir(db_root_dir+item)
            
            item_in2=os.listdir(db_root_dir+item+'/'+item_in1[0])
            segment_addr=''
            for i in range(0,len(item_in2)):
                state=item_in2[i].find('NA')
                if(state!=-1):
                    data_folder=i
                    break
            raw_addr=db_root_dir+item+'/'+item_in1[0]+'/'+item_in2[data_folder]
            dicom_images=os.listdir(raw_addr)
            images_path = list(map(lambda x: raw_addr+'/'+x, dicom_images))
            start_num=len(image_list)
            image_list.extend(images_path)
            end_num = len(image_list)
            Index[item.strip('\n')] = np.array([start_num, end_num])
            lab = np.sort(os.listdir(os.path.join(lab_root,item)))
            lab_path = list(map(lambda x: lab_root+'/'+item+'/'+x, lab))
            labels.extend(lab_path)
           

            assert (len(labels) == len(image_list))

        
        self.image_list = image_list
        self.labels = labels
        self.Index = Index
            # img_files = open('all_im.txt','w+')

            

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        target, target_gt = self.make_dicom_gt_pair(idx)
       
        seq_name1 = self.image_list[idx].split('/')[-4] #get video name
        if self.train:
            my_index = self.Index[seq_name1]
            search_id = np.random.randint(my_index[0], my_index[1])#min(len(self.video_list)-1, target_id+np.random.randint(1,self.range+1))
            search_id1 = np.random.randint(my_index[0], my_index[1])
            search, search_gt = self.make_dicom_gt_pair(search_id)
            search1, search_gt1 = self.make_dicom_gt_pair(search_id1)
            
            sample = {'target_0': target, 'target_gt_0': target_gt, 'target_1': search, 'target_gt_1': search_gt,
                      'target_2': search1, 'target_gt_2': search_gt1}
            #np.save('search1.npy',search)
            #np.save('search_gt.npy',search_gt)
            if self.seq_name is not None:
                fname = os.path.join(self.seq_name, "%05d" % idx)
                sample['fname'] = fname

            if self.transform is not None:
                sample = self.transform(sample)
       
        else:
            img, gt = self.make_video_gt_pair(idx)
            sample = {'image': img, 'gt': gt}
            if self.seq_name is not None:
                fname = os.path.join(self.seq_name, "%05d" % idx)
                sample['fname'] = fname
        
        
        
        return sample  #这个类最后的输出
    def dicom_normalizer(self,img):
        img = np.double(img)
        min,max=np.min(img),np.max(img)
        
        #to set minimum value =0
        img=img+abs(min)
        
        img=img/(max+abs(min))
        return img
    def get_pixels_hu(self,slice):
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
        


    def slice_windowing(self,slice, level=-600, window=1500):
        """
        Function to display an image slice
        Input is a numpy 2D array
        """
        max = level + window/2
        min = level - window/2
        slice = slice.clip(min,max)
        # plt.figure()
        # plt.imshow(slice.T, cmap="gray", origin="lower")
        # plt.savefig('L'+str(level)+'W'+str(window))
        return slice

    def make_dicom_gt_pair(self, idx): #这个函数存在的意义是为了getitem函数服务的
        """
        Make the image-ground-truth pair
        """
        img = pydicom.dcmread(self.image_list[idx])
        # cv2.imshow('',img)
        # cv2.waitKey(0)
        a = torch.randn(1, 2, 3, 4, 5)
        img=self.get_pixels_hu(img)
        img=self.slice_windowing(img)


        # histogram, bin_edges = np.histogram(img, bins=256, range=(0, 1))
        # plt.figure()
        # plt.title("Grayscale Histogram")
        # plt.xlabel("grayscale value")
        # plt.ylabel("pixel count")
        # plt.plot(bin_edges[0:-1], histogram)  


        if self.labels[idx] is not None and self.train:
            label =np.load(self.labels[idx])
            # label=self.dicom_normalizer(label)
            # cv2.imshow('',label)
            # cv2.waitKey(0)
        else:
            gt = np.zeros(img.shape[:-1], dtype=np.uint8)
            
         ## 已经读取了image以及对应的ground truth可以进行data augmentation了
        if self.train:  #scaling, cropping and flipping
             img, label = crop_dicom(img,label)
             scale = random.uniform(0.7, 1.3)
             flip_p = random.uniform(0, 1)
             img_temp = scale_im(img,scale)
             img_temp = flip(img_temp,flip_p)
             gt_temp = scale_gt(label,scale)
             gt_temp = flip(gt_temp,flip_p)
             
             img = img_temp
             label = gt_temp
             
        if self.inputRes is not None:
            img = cv2.resize(img, self.inputRes)
            #print('ok1')
            #scipy.misc.imsave('label.png',label)
            #scipy.misc.imsave('img.png',img)
            if self.labels[idx] is not None and self.train:
                label = cv2.resize(label, self.inputRes, interpolation = cv2.INTER_NEAREST)

        img = np.array(img, dtype=np.float32)
        #img = img[:, :, ::-1]
        t=np.array(self.meanval, dtype=np.float32)
        img = np.subtract(img, t)        
        # img = img.transpose((2, 0, 1))  # NHWC -> NCHW
        
        if self.labels[idx] is not None and self.train:
                gt = np.array(label, dtype=np.int32)
                gt[gt!=0]=1
                #gt = gt/np.max([gt.max(), 1e-8])
        #np.save('gt.npy')
        img=self.dicom_normalizer(img)
        img=img.astype(np.float32)
        return img, gt


    def get_img_size(self):
        img = cv2.imread(os.path.join(self.db_root_dir, self.video_list[0]))
        
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