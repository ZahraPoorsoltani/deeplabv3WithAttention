# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 17:53:20 2018

@author: carri
"""

import argparse
import pydicom

from ast import arg
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import torchvision.transforms as transforms

from PIL import Image
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch.nn as nn
# from utils.colorize_mask import cityscapes_colorize_mask, VOCColorize
from scipy import ndimage
# from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
from torchvision.utils import save_image

import  deeplab
from deeplab import GNNNet
my_scales = [0.75, 1.0, 1.5]


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="PSPnet")
    parser.add_argument("--dataset", type=str, default='pascal-context',
                        help="voc12, cityscapes, or pascal-context")

    # GPU configuration
    parser.add_argument("--cuda", default=False, help="Run on CPU or GPU")
    parser.add_argument("--gpus", type=str, default="0",
                        help="choose gpu device.")
    parser.add_argument("--seq_name", default='bmx-bumps')
    parser.add_argument("--use_crf", default='True')
    parser.add_argument("--sample_range", default=2)

    return parser.parse_args()


def configure_dataset_model(args):
    args.batch_size = 4  # 1 card: 5, 2 cards: 10 Number of images sent to the network in one step, 16 on paper
    args.maxEpoches = 15  # 1 card: 15, 2 cards: 15 epoches, equal to 30k iterations, max iterations= maxEpoches*len(train_aug)/batch_size_per_gpu'),
    args.data_dir = './dataset/DAVIS-2016'  # 37572 image pairs
    args.data_list = './dataset/DAVIS-2016/test_seqs.txt'  # Path to the file listing the images in the dataset
    args.ignore_label = 255  # The index of the label to ignore during the training
    args.input_size = '473, 473'  # Comma-separated string with height and width of images
    args.num_classes = 2  # Number of classes to predict (including background)
    args.img_mean = np.array((104.00698793, 116.66876762, 122.67891434),
                             dtype=np.float32)  # saving model file and log record during the process of training
    args.restore_from = './snapshots/attention_agnn_51.pth'  # './snapshots/davis_iteration_conf_gnn3_sa/co_attention_davis_55.pth' #resnet50-19c8e357.pth''/home/xiankai/PSPNet_PyTorch/snapshots/davis/psp_davis_0.pth' #
    args.snapshot_dir = './snapshots/davis_iteration/'  # Where to save snapshots of the model
    args.save_segimage = True
    args.seg_save_dir = "./result/test/davis_iteration_conf_gnn3_sa_org_scale_batch"
    args.vis_save_dir = "./result/test/davis_vis"
    args.corp_size = (473, 473)

def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap

def convert_state_dict(model,state_dict):
    new_state_model=model.state_dict().copy()
    for k,v in state_dict.items():
        name_splt = k.split('.')
        if name_splt[0] == 'backbone':
            name = k[9:]  # remove the 'backbone' prefix module.
        elif name_splt[0]=='classifier':
            name='layer5.'+k[11:]

        new_state_model[name] = v

    return new_state_model


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))  # Define a sigmoid method, the essence of which is 1/(1+e^-x)


def main():
    args = get_arguments()
    print("=====> Configure dataset and model")
    configure_dataset_model(args)
    print(args)

    print("=====> Set GPU for training")
    if args.cuda:
        print("====> Use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    model = GNNNet(num_classes=21)

    for param in model.parameters():
        param.requires_grad = False

    args.restore_from = './pretrained/best_deeplabv3plus_resnet101_voc_os16.pth'
    saved_state_dict = torch.load(args.restore_from, map_location=lambda storage, loc: storage)['model_state']
    # print(saved_state_dict.keys())
    # model.load_state_dict({k.replace('pspmodule.',''):v for k,v in torch.load(args.restore_from)['state_dict'].items()})



    new_state_dict=convert_state_dict(model,saved_state_dict)
    model.load_state_dict(new_state_dict)
    # model.load_state_dict(saved_state_dict["model"])

    model.eval()
    model.cuda()

    val_transform = transforms.Compose([
        transforms.Resize(513),
        transforms.CenterCrop(513),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_path='samples/2008_000004.jpg'
    result_path = r"samples/image_results.png"
    img = Image.open(img_path).convert('RGB')
    img = val_transform(img).unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = 'voc'
    with torch.no_grad():
        print(img.shape)
        img = img.to(device)
        # preds = outputs.detach().max(dim=1)[1].cpu().numpy()
        pred = model(img)
        pred=pred.max(dim=1)[1].cpu().numpy()[0, :, :]

        if dataset == 'voc':
            pred = voc_cmap()[pred].astype(np.uint8)


        Image.fromarray(pred).save(result_path)
        print("Prediction is saved in %s" % result_path)

if __name__ == '__main__':
    main()
