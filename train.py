from __future__ import print_function
from __future__ import division

import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torch.utils import data

from deeplab import AttentionDeeplabV3


from utils import ext_transforms as et, utils
from dataloaders.VOC2012 import VOCSegmentation

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=['deeplabv3_resnet50', 'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet',
                                 'deeplabv3plus_mobilenetSA', 'deeplabv3plus_mobilenetSAc',
                                 'deeplabv3plus_mobilenet_M', 'deeplabv3plus_mobilenetSAc_M',
                                 'deeplabv3plus_mobilenetSA_M'
                                 ], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--test_divide", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="weight of multi-model prediction (default: 0.5)")

    parser.add_argument("--sin_model", type=str, default='checkpoints/best_deeplabv3plus_mobilenet_voc_os16.pth',
                        help="The path of single model")
    parser.add_argument("--lr_policy", type=str, default='step', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--divide_data", type=int, default=0)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='logit',
                        choices=['cross_entropy', 'focal_loss', 'logit'], help="loss type (default: False)")
    parser.add_argument("--ye", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--select_class", nargs='+', default=None,
                        help="Select list")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=200,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    parser.add_argument("--start_class", type=int, default=1,
                        help='The start class (default: 1)')
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser



def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            # et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', divide_data=opts.divide_data ,transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', divide_data=opts.divide_data,transform=val_transform)
        test_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                   image_set='test', divide_data=opts.divide_data,transform=val_transform)


    return train_dst, val_dst, test_dst


def main():
    opts = get_argparser().parse_args()
    opts.num_classes = 21

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)



    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None:
        print("Error --ckpt, can't read model")
        return

    _, val_dst, test_dst = get_dataset(opts)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    test_loader = data.DataLoader(
        test_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    vis_sample_id = np.random.randint(0, len(test_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization

    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images


# ==========   Train Loop   ==========#
    utils.mkdir('checkpoints/multiple_model')

    # ==========   Dataset   ==========#

    train_dst, val_dst, test_dst = get_dataset(opts)
    num_train = len(train_dst)
    if num_train % opts.batch_size == 1 or (num_train ) % opts.batch_size == 1:  # Prevent only one sample in the batch
        droplast = True
    else:
        droplast = False
    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size, shuffle=True,drop_last=droplast)

    num_val = len(val_dst)
    if num_val % opts.val_batch_size == 1 or (num_val) % opts.val_batch_size == 1:
        droplast = True
    else:
        droplast = False
    val_loader = data.DataLoader(val_dst, batch_size=opts.val_batch_size, shuffle=True,drop_last=droplast)
    test_loader = data.DataLoader(test_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s , Train set: %d, Val set: %d, Test set: %d" % (
        opts.dataset , len(train_dst), len(val_dst), len(test_dst)))

    # ==========   Model   ==========#
    model = AttentionDeeplabV3(num_classes=21)
    utils.set_bn_momentum(model.body.backbone, momentum=0.01)

    # ==========   Params and learning rate   ==========#
    params_list = [
        {'params': model.body.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.body.classifier.parameters(), 'lr': 0.1 * opts.lr}  # opts.lr
    ]

    optimizer = torch.optim.Adam(params=params_list, lr=opts.lr, weight_decay=opts.weight_decay)

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    model.to(device)

    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0

    interval_loss = 0
    while True: #cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss/10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

            # if (cur_itrs) % opts.val_interval == 0:
            #     save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
            #               (opts.model, opts.dataset, opts.output_stride))
            #     print("validation...")
            #     model.eval()
            #     val_score, ret_samples = validate(
            #         opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
            #     print(metrics.to_str(val_score))
            #
            #     # print("testing")
            #     # test_score, ret_samples = validate(
            #     #     opts=opts, model=model, loader=test_loader, device=device, metrics=metrics,
            #     #     ret_samples_ids=vis_sample_id)
            #     # print(metrics.to_str(test_score))
            #
            #     if val_score['Mean IoU'] > best_score:  # save best model
            #         best_score = val_score['Mean IoU']
            #         save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
            #                   (opts.model, opts.dataset,opts.output_stride))
            #
            #     if vis is not None:  # visualize validation score and samples
            #         vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
            #         vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
            #         vis.vis_table("[Val] Class IoU", val_score['Class IoU'])
            #
            #         for k, (img, target, lbl) in enumerate(ret_samples):
            #             img = (denorm(img) * 255).astype(np.uint8)
            #             target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
            #             lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
            #             concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
            #             vis.vis_image('Sample %d' % k, concat_img)
            #     model.train()
            scheduler.step()

            if cur_itrs >=  opts.total_itrs:
                return


if __name__ == '__main__':
    main()
