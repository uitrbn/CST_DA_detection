# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import random

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='vgg16', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=6, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="models",
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of workers to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')                      
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether to perform class_agnostic bbox regression',
                      action='store_true')

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and display
  parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      action='store_true')

  parser.add_argument('--pretrain_file', dest='pretrain_file',
                      help='pretrain file',
                      type=str)

  args = parser.parse_args()
  return args


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

if __name__ == '__main__':

  args = parse_args()
  args.dataset = 'foggycityscapes'

  print('Called with args:')
  print(args)
  args.cuda = True
  args.net = 'vgg16'

  if args.dataset == 'foggycityscapes':
      args.imdb_name = 'foggycityscapes_train'
      args.imdbval_name = 'foggycityscapes_val'
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '40']
  else:
      raise Exception("Unknown dataset")

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  torch.manual_seed(cfg.RNG_SEED)
  random.seed(cfg.RNG_SEED)
  torch.cuda.manual_seed(cfg.RNG_SEED)
  torch.cuda.manual_seed_all(cfg.RNG_SEED)

  # torch.backends.cudnn.benchmark = True
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = False
  cfg.USE_GPU_NMS = args.cuda

  imdb_cityscapes, roidb_cityscapes, ratio_list_cityscapes, ratio_index_cityscapes = combined_roidb('fullycityscapes_train')
  train_size_cityscapes = len(roidb_cityscapes)

  imdb_cityscapes_pl, roidb_cityscapes_pl, ratio_list_cityscapes_pl, ratio_index_cityscapes_pl = combined_roidb('foggycityscapes_train')
  train_size_cityscapes_pl = len(roidb_cityscapes_pl)

  print('{:d} roidb entries in fullycityscapes_train'.format(len(roidb_cityscapes)))
  print('{:d} roidb entries in foggycityscapes_train'.format(len(roidb_cityscapes_pl)))

  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  sampler_batch_cityscapes = sampler(train_size_cityscapes, args.batch_size)
  
  dataset_cityscapes = roibatchLoader(roidb_cityscapes, ratio_list_cityscapes, ratio_index_cityscapes, args.batch_size, \
                                  imdb_cityscapes.num_classes, training=True)

  dataloader_cityscapes = torch.utils.data.DataLoader(dataset_cityscapes, batch_size=args.batch_size,
                                                  sampler=sampler_batch_cityscapes, num_workers=args.num_workers)

  sampler_batch_cityscapes_pl = sampler(train_size_cityscapes_pl, args.batch_size)

  dataset_cityscape_pl = roibatchLoader(roidb_cityscapes_pl, ratio_list_cityscapes_pl, ratio_index_cityscapes_pl, args.batch_size, \
                                        imdb_cityscapes_pl.num_classes, training=True)
  
  dataloader_cityscapes_pl = torch.utils.data.DataLoader(dataset_cityscape_pl, batch_size = args.batch_size, 
                                                  sampler=sampler_batch_cityscapes_pl, num_workers=args.num_workers)

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)
  gf_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()
    gf_boxes = gf_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)
  gf_boxes = Variable(gf_boxes)

  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb_cityscapes.classes, pretrained=True, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  if not args.resume:
    # sim10k
    pretrain_input_dir = 'models' + '/' + args.net + '/' + 'foggycityscapes'
    pretrain_load_name = os.path.join(pretrain_input_dir, 'da_detection_pretrained_cityscapes_{}_{}_{}.pth'.format(1, 10, 2964))

    print("load pretrained file : ", pretrain_load_name)
    pretrained_checkpoint = torch.load(pretrain_load_name)
    fasterRCNN.load_state_dict(pretrained_checkpoint['model'])
    if 'pooling_mode' in pretrained_checkpoint.keys():
        cfg.POOLING_MODE = pretrained_checkpoint['pooling_mode']
    
    # re-initial discriminator
  #   def weights_init(m):
  #     if isinstance(m, nn.Conv2d):
  #         torch.nn.init.xavier_uniform(m.weight.data)
  #   fasterRCNN.global_discriminator.apply(weights_init)
    def weight_reset(m):
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()
    fasterRCNN.local_discriminator.apply(weight_reset)

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr
  #tr_momentum = cfg.TRAIN.MOMENTUM
  #tr_momentum = args.momentum

  params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  RCNN_base_params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad and 'RCNN_base' in key:
      if 'bias' in key:
        RCNN_base_params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        RCNN_base_params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
    
  RCNN_RPN_CLS_params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad and 'RCNN' in key and 'RCNN_base' not in key:
      if 'bias' in key:
        RCNN_RPN_CLS_params += [{'params':[value],'lr':lr * 1/2 *(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        RCNN_RPN_CLS_params += [{'params':[value],'lr':lr * 1/2, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  RCNN_RPN_params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad and 'RCNN_base' not in key and 'discriminator' not in key and 'RCNN_rpn' in key:
      print(key)
      if 'bias' in key:
        RCNN_RPN_params += [{'params':[value],'lr':0.001 *(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        RCNN_RPN_params += [{'params':[value],'lr':0.001, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  print("==================")

  RCNN_CLS_params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad and 'RCNN_base' not in key and 'discriminator' not in key and 'RCNN_rpn' not in key:
      print(key)
      if 'bias' in key:
        RCNN_CLS_params += [{'params':[value],'lr':0.001 *(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        RCNN_CLS_params += [{'params':[value],'lr':0.001, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]


  if args.optimizer == "adam":
    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
    optimizer_fe = torch.optim.SGD(RCNN_base_params, momentum=cfg.TRAIN.MOMENTUM)
    optimizer_cl = torch.optim.SGD(RCNN_RPN_CLS_params, momentum=cfg.TRAIN.MOMENTUM)
    optimizer_cls = torch.optim.SGD(RCNN_CLS_params, momentum=cfg.TRAIN.MOMENTUM)
    optimizer_rpn = torch.optim.SGD(RCNN_RPN_params, momentum=cfg.TRAIN.MOMENTUM)

  backbone_params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad and 'RCNN_base' in key:
      if 'bias' in key:
        backbone_params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        backbone_params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  backbone_optimizer = torch.optim.SGD(backbone_params, momentum=cfg.TRAIN.MOMENTUM)


  if args.cuda:
    fasterRCNN.cuda()

  if args.mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)

  iters_per_epoch = int(train_size_cityscapes / args.batch_size)

  if args.use_tfboard:
    from tensorboardX import SummaryWriter
    logger = SummaryWriter("logs")

  assert not args.resume
  assert args.batch_size == 1
  assert args.cuda > 0

  for epoch in range(args.start_epoch, args.max_epochs + 1):
    fasterRCNN.train()
    loss_temp = 0
    start = time.time()

    if epoch % (args.lr_decay_step + 1) == 0:
      adjust_learning_rate(optimizer, args.lr_decay_gamma)
      lr *= args.lr_decay_gamma

    data_iter_cityscapes = iter(dataloader_cityscapes)
    data_iter_cityscapes_pl = iter(dataloader_cityscapes_pl)

    for step in range(iters_per_epoch):

      # train with target
      try:
        data_cityscapes_pl = next(data_iter_cityscapes_pl)
      except StopIteration:
        data_iter_cityscapes_pl = iter(dataloader_cityscapes_pl)
        data_cityscapes_pl = next(data_iter_cityscapes_pl)

      im_data.data.resize_(data_cityscapes_pl[0].size()).copy_(data_cityscapes_pl[0])
      im_info.data.resize_(data_cityscapes_pl[1].size()).copy_(data_cityscapes_pl[1])
      gt_boxes.data.resize_(data_cityscapes_pl[2].size()).copy_(data_cityscapes_pl[2])
      num_boxes.data.resize_(data_cityscapes_pl[3].size()).copy_(data_cityscapes_pl[3])

      gt_boxes = gt_boxes.zero_()

      ##############################################
      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      _, _, \
      rois_label, domain_cls_loss, local_dis_output, \
      rpn_cls_align_loss, discrepancy_loss_min, discrepancy_loss_max = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, 0, imdb_cityscapes_pl)


      if epoch < 4:
        loss_cityscapes_pl = domain_cls_loss.mean()
      else:
        if type(rpn_loss_cls) != type(0):
          loss_cityscapes_pl = domain_cls_loss.mean() + rpn_loss_cls.mean() * 0.1 + rpn_loss_box.mean() * 0.1 + rpn_cls_align_loss.mean() * 0.05
        else:
          loss_cityscapes_pl = domain_cls_loss.mean()
          print("skip target")

      if type(loss_cityscapes_pl) != type(0):

        if epoch > 1:
          optimizer.zero_grad()
          discrepancy_loss_min_mean = discrepancy_loss_min.mean() * 0.1
          discrepancy_loss_min_mean.backward(retain_graph=True)
          if args.net == 'vgg16':
              clip_gradient(fasterRCNN, 10.)
          optimizer_fe.step()

          optimizer.zero_grad()
          discrepancy_loss_max_mean = discrepancy_loss_max.mean() * 0.1
          discrepancy_loss_max_mean.backward(retain_graph=True)
          if args.net == 'vgg16':
              clip_gradient(fasterRCNN, 10.)
          optimizer_cls.step()
          optimizer_rpn.step()

        optimizer.zero_grad()
        loss_cityscapes_pl.backward()
        if args.net == 'vgg16':
            clip_gradient(fasterRCNN, 10.)
        optimizer.step()
        loss_temp += loss_cityscapes_pl.item()
        loss_target_domain = domain_cls_loss.mean().item()
        local_dis_output_target = local_dis_output
        rpn_cls_align_loss_output_target = rpn_cls_align_loss.mean()


      # source domain
      data_cityscapes = next(data_iter_cityscapes)

      im_data.data.resize_(data_cityscapes[0].size()).copy_(data_cityscapes[0])
      im_info.data.resize_(data_cityscapes[1].size()).copy_(data_cityscapes[1])
      gt_boxes.data.resize_(data_cityscapes[2].size()).copy_(data_cityscapes[2])
      num_boxes.data.resize_(data_cityscapes[3].size()).copy_(data_cityscapes[3])

      fasterRCNN.zero_grad()
      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label, domain_cls_loss, local_dis_output, \
      rpn_cls_align_loss, _, _ = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, 1, imdb_cityscapes)

      # source loss calculate and backward
      loss_cityscapes = rpn_loss_cls.mean() + rpn_loss_box.mean() \
          + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean() + domain_cls_loss.mean() #+ domain_cls_loss_global.mean()# + instance_domain_loss.mean()
      optimizer.zero_grad()
      loss_cityscapes.backward()
      if args.net == 'vgg16':
          clip_gradient(fasterRCNN, 10.)
      optimizer.step()
      loss_temp += loss_cityscapes.item()
      loss_source_domain = domain_cls_loss.mean().item()
      local_dis_output_source = local_dis_output
      rpn_cls_align_loss_output_source = 0

      if step % args.disp_interval == 0:
        end = time.time()
        if step > 0:
          loss_temp /= (args.disp_interval + 1)

        if args.mGPUs:
          loss_rpn_cls = rpn_loss_cls.mean().item()
          loss_rpn_box = rpn_loss_box.mean().item()
          loss_rcnn_cls = RCNN_loss_cls.mean().item()
          loss_rcnn_box = RCNN_loss_bbox.mean().item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt
        else:
          loss_rpn_cls = rpn_loss_cls.item() if type(0) != type(rpn_loss_cls) else 0
          loss_rpn_box = rpn_loss_box.item() if type(0) != type(rpn_loss_box) else 0
          loss_rcnn_cls = RCNN_loss_cls.item()
          loss_rcnn_box = RCNN_loss_bbox.item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt

        print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
        print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
        print("\t\t\tsource domain cls loss: %.4f, target domain cls loss: %.4f" % (loss_source_domain, loss_target_domain))
        print("\t\t\tsource domain dis output: %.4f, target domain dis output: %.4f" % (local_dis_output_source, local_dis_output_target))
        print("\t\t\tsource rpn cls align loss: %.4f, target rpn cls align loss : %.4f" %(rpn_cls_align_loss_output_source, rpn_cls_align_loss_output_target))
        if args.use_tfboard:
          info = {
            'loss': loss_temp,
            'loss_rpn_cls': loss_rpn_cls,
            'loss_rpn_box': loss_rpn_box,
            'loss_rcnn_cls': loss_rcnn_cls,
            'loss_rcnn_box': loss_rcnn_box
          }
          logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)

        loss_temp = 0
        start = time.time()

    
    save_name = os.path.join(output_dir, 'fasterRCNN_da_cityscapes_to_foggycityscapes_{}_{}_{}.pth'.format(args.session, epoch, step))
    save_checkpoint({
      'session': args.session,
      'epoch': epoch + 1,
      'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
      'optimizer': optimizer.state_dict(),
      'pooling_mode': cfg.POOLING_MODE,
      'class_agnostic': args.class_agnostic,
    }, save_name)
    print('save model: {}'.format(save_name))

  if args.use_tfboard:
    logger.close()
