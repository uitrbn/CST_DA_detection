#-*- coding: utf-8 -*-
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer, _AnchorTargetLayer_target
from model.utils.net_utils import _smooth_l1_loss

import numpy as np
import math
import pdb
import time

class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din):
        super(_RPN, self).__init__()
        
        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = cfg.FEAT_STRIDE[0]

        # define the convrelu layers processing input feature map
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2 # 2(bg/fg) * 9 (anchors)
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4 # 4(coords) * 9 (anchors)
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes, domain_label):

        batch_size = base_feat.size(0)

        # return feature map after convrelu layer
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)
        # get rpn classification score
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        # get rpn offsets to the anchor boxes
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'
		# rois_score has foreground for each rois, l2 distance is the smoothness of anchor heat map
        rois, roi_scores, roi_scores_with_grad = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                 im_info, cfg_key, domain_label, rpn_cls_prob))

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # generating training labels and build the rpn loss
        # do only when it's source, unless it's delay
        if self.training and domain_label == 1:
            assert gt_boxes is not None

            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes, domain_label))

            # compute classification loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)

            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())

            if rpn_keep.size()[0] != 0:
                self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            else:
                raise NotImplementedError()
                self.rpn_loss_cls = torch.zeros((batch_size)).cuda()
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            # compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                            rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])

        if domain_label == 1: # source
            return rois, self.rpn_loss_cls, self.rpn_loss_box, roi_scores, rpn_cls_prob
        else: # target
            return rois, self.rpn_loss_cls, self.rpn_loss_box, roi_scores, rpn_cls_score, rpn_cls_score_reshape, batch_size, rpn_bbox_pred, rpn_cls_prob, roi_scores_with_grad




class RPN_training_target(nn.Module):
    def __init__(self):
        super(RPN_training_target, self).__init__()

        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = cfg.FEAT_STRIDE[0]

        self.RPN_anchor_target_for_target = _AnchorTargetLayer_target(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # # TODO: assert classes number must be 9
        self.gt_boxes_category_counter = [0 for i in range(9)]

    def forward(self, gt_boxes, im_info, num_boxes, domain_label, gf_boxes, rpn_cls_score, rpn_cls_score_reshape, batch_size, rpn_bbox_pred, gt_score, gf_score):
        assert self.training

        rpn_data = self.RPN_anchor_target_for_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes, domain_label, gf_boxes, gt_score, gf_score))

        # compute classification loss
        rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
        rpn_label = rpn_data[0].view(batch_size, -1)

        rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
        rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)
        rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
        rpn_label = Variable(rpn_label.long())

        rpn_anchor_max_overlaps_cls = rpn_data[-3].view(batch_size, -1)
        rpn_anchor_max_overlaps_cls = torch.index_select(rpn_anchor_max_overlaps_cls.view(-1), 0, rpn_keep.data)
        rpn_anchor_max_overlaps_cls = Variable(rpn_anchor_max_overlaps_cls.long())

        rpn_anchor_gt_score = rpn_data[-2]
        rpn_anchor_gt_score = torch.index_select(rpn_anchor_gt_score.view(-1), 0, rpn_keep.data)
        rpn_anchor_gt_score = Variable(rpn_anchor_gt_score.float())
        
        rpn_anchor_gf_score = rpn_data[-1]
        rpn_anchor_gf_score = torch.index_select(rpn_anchor_gf_score.view(-1), 0, rpn_keep.data)
        rpn_anchor_gf_score = Variable(rpn_anchor_gf_score.float())
        
        rpn_anchor_gt_weight = (rpn_anchor_gt_score > 0.5).float() * (rpn_anchor_gt_score.pow(5))
        rpn_anchor_gf_weight = (rpn_anchor_gf_score > 0.5).float() * (rpn_anchor_gf_score.pow(5))
        rpn_score_weight = rpn_anchor_gt_weight * rpn_label.float() + rpn_anchor_gf_weight * (1 - rpn_label.float())
        rpn_score_weight = rpn_score_weight.detach()

        for i in range(9):
            self.gt_boxes_category_counter[i] += (torch.nonzero(rpn_anchor_max_overlaps_cls==i).numel())

        assert (rpn_anchor_max_overlaps_cls==0).sum() == 0
        weight = torch.FloatTensor([0, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,1,1,1,1,1,1,1, 1]).cuda()
        label_weight = weight[rpn_anchor_max_overlaps_cls]

        # ############# anchor target annotation box 
        # ���Կ��Ǹ���gt_boxes�е�������������ͬ������rpn loss�Ĺ��ס�
        
        if rpn_keep.size()[0] != 0:
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
        else:
            raise NotImplementedError()
            self.rpn_loss_cls = torch.zeros((batch_size)).cuda()
        fg_cnt = torch.sum(rpn_label.data.ne(0))

        rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1: -3]
        # rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

        # compute bbox regression loss
        rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
        rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
        rpn_bbox_targets = Variable(rpn_bbox_targets)

        self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                        rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])

        return self.rpn_loss_cls, self.rpn_loss_box