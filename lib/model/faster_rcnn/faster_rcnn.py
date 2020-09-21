# -*- coding: UTF-8 -*- 
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN, RPN_training_target
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.rpn.proposal_target_layer_cascade_target import _ProposalTargetLayerForTarget
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
from global_discriminator import ReverseLayerF, LocalDiscriminator, LossForLocal, LossForRPNCLS, LossForDiscrepancy
from pseudo_label import generate_boxes_from_cls_score, generate_boxes_from_cls_score_older_version
import global_variable

counter = 0

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_proposal_target_for_target = _ProposalTargetLayerForTarget(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

		# local discriminator
        from model.faster_rcnn.vgg16 import vgg16
        from model.faster_rcnn.resnet import resnet101
        if isinstance(self, vgg16):
            self.local_discriminator = LocalDiscriminator(256)
        else:
            self.local_discriminator = LocalDiscriminator(512)
        self.local_loss_layer = LossForLocal()

		# cls entropy minimization CLS训练
        self.loss_rpn_cls_layer = LossForRPNCLS(5)

		# generate rpn target for pl box RPN训练
        self.rpn_training_target = RPN_training_target()

		# minimize discrepancy (MCD)
        self.loss_for_discrepancy = LossForDiscrepancy()

    def forward(self, im_data, im_info, gt_boxes, num_boxes, domain_label, imdb=None):

        if not global_variable.ProcessDontCare and global_variable.ImdbIsKitti:
            # ignore dontcare when imdb is kitti
            # pdb.set_trace()
            gt_boxes[gt_boxes[:, :, -1]==3] = 0
            gt_boxes = torch.cat((gt_boxes[gt_boxes[:, :, -1]!=0], gt_boxes[gt_boxes[:, :, -1]==0]), 0).view(1, -1, 5)

        global counter

        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        local_base_feat = self.RCNN_base1(im_data)
        base_feat = self.RCNN_base2(local_base_feat)

        # feed base feature map to RPN to obtain rois
        if domain_label == 1: # source
            rois_before, rpn_loss_cls, rpn_loss_bbox, roi_scores_before, rpn_cls_prob = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes, domain_label)
        elif domain_label == 0: # target
            rois_before, rpn_loss_cls, rpn_loss_bbox, roi_scores_before, \
                rpn_cls_score, rpn_cls_score_reshape, batch_size, rpn_bbox_pred, rpn_cls_prob, roi_scores_with_grad \
                    = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes, domain_label)

        # if it is training phase, then use ground truth bboxes for refining
        if self.training:
            reversed_local_feat = ReverseLayerF.apply(local_base_feat, 0.5)
            local_dis_output, _= self.local_discriminator(reversed_local_feat)
            # 使用heat map global loss
            domain_cls_loss = self.local_loss_layer(local_dis_output, domain_label, rpn_cls_prob)
            local_dis_output_0 = local_dis_output.mean()

			# rpn filter and rpn target for cls and reg calculation
            if domain_label == 1: # source
                roi_data = self.RCNN_proposal_target(rois_before, gt_boxes, num_boxes)
                rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            else: # target
                roi_data = self.RCNN_proposal_target_for_target(rois_before, gt_boxes, num_boxes, roi_scores_before, roi_scores_with_grad)
                rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws, roi_scores, roi_scores_grad = roi_data
            
            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
            domain_cls_loss = 0
            local_dis_output_0 = 0

        rois = Variable(rois) if self.training else Variable(rois_before)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        rpn_cls_align_loss = 0
        if domain_label == 0 and self.training:
			# cls entropy minimization CLS训练
            rpn_cls_align_loss = self.loss_rpn_cls_layer(cls_score, roi_scores.squeeze())

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training and domain_label == 1:
            # classification loss
            if global_variable.ProcessDontCare:
                RCNN_loss_cls = F.cross_entropy(cls_score, rois_label, weight=torch.FloatTensor([1, 1, 1, 0]).cuda())
            else:
            	RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        if domain_label == 0 and self.training:
            # normal version
            # gt_boxes, gt_num, gf_boxes, gf_num, gt_score, gf_score = generate_boxes_from_cls_score_older_version(rois, cls_prob, imdb)
            # weight version
            gt_boxes, gt_num, gf_boxes, gf_num, gt_score, gf_score = generate_boxes_from_cls_score(rois, cls_prob, imdb)
            counter += 1
            if gt_num > 0 and gf_num > 0:
                gt_boxes = gt_boxes.cuda()
                gf_boxes = gf_boxes.cuda()
                rpn_loss_cls, rpn_loss_bbox = self.rpn_training_target(gt_boxes, im_info, num_boxes, domain_label, gf_boxes, rpn_cls_score, rpn_cls_score_reshape, batch_size, rpn_bbox_pred, gt_score, gf_score)
            else:
                print("gt num : {}, gf num : {}".format(gt_num, gf_num))
                rpn_loss_cls = 0
                rpn_loss_bbox = 0

        if domain_label == 0 and self.training:
            discrepancy_loss_min, discrepancy_loss_max = self.loss_for_discrepancy(cls_score, roi_scores_grad.squeeze())
        else:
            discrepancy_loss_min = 0
            discrepancy_loss_max = 0

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, domain_cls_loss, local_dis_output_0, rpn_cls_align_loss, discrepancy_loss_min, discrepancy_loss_max

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
