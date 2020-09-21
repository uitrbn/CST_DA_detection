import torch
import numpy as np
import pdb
import time
import cv2
import os
from scipy.misc import imread, imsave
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.utils.net_utils import save_net, load_net, vis_detections

import global_variable

empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))

try:
    xrange
except Exception:
    xrange = range



def generate_boxes_from_cls_score(rois, cls_prob, imdb, positive_threshold=0.9):
    assert rois.size(0) == 1, rois.size(0)
    assert rois.size(2) == 5, rois.size(2)
    assert cls_prob.size(0) == 1, cls_prob.size(0)
    assert imdb is not None


    vis = False
    thresh = 0.0
    max_per_image = 100
    class_agnostic = False

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]

    _ = torch.from_numpy(np.tile(boxes, (1, scores.shape[2])))
    pred_boxes = _.cuda()

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()
    if vis:
        im = cv2.imread(imdb.image_path_at(i))
        im2show = np.copy(im)
    
    det_box_for_current_image = [None for j in range(imdb.num_classes)]

    for j in xrange(imdb.num_classes):
        inds = torch.nonzero(scores[:,j]>thresh).view(-1)
        # if there is det
        if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if class_agnostic:
                cls_boxes = pred_boxes[inds, :]
            else:
                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            if vis:
                im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
            det_box_for_current_image[j] = cls_dets.cpu().numpy()
        else:
            det_box_for_current_image[j] = empty_array
    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
        image_scores = np.hstack([det_box_for_current_image[j][:, -1]
                                for j in xrange(1, imdb.num_classes)])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            for j in xrange(1, imdb.num_classes):
                keep = np.where(det_box_for_current_image[j][:, -1] >= image_thresh)[0]
                det_box_for_current_image[j] = det_box_for_current_image[j][keep, :]

    boxes_above_threshold = list()

    for i in range(1, len(det_box_for_current_image)):
        one_cls_box = det_box_for_current_image[i]
        if one_cls_box is None:
            boxes_above_threshold.append(one_cls_box)
        else:
            boxes_above_threshold.append(one_cls_box[one_cls_box[:, -1] > positive_threshold])

    gt_boxes = torch.FloatTensor(1, cfg.MAX_NUM_GT_BOXES, 5).zero_()
    gt_score = torch.FloatTensor(1, cfg.MAX_NUM_GT_BOXES, 1).zero_()

    det_box_above_num = [box.shape[0] for box in boxes_above_threshold]
    # print("det box above distribution on category : {}".format(str(det_box_above_num)))
    det_box_above_sum = sum(det_box_above_num)
    # print("sum : {}".format(det_box_above_sum))
    if det_box_above_sum != 0:
        all_gt_boxes = torch.FloatTensor(1, det_box_above_sum, 5).zero_() # the 5th position for category
        all_gt_boxes_cls_score = torch.FloatTensor(1, det_box_above_sum, 1).zero_() # the cls score for each det box

        start = 0
        for i, det_box_one_cls_above in enumerate(boxes_above_threshold):
            if det_box_one_cls_above is None:
                continue
            if det_box_one_cls_above.shape[0] == 0:
                continue
            all_gt_boxes[0, start:start + det_box_one_cls_above.shape[0], :4] = \
                torch.FloatTensor(det_box_one_cls_above[:, :4])
            all_gt_boxes[0, start:start + det_box_one_cls_above.shape[0], 4] = i + 1

            all_gt_boxes_cls_score[0, start:start + det_box_one_cls_above.shape[0], 0] = \
                torch.FloatTensor(det_box_one_cls_above[:, 4])

            start += det_box_one_cls_above.shape[0]

        if det_box_above_sum > 20:
            index = np.random.permutation(det_box_above_sum)[:20]
            gt_boxes = all_gt_boxes[:, index, :]
            gt_score = all_gt_boxes_cls_score[:, index, :]
        else:
            gt_boxes[:, :det_box_above_sum, :] = all_gt_boxes
            gt_score[:, :det_box_above_sum, :] = all_gt_boxes_cls_score
    else:
        pass

    # pdb.set_trace()

    gf_boxes = torch.FloatTensor(1, 20, 5).zero_()
    gf_score = torch.FloatTensor(1, 20, 1).zero_()

    boxes_background_threshold = det_box_for_current_image[0][det_box_for_current_image[0][:, -1] > positive_threshold]

    # pdb.set_trace()
    if boxes_background_threshold.shape[0] > 20:
        index = np.random.permutation(20)
        gf_boxes[:, :, :4] = torch.FloatTensor(boxes_background_threshold[index, :4])
        gf_boxes[:, :, 4] = 1 # TODO: what should it be?
        gf_score[0, :20, 0] = torch.FloatTensor(boxes_background_threshold[index, 4])
        det_box_below_sum = 20
    else:
        gf_boxes[:, :boxes_background_threshold.shape[0], :4] = torch.FloatTensor(boxes_background_threshold[:, :4])
        gf_boxes[:, :, 4] = 1
        gf_score[0, :boxes_background_threshold.shape[0], 0] = torch.FloatTensor(boxes_background_threshold[:, 4])
        det_box_below_sum = boxes_background_threshold.shape[0]

    return gt_boxes, det_box_above_sum, gf_boxes, det_box_below_sum, gt_score, gf_score


def generate_boxes_from_cls_score_older_version(rois, cls_prob, imdb):
    assert rois.size(0) == 1, rois.size(0)
    assert rois.size(2) == 5, rois.size(2)
    assert cls_prob.size(0) == 1, cls_prob.size(0)
    # assert cls_prob.size(1) == 128, cls_prob.size(1)
    assert imdb is not None

    vis = False
    thresh = 0.0
    max_per_image = 100
    class_agnostic = False

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]

    _ = torch.from_numpy(np.tile(boxes, (1, scores.shape[2])))
    pred_boxes = _.cuda()

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()
    if vis:
        im = cv2.imread(imdb.image_path_at(i))
        im2show = np.copy(im)
    
    det_box_for_current_image = [None for j in range(imdb.num_classes)]

    for j in xrange(1, imdb.num_classes):
        inds = torch.nonzero(scores[:,j]>thresh).view(-1)
        # if there is det
        if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if class_agnostic:
                cls_boxes = pred_boxes[inds, :]
            else:
                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            if vis:
                im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
            det_box_for_current_image[j] = cls_dets.cpu().numpy()
        else:
            det_box_for_current_image[j] = empty_array
    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
        image_scores = np.hstack([det_box_for_current_image[j][:, -1]
                                for j in xrange(1, imdb.num_classes)])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            for j in xrange(1, imdb.num_classes):
                keep = np.where(det_box_for_current_image[j][:, -1] >= image_thresh)[0]
                det_box_for_current_image[j] = det_box_for_current_image[j][keep, :]
    
    positive_threshold = 0.9
    negatives_threshold = 0.1

    boxes_above_threshold = list()
    boxes_below_threshold = list()
    for one_cls_box in det_box_for_current_image:
        if one_cls_box is None:
            boxes_above_threshold.append(one_cls_box)
            boxes_below_threshold.append(one_cls_box)
        else:
            boxes_above_threshold.append(one_cls_box[one_cls_box[:, -1] > positive_threshold])
            boxes_below_threshold.append(one_cls_box[one_cls_box[:, -1] < negatives_threshold])

    gt_boxes = torch.FloatTensor(1, cfg.MAX_NUM_GT_BOXES, 5).zero_()
    gf_boxes = torch.FloatTensor(1, 20, 5).zero_()

    gt_score = torch.FloatTensor(1, cfg.MAX_NUM_GT_BOXES, 1).zero_()
    gf_score = torch.FloatTensor(1, 20, 1).zero_()
    
    # copy car det box only
    det_box_car = boxes_above_threshold[1]
    det_box_bg = boxes_below_threshold[1]

    det_box_car = det_box_car[:cfg.MAX_NUM_GT_BOXES]
    det_box_bg = det_box_bg[np.random.permutation(len(det_box_bg))]
    det_box_bg = det_box_bg[-20:]

    car_box_num = det_box_car.shape[0]
    bg_box_num = det_box_bg.shape[0]

    for i in range(det_box_car.shape[0]):
        gt_boxes[0, i, :4] = torch.FloatTensor(det_box_car[i][:4])
        gt_boxes[0, i, 4] = 1
    for i in range(det_box_bg.shape[0]):
        gf_boxes[0, i, :4] = torch.FloatTensor(det_box_bg[i][:4])
        gf_boxes[0, i, 4] = 1

    return gt_boxes, car_box_num, gf_boxes, bg_box_num, gt_score, gf_score


