from __future__ import print_function
from __future__ import absolute_import

import xml.dom.minidom as minidom

import os
import numpy as np
import scipy.sparse
import subprocess
import math
import glob
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET
import pickle
from .imdb import imdb
from .imdb import ROOT_DIR
from . import ds_utils
# from .cityscapes_eval import cityscapes_eval
from .fullycityscapes_eval import fullycityscapes_eval

from model.utils.config import cfg

try:
    xrange
except NameError:
    xrange = range

class fullycityscapes(imdb):
    def __init__(self, image_set, root_path=None):
        imdb.__init__(self, 'fullycityscapes_' + image_set)

        if root_path is None:
            self._data_path = self._get_default_path()
        else:
            self._data_path = root_path

        self._image_path = os.path.join(self._data_path, 'leftImg8bit', image_set)
        self._label_path = os.path.join(self._data_path, 'gtFine', image_set)
        self._image_set = image_set

        self._image_index = self._load_index(self._image_path)

        self._roidb_handler = self.gt_roidb

        self._classes = ('__background__', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))

        self._salt = str(uuid.uuid4())
        self._comp_id = "comp12"

        self.config = {
            'cleanup' : False,
            'use_salt': True,
            'use_diff': False,
            'rpn_file': None,
            'min_size': 2
        }

        assert os.path.exists(self._image_path), "cityscapes image path not exists"
        assert os.path.exists(self._label_path), "cityscapes label path not exists"


    def _get_default_path(self):
        return os.path.join(cfg.DATA_DIR, 'fullycityscapes')

    def _load_index(self, path):
        image_index = list()
        for city in os.listdir(path):
            city_path = os.path.join(path, city)
            for image_name in os.listdir(city_path):
                image_index.append((city, image_name))
        return image_index
        
    def image_path_at(self, i):
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        return i

    def image_path_from_index(self, index):
        image_path = os.path.join(self._image_path, index[0], index[1])
        assert os.path.exists(image_path), "Path does not exists {}".format(image_path)
        return image_path

    def gt_roidb(self):
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_cityscapes_annotation(index)
            for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def _load_cityscapes_annotation(self, index):
        
        # not using pseudo label 
        city, image_filename = index
        filename = os.path.join(self._data_path, 'gtFine', self._image_set, \
            city, image_filename.split('leftImg8bit')[0] + 'gtFine_bounding_box.txt')
        annotation_file = open(filename)

        num_objs = len(annotation_file.readlines())

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)

        annotation_file.close()
        annotation_file = open(filename)

        for ix, line in enumerate(annotation_file.readlines()):
            _cls = line.split('#')[0]
            x1, y1, x2, y2 = eval(line.split('#')[1])
            x1 -= 1
            y1 -= 1
            x2 -= 1
            y2 -= 1

            difficult = 0
            ishards[ix] = difficult

            cls = self._class_to_ind[_cls]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2-x1 + 1) * (y2-y1 + 1)
        
        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {
            'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_ishard': ishards,
            'gt_overlaps': overlaps,
            'flipped' : False,
            'seg_areas' : seg_areas
        }

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def _get_cityscapes_results_file_template(self):
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        filedir = os.path.join(self._data_path, 'results', 'fullycityscapes', 'Main')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path
    
    def _write_cityscapes_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue

            print("Writing {} cityscapes result file".format(cls))
            filename = self._get_cityscapes_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index[1], dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(self._data_path, 'gtFine', self._image_set, '{:s}/{:s}_gtFine_bounding_box.txt')
        # if self._image_set == 'train':
        #     imagesetfileL
        cachedir = os.path.join(self._data_path, 'annotations_cache')
        aps = []

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_cityscapes_results_file_template().format(cls)
            rec, prec, ap = fullycityscapes_eval(
                filename, annopath, os.path.join(self._data_path, 'leftImg8bit', self._image_set), cls, cachedir, ovthresh=0.5
            )

            aps += [ap]
            print("AP for {} = {:.4f}".format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print("Mean AP = {:.4f}".format(np.mean(filter(lambda x:~np.isnan(x), aps))))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print("~~~~~~~~")
        print('')
    
    def evaluate_detections(self, all_boxes, output_dir):
        self._write_cityscapes_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_cityscapes_results_file_template().format(cls)
                os.remove(filename)
