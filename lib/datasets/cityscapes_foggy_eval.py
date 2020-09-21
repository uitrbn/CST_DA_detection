# -*- coding: UTF-8 -*-
import os
import pickle
import numpy as np

def foggy_cityscapes_eval(detpath, annopath, imagesetdir, classname, cachedir, ovthresh=0.5):
    # import pdb; pdb.set_trace()
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, "foggycityscapes_annots_all.pkl")

    imagenames = list()
    for city in os.listdir(imagesetdir):
        current_city_dir = os.path.join(imagesetdir, city)
        for imagename in os.listdir(current_city_dir):
            imagenames.append((city, imagename))
    
    # pdb.set_trace()
    if not os.path.isfile(cachefile):
        recs = {} #the annotation, a dict from imagename to image info
        for i, (city, imagename) in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(city, imagename.split('_leftImg8bit')[0]))
            if i % 100 == 0:
                print("Reading annotation for {:d}/{:d}".format(i+1, len(imagenames)))
        print("Saving cached annotations to {:s}".format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        with open(cachefile, 'rb') as f:
            try:
                recs = pickle.load(f)
            except:
                recs = pickle.load(f, encoding='bytes')

    # pdb.set_trace()
    class_recs = {} # recs for this classname, from imageid to image info
    npos = 0
    for (city, imagename) in imagenames:
        # the recs for this classname
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {
            'bbox': bbox,
            'difficult': difficult,
            'det': det
        }

    # pdb.set_trace()
    detfile = detpath.format(classname) #predication result for this classname
    with open(detfile, 'r') as f:
        lines = f.readlines()

    # for each line in detfile, x[0] is the image id, x[1] is the confidence, x[2:] is the bouding box
    # maybe one image has multiple lines
    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]) #bboxes, 2 dimensions

    nd = len(image_ids) # image ids num of this class
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    # pdb.set_trace()

    if BB.shape[0] > 0: #if the bbox number > 0
        # sort by confidence, ascending
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        for d in range(nd):
            R = class_recs[image_ids[d]] #standard answer
            bb = BB[d, :].astype(float) # the predicted boxes of this image
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0: # the gt boxes num of this image
                #compute overlaps
                # intersection
                #比较所有gtboxes和pred boxes的第一维度
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])

                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                    (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                    (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
            
            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1
            else:
                fp[d] = 1
    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = cityscapes_foggy_ap(rec, prec)

    return rec, prec, ap


def parse_rec(filename):
    objects = []
    labelfile = open(filename)
    for line in labelfile.readlines():
        obj_struct = {}
        # obj_struct['name'] = 'car'
        obj_struct['name'] = line.split('#')[0]
        obj_struct['pose'] = None
        obj_struct['truncated'] = None
        obj_struct['difficult'] = 0


        x1, y1, x2, y2 = eval(line.split('#')[1])
        obj_struct['bbox'] = [x1, y1, x2, y2]
        objects.append(obj_struct)
    return objects

def cityscapes_foggy_ap(rec, prec):
    mrec = np.concatenate(([0.], rec, [1.]))
    mprec = np.concatenate(([0.], prec, [0.]))

    for i in range(mprec.size - 1, 0, -1):
        mprec[i - 1] = np.maximum(mprec[i - 1], mprec[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]

    ap = np.sum((mrec[i + 1] - mrec[i]) * mprec[i + 1])
    
    return ap