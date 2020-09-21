from PIL import Image
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import numpy as np

root_dir = "./data/cityscapes"
orgin_picture_dir = root_dir + '/' + 'leftImg8bit'
annotation_dir = root_dir + '/' + 'gtFine'

train_val_test = "train"

current_picture_dir = orgin_picture_dir + '/' + train_val_test
current_annotation_dir = annotation_dir + '/' + train_val_test

city_list = os.listdir(current_picture_dir)

def store_bbox_to_file(filename, bounding_box_list):
    with open(filename, 'w') as f:
        for label, box in bounding_box_list:
            f.write(label + '#' + str(box) + '\n')
            # f.write(str(box) + '\n')

def show_image_with_bounding_box(filename, box_filename):
    im = np.array(Image.open(filename), dtype=np.uint8)
    # Create figure and axes
    fig,ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)
    # Create a Rectangle patch
    with open(box_filename) as f:
        rectList = list()
        all_box = f.readlines()
        print("box num : {}".format(len(all_box)))
        for box in all_box:
            box = box[:-1]
            box = eval(box)
            top_left = (box[0], box[1])
            rect = patches.Rectangle(top_left, box[2] - box[0], box[3] - box[1])
            # Add the patch to the Axes
            rectList.append(rect)
    ax.add_collection(PatchCollection(rectList,linewidth=1,edgecolor='r',facecolor='none'))
    plt.show()

for city in city_list:
    file_list = os.listdir(os.path.join(current_picture_dir, city))
    for file_jpg in file_list:
        
        file_id = file_jpg.split('_leftImg8bit')[0]
        picture_mat = Image.open(os.path.join(current_picture_dir, city, file_jpg))
        gtFine_color = file_id + '_gtFine_color.png'
        gtFine_instanceIds = file_id + '_gtFine_instanceIds.png'
        gtFine_labelId = file_id + '_gtFine_labelIds.png'
        gtFine_polygons = file_id + '_gtFine_polygons.json'

        print("{}  processing....., image size : {}".format(file_jpg, picture_mat.size))

        with open(os.path.join(current_annotation_dir, city, gtFine_polygons), 'r') as myfile:
            json_string=myfile.read().replace('\n', '')
        json_object = json.loads(json_string)
        json_objects_object = json_object['objects']

        bounding_box_list = list()

        for item in json_objects_object:
            minx = None
            miny = None
            maxx = None
            maxy = None
            # if item['label'] == 'car': # only create bounding box for car
            if item['label'] in ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']: # only create bounding box for car
                if item['label'] == 'motorcycle':
                    print("motorcycle")
                polygon = item['polygon']
                for pixel in polygon:
                    if minx is None or minx > pixel[0]: # todo : pixel[0] or pixel[1]?
                        minx = pixel[0]
                    if miny is None or miny > pixel[1]:
                        miny = pixel[1]
                    if maxx is None or maxx < pixel[0]:
                        maxx = pixel[0]
                    if maxy is None or maxy < pixel[1]:
                        maxy = pixel[1]
            if minx < 1:
                minx = 1
            if miny < 1:
                miny = 1
            if maxx > picture_mat.size[0]:
                maxx = picture_mat.size[0]
            if maxy > picture_mat.size[1]:
                maxy = picture_mat.size[1]
            bounding_box = (minx, miny, maxx, maxy)
            if None in bounding_box:
                continue
            bounding_box_list.append((item['label'], bounding_box))
        
        file_to_write_bounding_box = file_id + '_gtFine_bounding_box.txt'
        filepath_to_write_bounding_box = os.path.join(current_annotation_dir, city, file_to_write_bounding_box)
        store_bbox_to_file(filepath_to_write_bounding_box, bounding_box_list)

