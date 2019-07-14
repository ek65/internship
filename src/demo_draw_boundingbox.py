import cv2
import os 
import numpy as np
import subprocess
from dataset.imdb import imdb
from utils.util import bbox_transform_inv, batch_iou

_image_set = "train"
_data_root_path = "./data/gta"
_image_root_path = "./data/out"

# "GROUND_TRUTH" is for overlaying prediction box onto GT box
# "DATASET" is for any training/test dataset
IMAGE_TYPE = "GROUND_TRUTH" # OR "DATASET"

_image_path = os.path.join(_image_root_path, 'training_groundtruth')
_label_path = os.path.join(_data_root_path, 'prediction', 'test_example')
image_set_file = os.path.join(_data_root_path, "ImageSets", 'trainval.txt')

assert os.path.exists(image_set_file), \
        'File does not exist: {}'.format(image_set_file)

with open(image_set_file) as f:
  _image_idx = [x.strip() for x in f.readlines()]

# print(_image_idx)

idx2annotation = {}
for index in _image_idx:
  filename = os.path.join(_label_path, index+'.txt')
  with open(filename, 'r') as f:
    lines = f.readlines()
  f.close()
  bboxes = []
  im = cv2.imread(os.path.join(_image_path, 'out_'+index+".png"))

  for line in lines:
    obj = line.strip().split(' ')
    print(obj)

    if(IMAGE_TYPE == 'DATASET'):
      xmin = int(obj[4])
      ymin = int(obj[5])
      xmax = int(obj[6])
      ymax = int(obj[7])

    if(IMAGE_TYPE == 'GROUND_TRUTH'):
      xmin = int(obj[2])
      ymin = int(obj[3])
      xmax = int(obj[4])
      ymax = int(obj[5])


    print(filename)
    assert xmin >= 0.0 and xmin <= xmax, \
        'Invalid bounding box x-coord xmin {} or xmax {} at {}.txt' \
            .format(xmin, xmax, index)

    assert ymin >= 0.0 and ymin <= ymax, \
        'Invalid bounding box y-coord ymin {} or ymax {} at {}.txt' \
            .format(ymin, ymax, index)

    x, y, w, h = bbox_transform_inv([xmin, ymin, xmax, ymax])
    bboxes.append([x, y, w, h])

    # print(im.size())
    im = im.astype(np.float32, copy=False)
    im = cv2.resize(im, (1920, 1200))
    color = (0,0,255)
    cv2.rectangle(im, (xmin, ymin), (xmax, ymax), color, 1)

    file_name = index
    out_file_name = os.path.join("./data/out/test_example", 'out_'+file_name+".png")
    cv2.imwrite(out_file_name, im)
    print ('Image detection output saved to {}'.format(out_file_name))

  idx2annotation[index] = bboxes
  # print(idx2annotation[index])


