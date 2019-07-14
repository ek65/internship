#from gen_utils import *
import squeezedet as nn

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

import pickle
import random


def get_area_cap(b1, b2):
    x1_c, y1_c, l1, w1 = b1
    x2_c, y2_c, l2, w2 = b2

    left_1 = x1_c - (l1/2)
    right_1 = x1_c + (l1/2)
    top_1 = y1_c - (w1/2)
    bot_1 = y1_c + (w1/2)
    left_2 = x2_c - (l2/2)
    right_2 = x2_c + (l2/2)
    top_2 = y2_c - (w2/2)
    bot_2 = y2_c + (w2/2)

    # print("[r1,l1,t1,b1]")
    # print([right_1, left_1, top_1, bot_1])
    # print("[r2,l2,t2,b2]")
    # print([right_2, left_2, top_2, bot_2])

    left_cap = max(left_1, left_2)
    right_cap = min(right_1, right_2)
    top_cap = max(top_1, top_2)
    bot_cap = min(bot_1, bot_2)

    area_1 = l1*w1
    area_2 = l2*w2

    intersection_length = max(0, right_cap-left_cap)
    intersection_height = max(0, bot_cap-top_cap)

    area_cap = intersection_length*intersection_height

    return max(area_cap,0)


def iou(prediction, gt):

    x1_c, y1_c, l1, w1 = prediction
    x2_c, y2_c, l2, w2 = gt

    area_1 = l1*w1
    area_2 = l2*w2

    area_cap = get_area_cap((x1_c, y1_c, l1, w1), (x2_c, y2_c, l2, w2))
    # print("area_cap : ")
    # print(area_cap)

    # print("area1 : ")
    # print(area_1)
    # print("area2 : ")
    # print(area_2)

    return area_cap/float(area_1+area_2-area_cap)

def iou2(prediction, gt): 
    # output: intersection over the prediction box, not the union
    # purpose: to check nested detection

    x1_c, y1_c, l1, w1 = prediction
    x2_c, y2_c, l2, w2 = gt

    area_1 = l1*w1
    area_2 = l2*w2

    area_cap = get_area_cap((x1_c, y1_c, l1, w1), (x2_c, y2_c, l2, w2))

    return float(area_cap/area_1)


def box_2_kitti_format(box):
    '''Transform box for KITTI label format'''
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    left = int(x - w/2)
    right = int(x + w/2)
    top = int(y - h/2)
    bot = int(y + h/2)
    return [left,top,right,bot]

def kitti_2_box_format(label):

    '''Transform KITTI label format to box'''
    xl = label[0]
    yt = label[1]
    xr = label[2]
    yb = label[3]
    w = xr - xl
    h = yb - yt
    xc = int(xl + w/2)
    yc = int(yt + h/2)
    return [xc, yc, w, h]

def read_label(file_name):
    '''Read label from file (KITTI format)'''
    with open(file_name, "r") as f:
        content = f.readlines()
    content = [c.strip().split(" ") for c in content]
    labels = []
    for c in content:
        if c[0] == 'Car':
            labels += [[float(c[4]), float(c[5]), float(c[6]), float(c[7])]]

    return labels

def read_image_set(image_set):
    '''Read image names from image set'''
    with open(image_set, "r") as f:
        content = f.readlines()
    images = [c.strip() for c in content]

    gt_labels = dict()
    for i in images:
        labels = read_label(PREFIX_LABELS + i + '.txt')

        if labels != []:
            label_boxes = [ kitti_2_box_format(l) for l in labels ]
            gt_labels[i] = label_boxes
            # print("print ground_truth labels : ")
            # print(label_boxes)

    return gt_labels


def predict(net, image_set):
    '''Run neural net on images from image_set'''

    with open(image_set, "r") as f:
        content = f.readlines()
    images = [c.strip() for c in content]

    predictions = dict()
    for i in images:
        file_name = PREFIX_IMAGES + i + '.png'
        pred = nn.classify(file_name, net, NN_PROB_THRESH)

        car_preds = []
        for p in pred:
            if p[0] == 0:
                car_preds.append([0, p[1]] + [int(p[2][0]), int(p[2][1]), int(p[2][2]), int(p[2][3])])

        for p in car_preds:
            scale_x = 1.5385
            scale_y = 3.125
            p[2] = int(p[2]*scale_x)
            p[3] = int(p[3]*scale_y)
            p[4] = int(p[4]*scale_x)
            p[5] = int(p[5]*scale_y)

        predictions[i] = car_preds

    return predictions

def plot_boxes(image, gt, pred):
    '''Plot gt and prediction boxes'''
    im = np.array(Image.open(PREFIX_IMAGES + image + '.png'), dtype=np.uint8)
    fig,ax = plt.subplots(1)
    ax.imshow(im)
    for box in gt:
        rect = patches.Rectangle((box[0]-box[2]/2, box[1]-box[3]/2),box[2],box[3],linewidth=1,edgecolor='b',facecolor='none')
        ax.add_patch(rect)
    for box in pred:
        rect = patches.Rectangle((box[0]-box[2]/2, box[1]-box[3]/2),box[2],box[3],linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    plt.show()

def plot_results(x,ys):
    plt.plot(x, ys)
    plt.show()


def evaluate(gt, prediction, iou_thresh, iou2_thresh):
    '''Eddie: categorize whether the prediction is correct/incorrect'''
    # Criteria : 
    # 1) len(prediction)>len(gt) is still correct as long as all ground truth boxes are detected over iou_threshold i.e. double detection is okay
    # 2) predicting non-car to be car is incorrect
    # 3) nested detection is okay (as long as )

    # Input Format: 
    # gt, prediction : list of lists = [[xc1, yc1, w1, h1],[xc2, yc2, w2, h2], ..]
    # Note that in read_image_set() the format of (x,y) coordinates that
    # are read from label.txt files are converted to "box" format via
    # kitti_2_box_format() function

    # check whether boxes match
    already_detected = [False]*len(gt)

    for pred in prediction:
        detect = False

        for i in range(len(gt)):
            if iou(pred, gt[i]) > iou_thresh:
                detect = True
                already_detected[i] = True
                # print("predicted correctly")
                # print("iou(pred,gt[i]) : ")
                # print(iou(pred, gt[i]))
                
            else: # Enforcing Criteria #3
                if iou2(pred, gt[i]) > iou2_thresh: # nested pred_box in gt
                    detect = True
                    # print("predict_box included in gt")
                    # print("iou2(pred,gt[i]) : ")
                    # print(iou2(pred, gt[i]))
        

        if not detect: # Enforcing Criteria #2
            return False 

    # Enforcing Criteria #1
    for detection in already_detected:
        if detection != True:
            return False

    return True


def evaluate_set(net, image_set, verbose=True):
    '''Evaluate net on image_set'''

    gt_labels = read_image_set(image_set)
    predictions = predict(net, image_set)

    # print("ground_truth and predictions all successfully loaded")
    # print(predictions)

    for file in gt_labels:

        gt = gt_labels[file]
        preds = predictions[file]
        pred = []
        for p in preds:
            pred.append(p[2:])

        correct = evaluate(gt, pred, IOU_THRESH, IOU2_THRESH)
    
        # save the file/predicted label/configuration
        if correct:
            preds_directory = './data/gta/test_example/model_10000_prediction_label/correctly_detected/'
            write_predictions(preds, preds_directory,file)
            # print("correct detection")

        else:
            preds_directory = './data/gta/test_example/model_10000_prediction_label/incorrectly_detected/'
            write_predictions(preds, preds_directory,file)
            # print("incorrect detection")

        ## Uncomment "plot_boxes" line below to visualize (Red=pred/ Blue=gt)
        # plot_boxes(file, gt, pred)


    return correct, predictions


def write_predictions(prediction, preds_directory, filename):
    '''Write predictions into files'''
    
    f = open(preds_directory + filename + '.txt', "w")
    for p in prediction:
        kitti = box_2_kitti_format(p[2:])
        f.write("Car " + str(p[1]) + " " + " ".join([str(x) for x in kitti]) + '\n')
    f.close()


IOU_THRESH = 0.5
IOU2_THRESH = 0.7
NN_PROB_THRESH = 0.5

PREFIX = './data/gta/training/'
PREFIX_LABELS = PREFIX + 'label/' # directory path to label
PREFIX_IMAGES = PREFIX + 'image/' # directory path to image
checkpoint = './data/model_checkpoints/squeezeDet/model.ckpt-10000'

# This .txt is generated by following instruction in squeezeDet repo README
# go to ImageSet and execute command : 
# ls ../directory_path_to_image | grep ".png" | sed s/.png// > train.txt
# this command parses out only the names of images into a .txt file
test_set = './data/gta/ImageSets/train.txt' 


net = nn.init(checkpoint)
print("model initiated!!")
evaluate_set(net, test_set)



