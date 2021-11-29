from gen_utils import *
import squeezedet as nn
##重要逻辑段,用于对于模型的各种操作
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

import pickle
import random

from squeezeDet.src.utils.util import iou

IOU_THRESH = 0.5
NN_PROB_THRESH = 0.5

def read_label(file_name):
    '''从文件中读取特征值'''
    with open(file_name, "r") as f:
        content = f.readlines()
    content = [c.strip().split(" ") for c in content]
    labels = []
    for c in content:
        labels += [[float(c[4]), float(c[5]), float(c[6]), float(c[7])]]

    return labels

def read_image_set(image_set):
    '''从图像集中读取图像名称'''
    with open(image_set, "r") as f:
        content = f.readlines()
    images = [c.strip() for c in content]

    gt_labels = dict()
    for i in images:
        labels = read_label(PREFIX_LABELS + i + '.txt')
        label_boxes = [ kitti_2_box_format(l) for l in labels ]
        gt_labels[i] = label_boxes
    return gt_labels


def predict(net, image_set):
    '''对来自 image_set 的图像运行神经网络'''

    with open(image_set, "r") as f:
        content = f.readlines()
    images = [c.strip() for c in content]

    predictions = dict()
    for i in images:
        (boxes, probs, _) = nn.classify(PREFIX_IMAGES + i + '.png', net, NN_PROB_THRESH)
        predictions[i] = boxes

    return predictions


def average_precision(gt, prediction, iou_thresh):
    '''平均预测精度'''
    alread_detected = [False]*len(gt)
    tp = 0
    fp = 0
    fn = 0

    if not(prediction):
        return 0, 0

    for pred in prediction:
        detect = False
        for i in range(len(gt)):
            #输出相关数据
            if iou(pred, gt[i]) > iou_thresh:
                detect = True
                if not alread_detected[i]:
                    tp += 1
                    alread_detected[i] = True
                else:
                    fp += 1
        if not detect:
            fp += 1
    fn = alread_detected.count(False)
    ap = tp/float(tp+fp)
    recall = tp/float(tp+fn)
    return ap, recall


def plot_boxes(image, gt, pred):
    '''绘制 gt 和预测框'''
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


def get_ap_rec(res):
    '''从结果中获取检查点、平均精度和召回率'''
    cps = []
    aps = []
    recs = []
    for cp in res:
        (ap,rec) = res[cp]
        cps += [cp]
        aps += [ap]
        recs += [rec]
    order = np.argsort(cps)
    x = np.array(cps)[order]
    y1 = np.array(aps)[order]
    y2 = np.array(recs)[order]
    return x, y1, y2



def eval_set(net, image_set):
    '''在 image_set 上评估网络'''

    gt_labels = read_image_set(image_set)
    predictions = predict(net, image_set)

    tot_ap = 0
    tot_rec = 0
    for image in gt_labels:
        gt = gt_labels[image]
        pred = predictions[image]
        ap, rec = average_precision(gt, pred, IOU_THRESH)
        tot_ap += ap
        tot_rec += rec

    tot_ap = tot_ap/float(len(gt_labels))
    tot_rec = tot_rec/float(len(gt_labels))

    return tot_ap, tot_rec


def eval_train(checkpoint_dir, checkpoint_list, image_set):
    '''评估对存储检查点列表的培训'''
    results = dict()
    for cp in checkpoint_list:
        cp_path = checkpoint_dir + 'model.ckpt-' + str(cp)
        print('Evaluating: ' + cp_path)
        net = nn.init(cp_path)
        ap, recall = eval_set(net, image_set)
        results[cp] = (ap,recall)
    return results


def gen_augmented_set(image_set, i_start, i_end, n):
    '''使用索引在 i_start 和 i_end 之间的 n 个图像增强图像集'''

    for i in range(1,4):

        PREFIX = 'm_2_' + str(i) + '_'

        arr = range(i_start,i_end)
        random.shuffle(arr)

        arr = arr[0:n]

        with open(image_set, 'a') as f:
            for a in arr:
                f.write(PREFIX + str(a).zfill(6) + '\n')


def multiple_checkpoint_eval():

    checkpoint_list = range(0,5250,250)

    for j in range(1,4):
        checkpoint_dir = './data/train_0/' + 'checkpoint' + '/train/'
        res = eval_train(checkpoint_dir, checkpoint_list, image_set)
        pickle.dump( res, open( "save_mis_05_" + str(j) +".p", "wb" ) )


def avg_eval(result_list):
    '''计算结果的平均值'''
    all_ap = []
    all_rec = []
    for res_l in result_list:
        res = pickle.load( open( res_l, "rb" ) )
        cp, ap, rec = get_ap_rec(res)
        all_ap += [ap]
        all_rec += [rec]
    return cp, np.mean(all_ap,axis=0), np.mean(all_rec,axis=0)

checkpoint_list = [500]


dirs = 'mis_0', 'mis_halt', 'class_aug', 'mis_dist', 'mis_mix'
sets = 'test_mis.txt', 'test_halt.txt', 'test_aug.txt', 'test_mis.txt', 'test_mix.txt'

test = 3

PREFIX = '路径'
##/home/tommaso/squeezeDet/data/webots/test/
PREFIX_LABELS = PREFIX + 'labels/'
PREFIX_IMAGES = PREFIX + 'images/'
image_set = 'test.txt'


# ratio = 0
res = eval_train('路径', checkpoint_list, image_set)
##/home/tommaso/squeezeDet/data/webots/train/checkpoints
file_name = 'save_0_test_err.p'
pickle.dump( res, open( file_name, "wb" ) )
_, ap, rec = avg_eval([file_name])
print('ap: ' + str(np.max(ap)))
print('rec: ' + str(np.max(rec)))

