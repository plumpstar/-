from random import shuffle, random

from image_generation_utils import *
from image_mod_gen_utils import *
from analysis_nn import *
from update_library import *
##analysis_nn报错
##用于生成图像
lib = update_library()

def gen_image(sample):
    fg = []
    for s in range(len(sample[1])):
        fg += [fg_obj(fg_id=sample[1][s], x=sample[2][s], y=sample[3][s])]
    return gen_comp_img(lib, fg, bg_id=sample[0][0], brightness=sample[4][0], sharpness=sample[5][0], contrast=sample[5][0], color=sample[5][0])

def scale_sample(sample, sampling_domain):
    '''Scale a [0,1] sample in a given domain'''
    for i in range(len(sample)):
        sample[i] = sample[i]*(sampling_domain[i][1] - sampling_domain[i][0]) + sampling_domain[i][0]
    return sample



def random_sample(typ, domain, n_samples):

    sample = []
    for _ in range(n_samples):
        if typ == 'float':
            r = random.random()*(domain[1]-domain[0])+domain[0]
        elif typ == 'int':
            r = random.choice(range(domain[0],domain[1]))
        else:
            print('Error')
        sample.append(r)
    return sample

def random_config(domains, n_cars):
    # Scene and cars
    sample = [random_sample('int',domains[0],1)]
    sample += [random_sample('int',domains[1],n_cars)]
    # Modifications
    x_sample = []
    y_sample = []
    if n_cars > 0:
        step_x = float(domains[2][1])/n_cars
        step_y = float(domains[3][1])/n_cars

        base_x = 0
        base_y = 0
        for _ in range(n_cars):
            x_sample.append(random.uniform(base_x, base_x+step_x))
            y_sample.append(random.uniform(base_y, base_y+step_y))
            base_x += step_x
            base_y += step_y

    shuffle(x_sample)
    y_sample.sort(reverse=True)
    sample += [x_sample]
    sample += [y_sample]

    sample += [random_sample('float',domains[4],1)]
    sample += [random_sample('float',domains[5],1)]

    return sample


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



def save_image(img, file_name, path_data_set):
    '''Save image and label'''

    img_file_name = path_data_set + 'images/' + file_name + '.png'
    img.save(img_file_name)


def save_label(ground_boxes, file_name, path_data_set):
    '''Save label'''

    f = open(path_data_set + 'labels/' + file_name + '.txt', 'w')

    if len(ground_boxes) > 0:
        for box in ground_boxes[:-1]:
            label = [0,0,0] + box_2_kitti_format(box) + [0,0,0,0,0,0,0]
            label = list(map(str, label))
            label = " ".join(label)
            label  = "Car " + label + "\n"
            f.write(label)  # python will convert \n to os.linesep
        label = [0,0,0] + box_2_kitti_format(ground_boxes[-1]) + [0,0,0,0,0,0,0]
        label = list(map(str, label))
        label = " ".join(label)
        label  = "Car " + label
        f.write(label)  # python will convert \n to os.linesep
    f.close()


def pad_sample(conf):
    MAX_NUM_CARS = 3
    pad = MAX_NUM_CARS - len(conf[1])

    pt = conf[0]                # background
    pt += (conf[1] + [-1]*pad)  # car models
    pt += (conf[2] + [-1]*pad)  # x
    pt += (conf[3] + [-1]*pad)  # y
    pt += conf[4]               # image params
    pt += conf[5]
    pt += conf[6]
    pt += conf[7]

    return pt

def check_perspective(xs, ys, x_eps, y_eps):
    '''检查采样点之间的距离'''
    for yi in range(len(ys)):
        for yj in range(yi+1,len(ys)):
            if abs(ys[yi] - ys[yj]) < y_eps:
                if abs(xs[yi] - xs[yj]) < x_eps:
                    return False
    return True


def pred_2_detect(preds, gt, iou_tresh):
    '''将预测分配给检测'''
    detetctions = []
    for p in preds:
        detects = []
        for g in gt:
            detects += [iou(p, g) > iou_tresh]
        detetctions += [detects]
    assigns = []
    for d in detetctions:
        a = -1
        for i in range(len(d)):
            if d[i]:
                a = i
        assigns += [a]

    assigns = duplicate_false_positive(assigns, len(gt))
    return assigns

def duplicate_false_positive(detects, n_gt):
    '''纠正双重误报'''
    for i in range(n_gt):
        first_occ = True
        for j in range(len(detects)):
            if i == detects[j]:
                if first_occ:
                    first_occ = False
                else:
                    detects[j] = -1
    return detects


def prec_rec(detects, n_gt):
    '''计算精度和召回率'''

    tp = sum(d != -1 for d in detects)
    fp = sum(d == -1 for d in detects)
    fn = sum(g not in detects for g in range(n_gt))

    prec = tp/float(tp+fp)
    rec = tp/float(tp+fn)

    return prec, rec


def precision_recall(preds, gt_boxes, iou_tresh):
    detects = pred_2_detect(preds, gt_boxes, iou_tresh)
    return prec_rec(detects, len(gt_boxes))
