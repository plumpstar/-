from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import time
import sys
import os
import glob
## 链接squeezeDet
import numpy as np
import tensorflow as tf


import sys
from squeezeDet.src.config.kitti_squeezeDet_config import kitti_squeezeDet_config
from squeezeDet.src.nets.squeezeDet import SqueezeDet

ROOT_PATH = './squeezeDet/'
sys.path.insert(0, ROOT_PATH + 'src')

PROB_THRESH = 0.5

from config import *
from train import _draw_box
from nets import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('mode', 'image', """'image' or 'video'.""")
#tf.app.flags.DEFINE_string('checkpoint', ROOT_PATH + 'data/model_checkpoints/squeezeDet/model.ckpt-87000',"""Path to the model parameter file.""")
tf.app.flags.DEFINE_string('checkpoint', str('路径'),"""Path to the model parameter file.""")
##/home/tommaso/Desktop/logs/SqueezeDet/train/model.ckpt-2000
def init(checkpoint_path):

  with tf.Graph().as_default():

    mc = kitti_squeezeDet_config()
    mc.BATCH_SIZE = 1
    mc.LOAD_PRETRAINED_MODEL = False
    model = SqueezeDet(mc, FLAGS.gpu)

    FLAGS.checkpoint = str(checkpoint_path)

    saver = tf.train.Saver(model.model_params)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver.restore(sess, FLAGS.checkpoint)

    return (sess,mc,model)



def classify(im_path, conf, prob_thresh):

    (sess,mc,model) = conf
    im = cv2.imread(im_path)
    im = im.astype(np.float32, copy=False)
    im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
    input_image = im - mc.BGR_MEANS

    det_boxes, det_probs, det_class = sess.run(
        [model.det_boxes, model.det_probs, model.det_class],
        feed_dict={model.image_input:[input_image]})

    final_boxes, final_probs, final_class = model.filter_prediction(
     det_boxes[0], det_probs[0], det_class[0])

    keep_idx = [idx for idx in range(len(final_probs)) \
                if final_probs[idx] > prob_thresh]

    final_boxes = [final_boxes[idx] for idx in keep_idx]
    final_probs = [final_probs[idx] for idx in keep_idx]
    final_class = [final_class[idx] for idx in keep_idx]

    res = []
    for label, confidence, box in zip(final_class, final_probs, final_boxes):
        res.append((label,confidence,box))
    return res


