'''
该文件包含用于创建新图像的所有基本原语。
基本对象的特征是图像数据，从图像文件中采集
以及用户想要提供的任何其他附加参数。
'''

from PIL import Image
from image_generation_utils import *

class im_object(object):
    def __init__(self, im_data, add_details=None):
        self.image = im_data
        self.add_details = add_details


class back_object(im_object):
    def __init__(self, im_data=None, add_details=None, im_path=None):
        if im_data is None:
            im_data = Image.open(im_path).data
        self.bounding_boxes = None
        self.scaling = scale(1., 1.)
        super(back_object, self).__init__(im_data, add_details)

    def update_sample_box(self, boxes, scale=None):
        self.bounding_boxes = boxes
        self.homography_h = unit_to_bb_h(self.bounding_boxes)
        if scale is not None:
            self.scaling = scale



class fore_object(im_object):
    def __init__(self, im_data=None, add_details=None, im_path=None):
        if im_data is None:
            im_data = Image.open(im_path).data
        super(fore_object, self).__init__(im_data, add_details)
