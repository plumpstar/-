'''从评估文件中提取信息'''

eval_dir="C:/Users/蒋睿兮/Desktop/工具复现/test_m_0_2"
eval_file="C:/Users/蒋睿兮/Desktop/工具复现/test_1.txt"
##用于解析和评估数据
import matplotlib.pyplot as plt
import numpy as np


def get_data(file_name):
    '''从评估文件中获取数据'''
    f = open(file_name,"r")
    eval_cont = f.read()
    easy_acc_i = eval_cont.find("car_easy: ")
    medium_acc_i = eval_cont.find("car_medium: ")
    hard_acc_i = eval_cont.find("car_hard: ")
    easy_acc = float(eval_cont[easy_acc_i+10:easy_acc_i+15])
    medium_acc = float(eval_cont[medium_acc_i+12:medium_acc_i+17])
    hard_acc = float(eval_cont[hard_acc_i+10:hard_acc_i+15])
    return easy_acc, medium_acc, hard_acc

def get_data_dir(dir_path, checkpoints):
    '''从评估目录中获取数据'''
    easys = []
    mediums = []
    hards = []
    avgs = []
    for checkpoint in checkpoints:
        file_name = str(checkpoint) + '.txt'
        easy, medium, hard = get_data(dir_path + '/' + file_name)
        easys += [easy]
        mediums += [medium]
        hards += [hard]
        avgs += [np.mean([easy, medium, hard])]
    return easys, mediums, hards, avgs

checkpoints = range(0,4600,200)
easys, mediums, hards, avgs = get_data_dir(eval_dir, checkpoints)
plt.plot(checkpoints, easys, 'r-', checkpoints, mediums, 'g-', checkpoints, avgs, 'b-',)
plt.show()
