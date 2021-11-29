from scipy import spatial
import numpy as np

class ConfigDB:

    data = None

    def __init__(self):
        self.data = []


    def dim(self):
        '''定义数据维度'''
        return len(self.data)


    def add(self, conf):
        '''添加配置'''
        self.data += [conf]


    def dist(self, conf1, conf2):
        '''获取距离'''
        d = 0
        # 背景
        if conf1[0] != conf2[0]:
            d = d + 1

        # 汽车
        d = d + np.equal(conf1[1:4],conf2[1:4]).tolist().count(False)

        # 其他的特征
        d = d + np.linalg.norm(np.array(conf1[4:])-np.array(conf2[4:]))

        return d


    def dist_closest_neigh(self, conf):
        '''距最近存储点的距离'''
        min_dist = 999
        for dt in self.data:
            d = self.dist(dt,conf)
            if d < min_dist:
                min_dist = d
        return min_dist


    def eps_dist(self, conf, eps):
        '''判断是否离存储点过远'''
        for dt in self.data:
            if self.dist(dt,conf) < eps:
                return False
        return True


    def print_data(self):
        '''打印存储的数据'''
        for d in self.data:
            print(d)
