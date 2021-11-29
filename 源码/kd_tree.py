from scipy import spatial
import numpy as np

class KDTree:

    tree = None

    def dim(self, pt):
        '''数据维度'''
        return self.tree.data.shape


    def add(self, pt, config=True):
        if self.tree == None:
            self.tree = spatial.cKDTree([pt])
        else:
            self.tree = spatial.cKDTree(np.append(self.tree.data,[pt],axis=0))


    def query(self, pt, config=True):
        if config:
            pt = self.flat_config(pt)
        dist, neigh_idx = self.tree.query(pt)
        return dist, self.tree.data[neigh_idx]


    def esp_distant(self, pt, eps, add=False, config=True):
        if config:
            pt = self.flat_config(pt)
        if self.tree != None:
            dist, _  = self.tree.query(pt)
        else:
            dist = 999

        eps_dist = dist >= eps
        if( eps_dist and add ):
            self.add(pt)
        return eps_dist, dist


    def flat_config(self, conf):
        MAX_NUM_CARS = 3
        pad = MAX_NUM_CARS - len(conf[1])

        pt = conf[0]
        pt += (conf[1] + [-1]*pad)
        pt += (conf[2] + [-1]*pad)
        pt += (conf[3] + [-1]*pad)
        pt += (conf[3] + [-1]*pad)
        pt += conf[4]
        pt += conf[5]
        pt += conf[6]
        pt += conf[7]

        return pt
