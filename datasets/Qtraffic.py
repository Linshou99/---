import sys
sys.path.append('/home/智慧城市final/utils')
import pickle
import os.path as osp
from datetime import datetime
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.utils import *
from utils.Metrics import *
class Qtraffic(Dataset):
    def __init__(self, cfgs, split):
        self.root = cfgs['root']    # 根目录路径
        self.num_nodes = 6392
        self.num_features = 1
        self.in_len = 1
        self.out_len = 1
        self.num_intervals = 96
        self.interval = 15
        self.eigenmaps_k = cfgs.get('eigenmaps_k', 8)
        self.similarity_delta = cfgs.get('similarity_delta', 0.1)

        # 打开train.pkl或val.pkl或test.pkl
        with open(osp.join(self.root, f'{split}_all.pkl'), 'rb') as f:
            self.data = pickle.load(f)

        if split == 'train':
            mean, std = self.compute_mean_std()
            graph_conn = self.gen_graph_conn()  # provided by PVCGN
            graph_sml = self.gen_graph_sml()
            graphs = {'graph_conn': graph_conn, 'graph_sml': graph_sml}
            eigenmaps = self.gen_eigenmaps()
            transition_matrices = self.gen_transition_matrices(graphs)
            self.mean, self.std, eigenmaps, transition_matrices = totensor(
                [mean, std, eigenmaps, transition_matrices],
                dtype=torch.float32)
            graphs = totensor(graphs, dtype=torch.float32)
            self.statics = {'eigenmaps': eigenmaps,
                            'transition_matrices': transition_matrices,
                            'graphs': graphs}

    def __len__(self):
        return len(self.data['x'])

    def __getitem__(self, item):
        # traffic speed
        inputs = self.data['x'][item]
        targets = self.data['y'][item]
        # 一天内的第几个时间片，(batch_size, 1)、(batch_size, 1)
        inputs_time = self.data['xtime0'][item]
        targets_time = self.data['ytime0'][item]
        
        # 时间片属性，(batch_size, 6)、(batch_size, 6)
        inputs_attr1 = self.data['xtime1'][item]
        targets_attr1 = self.data['ytime1'][item]
        inputs_attr2 = self.data['xtime2'][item]
        targets_attr2 = self.data['ytime2'][item]
        inputs_attr3 = self.data['xtime3'][item]
        targets_attr3 = self.data['ytime3'][item]
        inputs_attr4 = self.data['xtime4'][item]
        targets_attr4 = self.data['ytime4'][item]
        inputs_attr5 = self.data['xtime5'][item]
        targets_attr5 = self.data['ytime5'][item]
        inputs_attr6 = self.data['xtime6'][item]
        targets_attr6 = self.data['ytime6'][item]

        inputs, targets = totensor([inputs, targets], dtype=torch.float32)
        inputs_time, targets_time, inputs_attr1, targets_attr1,inputs_attr2, targets_attr2,inputs_attr3, targets_attr3,inputs_attr4, targets_attr4,inputs_attr5, targets_attr5,inputs_attr6, targets_attr6 = totensor(
            [inputs_time, targets_time, inputs_attr1, targets_attr1,inputs_attr2, targets_attr2,inputs_attr3, targets_attr3,inputs_attr4, targets_attr4,inputs_attr5, targets_attr5,inputs_attr6, targets_attr6], dtype=torch.int64)
        # extras=(inputs_time， targets_time，inputs_attr, targets_attr)
        return inputs, targets, inputs_time, targets_time, inputs_attr1, targets_attr1,inputs_attr2, targets_attr2,inputs_attr3, targets_attr3,inputs_attr4, targets_attr4,inputs_attr5, targets_attr5,inputs_attr6, targets_attr6

    def compute_mean_std(self):
        mean = self.data['x'].mean()
        std = self.data['x'].std()
        return mean, std

    def gen_graph_conn(self):
        with open(osp.join(self.root, 'adj_matrix_filtered_6392.pkl'), 'rb') as f:
            graph_conn = pickle.load(f).astype(np.float32)  # symmetric, with self-loops

        return graph_conn
    
    def gen_graph_sml(self):
        with open(osp.join(self.root, 'sml_mat_6392.pkl'), 'rb') as f:
            graph_sml = pickle.load(f).astype(np.float32)  # symmetric, with self-loops

        return graph_sml
    
    def gen_eigenmaps(self):
        with open(osp.join(self.root, 'eigenmaps_6392.pkl'), 'rb') as f:
            eigenmaps = pickle.load(f).astype(np.float32)  # symmetric, with self-loops

        return eigenmaps


    def gen_transition_matrices(self, graphs):
        # transform adjacency matrices (value span: 0.0~1.0, A(i, j) is the weight from j to i)
        # to transition matrices
        S_conn = row_normalize(add_self_loop(graphs['graph_conn']))
        S_sml = row_normalize(add_self_loop(graphs['graph_sml']))
        S = np.stack((S_conn, S_sml), axis=0)
        return S

if __name__ == '__main__':
    cfgs = yaml.safe_load(open('../cfgs/Qtraffic.yaml'))['dataset']
    train_set = Qtraffic(cfgs, split='train')
    val_set = Qtraffic(cfgs, split='val')
    test_set = Qtraffic(cfgs, split='test')
    batch = train_set[0]