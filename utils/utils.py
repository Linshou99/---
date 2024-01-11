from datetime import datetime
import os.path as osp
import os
import logging

import yaml
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.linalg import eigh
import torch

# ?
class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
        self.xs = xs
        self.ys = ys

    def _shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return int(self.num_batch)

    def __iter__(self):
        return self.get_iterator()

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                x_i, y_i = totensor([x_i, y_i], dtype=torch.float32)
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


# 返回了一个包含数据集和模型参数的字典 dataset_model_args
def get_dataset_model_args(dataset, model):
    # yaml.safe_load(open(osp.join('cfgs', f'{dataset}_{model}.yaml')))：
    # 这行代码的作用是打开一个名为 {dataset}_{model}.yaml 的配置文件，
    # 该文件存储了有关数据集和模型的参数配置。
    # 这个文件通常是用 YAML 格式编写的，以便轻松地存储结构化的参数信息。
    # osp.join('cfgs', f'{dataset}_{model}.yaml')：
    # 这部分代码生成配置文件的路径，osp.join 用于将目录路径 'cfgs' 
    # 和文件名 {dataset}_{model}.yaml 连接在一起
    dataset_model_args = yaml.safe_load(open(osp.join('cfgs', f'{dataset}_{model}.yaml')))
    return dataset_model_args

# 创建新的实验目录的路径并返回
def create_exp_dir(dataset, model, name):
    # 函数使用 osp.join 函数构建了一个实验目录的路径 exp_dir。
    # 这个路径通常由三个部分组成，分别是 'exps'、dataset、model 和 name，
    # 它们通过路径分隔符连接起来，例如 '/'。
    exp_dir = osp.join('exps', dataset, model, name)
    # os.makedirs 函数来创建实验目录。
    # 如果该目录已经存在，那么 exist_ok=True 选项将确保不会引发错误
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

#计算模型所有参数的字节总数
def model_size(model, type_size=4):
    size = 0
    for p in model.parameters():
        size += p.numel() * type_size  # Bytes

    return size

def get_logger(exp_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    hdlr = logging.FileHandler(osp.join(exp_dir, 'log.txt'))
    console = logging.StreamHandler()
    fmtr = logging.Formatter('%(message)s')
    hdlr.setFormatter(fmtr)
    console.setFormatter(fmtr)

    logger.addHandler(hdlr)
    logger.addHandler(console)

    return logger

# 求平均
class Average(object):
    def __init__(self):
        self._sum = 0
        self._count = 0
    def add(self, value, count):
        self._sum += value
        self._count += count
    def average(self):
        return self._sum / self._count
    
# .to(device)
def move2device(x, device):
    if isinstance(x, list):
        y = []
        for item in x:
            y.append(move2device(item, device))
    elif isinstance(x, dict):
        y = {}
        for k, v in x.items():
            y[k] = move2device(v, device)
    elif x is None:
        y = None
    else:
        y = x.to(device)

    return y

# 归一化：zscore标准化和min-max归一化
def normalize(tensors, mean, std, type='zscore'):
    y = []
    for x in tensors:
        if type == 'zscore':
            y.append((x - mean) / std)
        elif type == 'maxmin':
            _max, _min = mean, std
            z = (x - _min) / (_max - _min)
            z = z * 2 - 1  # [-1, 1]
            y.append(z)
        else:
            raise ValueError('type should be zscore or maxmin')

    return y

# 反normalize，恢复normalize过的data
def denormalize(tensors, mean, std, type='zscore'):
    y = []
    for x in tensors:
        if type == 'zscore':
            y.append(std * x + mean)
        elif type == 'maxmin':
            _max, _min = mean, std
            z = (x + 1) / 2
            z = z * (_max - _min) + _min
            y.append(z)
        else:
            raise ValueError('type should be zscore or maxmin')

    return y

# def compute_graph_sml(data, delta):
#     n, c = data.shape
#     graph_sml = np.zeros((n, n), dtype=np.float32)
#     for i in range(n):
#         for j in range(i, n):
#             a = np.linalg.norm(data[i] - data[j])**2
#             b = np.minimum(np.linalg.norm(data[i])**2, np.linalg.norm(data[j])**2)
#             c = np.exp(-a / b)
#             if c > delta:
#                 graph_sml[j, i] = graph_sml[i, j] = c

#     return graph_sml

def zero_diagonals(x):
    y = x.copy()
    y[np.diag_indices_from(y)] = 0

    return y

# # 依据conn邻接矩阵生成eigenmaps,用于生成空间嵌入
# def compute_eigenmaps(adj_mx, k):
#     A = adj_mx.copy()
#     row, col = A.nonzero()
#     A[row, col] = A[col, row] = 1  # 0/1 matrix, symmetric

#     n_components = connected_components(csr_matrix(A), directed=False, return_labels=False)
#     assert n_components == 1  # the graph should be connected

#     n = A.shape[0]
#     A = zero_diagonals(A)
#     D = np.sum(A, axis=1)**(-1/2)
#     L = np.eye(n) - (A * D).T * D  # normalized Laplacian

#     _, v = eigh(L)
#     eigenmaps = v[:, 1:(k + 1)]  # eigenvectors corresponding to the k smallest non-trivial eigenvalues

    return eigenmaps

# 用于生成转移矩阵TM
def add_self_loop(A):
    B = A.copy()
    B[np.diag_indices_from(B)] = 1.0

    return B

# 用于生成转移矩阵TM
def row_normalize(A):
    A = A.astype(np.float32)
    S = (A.T * np.sum(A, axis=1)**(-1)).T

    return S

#list或dict转tensor
def totensor(x, dtype):
    if isinstance(x, list):
        y = []
        for item in x:
            y.append(totensor(item, dtype))
    elif isinstance(x, dict):
        y = {}
        for k, v in x.items():
            y[k] = totensor(v, dtype)
    elif x is None:
        y = None
    else:
        y = torch.as_tensor(x, dtype=dtype)

    return y


def save_metrics(rmse, mae, mape, file):
    data = torch.stack((rmse, mae, mape), dim=1).numpy()
    metrics = pd.DataFrame(data, columns=['rmse', 'mae', 'mape'])
    metrics.to_csv(file, index=False)

    return metrics


# def compute_normalized_laplacian(adj_mx):
#     A = zero_diagonals(adj_mx)  # remove self-loops
#     A = np.maximum(A, A.T)  # symmetrization

#     D = A.sum(1)  # degrees
#     D[D == 0] = np.inf
#     D_rs = D**(-1/2)

#     n = A.shape[0]
#     I = np.eye(n)
#     normalized_L = I - (A * D_rs).T * D_rs  # I - D^(-1/2)AD^(-1/2)

#     return normalized_L