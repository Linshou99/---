import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

df_pos_mercator = pd.read_csv('./data/event_link_set_all_beijing_1km_filtered.txt', sep='\t', header=None)
df_pos_mercator.columns = ['link_id','longitude','latitude']
indices_6392 = pd.read_pickle('./data/indices_6392.pkl')


num_nodes = len(indices_6392)
dis_mat = np.zeros((num_nodes, num_nodes))
m,n= 0,0
for i in tqdm(indices_6392):
    n=0
    for j in indices_6392:
        if i!=j and dis_mat[m][n] == 0:
            dis_mat[m][n] = np.linalg.norm(df_pos_mercator.iloc[i,1:] - df_pos_mercator.iloc[j,1:])
            dis_mat[n][m] = dis_mat[m][n]
        n+=1
    m+=1

with open('./data/dis_mat_6392.pkl', 'wb') as file:
    pickle.dump(dis_mat, file)
file.close()

dis_mat_ = pd.read_pickle('./data/dis_mat_6392.pkl')
# 判断是否是对称矩阵
if (dis_mat_ == dis_mat_.T).all():
    print("矩阵是对称矩阵。")
else:
    print("矩阵不是对称矩阵。")