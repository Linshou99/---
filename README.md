### 智慧城市课程设计2020级

#### 基于实时交通速度预测的动态路径规划的交通速度预测部分
```
|-- cfgs  # MGT模型配置文件
|   |-- Qtraffic_MGT.yaml
|-- datasets 
|   |-- Qtraffic.py # Qtraffic的dataset类构造
|-- data  # Qtraffic数据集以及预处理后的文件
|   |-- event_traffic_beijing_1km_mv_avg_15min_completion.pkl   #来源Qtraffic数据集，4.5w条路段的交通速度数据集
|   |-- road_network_sub-dataset.csv    #来源Qtraffic数据集，4.5w条路段的路网数据集，包含路网结构和路段属性信息
|   |-- time_feature_15min.pkl  # 来源Qtraffic数据集的时间特征提取
|   |-- event_link_set_all_beijing_1km_filtered.txt # 来源Qtraffic数据集的各路段经纬度坐标
|   |-- adj_matrix_filtered_6392.pkl # 构成强连通图的子路网的邻接矩阵
|   |-- eigenmaps_6392.pkl  # 子路网的特征映射矩阵
|   |-- indices_6392.pkl    # 子路网的路段在路网数据集的索引
|   |-- road_network_linkid_filtered_6392.pkl   # 子路网的路段id
|   |-- dis_mat_6392.pkl    # 子路网各路段间距离矩阵
|   |-- sml_mat_6392.pkl    # 子路网相似性矩阵
|   |-- inputs.pkl  # 模型输入（中间处理）
|   |-- targets.pkl # 模型目标输出（中间处理）
|   |-- train.pkl   # 训练集（模型的最终输入，inputs.pkl、target.pkl经preprocess_dataset.ipynb的处理结果，val.pkl、test.pkl同理）
|   |-- val.pkl     # 验证集（模型的最终输入）
|   |-- test.pkl    # 测试集（模型的最终输入）
|   |-- test_all.pkl    # 用于模型预测所有时间片子路网交通速度，训练集+验证集+测试集（模型的最终输入）
|-- models
|   |-- MGT.py  # MGT时空交通速度预测模型
|-- main.py # 执行模型训练和预测
|-- adj_mat.ipynb生成邻接矩阵
|-- adj_mat_filtered.ipyhb过滤生成强连通图子路网的邻接矩阵
|-- dis_mat.ipynb/dis_mat.py计算每条路段之间的距离，对称矩阵
|-- eigenmaps.ipynb特征映射：依据邻接矩阵进行拉普拉斯特征映射并降维，生成特征矩阵（方阵）
|-- preprocess_dataset.ipynb对/data中的train.pkl、val.pkl、test.pkl数据作进一步处理，生成最终模型dataset的输入
|-- real_predict_speed.ipynb画图（其中一张）
|-- sml_mat.ipynb依据原论文公式生成相似性矩阵
|-- exps/Qtraffic/MGT/E03  # 模型训练和预测结果，模型best.pth文件和预测结果文件output_all.pth见网盘
|   |-- metrics_all.csv # 总数据集（6932）的指标计算结果（训练集+验证集+测试集）
|   |-- metrics.csv # 测试集指标计算结果
|   |-- output.pth  # 测试集预测结果
|   `-- output_all.pth  # 总数据集预测结果
`-- utils  
```

#### 数据文件补充
/data以及模型训练的部分实验结果见百度网盘，链接及密码在报告最后

#### 模型训练及预测运行命令
```shell
pip install requirements.txt
# python main.py <dataset> MGT <experiment name> <CUDA device>
python main.py Qtraffic MGT E01 0
```

#### 参考
Qtraffic数据集https://github.com/JingqingZ/BaiduTraffic/tree/master

MGT时空交通预测模型 https://github.com/lonicera-yx/MGT
