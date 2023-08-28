#-*- codeing = utf-8 -*-
#@Time : 2023/8/28 18:39
#@Author : yuning
#@File : dhaSCA.py

import torch
import pandas as pd
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse
from scipy.sparse import coo_matrix
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset, download_url
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.nn.conv import GCNConv
import numpy as np                  # 创建numpy数组库
from sklearn.cluster import KMeans  # 聚类库
import matplotlib.pyplot as plt     # 绘图库
import os
from sklearn.cluster import spectral_clustering
from sklearn.metrics import v_measure_score
import warnings
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo
from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sklearn
import time

from Function_all import partition,exPCA,load_labels,T_SNE,k_means,L2_re
########################################数据data################################################
class MyOwnDataset_train(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    #返回数据集源文件名
    @property
    def raw_file_names(self):
        return ['D:\Pre-datasets\datasets\Zheng sorted\data_normalization.csv','D:\Pre-datasets\datasets\Zheng sorted\Pearson_filtered_0.85.csv']
    #返回process方法所需的保存文件名。你之后保存的数据集名字和列表里的一
    @property
    def processed_file_names(self):
        return ['data.pt']
    # #用于从网上下载数据集
    # def download(self):
    #     # Download to `self.raw_dir`.
    #     download_url(url, self.raw_dir)
        ...
    #生成数据集所用的方法
    def process(self):
        # Read data into huge `Data` list.
        # Read data into huge `Data` list.
        # 这里用于构建data

        datafeature_1 = pd.read_csv("D:\Pre-datasets\datasets\Zheng sorted\data_normalization.csv", index_col=0)
        allfeature_1 = torch.tensor(datafeature_1.values).to(torch.float32)


        N,_= allfeature_1.shape
        ratio = math.ceil(N * 0.3)
        nums = np.ones(N)
        nums[:ratio] = 0
        np.random.shuffle(nums)
        train_mask = np.array(nums, dtype=bool)
        test_mask = ~train_mask

          # 将label转换为tensor
        def load_labels():
            # def load_labels():
            labels = pd.read_csv("D:\Pre-datasets\datasets\Zheng sorted\Labels.csv", index_col=None)
            labels.columns = ['V1']
            class_mapping = {label: idx for idx, label in enumerate(np.unique(labels['V1']))}
            labels['V1'] = labels['V1'].map(class_mapping)
            del class_mapping
            labels = np.asarray(labels).reshape(-1)
            return labels

        label_1 = load_labels()
        num_label_1 = torch.tensor(label_1)
        # 处理邻接矩阵    D:\HeGraph\data.csv
        adj_1 = pd.read_csv("D:\Pre-datasets\datasets\Zheng sorted\Pearson_filtered_0.85.csv", index_col=0)  # 邻接矩阵应该维度是21000x21000
        adjedge_1 = torch.tensor(adj_1.values)
        my_matrix_1 = scipy.sparse.coo_matrix(adjedge_1)
        rescombine_1 = np.vstack((my_matrix_1.row, my_matrix_1.col))
        edge_adj_1 = torch.tensor(rescombine_1).to(torch.long)
        data = Data(x=allfeature_1, edge_index=edge_adj_1, y=num_label_1,train_mask=train_mask,test_mask=test_mask)

        # 放入datalist
        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

########################################数据data################################################

#加载数据成图
dataset_train= MyOwnDataset_train(root="D:\Pre-datasets\Graphs\Zheng sorted_0.85_AE")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data= dataset_train[0].to(device)

# 定义model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, enconder_layer_1,embeding_layer,deconder_layer_1,deconder_layer_2,out_channels,cluster, dropout=0.0):
        super(GCN, self).__init__()

        #encoder
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, enconder_layer_1))
        self.convs.append(GCNConv(enconder_layer_1, embeding_layer))
        self.dropout = dropout


        #decoder
        self.dec_1=torch.nn.Linear(embeding_layer,deconder_layer_1)
        self.dec_2 = torch.nn.Linear(deconder_layer_1, deconder_layer_2)
        self.dec_3 = torch.nn.Linear(deconder_layer_2, out_channels)


        #classify
        self.predict = torch.nn.Linear(embeding_layer, cluster)


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, raw_x, edge_index):

        #GAE
        x1 = self.convs[0](raw_x, edge_index)  # 1000->1256
        x_1 = F.relu(x1)
        x = F.dropout(x_1, p=self.dropout, training=self.training)

        x_2 = self.convs[1](x, edge_index)  # # 256->32
        embeding_x = F.relu(x_2)
        result=self.predict(embeding_x)
        #分类
        x = F.dropout(embeding_x, p=self.dropout, training=self.training)
        x_3=self.dec_1(embeding_x) #32->64
        x=F.relu(x_3)
        x_4=self.dec_2(x)
        x=F.relu(x_4)
        x_5=self.dec_3(x)
        x=F.softplus(x_5)
                             #32         13     1000
        return torch.concat([embeding_x,result,x], axis=1)




# 实例化model
model = GCN(in_channels=1000, enconder_layer_1=256,embeding_layer=32,deconder_layer_1=64,deconder_layer_2=256,cluster=10,out_channels=1000)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#,weight_decay=0.94
# 查看各层之间的参数


#划分训练和测试数据
pos_partions=partition(data.edge_index,10,None)
train_pos_edge_index=torch.cat((pos_partions[0:10]),dim=1)
#test_pos_edge_index=torch.cat(((pos_partions[7: ])),dim=1)
l2_regularization=5e-1
#定义train()
def train():
    model.train()
    optimizer.zero_grad()
    out= model(data.x, train_pos_edge_index)  # 前面我们提到了，GCN是实现了edge_index和adj_t两种形式的
    out_data = out.split([32,10,1000], dim=1)
    #拼接PCA
    out_classify = out_data[1].log_softmax(dim=-1)  # 分类的损失##################################更换的第一个位置
    loss1=torch.nn.MSELoss()(out_data[2], (data.x))
    loss2=torch.nn.CrossEntropyLoss()(out_classify[data.train_mask], (data.y[data.train_mask]))
    #loss =0.6* loss2 + 0.4 * loss1
    lossR = L2_re(model.parameters(), l2_regularization)
    loss = 0.6* loss2 + 0.4* loss1 + 0.01*lossR #
    loss.backward()
    optimizer.step()
    return loss.item()

#定义test()
def test():
    model.eval()
    out = model(data.x, train_pos_edge_index)
    out_data = out.split([32,10,1000], dim=1)
    labels=load_labels()
    #exPCA(out_data[1],10)
    #T_SNE(out_data[0], labels, 10)
    out= out_data[1].log_softmax(dim=-1)  # 分类的损失######更换的第一个位置################################################
    y_pred = out.argmax(axis=-1)
    #test_acc=f1_score(labels, y_pred, average='macro')
    #correct = y_pred[data.test_mask] == data.y[data.test_mask]
    #test_acc = correct.sum().float() / data.test_mask.sum()
    F1 = f1_score(labels, y_pred, average='macro')
    #return test_acc,out_data[2]
    return F1


#300epoch
for epoch in range(300):
    loss = train()
    test_acc=test()
    print(
        f'Epoch: {epoch:02d}, '
        f'F1: {100 * test_acc:.3f}%'
        )


