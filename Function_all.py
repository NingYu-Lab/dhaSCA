#-*- codeing = utf-8 -*-
#@Time : 2023/8/28 18:42
#@Author : yuning
#@File : function.py
import math
import torch
import umap
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
# from factor_analyzer import FactorAnalyzer
# from factor_analyzer.factor_analyzer import calculate_kmo
from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.cluster import SpectralClustering
import pandas as pd
import sklearn
import numpy as np
from sklearn.cluster import KMeans  # 聚类库
import matplotlib.pyplot as plt     # 绘图库
import os
from sklearn.metrics.cluster import adjusted_rand_score

#划分数据集
def partition(edge_index, k, seed=None):
   #edge_index is directed,[2,*]
    # equal k parts
    size = math.ceil(edge_index.size(1) / float(k))
    partitions = []
    row, col = edge_index
    perm = torch.randperm(row.size(0))  # random
    row, col = row[perm], col[perm]
    for i in range(k - 1):
        r, c = row[i * size: (i + 1) * size], col[i * size: (i + 1) * size]
        part = torch.stack([r, c], dim=0)
        partitions.append(part)
    i = i + 1
    r, c = row[i * size:], col[i * size:]
    part = torch.stack([r, c], dim=0)
    partitions.append(part)
    return partitions


def  SpectralClusteringF(data, cluster):
    label_1 = load_labels()
    num_label_1 = torch.tensor(label_1)
    y_pred = SpectralClustering(n_clusters=cluster, assign_labels='discretize', random_state=0).fit(data)
    labelPred = y_pred.labels_
    NMI = normalized_mutual_info_score(num_label_1, labelPred)
    print("NMI", NMI)




def exPCA(x,cluster):
    Pca_embeding_2 = (pd.DataFrame(x)).iloc[:, :].values

    #PCA
    pca = PCA(n_components=32, svd_solver='full')
    embeding_1 = pca.fit_transform(Pca_embeding_2)
    '''
    #FastICA
    transformer = FastICA(n_components=32, whiten=True)
    embeding_1 = transformer.fit_transform(Pca_embeding_2)

    # NMF
    model = NMF(n_components=32, init='random', random_state=0)
    embeding_1 = model.fit_transform(Pca_embeding_2)
    # UMap
    embeding_1 = umap.UMAP( n_components=32).fit_transform(Pca_embeding_2)
        '''
    label=load_labels()
    k_means(embeding_1, cluster)



def load_labels():
    # def load_labels():
    labels = pd.read_csv("D:\Pre-datasets\datasets\Zheng sorted\Labels.csv", index_col=None)
    labels.columns = ['V1']
    class_mapping = {label: idx for idx, label in enumerate(np.unique(labels['V1']))}
    labels['V1'] = labels['V1'].map(class_mapping)
    del class_mapping
    labels = np.asarray(labels).reshape(-1)
    return labels

def T_SNE(embs,labels,cluster):
    embs = (pd.DataFrame(embs)).iloc[:, :].values
    tsne = TSNE(n_components=2, learning_rate=200, metric='cosine', n_jobs=-1)
    tsne.fit_transform(embs)
    outs_2d = np.array(tsne.embedding_)
    css4 = list(mcolors.CSS4_COLORS.keys())
    # 我选择了一些较清楚的颜色，更多的类时也能画清晰
    color_ind = [10, 11, 13,14,146, 30, 110, 32, 110, 38, 40, 47, 51,
                 55, 60, 65, 82, 85, 88, 106, 110, 115, 118, 120, 125, 131, 135, 139, 142, 146, 147,2, 7, 9, 10, 11, 13,14]
    # color_ind = [14, 17, 19, 20, 21, 25, 28, 30, 31, 32, 37, 38, 40, 47, 51,
    #              55, 60, 65, 82, 85, 88, 106, 110, 115, 118, 120, 125, 131, 135, 139, 142, 146, 147, 2, 7, 9, 10, 11
    css4 = [css4[v] for v in color_ind]
    for lbi in range(cluster):
        temp = outs_2d[labels == lbi]
        plt.plot(temp[:, 0], temp[:, 1], '.', color=css4[lbi])
    #plt.title('gaeSCA')
    plt.savefig('./test2.jpg',dpi=300)
   # plt.savefig('D:\Pre-datasets\\results\A\savefig_example_dpi.png', dpi=300)


def k_means(x,clusters):
    X=x
    label_1 = load_labels()
    num_label_1 = torch.tensor(label_1)
    #K-Means聚类
    y_pred = KMeans(n_clusters=clusters).fit(X)
    labelPred = y_pred.labels_
    ARI=adjusted_rand_score(num_label_1, labelPred)
    print("ARI",ARI)


    #l2_regularization=5e-4
    def loss(self, y1, y_target1,y2, y_target2,l2_regularization):

        loss1 = torch.nn.MSELoss()(y1, y_target1)#重建损失
        loss2 = torch.nn.NLLLoss()(y2,y_target2)#分类损失
        loss = 1 * loss1 + 1 * loss2

        l2_loss = 0.0
        for param in self.parameters():
            data = param* param
            l2_loss += data.sum()


        loss += 0.2* l2_regularization* l2_loss

        return loss

def L2_re(Parameters,l2_regularization):
    l2_loss = 0.0
    for param in Parameters:
        data = param * param
        l2_loss += data.sum()
    loss=l2_regularization * l2_loss
    return loss