import torch
from dataset.dataLoader import DL
from network import Model
import torch.nn as nn
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import scipy.io as io
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self._init_data()
        self._init_model()

    # 初始化模型
    def _init_model(self):
        self.net = Model(self.args).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), )
        self.svm = SVC(kernel='rbf')
        self.pca = PCA(n_components=self.args.K)
        # self.pca = manifold.TSNE(n_components=self.args.K, init='pca')

    def _init_data(self):
        self.data = DL(self.args)
        self.dl = self.data.dl

    # 提取特征
    def feature_extract(self):
        outputs = []
        labels = []
        print("进行特征提取...")
        for inputs, targets in tqdm(self.dl, ncols=90):
            inputs = inputs.to(self.device)
            targets = targets.numpy()
            output = self.net(inputs).detach().cpu().numpy()
            outputs.append(output)
            labels.append(targets)

        # 沿第一个维度，拼接所有批次的特征和标签
        X = np.concatenate(outputs, axis=0)
        y = np.concatenate(labels, axis=0)

        # 创建字典，存储特征和标签
        data = {'X': X, 'y': y}
        io.savemat('results/%s.mat' % (self.args.dataset + str(self.args.ratio)), data)

    def train(self):
        print("数据集: ", self.args.dataset)
        print("train ratio: ", self.args.ratio)

        print("读取数据集...")
        data = io.loadmat('results/%s.mat' % (self.args.dataset + str(self.args.ratio)))
        X, y = data['X'], data['y'].squeeze()
        print("pca降维...")
        X = self.pca.fit_transform(X)
        print("划分数据集...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.args.ratio, random_state=42)
        # 训练SVM分类器
        self.svm.fit(X_train, y_train)
        # 预测
        pred = self.svm.predict(X_test)
        # 计算评估结果
        OA = accuracy_score(y_test, pred)
        AA = balanced_accuracy_score(y_test, pred)
        cm = confusion_matrix(y_test, pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm_normalized * 100).astype(int)
        print('OA: %.6f' % OA)
        print('AA: %.6f' % AA)

        # 可视化混淆矩阵
        plt.figure(figsize=(8, 6))
        heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', cbar=True,
                              xticklabels=True, yticklabels=True,
                              vmin=0, vmax=cm.max(), square=True,
                              annot_kws={"size": 8})
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # 保存混淆矩阵图片
        confusion_matrix_path = 'results/cm/confusion_matrix_%s.png' % (self.args.dataset + str(self.args.ratio))
        plt.savefig(confusion_matrix_path)
        print('混淆矩阵已保存至:', confusion_matrix_path)
