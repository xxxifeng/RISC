import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import scipy.io as io

def TSNE_visual(dataset, ratio):
    # 读取数据集 
    data = io.loadmat('results/%s.mat' % (dataset+str(ratio)))
    X, y = data['X'], data['y'].squeeze()

    # 划分数据集，选定测试部分的图像
    print("划分数据集...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=ratio,random_state=42)
    X = X_test
    y = y_test

    # 初始化T-SNE模型
    tsne = TSNE(n_components=2, random_state=42)

    # 进行降维
    X_embedded = tsne.fit_transform(X)

    # 将降维后的数据与标签组合成DataFrame，方便后续绘图
    df = pd.DataFrame()
    df["comp-1"] = X_embedded[:,0]
    df["comp-2"] = X_embedded[:,1]
    df["label"] = y

    # 创建图形
    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        x='comp-1', y='comp-2',
        hue="label",
        palette=sns.color_palette("hsv", len(set(y))),
        data=df,
        legend=None,
        linewidth=0,
        alpha=0.6,
        s=15
    )
    plt.title('T-SNE Visualization of the Dataset')

    # 指定保存图像的路径和文件名
    output_path = 'results/TSNE/%s.png' % (dataset+str(ratio))

    # 保存图像到指定位置
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # dpi参数设置图像分辨率，bbox_inches='tight'去除不必要的空白

def line_chart(list1,list2):
    x_axis = np.arange(0.2, 0.91, 0.1)
    plt.figure(figsize=(10, 6))
    color_nwpu = '#FF7F50'  
    color_aid = '#6495ED'  
    plt.plot(x_axis, nwpu_oa, color=color_nwpu, marker='o', linestyle='-', linewidth=2, label='NWPU OA')
    plt.plot(x_axis, aid_oa, color=color_aid, marker='s', linestyle='--', linewidth=2, label='AID OA')

    plt.title('Comparison of NWPU OA and AID OA', fontsize=16)
    plt.xlabel('Training Ratio', fontsize=14)
    plt.ylabel('Overall Accuracy (OA)', fontsize=14)
    plt.xticks(x_axis, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='lower right', fontsize=12, frameon=True, facecolor='white', edgecolor='black')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.3)
    plt.ylim(0.7, 0.95) 
    save_path = 'results/line_chart.png'
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    dataset = 'AID'
    ratio = 0.8
    TSNE_visual(dataset, ratio)
