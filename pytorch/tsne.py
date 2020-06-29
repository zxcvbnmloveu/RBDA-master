# coding='utf-8'
"""t-SNE 对手写数字进行可视化"""
from time import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
from sklearn import datasets
from sklearn.manifold import TSNE
import os


def get_data():
    data = np.load('./tsne/lamb_change_1000/CCDAN_t.npy')
    label = np.load('./tsne/lamb_change_1000/CCDAN_t_l.npy')

    print(data.shape)
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    color = label
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        # plt.text(data[i, 0], data[i, 1], str(label[i]),
        #          color=plt.cm.Set1(label[i] / 31.),
        #          fontdict={'weight': 'bold', 'size': 9})
        plt.scatter(data[:, 0], data[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def main():
    data, label, n_samples, n_features = get_data()
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, label,
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0))
    plt.show(fig)

def trans_mat():

    # our_s = np.load('./tsne/lamb_change_1000/fea_3wA_W1CDRL_s.npy')
    # our_t = np.load('./tsne/lamb_change_1000/fea_3wA_W1CDRL_t.npy')
    # io.savemat('./tsne/fea_3wA_WCDRL_s.mat', {'name': our_s})
    # io.savemat('./tsne/fea_3wA_WCDRL_t.mat', {'name': our_t})
    our_s = np.load('./snapshot/model/AWCDAN_s.npy')
    our_t = np.load('./snapshot/model/AWCDAN_t.npy')
    io.savemat('./tsne/AWCDAN_s.mat', {'name': our_s})
    io.savemat('./tsne/AWCDAN_t.mat', {'name': our_t})
    # our_s = np.load('./tsne/lamb_change_1000/fea_A_W2CDRL_s_l.npy')
    # our_t = np.load('./tsne/lamb_change_1000/fea_A_W2CDRL_t_l.npy')
    # io.savemat('./tsne/fea_AW2CDRL_s_l.mat', {'name': our_s})
    # io.savemat('./tsne/fea_AW2CDRL_t_l.mat', {'name': our_t})


if __name__ == '__main__':
    # trans_mat()

    filePath = './snapshot/model/'
    list = os.listdir(filePath)
    for l in list:
        npy = np.load(filePath+l)
        print(l)
        io.savemat('./tsne/'+l[:-4]+'.mat', {'name': npy})
        print(l[:-4]+'.mat')
