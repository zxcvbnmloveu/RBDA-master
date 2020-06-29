import random

import torch
from torchvision import models
import torch.nn as nn
import numpy as np
import math

import losssntg
from losssntg import Entropy
import matplotlib.pyplot as plt
import torch.nn.functional as F

import time

class KMEANS:
    def __init__(self, n_clusters=20, max_iter=None, verbose=False):

        self.n_cluster = n_clusters
        self.n_clusters = n_clusters
        self.labels = None
        self.dists = None  # shape: [x.shape[0],n_cluster]
        self.centers = None
        self.variation = torch.Tensor([float("Inf")]).cuda()
        self.verbose = verbose
        self.started = False
        self.representative_samples = None
        self.max_iter = max_iter
        self.count = 0

    def fit(self, x):
        # 随机选择初始中心点，想更快的收敛速度可以借鉴sklearn中的kmeans++初始化方法
        # init_row = torch.randint(0, x.shape[0], (self.n_clusters,))
        init_row = torch.tensor(random.sample(range(0, x.shape[0]), self.n_clusters))
        init_points = x[init_row]
        self.centers = init_points
        while True:
            # 聚类标记
            self.nearest_center(x)

            # 更新中心点
            self.update_center(x)
            if self.verbose:
                print(self.variation, torch.argmin(self.dists, (0)))
            if torch.abs(self.variation) < 1e-2 and self.max_iter is None:
                break
            elif self.max_iter is not None and self.count == self.max_iter:
                break

            self.count += 1

        # self.representative_sample()
        print(self.count)
        return self.centers

    def match_centers(self,s_centroids,t_centroids):

        n_class, d = s_centroids.size()

        used = [0] * n_class
        index = [0] * n_class
        for i in range(n_class):
            min_dis = 1e10
            for j in range(n_class):
                if used[j] == 0:
                    dis = ((s_centroids[i] - t_centroids[j]) ** 2).mean(0)
                    if dis < min_dis:
                        min_dis = dis
                        index[i] = j
            used[index[i]] = 1

        index = torch.tensor(index).cuda()
        return t_centroids.index_select(0, index)

    def nearest_center(self, x):
        labels = torch.empty((x.shape[0],)).long().cuda()
        dists = torch.empty((0, self.n_clusters)).cuda()
        for i, sample in enumerate(x):
            dist = torch.sum(torch.mul(sample - self.centers, sample - self.centers), (1))
            labels[i] = torch.argmin(dist)
            dists = torch.cat([dists, dist.unsqueeze(0)], (0))
        self.labels = labels
        if self.started:
            self.variation = torch.sum(self.dists - dists)
        self.dists = dists
        self.started = True

    def update_center(self, x):
        centers = torch.empty((0, x.shape[1])).cuda()
        for i in range(self.n_clusters):
            mask = self.labels == i
            cluster_samples = x[mask]
            centers = torch.cat([centers, torch.mean(cluster_samples, (0)).unsqueeze(0)], (0))
        self.centers = centers

    def representative_sample(self):
        # 查找距离中心点最近的样本，作为聚类的代表样本，更加直观
        self.representative_samples = torch.argmin(self.dists, (0))

def update_fs_centers(loader, model, n_class):
    """
    calculate feature space centers
    :param loader:
    :param model:
    :return: fs_centers n_class*fea_dim
    """
    iter_source = iter(loader['source'])
    model.train(False)
    model.train(False)
    with torch.no_grad():
        total_s_sum_feature = torch.zeros(n_class, 256).cuda()
        total_s_n_classes = torch.zeros(n_class).cuda()
        for i in range(len(loader['source'])):
            inputs_source, labels_source = iter_source.next()
            inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
            features_source, outputs_source = model(inputs_source)
            n, d = features_source.shape
            # image number in each class
            ones = torch.ones_like(labels_source, dtype=torch.float).cuda()
            zeros = torch.zeros(n_class)
            zeros = zeros.cuda()
            s_n_classes = zeros.scatter_add(0, labels_source, ones)
            # calculating centroids, sum and divide
            zeros = torch.zeros(n_class, d)
            zeros = zeros.cuda()
            s_sum_feature = zeros.scatter_add(0, torch.transpose(labels_source.repeat(d, 1), 1, 0), features_source)
            total_s_sum_feature += s_sum_feature
            total_s_n_classes += s_n_classes
            # print(total_s_sum_feature,total_s_n_classes)
        fs_centers = torch.div(total_s_sum_feature, total_s_n_classes.view(n_class, 1))
        return fs_centers.cuda()

def update_ls_centers(loader, model,n_class):
    """
    calculate label space centers
    :param loader:
    :param model:
    :return: ls_centers n_class*n_class
    """
    iter_source = iter(loader['source'])
    model.train(False)
    with torch.no_grad():
        total_s_sum_feature = torch.zeros(n_class, n_class).cuda()
        total_s_n_classes = torch.zeros(n_class).cuda()
        for i in range(len(loader['source'])):
            inputs_source, labels_source = iter_source.next()
            inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
            features_source, outputs_source = model(inputs_source)

            # softmax_out_source = nn.Softmax(dim=1)(outputs_source)

            # use temperature
            temperature = 2
            softmax_out_source = nn.Softmax(dim=1)(outputs_source/temperature)

            n, d = softmax_out_source.shape
            # image number in each class
            ones = torch.ones_like(labels_source, dtype=torch.float).cuda()
            zeros = torch.zeros(n_class)
            zeros = zeros.cuda()
            s_n_classes = zeros.scatter_add(0, labels_source, ones)
            # calculating centroids, sum and divide
            zeros = torch.zeros(n_class, d)
            zeros = zeros.cuda()
            s_sum_feature = zeros.scatter_add(0, torch.transpose(labels_source.repeat(d, 1), 1, 0), softmax_out_source)
            total_s_sum_feature += s_sum_feature
            total_s_n_classes += s_n_classes
            # print(total_s_sum_feature,total_s_n_classes)
        ls_centers = torch.div(total_s_sum_feature, total_s_n_classes.view(n_class, 1))
        return ls_centers.cuda()


# corresponding ls_c and fs_c for each batch
def cal_batch_fclc(label,fs_centers,ls_centers):
    num_classes = 31
    n,f_dim,l_dim = label.size(0), fs_centers.size(1),ls_centers.size(1)
    batch_fc = torch.zeros(n,f_dim).cuda()
    batch_lc = torch.zeros(n,l_dim).cuda()

    for i in range(n):
        batch_fc[i] = fs_centers[label[i]]
        batch_lc[i] = ls_centers[label[i]]

    return batch_fc,batch_lc

def sample_select(softmax_out_target,features_tar,ad_net,iter):
    # with torch.no_grad():
    ad_out = ad_net(features_tar, test=True).view(-1)
    entropy = losssntg.Entropy(softmax_out_target)
    entropy = 1.0 + torch.exp(-entropy)
    threshold = 1.6 - 0.1*iter/2000
    # w_e = torch.threshold(entropy,threshold,0)

    # return ad_out*w_e

    w_e = torch.threshold(entropy,threshold,-1)
    w_e = torch.threshold(-w_e, 0, 0)
    return 1-w_e

def cal_pseudolabel_w(loader,ad_net,base_net):
    iter_tar = iter(loader['target'])
    ad_net.train(False)
    base_net.train(False)
    with torch.no_grad():
        D_output_false = []
        F_output_false = []
        D_output_true = []
        F_output_true = []
        F_entropy_false = []
        F_max_false = []
        F_entropy_true = []
        F_max_true = []

        for i in range(len(loader['target'])):
            inputs_tar, labels_tar = iter_tar.next()
            inputs_tar, labels_tar = inputs_tar.cuda(), labels_tar.cuda()
            features_tar, outputs_tar = base_net(inputs_tar)
            softmax_out_target = nn.Softmax(dim=1)(outputs_tar)
            pseudo_label = torch.max(softmax_out_target, 1)[1]
            ad_out = ad_net(features_tar,test=True)
            entropy = losssntg.Entropy(softmax_out_target)
            entropy = 1.0 + torch.exp(-entropy)

            pred = (labels_tar == pseudo_label)
            for j in range(inputs_tar.size(0)):
                if pred[j].item() == 0:
                    D_output_false.append(ad_out[j].item())
                    F_entropy_false.append(entropy[j].item())
                    F_max_false.append(softmax_out_target[j].max().item())

                elif pred[j].item() == 1:
                    D_output_true.append(ad_out[j].item())
                    F_entropy_true.append(entropy[j].item())
                    F_max_true.append(softmax_out_target[j].max().item())

    return {'D_output_false':D_output_false,'D_output_true':D_output_true,
            'F_entropy_false':F_entropy_false,'F_entropy_true':F_entropy_true,
            'F_max_false':F_max_false,'F_max_true':F_max_true}

class VAT(nn.Module):
    def __init__(self, model):
        super(VAT, self).__init__()
        self.n_power = 1
        self.XI = 1e-6
        self.model = model
        self.epsilon = 3.5

    def forward(self, X, logit):
        vat_loss = self.virtual_adversarial_loss(X, logit)
        return vat_loss

    def generate_virtual_adversarial_perturbation(self, x, logit):
        d = torch.randn_like(x, device='cuda')

        for _ in range(self.n_power):
            d = self.XI * self.get_normalized_vector(d).requires_grad_()
            _, logit_m = self.model(x + d)
            dist = self.kl_divergence_with_logit(logit, logit_m)
            grad = torch.autograd.grad(dist, [d])[0]
            d = grad.detach()

        return self.epsilon * self.get_normalized_vector(d)

    def kl_divergence_with_logit(self, q_logit, p_logit):
        q = F.softmax(q_logit, dim=1)
        qlogq = torch.sum(q * F.log_softmax(q_logit, dim=1), dim=1)
        qlogp = torch.sum(q * F.log_softmax(p_logit, dim=1), dim=1)
        return qlogq - qlogp

    def get_normalized_vector(self, d):
        return F.normalize(d.view(d.size(0), -1), p=2, dim=1).reshape(d.size())

    def virtual_adversarial_loss(self, x, logit):
        r_vadv = self.generate_virtual_adversarial_perturbation(x, logit)
        logit_p = logit.detach()
        _, logit_m = self.model(x + r_vadv)
        loss = self.kl_divergence_with_logit(logit_p, logit_m)
        return loss


if __name__ == '__main__':
    n,f_dim,l_dim = 4,3,2
    fs_centers = torch.tensor([[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
    ls_centers = torch.tensor([[0.5,0.5],[0.3,0.7],[0.1,0.9],[0.2,0.8]])
    label = torch.tensor([3,2,1,1])

    t_label =torch.tensor([4,1,1,1])

    q = F.softmax(ls_centers, dim=1)
    qlogq = torch.sum(q * F.log_softmax(ls_centers, dim=1), dim=1)
    qlogp = torch.sum(q * F.log_softmax(ls_centers, dim=1), dim=1)
    print(qlogq - qlogp)