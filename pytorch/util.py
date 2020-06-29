import random

import torch
from torchvision import models
import torch.nn as nn
import numpy as np
import math
import torch.utils.data as data
import losssntg
from losssntg import Entropy
import matplotlib.pyplot as plt
import torch.nn.functional as F

import time

def disp_f_c(features,label,centers):
    n_class = centers.size(0)
    n, d = features.size()
    centers_sntg = centers.repeat(n, 1).reshape(n * n_class, d)
    features_sntg = features.repeat(1, n_class).reshape(n * n_class, d)
    dis = 100*((centers_sntg - features_sntg) ** 2).mean(1).reshape(n,n_class)
    return torch.cat((dis,label.reshape(-1,1).float()),dim=1)

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
    with torch.no_grad():

        arr = [[] for i in range(n_class)]
        f_c = torch.zeros(n_class, 256)

        for i in range(len(loader['source'])):
            inputs_source, labels_source = iter_source.next()
            inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
            features_source, outputs_source = model(inputs_source)
            n, d = features_source.shape
            # image number in each class

            for l in range(n):
                arr[int(labels_source[l].item())].append(features_source[l])

        for i in range(n_class):
            f_c[i] = torch.cat(arr[i]).reshape(-1, d).median(dim=0)[0]

        return f_c.cuda()

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
        with torch.no_grad():

            arr = [[] for i in range(n_class)]
            l_c = torch.zeros(n_class, n_class)

            for i in range(len(loader['source'])):
                inputs_source, labels_source = iter_source.next()
                inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
                features_source, outputs_source = model(inputs_source)
                n, d = features_source.shape
                # softmax_out_source = nn.Softmax(dim=1)(outputs_source)

                # use temperature
                temperature = 2
                softmax_out_source = nn.Softmax(dim=1)(outputs_source / temperature)

                for l in range(n):
                    arr[int(labels_source[l].item())].append(softmax_out_source[l])

            for i in range(n_class):
                l_c[i] = torch.cat(arr[i]).reshape(-1, n_class).mean(dim=0)[0]

        return l_c.cuda()

def usc(y_s,s_feature,softmax_out_source,n_class,s_centroid,decay = 0.3):

    n,d = s_feature.size()

    # get labels
    s_labels = y_s

    # image number in each class
    ones = torch.ones_like(s_labels, dtype=torch.float)
    zeros = torch.zeros(n_class).cuda()

    s_n_classes = zeros.scatter_add(0, s_labels, ones)

    # image number cannot be 0, when calculating centroids
    ones = torch.ones_like(s_n_classes)
    s_n_classes = torch.max(s_n_classes, ones)

    # calculating centroids, sum and divide
    zeros = torch.zeros(n_class, d).cuda()
    s_sum_feature = zeros.scatter_add(0, torch.transpose(s_labels.repeat(d, 1), 1, 0), s_feature)

    zeros = torch.zeros(n_class, n_class).cuda()
    s_sum_feature2 = zeros.scatter_add(0, torch.transpose(s_labels.repeat(n_class, 1), 1, 0), softmax_out_source)

    feature = torch.cat((s_sum_feature,s_sum_feature2),dim=1)

    current_s_centroid = torch.div(feature, s_n_classes.view(n_class, 1))

    #set decay
    de = torch.threshold(current_s_centroid,1e-6,-1)
    de = 1 - torch.threshold(-de,0,0)
    decay = decay*de

    # Moving Centroid
    s_centroid = (1 - decay) * s_centroid + decay * current_s_centroid
    return s_centroid

def utc(softmax_out_target,t_feature,n_class,t_centroid,decay = 0.3):

    n,d = t_feature.size()

    weight = softmax_out_target

    weight_sum = weight.sum(0)

    # [bs, nclass, d])
    features_target_sl = t_feature.repeat(1, n_class).reshape(-1, n_class, d)

    features_target_sl2 = softmax_out_target.repeat(1, n_class).reshape(-1, n_class, n_class)

    feature = torch.cat((features_target_sl,features_target_sl2),dim=2)

    weight = weight.reshape(-1, 1).expand(-1, d+n_class).reshape(n, n_class, d+n_class)

    t_sum_feature = (feature * weight).sum(0)
    t_n_classes = weight_sum+1e-6

    current_t_centroid = torch.div(t_sum_feature, t_n_classes.view(n_class, 1))

    #set decay
    de = torch.threshold(current_t_centroid,1e-6,-1)
    de = 1 - torch.threshold(-de,0,0)
    decay = decay*de

    # Moving Centroid
    t_centroid = (1 - decay) * t_centroid + decay * current_t_centroid
    return t_centroid

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

def sample_select(softmax_out_target,iter,type):
    # with torch.no_grad():
    # ad_out = ad_net(features_tar, test=True).view(-1)
    if type == 1:
        entropy = losssntg.Entropy(softmax_out_target)
        conf = torch.exp(-entropy)
        th = (0.3*conf.median()+0.7*conf.max())
        # th = conf.mean()

    elif type == 2:
        conf = softmax_out_target.max(dim=1)[0]
        th = (0.3*conf.median()+0.7*conf.max())
        # th = conf.mean()

    # w = torch.where(conf<th, torch.full_like(w, 0),w)
    sel = torch.where(conf<th, torch.full_like(conf, 0),torch.full_like(conf, 1))
    w = ((conf/(1-th))*sel).cuda().detach()
    return w,sel.sum()

    # w_e2 = torch.threshold(entropy,threshold,-1)
    # w_e2 = 1-torch.threshold(-w_e2, 0, 0)
    # # return w_e2

def diversity_loss(softmax_out_target,softmax_out_source,w):
    d = softmax_out_target.size(1)
    sel_sout_t = softmax_out_target[(w>0).nonzero()].reshape(-1,d)
    softmax_out = torch.cat((sel_sout_t,softmax_out_source),dim=0)
    dloss = softmax_out.mean(dim=0)
    epsilon = 1e-5
    entropy = -dloss * torch.log(dloss + epsilon)
    entropy = torch.sum(entropy, dim=0)
    dloss = torch.exp(-entropy)
    return dloss

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
            # ad_out = ad_net(features_tar,test=True)
            entropy = losssntg.Entropy(softmax_out_target)
            entropy = 1.0 + torch.exp(-entropy)

            pred = (labels_tar == pseudo_label)
            for j in range(inputs_tar.size(0)):
                if pred[j].item() == 0:
                    # D_output_false.append(ad_out[j].item())
                    F_entropy_false.append(entropy[j].item())
                    F_max_false.append(softmax_out_target[j].max().item())

                elif pred[j].item() == 1:
                    # D_output_true.append(ad_out[j].item())
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

class DummyDataset(data.Dataset):
    """Slice dataset and eeplace labels of dataset with pseudo ones."""

    def __init__(self, original_dataset, pseudo_labels):
        """Init DummyDataset."""
        super(DummyDataset, self).__init__()
        self.dataset = original_dataset
        self.pseudo_labels = pseudo_labels

    def __getitem__(self, index):
        """Get images and target for data loader."""
        images, label = self.dataset[index]
        return images,label,self.pseudo_labels[index]

    def __len__(self):
        return len(self.pseudo_labels)

def make_data_loader(dataset, batch_size=34,
                     shuffle=True, sampler=None):
    """Make dataloader from dataset."""
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler)
    return data_loader

def get_sampled_data_loader(dataset):
    """Get data loader for sampled dataset."""
    # get indices
    sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return make_data_loader(dataset, sampler=sampler, shuffle=False)

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