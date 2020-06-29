import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb
import torchsnooper

def Dis_weight(features,label,centroid):
    """
    :param features: bs*fea
    :param label: bs
    :param centroid:cls_num*fea
    :return: bs
    """
    fea_cen = torch.tensor([]).cuda()
    for i in range(features.size(0)):
        fea_cen = torch.cat((fea_cen, centroid[label[i]]), 0)
    fea_cen = fea_cen.view(-1,features.size(1))
    dis = torch.cosine_similarity(features, fea_cen, dim=1).view(-1)
    return 1 + dis.pow(2)

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

# def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None,s_centroid=None,t_centroid=None,labels_source=None,epoch = 0,n_class = 31):
    outputs_source, outputs_target = input_list[2].detach(),input_list[3].detach()
    features_source, features_target  = input_list[0],input_list[1]
    softmax_out_source = nn.Softmax(dim=1)(outputs_source)
    softmax_out_target = nn.Softmax(dim=1)(outputs_target)
    feature = torch.cat((features_source, features_target), dim=0)
    softmax_output = torch.cat((softmax_out_source, softmax_out_target), dim=0).detach()
    s_labels, t_labels = labels_source, torch.max(softmax_out_target, 1)[1]
    # softmax_output = input_list[1].detach()
    # feature = input_list[0]
    if s_centroid is not None:
        n, d = features_target.shape
        decay = 0.3
        # image number in each class
        # n_class = 31 #office31,clef12 记得改！！！！！
        n_class = n_class

        entropy_t = Entropy(softmax_out_target)
        # 计算最肯定的前一半样本
        sort = torch.argsort(entropy_t, dim=0)
        index = []
        for i, v in enumerate(sort):
            # print(i,v)
            if v > 17:
                index.append(i)
        features_target_selected = features_target[index]
        t_labels_selected = t_labels[index]
        # features_target_selected = features_target
        # t_labels_selected = t_labels
        ones = torch.ones_like(s_labels, dtype=torch.float)
        zeros = torch.zeros(n_class)
        zeros = zeros.cuda()
        s_n_classes = zeros.scatter_add(0, s_labels, ones)
        t_n_classes = zeros.scatter_add(0, t_labels, ones)

        # image number cannot be 0, when calculating centroids
        ones = torch.ones_like(s_n_classes)
        s_n_classes = torch.max(s_n_classes, ones)
        t_n_classes = torch.max(t_n_classes, ones)

        # calculating centroids, sum and divide
        zeros = torch.zeros(n_class, d)
        zeros = zeros.cuda()
        s_sum_feature = zeros.scatter_add(0, torch.transpose(s_labels.repeat(d, 1), 1, 0), features_source)
        t_sum_feature = zeros.scatter_add(0, torch.transpose(t_labels_selected.repeat(d, 1), 1, 0), features_target_selected)
        current_s_centroid = torch.div(s_sum_feature, s_n_classes.view(n_class, 1))
        current_t_centroid = torch.div(t_sum_feature, t_n_classes.view(n_class, 1))
        for i in range(current_s_centroid.size(0)):
            if torch.equal(current_s_centroid[i][:], torch.zeros_like(current_s_centroid[i][:])):
                current_s_centroid[i][:] = s_centroid[i][:]
        for i in range(current_t_centroid.size(0)):
            if torch.equal(current_t_centroid[i][:], torch.zeros_like(current_t_centroid[i][:])):
                current_t_centroid[i][:] = t_centroid[i][:]
        # Moving Centroid
        s_centroid = (1 - decay) * s_centroid + decay * current_s_centroid
        t_centroid = (1 - decay) * t_centroid + decay * current_t_centroid

    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    batch_size = softmax_output.size(0) // 2
    # dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()

    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask


        # Dis_weight_s = Dis_weight(features_source, s_labels, t_centroid)
        # Dis_weight_s.register_hook(grl_hook(coeff))
        # s_mask = torch.ones_like(labels_source).float()
        # d_weight_s = torch.cat((Dis_weight_s,s_mask), dim=0)
        # Dis_weight_t = Dis_weight(features_target, t_labels, t_centroid)
        # Dis_weight_t.register_hook(grl_hook(coeff))
        # t_mask = torch.ones_like(labels_source).float()
        # d_weight_t = torch.cat((t_mask,Dis_weight_t), dim=0)
        #
        # weight = d_weight_s * source_weight / torch.sum(d_weight_s * source_weight).detach().item() + \
        #          d_weight_t * target_weight / torch.sum(d_weight_t * target_weight).detach().item()

        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()

        transfer_loss = torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        transfer_loss = nn.BCELoss()(ad_out, dc_target)

    return s_centroid,t_centroid,transfer_loss

def DANN(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)

if __name__ == '__main__':
    a1 = torch.rand(1,4)
    a1_e = Entropy(a1)
