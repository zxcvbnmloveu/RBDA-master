import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb

def Dis_weight1(features,label,centroid):
    n_class = centroid.size(0)
    n, d = features.size()
    # calculate the sample-centorids dis
    features_exp = features.repeat(1, n_class).reshape(-1, d)
    # n*n_class,fea_num
    centroid_exp = centroid.repeat(n, 1)
    # n*n_class,fea_num
    dis = ((features_exp - centroid_exp) ** 2).mean(1)
    add = torch.tensor([i for i in range(0, n_class * n, n_class)]).cuda()
    index_dis_closest = label + add
    index_dis_others = torch.tensor([i for i in range(n_class * n) if i not in index_dis_closest])
    dis_closest = dis[index_dis_closest]
    dis_others = dis[index_dis_others].reshape(n, -1)
    # print(dis_closest*(n_class-1)/dis_others)
    return 2 - (n_class - 1) * dis_closest / dis_others.sum(1)[0]

def Dis_weight2(features,label,centroid):
    n_class = centroid.size(0)
    n, d = features.size()
    # calculate the sample-centorids dis
    features_exp = features.repeat(1, n_class).reshape(-1, d)
    # n*n_class,fea_num
    centroid_exp = centroid.repeat(n, 1)
    # n*n_class,fea_num
    dis = ((features_exp - centroid_exp) ** 2).mean(1)
    add = torch.tensor([i for i in range(0, n_class * n, n_class)]).cuda()
    index_dis_closest = label + add
    index_dis_others = torch.tensor([i for i in range(n_class * n) if i not in index_dis_closest])
    dis_closest = dis[index_dis_closest]
    dis_others = dis[index_dis_others].reshape(n,-1)
    # print(dis_closest/dis_others.min(1)[0])
    return 2-dis_closest/dis_others.min(1)[0]

def Dis_weight3(features,label,centroid):
    n_class = centroid.size(0)
    n, d = features.size()
    # calculate the sample-centorids dis
    features_exp = features.repeat(1, n_class).reshape(-1, d)
    # n*n_class,fea_num
    centroid_exp = centroid.repeat(n, 1)
    # n*n_class,fea_num
    dis = ((features_exp - centroid_exp) ** 2).mean(1)
    add = torch.tensor([i for i in range(0, n_class * n, n_class)]).cuda()
    index_dis_label = label + add
    dis_label = dis[index_dis_label]
    dis_others = dis.reshape(n,-1)
    p = (dis_label - dis_others.min(1)[0]) ** 2
    # if dis_label == min: p=1
    return torch.exp(-1.*p)

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

def sntg_loss(n_class,features,label,m=2):
    class_num = n_class
    label = label
    ones = torch.sparse.torch.eye(class_num).cuda()
    onehot = ones.index_select(0, label).cuda()
    onehot_transpose = onehot.permute(1, 0)
    z = torch.mm(onehot, onehot_transpose).view(-1)

    # print(n_class,label.size(),features.size())

    n, d = features.size()
    features_ext = features.expand(n, n, -1).cuda()
    features_ext_transpose = features_ext.permute(1, 0, 2).cuda()
    features_ext = features_ext.reshape(n * n, -1)
    features_ext_transpose = features_ext_transpose.reshape(n * n, -1)
    dis = ((features_ext - features_ext_transpose) ** 2).mean(1)

    sntg_loss = (z * dis + (1 - z) * torch.relu(m - dis)).mean()
    # print(sntg_loss)
    return sntg_loss,dis

def sntg_loss2(features,centers,m=2,i = 0):
    # sl sntg loss
    n_class = centers.size(0)
    n, d = features.size()

    centers_sntg = centers.repeat(n, 1).reshape(n*n_class,d)
    features_sntg = features.repeat(1, n_class).reshape(n*n_class,d)
    # n, n_class, d

    d_m = 0.3
    m = 2
    # weight_p = nn.Threshold(0.5,0)(softmax_output)
    # weight_n = nn.Threshold(0.7,0)(1-softmax_output)

    # print(weight_p**(1/2))
    # print(weight_n**(1/2))
    # weight_n = torch.relu(-1*weight)

    dis = ((centers_sntg - features_sntg) ** 2).mean(1).reshape(n,n_class)

    dis_step1 = (dis.min(1)[0] + 1e-5).reshape(-1, 1).repeat(1, n_class) - dis
    dis_step2 = torch.relu(dis_step1)
    dis_step3 = dis_step2 / dis_step2.max(1)[0].reshape(-1, 1).repeat(1, n_class)
    weight = dis_step3

    # sntg_loss2 = weight_p*dis + weight_n*torch.relu(m - dis)
    sntg_loss2 = weight*dis
    return sntg_loss2.mean(),dis

def sntg_loss3(features,centers):
    # sl sntg loss
    n_class = centers.size(0)
    n, d = features.size()

    centers = centers.repeat(n, 1).reshape(n*n_class,d)
    features = features.repeat(1, n_class).reshape(n*n_class,d)
    # n, n_class, d

    dis = ((centers - features) ** 2).mean(1)
    return dis.reshape(n,n_class)

def DANNdis(input_list, ad_net, entropy=None, coeff=None,random_layer=None,iter=0,n_class=31,w=0):
    dis_s, dis_t = input_list[2].detach(), input_list[3].detach()
    features_source, features_target = input_list[0], input_list[1]
    feature = torch.cat((features_source, features_target), dim=0)
    softmax_output = torch.cat((dis_s, dis_t), dim=0).detach()

    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()

    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0 + torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0) // 2:] = 0
        source_weight = entropy * source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0) // 2] = 0
        target_weight = entropy * target_mask

    else:
        transfer_loss = nn.BCELoss()(ad_out, dc_target)

    if w == 0:
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()

        transfer_loss = torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(
            weight).detach().item()

    return transfer_loss

def CDANori(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    batch_size = softmax_output.size(0) // 2
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
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)

# def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None,labels_source=None,iter = 0,n_class = 31,s_c = None,t_c = None, w = 0):
    softmax_out_source, softmax_out_target = input_list[2].detach(),input_list[3].detach()
    features_source, features_target  = input_list[0],input_list[1]
    feature = torch.cat((features_source, features_target), dim=0)
    softmax_output = torch.cat((softmax_out_source, softmax_out_target), dim=0).detach()
    s_labels, t_labels = labels_source, torch.max(softmax_out_target, 1)[1]
    # softmax_output = input_list[1].detach()
    # feature = input_list[0]

    #
    # if random_layer is None:
    #     op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
    #     ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    # else:
    #     random_out = random_layer.forward([feature, softmax_output])
    #     ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    ad_out = ad_net(feature)

    batch_size = softmax_output.size(0) // 2
    # dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()

    # sntgloss_s,_ = sntg_loss(n_class, features_source, s_labels)
    # sntgloss_t,_ = sntg_loss(n_class, features_target, t_labels)
    # sntgloss,_ = sntg_loss(n_class,feature,torch.cat((s_labels, t_labels), dim=0))

    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0 + torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0) // 2:] = 0
        source_weight = entropy * source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0) // 2] = 0
        target_weight = entropy * target_mask

    else:
        transfer_loss = nn.BCELoss()(ad_out, dc_target)

    if w == 0:

        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()

        transfer_loss = torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    elif w == 1:

        # tl/sum
        Dis_weight_s1 = Dis_weight1(features_source, s_labels, s_c)
        Dis_weight_s1.register_hook(grl_hook(coeff))
        s_mask1 = torch.ones_like(labels_source).float()
        d_weight_s1 = torch.cat((Dis_weight_s1, s_mask1), dim=0)
        Dis_weight_t1 = Dis_weight1(features_target, t_labels, t_c)
        Dis_weight_t1.register_hook(grl_hook(coeff))
        t_mask1 = torch.ones_like(labels_source).float()
        d_weight_t1 = torch.cat((t_mask1, Dis_weight_t1), dim=0)

        if iter<1000:
            weight = source_weight / torch.sum(source_weight).detach().item() + \
                     target_weight / torch.sum(target_weight).detach().item()
            transfer_loss = torch.sum(
                weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
        else:
            weight = d_weight_s1 * source_weight / torch.sum(d_weight_s1 * source_weight).detach().item() + \
                     d_weight_t1 * target_weight / torch.sum(d_weight_t1 * target_weight).detach().item()
            transfer_loss = torch.sum(
                weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()

    elif w == 2:

        # tl/min
        Dis_weight_s2 = Dis_weight2(features_source, s_labels, s_c)
        Dis_weight_s2.register_hook(grl_hook(coeff))
        s_mask2 = torch.ones_like(labels_source).float()
        d_weight_s2 = torch.cat((Dis_weight_s2, s_mask2), dim=0)
        Dis_weight_t2 = Dis_weight2(features_target, t_labels, t_c)
        Dis_weight_t2.register_hook(grl_hook(coeff))
        t_mask2 = torch.ones_like(labels_source).float()
        d_weight_t2 = torch.cat((t_mask2, Dis_weight_t2), dim=0)

        if iter < 0:
            weight = source_weight / torch.sum(source_weight).detach().item() + \
                     target_weight / torch.sum(target_weight).detach().item()
            transfer_loss = torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
        else:
            weight = d_weight_s2*source_weight / torch.sum(d_weight_s2*source_weight).detach().item() + \
                     d_weight_t2*target_weight / torch.sum(d_weight_t2*target_weight).detach().item()
            transfer_loss = torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum( weight).detach().item()

    elif w == 3:

        Dis_weight_s3 = Dis_weight3(features_source, s_labels, s_c)
        Dis_weight_s3.register_hook(grl_hook(coeff))
        s_mask3 = torch.ones_like(labels_source).float()
        d_weight_s3 = torch.cat((Dis_weight_s3, s_mask3), dim=0)
        Dis_weight_t3 = Dis_weight3(features_target, t_labels, t_c)
        Dis_weight_t3.register_hook(grl_hook(coeff))
        t_mask3 = torch.ones_like(labels_source).float()
        d_weight_t3 = torch.cat((t_mask3, Dis_weight_t3), dim=0)

        if iter < 0:
            weight = source_weight / torch.sum(source_weight).detach().item() + \
                     target_weight / torch.sum(target_weight).detach().item()
            transfer_loss = torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
        else:
            weight = d_weight_s3 * source_weight / torch.sum(d_weight_s3 * source_weight).detach().item() + \
                     d_weight_t3 * target_weight / torch.sum(d_weight_t3 * target_weight).detach().item()
            transfer_loss = torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()

    elif w == 4:
        transfer_loss = nn.BCELoss()(ad_out, dc_target)

    # return transfer_loss, sntgloss_s, sntgloss_t
    return transfer_loss

def DANN(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    adloss = nn.BCELoss()(ad_out, dc_target)
    return adloss

def adloss2(softmax_out, ad_net):
    ad_out = ad_net(softmax_out)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)

def NEW(features, ad_net,softmax_out,coeff=None):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    entropy = Entropy(softmax_out)
    ad_out = ad_net(features)
    entropy.register_hook(grl_hook(coeff))
    entropy = 1.0 + torch.exp(-entropy)
    source_mask = torch.ones_like(entropy)
    source_mask[features.size(0) // 2:] = 0
    source_weight = entropy * source_mask
    target_mask = torch.ones_like(entropy)
    target_mask[0:features.size(0) // 2] = 0
    target_weight = entropy * target_mask
    weight = source_weight / torch.sum(source_weight).detach().item() + \
             target_weight / torch.sum(target_weight).detach().item()
    return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(
        weight).detach().item()

def NEW2(features, ad_net,labels_source,softmax_out_source,softmax_out_target):
    ad_out = ad_net(features)
    n_class = ad_out.size(1)-1
    batch_size = ad_out.size(0) // 2

    dc_target = torch.cat((labels_source,n_class*torch.ones_like(labels_source)))
    ad_loss_D = nn.CrossEntropyLoss()(ad_out,dc_target)

    output_s = ad_out[:batch_size,:]
    output_t = ad_out[batch_size:,:]


    zeros = torch.zeros(batch_size, 1).cuda()
    ls = torch.cat((softmax_out_source, zeros), dim=1)
    lt = torch.cat((softmax_out_target, zeros), dim=1)

    ad_loss_G_s = - torch.sum(ls.detach() * nn.LogSoftmax(1)(output_s))/float(batch_size)
    ad_loss_G_t = - torch.sum(lt.detach() * nn.LogSoftmax(1)(output_t))/float(batch_size)

    return ad_loss_D,ad_loss_G_s,ad_loss_G_t

if __name__ == '__main__':
    a1 = torch.rand(1,4)
    a1_e = Entropy(a1)
