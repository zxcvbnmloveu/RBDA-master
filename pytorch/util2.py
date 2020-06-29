import torch
import torch.utils.data as data
from torch import nn


def t2(loader,model,n_class):

    iter_test = iter(loader["test"])
    with torch.no_grad():
        start_test = True
        for i in range(len(loader['test'])):
            inputs_target, labels_target = iter_test.next()
            inputs_target = inputs_target.cuda()
            labels_target = labels_target.cuda()
            features_target, outputs_target = model(inputs_target)
            softmax_out_target = nn.Softmax(dim=1)(outputs_target)
            max_pred,t_labels = torch.max(softmax_out_target, 1)
            max_pred, t_labels = max_pred.cuda(),t_labels.cuda()

            if start_test:
                all_feature = features_target.float().cpu()
                all_tlabel = t_labels.float()
                all_pred = max_pred
                all_label = labels_target.float()
                start_test = False
            else:
                all_feature = torch.cat((all_feature, features_target.float().cpu()), 0)
                all_tlabel = torch.cat((all_tlabel, t_labels.float()), 0)
                all_pred = torch.cat((all_pred, max_pred), 0)
                all_label = torch.cat((all_label, labels_target.float()), 0)

    print(all_tlabel.size())
    fea = all_feature
    t_labels = all_tlabel
    prob = all_pred

    index_cate = [[] for i in range(n_class)]

    #记录样本的编号
    for l in range(fea.size(0)):
        index_cate[int(t_labels[l].item())].append(l)

    prob_arr = [[] for i in range(n_class)]
    # n_class,n_c
    true_samples = [[] for i in range(n_class)]

    #记录样本概率
    for i in range(n_class):
        n_c = len(index_cate[i])
        for j in range(n_c):
            prob_arr[i].append(prob[index_cate[i][j]].view(-1))

    pl = [[] for i in range(n_class)]
    for i in range(n_class):
        n_c = len(index_cate[i])
        for j in range(n_c):
            pl[i].append(t_labels[index_cate[i][j]].view(-1))

    tl = [[] for i in range(n_class)]
    for i in range(n_class):
        n_c = len(index_cate[i])
        for j in range(n_c):
            tl[i].append(all_label[index_cate[i][j]].view(-1))

    for i in range(n_class):
        _, idx1 = torch.sort(torch.cat(prob_arr[i]))
        _, sort = torch.sort(idx1)
        n_c = len(index_cate[i])
        print("class: "+str(i))
        print(sort)
        print(torch.cat(tl[i])==torch.cat(pl[i]))
        print(torch.cat(prob_arr[i]))

    # 选出每类概率很大的样本作为真实样本
    for i in range(n_class):
        n_c = len(index_cate[i])
        print(i,len(prob_arr[i]))
        max_pred1 = prob_arr[i][0].item()
        max_fea1_index = 0
        max_pred2 = prob_arr[i][0].item()
        max_fea2_index = 0
        if max_pred1>max_pred2:
            tmp = max_pred1,max_fea1_index
            max_pred1,max_fea1_index = max_pred2,max_fea2_index
            max_pred2,max_fea2_index = tmp
        for j in range(n_c):
            if prob_arr[i][j].item() > 0.99:
                if (prob_arr[i][j].item() > max_pred1):
                    max_pred1 = prob_arr[i][j].item()
                    if (max_pred1 > max_pred2):
                        tmp = max_pred1, max_fea1_index
                        max_pred1, max_fea1_index = max_pred2, max_fea2_index
                        max_pred2, max_fea2_index = tmp
                true_samples[i].append(fea[index_cate[i][j]])

        if len(true_samples[i]) == 0:
            true_samples[i].append(fea[index_cate[i][max_fea1_index]])
            true_samples[i].append(fea[index_cate[i][max_fea2_index]])
        if len(true_samples[i]) == 1:
            true_samples[i].append(fea[index_cate[i][max_fea2_index]])


    #给样本评分
    select_index = []

    # 必须大于两个确定样本才能计算
    th = []
    for i in range(n_class):
        n_c = len(index_cate[i])
        index = index_cate[i]
        dis2 = 0
        for j,v1 in enumerate(true_samples[i]):
            max_dis = 0
            for k,v2 in enumerate(true_samples[i]):
                if j!=k:
                    dis = nn.MSELoss()(v1, v2)
                    if dis > max_dis:
                        max_dis = dis
            dis2 += max_dis
        th.append(dis2)


    for i in range(n_class):
        n_c = len(index_cate[i])
        index = index_cate[i]
        cnt = n_c
        for j in range(n_c):
            dis = 0
            v = fea[index_cate[i][j]]
            for k, vi in enumerate(true_samples[i]):
                dis += nn.MSELoss()(v, vi)

            if dis > th[i]*0.85:
                cnt -= 1
                select_index.append(index[j])
        print(i,len(true_samples[i]),cnt)

    cnt = 0
    for i in range(n_class):
        cnt += len(true_samples[i])
    print(cnt)

    t_labels[select_index]=-1

    return t_labels
