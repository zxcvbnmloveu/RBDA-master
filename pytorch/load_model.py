import argparse
import os

from torch import nn, optim

import lr_schedule, pre_process as prep, network, losssntg
import torch
import torch.nn
import numpy as np
from data_list import ImageList
from torch.utils.data import DataLoader

import network
from util import DummyDataset, make_data_loader, get_sampled_data_loader


def train_homo_cl(loader,basenet):
    homo_cl = network.Homo_classifier(256,1024).cuda()
    parameter_list = homo_cl.get_parameters()

    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]
    len_train_source = len(loader["source"])

    loss_cnt = 0
    for i in range(1):
        # homo_cl.train(True)
        optimizer.zero_grad()
        if i % len_train_source == 0:
            iter_source = iter(loader["source"])
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        inputs_source, labels_source = iter_source.next()
        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        features_source, _ = basenet(inputs_source)

        bz = inputs_source.size(0)

        arr = np.arange(bz)
        np.random.shuffle(arr)
        index1 = arr[:bz // 2]
        index2 = arr[bz // 2:]
        # input = torch.cat((features_source[index1],features_source[index2]),dim=1).cuda()
        input = (features_source[index1]-features_source[index2])**2
        label = (labels_source[index1] == labels_source[index2]).float().cuda()
        label = label.reshape(-1, 1)
        out = homo_cl(input)
        loss = nn.BCELoss()(out, label)
        loss.backward()
        optimizer.step()

        loss_cnt+=loss.item()
        if i%50== 0 and i!=0:
            print(loss_cnt)
            loss_cnt=0
    return homo_cl

def t1(fea,labels,n_class):
    t_labels = labels.clone().detach()
    index_cate = [[] for i in range(n_class)]

    for l in range(fea.size(0)):
        index_cate[int(t_labels[l].item())].append(l)

    density_arr = [[] for i in range(n_class)]
    # n_class,n_c
    dis_arr = [[] for i in range(n_class)]
    # n_class,n_c*n_c
    dis_arr2 = [[] for i in range(n_class)]
    # n_class,n_c
    centers_index = []

    for i in range(n_class):
        for j in index_cate[i]:
            for l in index_cate[i]:
                dis_arr[i].append(nn.MSELoss()(fea[j], fea[l]).view(-1))

    for i in range(n_class):
        n_c = len(index_cate[i])
        for j in range(n_c):
            # print(torch.cat((dis_arr[i][j*n_c:(j+1)*n_c])))
            density_arr[i].append(torch.sum(torch.cat((dis_arr[i][j*n_c:(j+1)*n_c]))).view(-1))

    for i in range(n_class):
        centers_index.append(torch.cat(density_arr[i]).min(0)[1].item())

    # 中心离各个样本之间的距离
    for i in range(n_class):
        n_c = len(index_cate[i])
        c = centers_index[i]
        dis_arr2[i] = (dis_arr[i][c*n_c:(c+1)*n_c])

    # print(index_cate)
    # print(density_arr)
    # print(centers_index)
    # print(dis_arr2)
    select_index = []
    for i in range(n_class):
        sort = (torch.argsort(torch.cat(dis_arr2[i])))
        index = index_cate[i]
        # print(sort)
        # print(index)
        n_c = sort.size(0)
        # 注意距离相同的话，可能输出较多个。
        # print('th',n_c,int(n_c*0.5)-1)
        for j in range(n_c):
            if sort[j].item() > int(n_c*0.3)-1:
                select_index.append(index[j])
                # print('select',sort[j].item(),index[j])
        # print(select_index)
        t_labels[select_index]=-1
    return t_labels
        # print(t_labels)

def t2(fea,labels_t,prob,n_class):
    t_labels = labels_t.clone().detach()
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

    # 选出每类概率很大的样本作为真实样本
    for i in range(n_class):
        n_c = len(index_cate[i])
        max_pred1 = prob_arr[i][0].item()
        max_fea1_index = 0
        max_pred2 = prob_arr[i][1].item()
        max_fea2_index = 1
        if max_pred1>max_pred2:
            tmp = max_pred1,max_fea1_index
            max_pred1,max_fea1_index = max_pred2,max_fea2_index
            max_pred2,max_fea2_index = tmp
        for j in range(n_c):
            if prob_arr[i][j].item() > 0.999:
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

            if dis > th[i]*0.75:
                cnt -= 1
                select_index.append(index[j])
        print(i,len(true_samples[i]),cnt)

    cnt = 0
    for i in range(n_class):
        cnt += len(true_samples[i])
    print(cnt)

    t_labels[select_index]=-1

    return t_labels

def t3(fea,labels_t,prob,dsets,model,n_class):
    t_labels = labels_t.clone().detach()
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

    # 选出每类概率很大的样本作为真实样本
    for i in range(n_class):
        n_c = len(index_cate[i])
        max_pred1 = prob_arr[i][0].item()
        max_fea1_index = 0
        max_pred2 = prob_arr[i][1].item()
        max_fea2_index = 1
        if max_pred1>max_pred2:
            tmp = max_pred1,max_fea1_index
            max_pred1,max_fea1_index = max_pred2,max_fea2_index
            max_pred2,max_fea2_index = tmp
        for j in range(n_c):
            if prob_arr[i][j].item() > 0.999:
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
        th.append(dis2*0.65)


    for i in range(n_class):
        n_c = len(index_cate[i])
        index = index_cate[i]
        cnt = n_c
        for j in range(n_c):
            dis = 0
            v = fea[index_cate[i][j]]
            for k, vi in enumerate(true_samples[i]):
                dis += nn.MSELoss()(v, vi)

            if dis > th[i]:
                cnt -= 1
                select_index.append(index[j])
        print(i,len(true_samples[i]),cnt)

    cnt = 0
    for i in range(n_class):
        cnt += len(true_samples[i])
    print(cnt)

    t_labels[select_index]=-1

    data_loader = get_sampled_data_loader(dsets["source"])
    iter_source = iter(data_loader)

    start_test = True
    for i in range(len(data_loader)):
        inputs_source, labels_source = iter_source.next()
        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        features_source, _ = model(inputs_source)

        n = inputs_source.size(0)
        select_index = []
        for l in range(n):
            dis = 0
            v = features_source[l]
            for k, vi in enumerate(true_samples[labels_source[l].item()]):
                dis += nn.MSELoss()(v, vi.cuda()).cpu()
            if dis > th[labels_source[l].item()]:
                select_index.append(l)

        labels_source[select_index] = -1
        if start_test:
            all_label = labels_source.float()

            start_test = False
        else:
            all_label = torch.cat((all_label, labels_source.float()), 0)

    return t_labels,all_label

def load_model(path):
    PATH = path+'/2999_model.pth.tar'
    # model = network.ResNetFc(**{'resnet_name': 'ResNet50', 'use_bottleneck': True, 'bottleneck_dim': 256, 'new_cls': True, 'class_num': 12})
    # model_dict = torch.load(PATH).state_dict()
    # # 载入参数
    # model.load_state_dict(model_dict)
    # model.load_state_dict(torch.load(PATH))

    model = torch.load(path)
    return model

def save_feature(loader, model,path):
    start_test = True
    iter_test = iter(loader['target'])
    model.train(False)
    with torch.no_grad():
        for i in range(len(loader['target'])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            labels = labels.cuda()
            features, _ = model(inputs)
            if start_test:
                all_feature = features.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_feature = torch.cat((all_feature, features.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
        all_feature = all_feature.cpu().numpy()
        all_label = all_label.cpu().numpy()
        np.save(path+'/fea_t.npy',all_feature)
        np.save(path+'/fea_t_l.npy',all_label)
    start_test = True
    iter_test = iter(loader['source'])
    with torch.no_grad():
        for i in range(len(loader['source'])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            labels = labels.cuda()
            features, _ = model(inputs)
            if start_test:
                all_feature = features.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_feature = torch.cat((all_feature, features.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
        all_feature = all_feature.cpu().numpy()
        all_label = all_label.cpu().numpy()
        np.save(path + '/fea_s.npy', all_feature)
        np.save(path + '/fea_s_l.npy', all_label)

def image_classification_test(loader, model, test_10crop=False):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    _, predict_out = model(inputs[j])
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)

        else:
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels.cuda()
                _, outputs = model(inputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float().cpu()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float().cpu()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy

def fun1(loader, model, n_class, dsets):
    iter_source = iter(loader['source'])
    # iter_test = iter(loader["test1"])

    model.train(False)
    model.eval()

    iter_test = iter(loader["test"])
    with torch.no_grad():

        arr1 = [[] for i in range(n_class)]
        arr2 = [[] for i in range(n_class)]

        d = 256
        f_c = torch.zeros(n_class, 256)
        start_test = True
        for i in range(len(loader['test'])):
            inputs_target, labels_target = iter_test.next()
            inputs_target, labels_target = inputs_target.cuda(), labels_target.cuda()
            features_target, outputs_target = model(inputs_target)
            softmax_out_target = nn.Softmax(dim=1)(outputs_target)
            t_labels = torch.max(softmax_out_target, 1)[1].cuda()
            max_pred = torch.max(softmax_out_target, 1)[0].cuda()

            if start_test:
                all_feature = features_target.float().cpu()
                all_tlabel = t_labels.float()
                all_label = labels_target.float()
                all_pred = max_pred

                start_test = False
            else:
                all_feature = torch.cat((all_feature, features_target.float().cpu()), 0)
                all_tlabel = torch.cat((all_tlabel, t_labels.float()), 0)
                all_label = torch.cat((all_label, labels_target.float()), 0)
                all_pred = torch.cat((all_pred, max_pred), 0)

        print((all_label==all_tlabel.float()).sum(),all_tlabel.size())

        # pseudo_labels = t1(all_feature,all_tlabel,n_class)
        # pseudo_labels = t2(all_feature,all_tlabel,all_pred,n_class)
        pseudo_labels,label_s = t3(all_feature, all_tlabel, all_pred, dsets, model, n_class)

        for i in range(all_label.size(0)):
            arr1[int(all_tlabel[i].item())].append(all_label[i].view(-1))
            arr2[int(all_tlabel[i].item())].append(pseudo_labels[i].view(-1))

        log.write(path+'\n')

        cnt1 = 0
        cnt2 = 0
        cnt3 = 0
        cnt4 = 0

        for i in range(n_class):
            a1 = torch.cat(arr1[i])
            a2 = torch.cat(arr2[i])
            # print(a1)
            # print(a2)
            judge = (a1 == (torch.ones_like(a1)*i))
            s,r = a1.size(0),judge.sum().item() #每类总数，每类分对个数
            s1,r1 = (a2!=(torch.ones_like(a2)*-1)).sum().item(),(a1==a2).sum().item()#挑选总数，挑选分对个数
            cnt1 += r1
            cnt2 += s1
            cnt3 += r
            cnt4 += s

            log.write('class:{:d}, 每类总数:{:.2f}, 分对个数:{:.2f}, r/s:{:.2f}'
                      '挑选总数:{:.2f}, 挑选分对个数:{:.2f}, r/s:{:.2f}'
                      '\n'.format(i,s,r,r/s,s1,r1,0))

        print(str(cnt3 )+' '+str(cnt4)+' '+str(cnt1)+' '+str(cnt2))
        log.write(str(cnt3 )+' '+str(cnt4)+' '+str(cnt1)+' '+str(cnt2))
            # print(len(arr1[i]),len(arr2[i]))
        #     f_c[i] = torch.cat(arr1[i]).reshape(-1, d).median(dim=0)[0]
        log.write('\n')

    # all_tlabel = all_tlabel.cpu()
    # pseudo_labels = np.where(all_pred.cpu()<0.999,torch.ones_like(all_tlabel)*-1.,all_tlabel)
    # print(len(pseudo_labels),(pseudo_labels==-1.).sum())
    # target_dataset_labelled = DummyDataset(dsets["test"], pseudo_labels)
    # target_dataloader_labelled = DataLoader(target_dataset_labelled, batch_size=34, \
    #                                     shuffle=True, num_workers=0, drop_last=True)
    #
    # source_dataset_labelled = DummyDataset(dsets["source"], label_s)
    # source_dataloader_labelled = DataLoader(source_dataset_labelled, batch_size=34,
    #                                         shuffle=True, num_workers=0, drop_last=True)
    #
    # parameter_list = model.get_parameters()
    # ## set optimizer
    # optimizer_config = config["optimizer"]
    # optimizer = optimizer_config["type"](parameter_list, **(optimizer_config["optim_params"]))
    # param_lr = []
    #
    # for param_group in optimizer.param_groups:
    #     param_lr.append(param_group["lr"])
    # schedule_param = optimizer_config["lr_param"]
    # lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]
    #
    # loss1_cnt = 0
    # loss2_cnt = 0
    #
    # for i in range(501):
    #
    #     if i % 10 == 0:
    #         model.train(False)
    #         temp_acc = image_classification_test(loader, model)
    #         log_str = "iter: {:d} precision: {:.5f}".format(i,temp_acc)
    #         print(log_str)
    #
    #     model.train(True)
    #     optimizer = lr_scheduler(optimizer, i, **schedule_param)
    #     optimizer.zero_grad()
    #
    #     if i % len(target_dataloader_labelled) == 0:
    #         iter_labelled = iter(target_dataloader_labelled)
    #     if i%len(source_dataloader_labelled) == 0:
    #         iter_s = iter(source_dataloader_labelled)
    #
    #     input_s, _, pseu_l_s = iter_s.next()
    #     input_t, true_t, pseu_l_t = iter_labelled.next()
    #     input_s, pseu_l_s = input_s.cuda(), pseu_l_s.cuda()
    #     input_t, pseu_l_t = input_t.cuda(), pseu_l_t.cuda()
    #     # input_t, true_t = input_t.cuda(), true_t.cuda()
    #
    #     _,output_s = model(input_s)
    #     _,output_t = model(input_t)
    #
    #     loss1 = nn.CrossEntropyLoss(ignore_index=-1)(output_s,pseu_l_s.long())
    #     loss2 = nn.CrossEntropyLoss(ignore_index=-1)(output_t,pseu_l_t.long())
    #     print(pseu_l_t)
    #     print(loss2)
    #
    #
    #     loss = loss2
    #     loss.backward()
    #     optimizer.step()
    #
    #     loss1_cnt += loss1.item()
    #     loss2_cnt += loss2.item()
    #
    #     if i % 50 == 0 and i != 0:
    #         print(loss1_cnt,loss2_cnt)
    #         loss1_cnt = 0
    #         loss2_cnt = 0

def run(config,path):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = 34
    test_bs = 34
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
                                        shuffle=True, num_workers=0, drop_last=True)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
                                        shuffle=True, num_workers=0, drop_last=True)
    dset_loaders["test1"] = DataLoader(dsets["target"], batch_size=train_bs, \
                                        shuffle=False, num_workers=1, drop_last=False)

    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["test"] = [ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                       transform=prep_dict["test"][i]) for i in range(10)]
            dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, \
                                               shuffle=False, num_workers=0) for dset in dsets['test']]
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                  transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                          shuffle=False, num_workers=0)

    n_class = config["network"]["params"]["class_num"]
    # for p in [499,999,1499,1999,2499,2999]:
    #     PATH = path + '/'+ str(p) + '_model.pth.tar'
    #
    #     model = load_model(PATH)
    #     base_network = model.cuda()
    #     fun1(dset_loaders, base_network, n_class)

    PATH = path+'/2999_model.pth.tar'

    model = load_model(PATH)
    base_network = model.cuda()
    # homo_cl = train_homo_cl(dset_loaders, base_network)

    fun1(dset_loaders, base_network, n_class, dsets)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--method', type=str, default='CDAN+E')
                        # , choices=['CDAN', 'CDAN+E', 'DANN','NEW','DANN+dis','CDRL','DANN+cluster'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50',
                        choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13",
                                 "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
    parser.add_argument('--dset', type=str, default='office', choices=['office', 'image-clef', 'visda', 'office-home'],
                        help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='./data/office/amazon_list.txt',
                        help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='./data/office/webcam_list.txt',
                        help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output1 model")
    parser.add_argument('--output_dir', type=str, default='san',
                        help="output1 directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    parser.add_argument('--save_name', type=str, default='result',
                        help="save_name")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    # train config
    config = {}
    config['method'] = args.method
    config["gpu"] = args.gpu_id
    config["num_iterations"] = 6001
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True

    config["prep"] = {"test_10crop": False, 'params': {"resize_size": 256, "crop_size": 224, 'alexnet': False}}
    config["loss"] = {"trade_off": 1.0}
    if "AlexNet" in args.net:
        config["prep"]['params']['alexnet'] = True
        config["prep"]['params']['crop_size'] = 227
        config["network"] = {"name": network.AlexNetFc, \
                             "params": {"use_bottleneck": True, "bottleneck_dim": 256, "new_cls": True}}
    elif "ResNet" in args.net:
        config["network"] = {"name": network.ResNetFc, \
                             "params": {"resnet_name": args.net, "use_bottleneck": True, "bottleneck_dim": 256,
                                        "new_cls": True}}
    elif "VGG" in args.net:
        config["network"] = {"name": network.VGGFc, \
                             "params": {"vgg_name": args.net, "use_bottleneck": True, "bottleneck_dim": 256,
                                        "new_cls": True}}
    config["loss"]["random"] = args.random
    config["loss"]["random_dim"] = 1024
    config["optimizer"] = {"type": optim.SGD, "optim_params": {'lr': args.lr, "momentum": 0.9, \
                                                               "weight_decay": 0.0005, "nesterov": True},
                           "lr_type": "inv", \
                           "lr_param": {"lr": args.lr, "gamma": 0.001, "power": 0.75}}
    config["dataset"] = args.dset
    config["data"] = {"source": {"list_path": args.s_dset_path, "batch_size": 34},#原本36
                      "target": {"list_path": args.t_dset_path, "batch_size": 34},
                      "test": {"list_path": args.t_dset_path, "batch_size": 20}}
    if config["dataset"] == "office":
        if ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path) or \
                ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path) or \
                ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path) or \
                ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        elif ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or \
                ("dslr" in args.s_dset_path and "webcam" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.0003  # optimal parameters

        config["network"]["params"]["class_num"] = 31
    elif config["dataset"] == "image-clef":
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = 12
    elif config["dataset"] == "visda":
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = 12
        config['loss']["trade_off"] = 1.0
    elif config["dataset"] == "office-home":
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = 65


    # path = './snapshot/310_CPCDAN'
    # log = open(path + '/ana' + '.log', 'w')
    # run(config,path)
    #
    path = './snapshot/310_WACDAN'
    log = open(path + '/ana2' + '.log', 'w')
    run(config, path)

    # path = './snapshot/310_WACDAN'
    # log = open(path + '/ana' + '.log', 'w')
    # run(config, path)
