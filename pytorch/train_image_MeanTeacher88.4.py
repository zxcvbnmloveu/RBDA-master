will be available soon
import argparse
import os
import os.path as osp
import math
import sys
import time
from util_MT import ConditionalEntropyLoss, VAT, compute_aug_loss2, CosineMarginLoss, compute_cbloss, compute_cbloss2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchsnooper
from torch.utils.data import DataLoader
import lr_schedule, pre_process as prep, network, losssntg
from data_list import ImageList
from util import sample_select, update_fs_centers, update_ls_centers, cal_batch_fclc, usc, utc, disp_f_c, diversity_loss
from util import cal_pseudolabel_w
# seed = 3
# torch.manual_seed(seed)            # 为CPU设置随机种子
# torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
# torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
from util_MT import EMAWeightOptimizer, compute_aug_loss

def output1(mes):
    print(mes)
    log1.write(mes+'\n')

def output2(mes):
    print(mes)
    log2.write(str(mes)+'\n')

def adaptation_factor(x):
    if x>= 1.0:
        return 1.0
    den = 1.0 + math.exp(-10 * x)
    lamb = 2.0 / den - 1.0
    return lamb

def cal_decay(start,end,i):
    i = i+1
    k = 0.0005
    decay = (start-end)*math.exp(-k*i) + end

    return decay


def image_classification_test(loader, model, test_10crop=False,n_class = 12):
    start_test = True
    with torch.no_grad():
        real = [0] * n_class
        pre = [0] * n_class
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
                labels = labels.cpu()
                _, outputs = model(inputs)
                # for i in range(len(labels)):
                #     real[labels[i].item()] += 1
                #     if labels[i] == outputs.max(1)[1][i].cpu():
                #         pre[labels[i].item()] +=1
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)

    # output1([pre[i]/real[i] for i in range(n_class)])
    # cnt = sum([pre[i]/real[i] for i in range(n_class)])/n_class
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy
    # return cnt

def train(config):
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
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
                                        shuffle=True, num_workers=0, drop_last=True)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
                                        shuffle=True, num_workers=0, drop_last=True)

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

    # class_num = config["network"]["params"]["class_num"]
    n_class = config["network"]["params"]["class_num"]

    ## set base network
    net_config = config["network"]
    base_network_stu = net_config["name"](**net_config["params"]).cuda()
    base_network_tea = net_config["name"](**net_config["params"]).cuda()

    ## add additional network for some methods
    if config["loss"]["random"]:
        random_layer = network.RandomLayer([base_network_stu.output_num(), n_class], config["loss"]["random_dim"])
        ad_net = network.AdversarialNetwork(config["loss"]["random_dim"], 1024)
    else:
        random_layer = None

        if config['method'] == 'DANN':
            ad_net = network.AdversarialNetwork(base_network_stu.output_num(), 1024)#DANN
        else:
            ad_net = network.AdversarialNetwork(base_network_stu.output_num() * n_class, 1024)

    if config["loss"]["random"]:
        random_layer.cuda()
    ad_net = ad_net.cuda()
    ad_net2 = network.AdversarialNetwork(n_class, n_class*4)
    ad_net2.cuda()

    parameter_list = base_network_stu.get_parameters() + ad_net.get_parameters()

    teacher_params = list(base_network_tea.parameters())
    for param in teacher_params:
        param.requires_grad = False

    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                                         **(optimizer_config["optim_params"]))

    teacher_optimizer = EMAWeightOptimizer(base_network_tea, base_network_stu, alpha=0.99)

    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    ## train
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    best_acc = 0.0

    output1(log_name)

    loss1, loss2, loss3, loss4, loss5,loss6 = 0, 0, 0, 0, 0,0

    output1('    =======    DA TRAINING    =======    ')

    best1 = 0
    f_t_result = []
    max_iter = config["num_iterations"]
    for i in range(max_iter+1):
        if i % config["test_interval"] == config["test_interval"] - 1 and i > 1500:
            base_network_tea.train(False)
            base_network_stu.train(False)

            # print("test")
            if 'MT' in config['method']:
                temp_acc = image_classification_test(dset_loaders,
                                                     base_network_tea, test_10crop=prep_config["test_10crop"])
                if temp_acc > best_acc:
                    best_acc = temp_acc

                log_str = "iter: {:05d}, tea_precision: {:.5f}".format(i, temp_acc)
                output1(log_str)
                #
                # if i > 20001 and best_acc < 0.69:
                #     break
                #
                # if temp_acc < 0.67:
                #     break
                #
                # if i > 30001 and best_acc < 0.71:
                #     break
                    # torch.save(base_network_tea, osp.join(path, "_model.pth.tar"))

                # temp_acc = image_classification_test(dset_loaders,
                #                                      base_network_stu, test_10crop=prep_config["test_10crop"])
                if temp_acc > best_acc:
                    best_acc = temp_acc
                    torch.save(base_network_stu, osp.join(path, "_model.pth.tar"))
                # log_str = "iter: {:05d}, stu_precision: {:.5f}".format(i, temp_acc)
                # output1(log_str)
            else:
                temp_acc = image_classification_test(dset_loaders,
                                                     base_network_stu, test_10crop=prep_config["test_10crop"])
                if temp_acc > best_acc:
                    best_acc = temp_acc
                    torch.save(base_network_stu, osp.join(path,"_model.pth.tar"))
                log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)
                output1(log_str)

        loss_params = config["loss"]
        ## train one iter
        base_network_stu.train(True)
        base_network_tea.train(True)

        ad_net.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()

        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])

        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()

        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        features_source, outputs_source = base_network_stu(inputs_source)

        features_target_stu, outputs_target_stu = base_network_stu(inputs_target)
        features_target_tea, outputs_target_tea = base_network_tea(inputs_target)

        softmax_out_source = nn.Softmax(dim=1)(outputs_source)
        softmax_out_target_stu = nn.Softmax(dim=1)(outputs_target_stu)
        softmax_out_target_tea = nn.Softmax(dim=1)(outputs_target_tea)

        features = torch.cat((features_source, features_target_stu), dim=0)

        if 'MT' in config['method']:
            softmax_out = torch.cat((softmax_out_source, softmax_out_target_tea), dim=0)

        else:
            softmax_out = torch.cat((softmax_out_source, softmax_out_target_stu), dim=0)

        vat_loss = VAT(base_network_stu).cuda()

        n, d = features_source.shape
        decay = cal_decay(start=1,end=0.6,i = i)
        # image number in each class
        s_labels = labels_source
        t_max, t_labels = torch.max(softmax_out_target_tea, 1)
        t_max, t_labels = t_max.cuda(), t_labels.cuda()
        if config['method'] == 'DANN+dis' or config['method'] == 'CDRL':
            pass

        elif config['method'] == 'RESNET':
            classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
            total_loss = classifier_loss

        elif config['method'] == 'CDAN+E':
            entropy = losssntg.Entropy(softmax_out)
            ad_loss = losssntg.CDANori([features, softmax_out], ad_net, entropy, network.calc_coeff(i),
                                          random_layer)

            transfer_loss = loss_params["trade_off"] * ad_loss
            classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
            total_loss = transfer_loss + classifier_loss


        elif config['method'] == 'CDAN':
            ad_loss = losssntg.CDANori([features, softmax_out], ad_net, None, None, random_layer)

            transfer_loss = loss_params["trade_off"] * ad_loss
            classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
            total_loss = transfer_loss + classifier_loss


        elif config['method'] == 'DANN':

            ad_loss = losssntg.DANN(features, ad_net)
            transfer_loss = loss_params["trade_off"] * ad_loss
            classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
            total_loss = transfer_loss + classifier_loss


        elif config['method'] == 'CDAN+MT':
            th = config['th']

            ad_loss = losssntg.CDANori([features, softmax_out], ad_net, None, None, random_layer)
            unsup_loss = compute_aug_loss(softmax_out_target_stu, softmax_out_target_tea, n_class,confidence_thresh=th)
            unsup_loss = compute_aug_loss2(softmax_out_target_stu, softmax_out_target_tea, n_class,confidence_thresh=th)

            transfer_loss = loss_params["trade_off"] * ad_loss \
                            + 0.01 * unsup_loss
            classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
            total_loss = transfer_loss + classifier_loss

        elif config['method'] == 'CDAN+MT+VAT':
            cent = ConditionalEntropyLoss().cuda()
            ad_loss = losssntg.CDANori([features, softmax_out], ad_net, None, None, random_layer)
            unsup_loss = compute_aug_loss(softmax_out_target_stu, softmax_out_target_tea, n_class)
            loss_trg_cent = 1e-2 * cent(outputs_target_stu)
            loss_trg_vat = 1e-2 * vat_loss(inputs_target, outputs_target_stu)
            transfer_loss = loss_params["trade_off"] * ad_loss \
                            + 0.001*(unsup_loss + loss_trg_cent + loss_trg_vat)
            classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
            total_loss = transfer_loss + classifier_loss

        elif config['method'] == 'CDAN+MT+cent+VAT+temp':
            th = 0.7
            ad_loss = losssntg.CDANori([features, softmax_out], ad_net, None, None, random_layer)
            unsup_loss = compute_aug_loss(softmax_out_target_stu, softmax_out_target_tea, n_class,confidence_thresh=th)
            cent = ConditionalEntropyLoss().cuda()
            # loss_src_vat = vat_loss(inputs_source, outputs_source)
            loss_trg_cent = 1e-2 * cent(outputs_target_stu)
            loss_trg_vat = 1e-2 * vat_loss(inputs_target, outputs_target_stu)
            transfer_loss = loss_params["trade_off"] * ad_loss \
                            + unsup_loss + loss_trg_cent + loss_trg_vat
            # temperature
            classifier_loss = nn.NLLLoss()(nn.LogSoftmax(1)(outputs_source / 1.05), labels_source)
            total_loss = transfer_loss + classifier_loss

        elif config['method'] == 'CDAN+MT+cent+VAT+weightCross+T':

            if i % len_train_target == 0:

                if i != 0:
                    # print(cnt)
                    cnt = torch.tensor(cnt).float()
                    weight = cnt.sum() - cnt
                    weight = weight.cuda()
                else:
                    weight = torch.ones(n_class).cuda()

                cnt = [0] * n_class


            for j in t_labels:
                cnt[j.item()] += 1


            a = config['a']
            b = config['b']
            th = config['th']
            temp = config['temp']

            ad_loss = losssntg.CDANori([features, softmax_out], ad_net, None, None, random_layer)
            unsup_loss = compute_aug_loss2(softmax_out_target_stu, softmax_out_target_tea, n_class,confidence_thresh=th)
            # cbloss = compute_cbloss(softmax_out_target_stu, n_class, cls_balance=0.05)
            # unsup_loss = compute_aug_loss(softmax_out_target_stu, softmax_out_target_tea, n_class,confidence_thresh=th)

            cent = ConditionalEntropyLoss().cuda()

            # loss_src_vat = vat_loss(inputs_source, outputs_source)
            loss_trg_cent = 1e-2 * cent(outputs_target_stu)
            loss_trg_vat = 1e-2 * vat_loss(inputs_target, outputs_target_stu)
            classifier_loss = nn.CrossEntropyLoss(weight=weight)(outputs_source/temp, labels_source)
            # classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
            transfer_loss = loss_params["trade_off"] * ad_loss \
                + a*unsup_loss + b*(loss_trg_vat+loss_trg_cent)

            total_loss = transfer_loss + classifier_loss

        elif config['method'] == 'CDAN+MT+E+VAT+weightCross+T':
            entropy = losssntg.Entropy(softmax_out)
            if i % len_train_target == 0:

                if i != 0:
                    # print(cnt)
                    cnt = torch.tensor(cnt).float()
                    weight = cnt.sum() - cnt
                    weight = weight.cuda()
                else:
                    weight = torch.ones(n_class).cuda()

                cnt = [0] * n_class

            for j in t_labels:
                cnt[j.item()] += 1

            a = config['a']
            b = config['b']
            th = config['th']
            temp = config['temp']

            ad_loss = losssntg.CDANori([features, softmax_out], ad_net, entropy, network.calc_coeff(i),
                                       random_layer)            # unsup_loss = compute_aug_loss2(softmax_out_target_stu, softmax_out_target_tea, n_class,confidence_thresh=th)
            # cbloss = compute_cbloss(softmax_out_target_stu, n_class, cls_balance=0.05)
            unsup_loss = compute_aug_loss(softmax_out_target_stu, softmax_out_target_tea, n_class, confidence_thresh=th)

            cent = ConditionalEntropyLoss().cuda()

            # loss_src_vat = vat_loss(inputs_source, outputs_source)
            loss_trg_cent = 1e-2 * cent(outputs_target_stu)
            loss_trg_vat = 1e-2 * vat_loss(inputs_target, outputs_target_stu)
            classifier_loss = nn.CrossEntropyLoss(weight=weight)(outputs_source / temp, labels_source)
            # classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
            transfer_loss = loss_params["trade_off"] * ad_loss \
                            + a * unsup_loss + b * (loss_trg_vat + loss_trg_cent)

            total_loss = transfer_loss + classifier_loss

        # adout
        # ad_out1 = ad_net(features_source)
        # w = 1-ad_out1
        # c = w * nn.CrossEntropyLoss(reduction='none')(outputs_source, labels_source)
        # classifier_loss = c.mean()
        # total_loss = transfer_loss + classifier_loss
        total_loss.backward()
        optimizer.step()
        teacher_optimizer.step()

        loss1 += ad_loss.item()
        loss2 += classifier_loss.item()
        # loss3 += unsup_loss.item()
        # loss4 += loss_trg_cent.item()
        # loss5 += loss_trg_vat.item()
        # loss6 += cbloss.item()
        # dis_sloss_l += sloss_l.item()

        if i % 50 == 0 and i != 0:
            output1('iter:{:d}, ad_loss_D:{:.2f}, closs:{:.2f}, unsup_loss:{:.2f}, loss_trg_cent:{:.2f}, loss_trg_vatcd:{:.2f}, cbloss:{:.2f}'
                    .format(i, loss1, loss2, loss3, loss4, loss5,loss6))
            loss1, loss2, loss3, loss4, loss5, loss6 = 0, 0, 0, 0, 0, 0

    # torch.save(best_model, osp.join(path, "best_model.pth.tar"))
    return best_acc


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

    parser.add_argument('--num_iterations', type=int, default=10000, help="num_iterations")

    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output1 model")
    parser.add_argument('--output_dir', type=str, default='zan',
                        help="output1 directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    parser.add_argument('--save_name', type=str, default='result',
                        help="save_name")
    parser.add_argument('--a', type=float, default=0.001,
                        help="th")
    parser.add_argument('--b', type=float, default=0.001,
                        help="th")
    parser.add_argument('--c', type=float, default=0.001,
                        help="th")
    parser.add_argument('--temp', type=float, default=1.05,
                        help="th")
    parser.add_argument('--th', type=float, default=0.7,
                        help="th")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    # train config
    config = {}
    config['a'] = args.a
    config['b'] = args.b
    config['c'] = args.c
    config['temp'] = args.temp
    config['th'] = args.th
    # transfer_loss = loss_params["trade_off"] * ad_loss \
    #                 + a * unsup_loss + b * (loss_trg_vat + loss_trg_cent)

    config['method'] = args.method
    config["gpu"] = args.gpu_id
    config["num_iterations"] = args.num_iterations
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["output_path"] = "../snapshot/" + args.output_dir
    if not osp.exists(config["output_path"]):
        os.system('mkdir -p ' + config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log1.txt"), "w")
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])

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
    config["data"] = {"source": {"list_path": args.s_dset_path, "batch_size": 20},#best34
                      "target": {"list_path": args.t_dset_path, "batch_size": 20},
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
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
    # config["out_file"].write(str(config))
    # config["out_file"].flush()

    now = time.localtime(time.time())
    log_name = str(now.tm_mon) + str(now.tm_mday) + '_' \
               + args.save_name + config['method']
    path = './snapshot/'+log_name
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)

    log1 = open(path + '/acc' + '_' + str(now.tm_hour) + str(now.tm_min) + '.log', 'w')
    log2 = open(path + '/dis.log', 'w')
    output1(str(config))
    output1(sys.argv[0][sys.argv[0].rfind(os.sep) + 1:])
    output2(sys.argv[0][sys.argv[0].rfind(os.sep) + 1:])
    log_test_error = open('./snapshot/' + 'test_error_ours' + '.log', 'w')

    best_acc = train(config)
    output1(str(best_acc))

