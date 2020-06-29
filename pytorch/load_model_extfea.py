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

def load_model(path):
    PATH = path+'/2999_model.pth.tar'
    # model = network.ResNetFc(**{'resnet_name': 'ResNet50', 'use_bottleneck': True, 'bottleneck_dim': 256, 'new_cls': True, 'class_num': 12})
    # model_dict = torch.load(PATH).state_dict()
    # # 载入参数
    # model.load_state_dict(model_dict)
    # model.load_state_dict(torch.load(PATH))

    model = torch.load(path)
    return model

def save_feature(loader,model,save_path,save_name):
    save_name = save_name
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
                all_label = labels.float().cpu()
                start_test = False
            else:
                all_feature = torch.cat((all_feature, features.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float().cpu()), 0)
        all_feature = all_feature.cpu().numpy()
        all_label = all_label.cpu().numpy()
        np.save(save_path+save_name+'_t.npy',all_feature)
        # np.save(save_path+'/fea_t_l'+save_name,all_label)
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
        np.save(save_path+save_name+'_s.npy',all_feature)
        # np.save(save_path + '/fea_s_l'+save_name, all_label)

def image_classification_test(loader, model, test_10crop=False, n_class = 31):
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
            real = [0] * n_class
            pre = [0] * n_class
            cnt3 = [0] * n_class

            for i in range(len(loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels.cuda()
                _, outputs = model(inputs)
                for i in range(len(labels)):
                    real[labels[i].item()] += 1
                    cnt3[outputs.max(1)[1][i].item()] += 1
                    if labels[i] == outputs.max(1)[1][i]:
                        pre[labels[i].item()] +=1

                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float().cpu()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float().cpu()), 0)

                    # print(labels[:5])
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    print(real)
    print(pre)
    print(cnt3)
    cnt = [round((pre[i]/real[i])*100,1) for i in range(n_class)]
    print(cnt)
    print([round((pre[i]/cnt3[i])*100,1) for i in range(n_class)])

    print(sum([pre[i]/real[i] for i in range(n_class)])/n_class)

    return accuracy

def run(config,model):
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
                                        shuffle=True, num_workers=0, drop_last=False)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
                                        shuffle=True, num_workers=0, drop_last=False)

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

    model.train(False)
    temp_acc = image_classification_test(dset_loaders, model, n_class = n_class)
    log_str = "precision: {:.5f}".format(temp_acc)
    print(log_str)


def save(config,model,save_name):
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
                                        shuffle=True, num_workers=0, drop_last=False)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
                                        shuffle=True, num_workers=0, drop_last=False)

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

    model.train(False)
    save_feature(dset_loaders,model,'./snapshot/model/',save_name)


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
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
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



    # 配置数据集
    choices = ["office", "image-clef", "visda", "office-home"]
    config["dataset"] = choices[2]
    task_office = ['./data/office/amazon_list.txt', './data/office/webcam_list.txt', './data/office/dslr_list.txt']
    task_visda = ['./data/visda-2017/train_list.txt','./data/visda-2017/validation_list.txt']
    args.s_dset_path = task_visda[0]
    args.t_dset_path = task_visda[1]

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

    config["data"] = {"source": {"list_path": args.s_dset_path, "batch_size": 34},  # 原本36
                      "target": {"list_path": args.t_dset_path, "batch_size": 34},
                      "test": {"list_path": args.t_dset_path, "batch_size": 20}}

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # name = '512_DACDAN+MT+cent+VAT+weightCross+T'
    name = '527_visdaCDAN+MT+cent+VAT+weightCross+T'

    path = './snapshot/'+name+ '/_model.pth.tar'

    model = load_model(path).cuda()
    model.train(False)
    run(config,model)
    print('others')
    name = '69_visdaCDAN+MT+cent+VAT+weightCross+T'
    path = './snapshot/' + name + '/_model.pth.tar'
    model = load_model(path).cuda()
    run(config,model)

    # save(config, model, '64AOT')

