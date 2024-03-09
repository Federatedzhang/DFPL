#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy, sys
import time
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
import random
import torch.utils.model_zoo as model_zoo
from pathlib import Path

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
mod_dir = (Path(__file__).parent / ".." / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))

from resnet import resnet18
from options import args_parser
from update import LocalUpdate, save_protos, LocalTest, test_inference_new_het_lt
from models import CNNMnist, CNNFemnist
from utils import get_dataset, average_weights, exp_details, proto_aggregation, agg_func, average_weights_per, average_weights_sem

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def BADFPL(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list,a,b,tsum):
    summary_writer = SummaryWriter('../tensorboard/'+ args.dataset +'BADFPL' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds) + 'r')
    acc_all = []
    loss_all = []
    global_protos = []
    idxs_users = np.arange(args.num_users)

    train_loss, train_accuracy = [], []

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos = [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')


        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss, acc, protos = local_model.update_weights_het(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            agg_protos = agg_func(protos)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            local_protos[idx] = agg_protos


        # update global weights
        local_weights_list = local_weights

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model

        # update global weights
        global_protos = proto_aggregation(local_protos)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos)

    print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
    np.save("a/"+str(args.ways)+"-"+str(args.stdev)+"/Acctsum=" + str(tsum) + "-b=" + str(b) + "-a=" + str(a) + "-T=" + str(args.rounds) + "-E=" + str(
        args.train_ep) + "-avg=" + str(args.ways) + "-std=" + str(args.stdev) + "-mnist+" + "-lr=" + str(
        args.lr) + "-N=" + str(args.num_users), np.mean(acc_list_l))
    np.save("a/"+str(args.ways)+"-"+str(args.stdev)+"/Losstsum=" + str(tsum) + "-b=" + str(b) + "-a=" + str(a) + "-T=" + str(args.rounds) + "-E=" + str(
        args.train_ep) + "-avg=" + str(args.ways) + "-std=" + str(args.stdev) + "-mnist+" + "-lr=" + str(
        args.lr) + "-N=" + str(args.num_users), np.mean(loss_list))
    # save protos
    if args.dataset == 'mnist':
        save_protos(args, local_model_list, test_dataset, user_groups_lt)


if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    exp_details(args)
    tsum = 100
    for uu in range(3,5):
        args.ways = uu
        for tt in range(1,3):
            args.stdev = tt
            for aa in range(1,5):
                a = aa
                b = 4
                for T in range(1,15):
                    args.rounds = T
                    args.train_ep = int((tsum - b * T) / (a * T))
                    if args.train_ep <= 1:
                        args.train_ep = 1
                    print(args.train_ep )
                # set random seeds
                    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    if args.device == 'cuda':
                        torch.cuda.set_device(args.gpu)
                        torch.cuda.manual_seed(args.seed)
                        torch.manual_seed(args.seed)
                    else:
                        torch.manual_seed(args.seed)
                    np.random.seed(args.seed)
                    random.seed(args.seed)

                    # load dataset and user groups
                    n_list = np.random.randint(max(2, args.ways - args.stdev), min(args.num_classes, args.ways + args.stdev + 1), args.num_users)
                    if args.dataset == 'mnist':
                        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
                    elif args.dataset == 'cifar10':
                        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
                    elif args.dataset =='cifar100':
                        k_list = np.random.randint(args.shots, args.shots + 1, args.num_users)
                    elif args.dataset == 'femnist':
                        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)

                    train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt = get_dataset(args, n_list, k_list)

                    # Build models
                    local_model_list = []
                    for i in range(args.num_users):
                        if args.dataset == 'mnist':
                            if args.mode == 'model_heter':
                                if i<7:
                                    args.out_channels = 18
                                elif i>=7 and i<14:
                                    args.out_channels = 20
                                else:
                                    args.out_channels = 22
                            else:
                                args.out_channels = 20

                            local_model = CNNMnist(args=args)

                        elif args.dataset == 'femnist':
                            if args.mode == 'model_heter':
                                if i<7:
                                    args.out_channels = 18
                                elif i>=7 and i<14:
                                    args.out_channels = 20
                                else:
                                    args.out_channels = 22
                            else:
                                args.out_channels = 20
                            local_model = CNNFemnist(args=args)

                        elif args.dataset == 'cifar100' or args.dataset == 'cifar10':
                            if args.mode == 'model_heter':
                                if i<10:
                                    args.stride = [1,4]
                                else:
                                    args.stride = [2,2]
                            else:
                                args.stride = [2, 2]
                            resnet = resnet18(args, pretrained=False, num_classes=args.num_classes)
                            initial_weight = model_zoo.load_url(model_urls['resnet18'])
                            local_model = resnet
                            initial_weight_1 = local_model.state_dict()
                            for key in initial_weight.keys():
                                if key[0:3] == 'fc.' or key[0:5]=='conv1' or key[0:3]=='bn1':
                                    initial_weight[key] = initial_weight_1[key]

                            local_model.load_state_dict(initial_weight)

                        local_model.to(args.device)
                        local_model.train()
                        local_model_list.append(local_model)

                    if args.mode == 'task_heter':
                        BADFPL(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list,a,b,tsum)
