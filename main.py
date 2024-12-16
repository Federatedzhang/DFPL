# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from pathlib import Path
import copy, sys
import torch
from resnet import resnet18
from options import parse_arguments
import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_lt
from sampling import cifar_iid, cifar10_noniid, cifar10_noniid_lt
import numpy as np
from update import LocalUpdate,test_inference_new_het_lt
import torch.utils.model_zoo as model_zoo
def agg_func(protos):
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]
    return protos
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde',
}
def proto_aggregation(local_protos_list):
    agg_protos_label = dict()
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]
    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = [proto / len(proto_list)]
        else:
            agg_protos_label[label] = [proto_list[0].data]
    return agg_protos_label
def get_dataset(args, n_list, k_list):
    data_dir = args.data_dir + args.dataset
    if args.dataset == 'mnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True,transform=apply_transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
        if args.iid:
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            if args.unequal:
                user_groups = mnist_noniid_unequal(args, train_dataset, args.num_users)
            else:
                user_groups, classes_list = mnist_noniid(args, train_dataset, args.num_users, n_list, k_list)
                user_groups_lt = mnist_noniid_lt(args, test_dataset, args.num_users, n_list, k_list, classes_list)
                classes_list_gt = classes_list
    elif args.dataset == 'cifar10':
        trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225])])
        trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=trans_cifar10_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=trans_cifar10_val)
        if args.iid:
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            if args.unequal:
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups, classes_list, classes_list_gt = cifar10_noniid(args, train_dataset, args.num_users, n_list, k_list)
                user_groups_lt = cifar10_noniid_lt(args, test_dataset, args.num_users, n_list, k_list, classes_list)
    return train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt
def DFPL(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):
    global_protos = []
    idxs_users = np.arange(args.num_users)
    train_loss, train_accuracy = [], []
    for round in range(args.rounds):
        local_weights, local_losses, local_protos = [], [], {}
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss, acc, protos = local_model.update_weights_het(args, global_protos, model=copy.deepcopy(local_model_list[idx]))
            agg_protos = agg_func(protos)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            local_protos[idx] = agg_protos
        local_weights_list = local_weights
        """"
        The client broadcasts its own local model updates.
        Execute the POW algorithm.
        The specific POW algorithm is omitted.
        """
        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model
        global_protos = proto_aggregation(local_protos)
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos)
    print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    torch.cuda.set_device('cuda:0')
    n_list = np.random.randint(max(2, args.ways - args.stdev), min(args.num_classes, args.ways + args.stdev + 1), args.num_users)
    if args.dataset == 'mnist':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
    elif args.dataset == 'cifar10':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
    train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt = get_dataset(args, n_list, k_list)
    local_model_list = []
    for i in range(args.num_users):
        if args.dataset == 'mnist':
            local_model = CNNMnist(args=args)
        elif  args.dataset == 'cifar10':
            args.stride = [2, 2]
            resnet = resnet18(args, pretrained=True, num_classes=args.num_classes)
            initial_weight = model_zoo.load_url(model_urls['resnet18'])
            local_model = resnet
            initial_weight_1 = local_model.state_dict()
            for key in initial_weight.keys():
                if key[0:3] == 'fc.' or key[0:5]=='conv1' or key[0:3]=='bn1':
                    initial_weight[key] = initial_weight_1[key]
            local_model.load_state_dict(initial_weight)
        local_model.to('cuda:0')
        local_model.train()
        local_model_list.append(local_model)
    DFPL(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list)