import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)
class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.trainloader = self.train_val_test(dataset, list(idxs))
        self.device = 'cuda:0'
        self.criterion = nn.NLLLoss().to(self.device)
    def train_val_test(self, dataset, idxs):
        idxs_train = idxs[:int(1 * len(idxs))]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        return trainloader
    def update_weights_het(self, args, global_protos, model):
        model.train()
        epoch_loss = {'total':[],'1':[], '2':[], '3':[]}
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        for iter in range(20):
            batch_loss = {'total':[],'1':[], '2':[], '3':[]}
            agg_protos_label = {}
            for batch_idx, (images, label_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), label_g.to(self.device)
                model.zero_grad()
                log_probs, protos = model(images)
                loss1 = self.criterion(log_probs, labels)
                loss_mse = nn.MSELoss()
                if len(global_protos) == 0:
                    loss2 = 0*loss1
                else:
                    proto_new = copy.deepcopy(protos.data)
                    i = 0
                    for label in labels:
                        if label.item() in global_protos.keys():
                            proto_new[i, :] = global_protos[label.item()][0].data
                        i += 1
                    loss2 = loss_mse(proto_new, protos)
                loss = loss1 + loss2 * args.ld
                loss.backward()
                optimizer.step()
                for i in range(len(labels)):
                    if label_g[i].item() in agg_protos_label:
                        agg_protos_label[label_g[i].item()].append(protos[i,:])
                    else:
                        agg_protos_label[label_g[i].item()] = [protos[i,:]]
                log_probs = log_probs[:, 0:args.num_classes]
                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()
                batch_loss['total'].append(loss.item())
            epoch_loss['total'].append(sum(batch_loss['total'])/len(batch_loss['total']))
        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        return model.state_dict(), epoch_loss, acc_val.item(), agg_protos_label
    def inference(self, model):
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        accuracy = correct/total
        return accuracy, loss

def test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_gt, global_protos=[]):
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()
    device = 'cuda:0'
    acc_list_g = []
    loss_list = []
    for idx in range(args.num_users):
        model = local_model_list[idx]
        model.to('cuda:0')
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)
        model.eval()
        if global_protos!=[]:
            for batch_idx, (images, labels) in enumerate(testloader):
                images, labels = images.to(device), labels.to(device)
                model.zero_grad()
                outputs, protos = model(images)
                a_large_num = 100
                dist = a_large_num * torch.ones(size=(images.shape[0], args.num_classes)).to(device)
                for i in range(images.shape[0]):
                    for j in range(args.num_classes):
                        if j in global_protos.keys() and j in classes_list[idx]:
                            d = loss_mse(protos[i, :], global_protos[j][0])
                            dist[i, j] = d
                _, pred_labels = torch.min(dist, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)
                proto_new = copy.deepcopy(protos.data)
                i = 0
                for label in labels:
                    if label.item() in global_protos.keys():
                        proto_new[i, :] = global_protos[label.item()][0].data
                    i += 1
                loss2 = loss_mse(proto_new, protos)
                if device == 'cuda:0':
                    loss2 = loss2.cpu().detach().numpy()
                else:
                    loss2 = loss2.detach().numpy()
            acc = correct / total
            print('| User: {} | Global Test Acc with protos: {:.5f}'.format(idx, acc))
            acc_list_g.append(acc)
            loss_list.append(loss2)
    return acc_list_g, acc_list_g, loss_list