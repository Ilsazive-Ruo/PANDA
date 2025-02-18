import pandas as pd
import torch
from dgl import load_graphs
import sys
import dgl
import module2
import tqdm
import openpyxl
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from dgllife.model.gnn import mpnn
from dgllife.model.model_zoo import mlp_predictor
from torch.utils.data import Dataset, DataLoader
from dgllife.model.readout.attentivefp_readout import AttentiveFPReadout
from dgllife.model.gnn.attentivefp import AttentiveFPGNN
from dgl.data.utils import load_info
from dgllife.utils import Meter
from dgllife.utils import CanonicalBondFeaturizer, CanonicalAtomFeaturizer
from dgllife.utils import RandomSplitter


def load_gdata(g_file, d_file):
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print('device:', device)
    graphs = load_graphs(g_file)

    g_names = load_info(d_file)
    g_dicts = {}
    for i in range(len(g_names)):
        g_dicts[g_names[i]] = dgl.add_self_loop(graphs[0][i])

    return g_dicts


def load_FSASet(file, g_dict):
    print('loading datas...')
    dataset_file = pd.read_csv(file)
    print(dataset_file.head())
    graph_pairs, labels, MPs = [], [], []
    for index in dataset_file.index:
        if dataset_file['API'][index] in g_dict.keys() and dataset_file['excipient'][index] in g_dict.keys():
            graph_pairs.append([g_dict[dataset_file['API'][index]], g_dict[dataset_file['excipient'][index]]])
            MPs.append([dataset_file['Ca'][index], dataset_file['Qa'][index],
                        dataset_file['Cb'][index], dataset_file['Qb'][index]])
            labels.append(dataset_file['class'][index])
    print('total samples:', len(labels))
    labels = np.array(labels).reshape(-1, 1)

    return graph_pairs, labels, MPs


# 自定义的 collate_fn 函数，处理图对批次
def collate_graph_pairs(data):
    # 从批次中提取图对
    g1_list, g2_list, labels, MPs = zip(*data)  # zip 解压批次

    # 使用 DGL 的 batch 函数将多个图合并成一个图
    batched_g1 = dgl.batch(g1_list)  # 合并图 1
    batched_g2 = dgl.batch(g2_list)  # 合并图 2
    MPs =torch.stack(MPs, dim=0)
    labels = torch.stack(labels, dim=0)

    # 返回合并后的图对
    return batched_g1, batched_g2, labels, MPs


def run_a_train_epoch(model, data_loader, loss_criterion, optimizer, Epoch, TEpoch, device='cuda:0'):
    model.train()
    train_meter = Meter()

    running_loss = 0.0
    loop = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))
    for step, batch_data in loop:

        bg1, bg2, b_labels, bMPs = batch_data
        bg1 = bg1.to(device)
        bg2 = bg2.to(device)
        bMPs = bMPs.to(device)
        b_labels = b_labels.to(device)
        pred = model(bg1, bg2, bMPs)
        # print('pred shape:', pred.shape)
        loss = loss_criterion(pred, b_labels).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(pred, b_labels)

        running_loss += loss.item()

        loop.set_description(f'Epoch [{Epoch}/{TEpoch}]')
        loop.set_postfix(loss=running_loss / (step + 1))

    return (np.mean(train_meter.compute_metric('pr_auc_score')), np.mean(train_meter.compute_metric('roc_auc_score')),
            np.mean(train_meter.compute_metric('mae')))


def run_an_eval_epoch(model, data_loader, device='cuda:0'):
    with torch.no_grad():
        model.eval()
        pre, y_t = [], []
        eval_meter = Meter()

        for _, batch_data in enumerate(data_loader):
            bg1, bg2, b_labels, bMPs = batch_data
            bg1 = bg1.to(device)
            bg2 = bg2.to(device)
            bMPs = bMPs.to(device)
            b_labels = b_labels.to(device)
            pred = model(bg1, bg2, bMPs)
            eval_meter.update(pred, b_labels)
            pre = np.append(pre, pred.detach().cpu().numpy())
            y_t = np.append(y_t, labels.detach().cpu().numpy())

        return (np.mean(eval_meter.compute_metric('pr_auc_score')), np.mean(eval_meter.compute_metric('roc_auc_score')),
                np.mean(eval_meter.compute_metric('mae')), pre, y_t)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser('train')
    parser.add_argument('-g', '--g-file', type=str, required=True,
                        help='path of graph.bin')
    parser.add_argument('-d', '--d-file', type=str, required=True,
                        help='path of info.pkl')
    parser.add_argument('-s', '--set-file', type=str, required=True,
                        help='path of dataset.csv')
    parser.add_argument('-b', '--batch-size', type=int, default=128,
                        help='batch size')
    parser.add_argument('-e', '--epochs', type=int, default=300,
                        help='num of training epochs')
    args = parser.parse_args().__dict__

    node_feat_size = CanonicalAtomFeaturizer().feat_size('h')
    bond_feat_size = CanonicalBondFeaturizer().feat_size('e')

    g_dict = load_gdata(args['g_file'], args['d_file'])

    graph_pairs, labels, MPs = load_FSASet(args['set_file'], g_dict)
    le = OneHotEncoder(sparse_output=False)
    labels = le.fit_transform(labels)

    MPs = torch.tensor(MPs, dtype=torch.float)
    labels = torch.tensor(labels)
    print('label shape:', labels.shape)

    dataset = module2.FSADataset(graph_pairs, labels, MPs)

    train_set, val_set, test_set = RandomSplitter.train_val_test_split(
        dataset, frac_train=0.9, frac_val=0.1, frac_test=0)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=args['batch_size'],
                              shuffle=True,
                              num_workers=12,
                              pin_memory=True,
                              collate_fn=collate_graph_pairs)

    val_loader = DataLoader(dataset=val_set,
                            batch_size=args['batch_size'],
                            shuffle=True,
                            num_workers=12,
                            pin_memory=True,
                            collate_fn=collate_graph_pairs)

    # model = module2.SYN_NC(node_size=node_feat_size, edge_size=bond_feat_size, hidden_size=node_feat_size)
    pre_model = torch.load('models/SYN_1227/model.pth')
    model = module2.FSA(pre_model=pre_model, fusion_feature_size=node_feat_size)
    model = model.to('cuda:0')
    loss_criterion = torch.nn.SmoothL1Loss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print('initializing training...')
    for epoch in range(args['epochs']):
        train_pr, train_auc, train_mae = run_a_train_epoch(model, train_loader, loss_criterion, optimizer,
                                                           Epoch=epoch, TEpoch=args['epochs'])
        val_pr, val_auc, val_mae, pre, y_t = run_an_eval_epoch(model, val_loader)
        print('epoch {:d} | train pr {:.4f} | val pr {:.4f}'.format(epoch, train_pr, val_pr))
        print('             train auc {:.4f} | val auc {:.4f}'.format(train_auc, val_auc))

    torch.save(model, 'models/SA/FSA_model.pth')


# python transfer_FSA.py -g data/gra/Sa/SA_dgl_graph.bin -d data/gra/SA/SA_info.pkl -s data/FSADATA.csv -b 128 -e 300

