import torch
from dgl import load_graphs
import sys
import dgl
import numpy as np
from sklearn import metrics
from dgllife.model.gnn import mpnn
from dgllife.model.model_zoo import mlp_predictor
from torch.utils.data import Dataset, DataLoader
from dgllife.model.readout.attentivefp_readout import AttentiveFPReadout
from dgllife.model.gnn.attentivefp import AttentiveFPGNN
from dgl.data.utils import load_info
from dgllife.utils import Meter
from dgllife.utils import CanonicalBondFeaturizer, CanonicalAtomFeaturizer
from dgllife.utils import RandomSplitter


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)


class AGN(torch.nn.Module):
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats=128,
                 edge_hidden_feats=128,
                 predictor_hidden_feats=512,
                 num_time_steps=2,
                 dropout=0.2,
                 n_tasks=1,
                 num_step_message_passing=6,
                 ):
        super(AGN, self).__init__()
        self.gnn = mpnn.MPNNGNN(node_in_feats=node_in_feats,
                                node_out_feats=node_out_feats,
                                edge_in_feats=edge_in_feats,
                                edge_hidden_feats=edge_hidden_feats,
                                num_step_message_passing=num_step_message_passing
                                )
        self.readout = AttentiveFPReadout(feat_size=node_out_feats,
                                          num_timesteps=num_time_steps,
                                          dropout=dropout)
        self.predict = mlp_predictor.MLPPredictor(in_feats=node_out_feats,
                                                  hidden_feats=predictor_hidden_feats,
                                                  n_tasks=n_tasks,
                                                  dropout=dropout)

    def forward(self, g, node_feats, edge_feats):
        n_node_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, n_node_feats)

        return self.predict(graph_feats)



def load_data(g_file, d_file):
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print('device:', device)
    graphs, label_dict = load_graphs(g_file)
    labels = label_dict['glabel']
    data_dict = load_info(d_file)
    graph_list = [g.to(device) for g in graphs]

    return graph_list, labels, data_dict


def collate_mol(data):
    graphs, labels = map(list, zip(*data))
    bg = dgl.batch(graphs)
    labels = torch.stack(labels, dim=0)

    return bg, labels


def run_a_train_epoch(model, data_loader, loss_criterion, optimizer, device='cuda:0'):
    model.train()
    train_meter = Meter()
    for _, batch_data in enumerate(data_loader):
        bg, labels = batch_data
        bg = bg.to(device)
        pred = model(bg, bg.ndata['h'], bg.edata['e'])
        loss = loss_criterion(pred, labels).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(pred, labels)
    return np.mean(train_meter.compute_metric('pr_auc_score')), np.mean(train_meter.compute_metric('roc_auc_score'))


def run_an_eval_epoch(model, data_loader, device='cuda:0'):
    model.eval()
    eval_meter = Meter()
    for _, batch_data in enumerate(data_loader):
        bg, labels = batch_data
        bg = bg.to(device)
        pred = model(bg, bg.ndata['h'], bg.edata['e'])
        eval_meter.update(pred, labels)
    return np.mean(eval_meter.compute_metric('pr_auc_score')), np.mean(eval_meter.compute_metric('roc_auc_score'))


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser('train')
    parser.add_argument('-g', '--g-file', type=str, required=True,
                        help='path of graph.bin')
    parser.add_argument('-d', '--d-file', type=str, required=True,
                        help='path of info.pkl')
    parser.add_argument('-b', '--batch-size', type=int, default=128,
                        help='batch size')
    parser.add_argument('-e', '--epochs', type=int, default=300,
                        help='num of training epochs')
    args = parser.parse_args().__dict__
    graph, label, datas = load_data(args['g_file'], args['d_file'])
    print(len(graph))
    label = label.reshape(len(label), 1).to('cuda')
    dataset = MyDataset(graph, label)
    print(dataset)
    train_set, val_set, test_set = RandomSplitter.train_val_test_split(
        dataset, frac_train=0.8, frac_val=0.1, frac_test=0.1)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=args['batch_size'],
                              shuffle=True,
                              collate_fn=collate_mol)

    val_loader = DataLoader(dataset=val_set,
                            batch_size=args['batch_size'],
                            shuffle=True,
                            collate_fn=collate_mol)

    test_loader = DataLoader(dataset=test_set,
                             batch_size=args['batch_size'],
                             shuffle=True,
                             collate_fn=collate_mol)

    node_in_feat = CanonicalAtomFeaturizer()
    edge_in_feat = CanonicalBondFeaturizer()
    model = AGN(
        node_in_feats=node_in_feat.feat_size(),
        edge_in_feats=edge_in_feat.feat_size(),
    )
    model = model.to('cuda:0')
    loss_criterion = torch.nn.SmoothL1Loss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    print(label.shape)
    for epoch in range(args['epochs']):
        train_pr, train_auc = run_a_train_epoch(model, train_loader, loss_criterion, optimizer)
        val_pr, val_auc = run_an_eval_epoch(model, val_loader)
        print('epoch {:d} | train pr {:.4f} | val pr {:.4f}'.format(epoch, train_pr, val_pr))
        print('             train auc {:.4f} | val auc {:.4f}'.format(train_auc, val_auc))
    torch.save(model, 'data/model.pth')
    model = torch.load('data/model.pth', weights_only=True)
    test_r2, test_mae = run_an_eval_epoch(model, test_loader)
    print('test pr {:.4f}'.format(test_r2))
    print('test mae {:.4f}'.format(test_mae))

# python module.py -g data/test_dgl_graph.bin -d data/test_info.pkl -b 8 -e 1000
