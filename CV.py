import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from dgllife.utils import CanonicalBondFeaturizer, CanonicalAtomFeaturizer
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
import tqdm
import numpy as np
import PANDA
import train


def run_a_fine_tuning_epoch(model, data_loader, optimizer, Epoch, TEpoch, task='PV', device='cuda:0'):
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='mean')

    running_loss = 0.0
    loop = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))

    y_true, y_pred = [], []
    for step, batch_data in loop:
        graph_a = batch_data['graph_a'].to(device)
        graph_e = batch_data['graph_e'].to(device)
        extra_feat = batch_data['extra_feat'].to(device)

        out1, out2, out3 = model(graph_a, graph_e, extra_feat)
        if task == 'PV':
            task_type = 'multiclass'
            y = batch_data['labels']['task1'].to(device)
            loss = criterion(out1, y)
            y_pred.append(out1.softmax(dim=-1).detach().cpu())

        elif task == 'SA':
            task_type = 'binary'
            y = batch_data['labels']['task2'].to(device)
            loss = criterion(out2, y)
            y_pred.append(torch.argmax(out2, dim=1).detach().cpu())

        else:
            task_type = 'binary'
            y = batch_data['labels']['task3'].to(device)
            loss = criterion(out3, y)
            y_pred.append(torch.argmax(out3, dim=1).detach().cpu())

        y_true.append(y.detach().cpu())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        loop.set_description(f'Epoch [{Epoch}/{TEpoch}]')
        loop.set_postfix(loss=running_loss / (step + 1))

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    metrics, _, _ = train.compute_metrics(y_true.numpy(), y_pred.numpy(), task_type=task_type)

    return metrics


def run_a_val_fine_tuning_epoch(model, data_loader, task='PV', device='cuda:0'):
    with torch.no_grad():
        model.eval()
        criterion = nn.CrossEntropyLoss(reduction='mean')

        running_loss = 0.0
        loop = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))

        y_true, y_pred, val_loss = [], [], 0
        for step, batch_data in loop:
            graph_a = batch_data['graph_a'].to(device)
            graph_e = batch_data['graph_e'].to(device)
            extra_feat = batch_data['extra_feat'].to(device)

            out1, out2, out3 = model(graph_a, graph_e, extra_feat)
            if task == 'PV':
                task_type = 'multiclass'
                y = batch_data['labels']['task1'].to(device)
                loss = criterion(out1, y)
                y_pred.append(out1.softmax(dim=-1).detach().cpu())

            elif task == 'SA':
                task_type = 'binary'
                y = batch_data['labels']['task2'].to(device)
                loss = criterion(out2, y)
                y_pred.append(torch.argmax(out2, dim=1).detach().cpu())

            else:
                task_type = 'binary'
                y = batch_data['labels']['task3'].to(device)
                loss = criterion(out3, y)
                y_pred.append(torch.argmax(out3, dim=1).detach().cpu())

            y_true.append(y.detach().cpu())

            running_loss += loss.item()

            loop.set_description(f'val')
            loop.set_postfix(loss=running_loss / (step + 1))

        val_loss = running_loss / (step + 1)

        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        metrics, roc, pr = train.compute_metrics(y_true.numpy(), y_pred.numpy(), task_type=task_type)

    return metrics, val_loss, roc, pr


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser('CV')
    parser.add_argument('-b', '--batch-size', type=int, default=128,
                        help='batch size')
    parser.add_argument('-e', '--epochs', type=int, default=300,
                        help='num of training epochs')
    args = parser.parse_args().__dict__

    node_feat_size = CanonicalAtomFeaturizer().feat_size('h')
    print(node_feat_size)
    bond_feat_size = CanonicalBondFeaturizer().feat_size('e')

    print('loading syn data')
    g_syn = train.load_gdata('data/gra/SYN/SYN_dgl_graph.bin',
                       "data/gra/SYN/SYN_info.pkl")
    syn_gra_a, syn_gra_e, syn_labels = train.load_syn_sa_dataset('data/syn_dataset.csv', g_syn)

    print('loading sa data')
    g_sa = train.load_gdata('data/gra/SA/SA_dgl_graph.bin',
                      "data/gra/SA/SA_info.pkl")
    sa_gra_a, sa_gra_e, sa_labels = train.load_syn_sa_dataset('data/sa_dataset.csv', g_sa)

    print('loading fsa data')
    g_fsa = train.load_gdata('data/gra/SA/SA_dgl_graph.bin',
                       "data/gra/SA/SA_info.pkl")
    fsa_gra_a, fsa_gra_e, fsa_labels, mcs = train.load_fsa_dataset('data/fsa_dataset.csv', g_fsa)

    g_all = train.load_gdata('data/gra/SA2/SA3_dgl_graph.bin',
                       "data/gra/SA2/SA3_info.pkl")
    print('g_all:', g_all)
    sa_add_gra_a, sa_add_gra_e, sa_add_labels = train.load_syn_sa_dataset('data/sa_add.csv', g_all)
    print(len(sa_add_gra_a), len(sa_add_gra_e), len(sa_add_labels))
    sa_add_set = train.PandaDataset(graphs_a=sa_add_gra_a,
                                    graphs_e=sa_add_gra_e,
                                    labels_task1=[0] * len(sa_add_gra_a),
                                    labels_task2=sa_add_labels,
                                    labels_task3=[0] * len(sa_add_gra_a),
                                    extra_feats=torch.zeros(len(sa_add_gra_a), 4),
                                    task_ids=[1] * len(sa_add_gra_a))

    syn_set = train.PandaDataset(graphs_a=syn_gra_a,
                                 graphs_e=syn_gra_e,
                                 labels_task1=syn_labels,
                                 labels_task2=[0] * len(syn_gra_a),
                                 labels_task3=[0] * len(syn_gra_a),
                                 extra_feats=torch.zeros(len(syn_gra_a), 4),
                                 task_ids=[0] * len(syn_gra_a))
    sa_set = train.PandaDataset(graphs_a=sa_gra_a,
                                graphs_e=sa_gra_e,
                                labels_task1=[0] * len(sa_gra_a),
                                labels_task2=sa_labels,
                                labels_task3=[0] * len(sa_gra_a),
                                extra_feats=torch.zeros(len(sa_gra_a), 4),
                                task_ids=[1] * len(syn_gra_a))
    fsa_set = train.PandaDataset(graphs_a=fsa_gra_a,
                                 graphs_e=fsa_gra_e,
                                 labels_task1=[0] * len(fsa_gra_a),
                                 labels_task2=[0] * len(fsa_gra_a),
                                 labels_task3=fsa_labels,
                                 extra_feats=torch.tensor(mcs),
                                 task_ids=[2] * len(syn_gra_a))

    sa_set = ConcatDataset([sa_set, sa_add_set])

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    train_subset, val_subset = ({'syn_set': {}, 'fsa_set': {}, 'sa_set': {}},
                                {'syn_set': {}, 'fsa_set': {}, 'sa_set': {}})
    for fold, (train_idx, val_idx) in enumerate(kf.split(syn_set, syn_labels)):
        train_subset['syn_set'][str(fold)] = Subset(syn_set, train_idx)
        val_subset['syn_set'][str(fold)] = Subset(syn_set, val_idx)

    for fold, (train_idx, val_idx) in enumerate(kf.split(sa_set, np.concatenate([sa_labels, sa_add_labels]))):
        train_subset['sa_set'][str(fold)] = Subset(sa_set, train_idx)
        val_subset['sa_set'][str(fold)] = Subset(sa_set, val_idx)

    for fold, (train_idx, val_idx) in enumerate(kf.split(fsa_set, fsa_labels)):
        train_subset['fsa_set'][str(fold)] = Subset(fsa_set, train_idx)
        val_subset['fsa_set'][str(fold)] = Subset(fsa_set, val_idx)

    for folds in range(10):
        train_sa = ConcatDataset([train_subset['sa_set'][str(folds)]] * 15)
        train_fsa = ConcatDataset([train_subset['fsa_set'][str(folds)]] * 50)

        train_set = ConcatDataset([train_subset['syn_set'][str(folds)],
                                   train_sa,
                                   train_fsa])
        val_set = ConcatDataset([val_subset['syn_set'][str(folds)],
                                 val_subset['sa_set'][str(folds)],
                                 val_subset['fsa_set'][str(folds)]])

        stage1_loader = DataLoader(dataset=train_set,
                                   batch_size=128,
                                   shuffle=True,
                                   num_workers=12,
                                   pin_memory=True,
                                   collate_fn=train.collate_fn,
                                   persistent_workers=True)
        stage1_val_loader = DataLoader(dataset=val_set,
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=12,
                                       pin_memory=True,
                                       collate_fn=train.collate_fn,
                                       persistent_workers=True)

        SA_loader = DataLoader(dataset=train_sa,
                               batch_size=args['batch_size'],
                               shuffle=True,
                               num_workers=12,
                               pin_memory=True,
                               collate_fn=train.collate_fn,
                               persistent_workers=True)
        SA_val_loader = DataLoader(dataset=val_subset['sa_set'][str(folds)],
                                   batch_size=args['batch_size'],
                                   shuffle=True,
                                   num_workers=12,
                                   pin_memory=True,
                                   collate_fn=train.collate_fn,
                                   persistent_workers=True)

        FSA_loader = DataLoader(dataset=train_fsa,
                                batch_size=args['batch_size'],
                                shuffle=True,
                                num_workers=12,
                                pin_memory=True,
                                collate_fn=train.collate_fn,
                                persistent_workers=True)

        FSA_val_loader = DataLoader(dataset=val_subset['fsa_set'][str(folds)],
                                    batch_size=args['batch_size'],
                                    shuffle=True,
                                    num_workers=12,
                                    pin_memory=True,
                                    collate_fn=train.collate_fn,
                                    persistent_workers=True)

        PV_loader = DataLoader(dataset=train_subset['syn_set'][str(folds)],
                               batch_size=args['batch_size'],
                               shuffle=True,
                               num_workers=12,
                               pin_memory=True,
                               collate_fn=train.collate_fn,
                               persistent_workers=True)

        PV_val_loader = DataLoader(dataset=val_subset['syn_set'][str(folds)],
                                   batch_size=args['batch_size'],
                                   shuffle=True,
                                   num_workers=12,
                                   pin_memory=True,
                                   collate_fn=train.collate_fn,
                                   persistent_workers=True)

        print('initializing model')
        feature_extractor = PANDA.ARCHIT(in_features=node_feat_size,
                                         edge_dim=bond_feat_size,
                                         hidden_features=node_feat_size,
                                         out_features=node_feat_size)
        feature_extractor.load_state_dict(torch.load('weights/pretrained_encoder.pth'))
        model = PANDA.Panda(encoder=feature_extractor,
                            node_size=node_feat_size,
                            hidden_size=1024)

        model.to('cuda:0')

        print('initializing warmup...')
        early_stopping = PANDA.EarlyStopping(patience=5, verbose=True, delta=0.005, path='weights/test/Panda.pth')
        for epoch in range(50):
            print('Folds: {}'.format(folds))
            FT_pv = train.run_a_fine_tuning_epoch(model, PV_loader,
                                                   Epoch=epoch, TEpoch=args['epochs'], task='PV', stage='WU')
            print('epoch {:d} | pv f1 {:.4f} | pv acc {:.4f} | pv auc {:.4f}'.format(
                epoch, FT_pv['f1'], FT_pv['accuracy'], FT_pv['roc_auc']))
            FT_pv_val, pv_loss, _, _ = train.run_a_val_fine_tuning_epoch(model, PV_val_loader, task='PV')
            print('val | pv loss {} | pv f1 {:.4f} | pv acc {:.4f} | pv auc {:.4f}'.format(
                pv_loss, FT_pv_val['f1'], FT_pv_val['accuracy'], FT_pv_val['roc_auc']))

            if early_stopping(pv_loss, model):
                print("warmup stopped.")
                break
        early_stopping.load_best_model(model)

        early_stopping = PANDA.EarlyStopping(patience=5, verbose=True, delta=0.005, path='weights/test/Panda.pth')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        pcgrad = train.PCGrad(optimizer)
        for epoch in range(2):
            print('Folds: {}'.format(folds))
            train_acc_syn, train_acc_sa, train_acc_fsa = train.run_a_train_epoch(model, stage1_loader, pcgrad,
                                                                                  Epoch=epoch, TEpoch=args['epochs'])

            print('epoch {:d} | train f1 syn {:.4f} | train f1 sa {:.4f} | train f1 fsa {:.4f}'.format(
                epoch, train_acc_syn['f1'], train_acc_sa['f1'], train_acc_fsa['f1']))

            val_res = train.evaluate(model, stage1_val_loader, 'cuda:0')
            print('          val   f1 syn {:.4f} | val   f1 sa {:.4f} | val   f1 fsa {:.4f}'.format(
                val_res['syn']['metrics']['f1'], val_res['sa']['metrics']['f1'],
                val_res['fsa']['metrics']['f1']))

            # 检查提前终止条件
            ave_val_acc = (val_res['syn']['metrics']['f1'] + val_res['sa']['metrics']['f1'] * 2 +
                           val_res['fsa']['metrics']['f1']) / 4
            if early_stopping(1 - ave_val_acc, model):
                print("Training stopped early.")
                break

        early_stopping.load_best_model(model)

        print('initializing SA fine tuning...')
        early_stopping = PANDA.EarlyStopping(patience=5, verbose=True, delta=0.001, path='weights/test/Panda.pth')
        for epoch in range(args['epochs']):
            print('Folds: {}'.format(folds))
            FT_sa = train.run_a_fine_tuning_epoch(model, SA_loader, Epoch=epoch, TEpoch=args['epochs'], task='SA')
            print('epoch {:d} | sa f1 {:.4f} | sa acc {:.4f} | sa auc {:.4f}'.format(
                epoch, FT_sa['f1'], FT_sa['accuracy'], FT_sa['roc_auc']))
            FT_sa_val, sa_loss, sa_roc, sa_pr = train.run_a_val_fine_tuning_epoch(model, SA_val_loader, task='SA')
            print('val | sa loss {} | sa f1 {:.4f} | sa acc {:.4f} | sa auc {:.4f}'.format(
                sa_loss, FT_sa_val['f1'], FT_sa_val['accuracy'], FT_sa_val['roc_auc']))

            if early_stopping(1-FT_sa_val['f1'], model, FT_sa_val, sa_roc, sa_pr):
                print("SA fine tuning stopped.")
                break
        early_stopping.load_best_model(model)
        train.save_metrics_and_curves(metrics=early_stopping.metrix,
                                      roc_curves=early_stopping.roc,
                                      pr_curves=early_stopping.pr,
                                      output_dir='./cv/' + str(folds) + 'sa_test')

        print('initializing FSA fine tuning...')
        early_stopping = PANDA.EarlyStopping(patience=5, verbose=True, delta=0.001, path='weights/test/Panda.pth')
        for epoch in range(args['epochs']):
            print('Folds: {}'.format(folds))
            FT_fsa = train.run_a_fine_tuning_epoch(model, FSA_loader, Epoch=epoch, TEpoch=args['epochs'], task='FSA')
            print('epoch {:d} | fsa f1 {:.4f} | fsa acc {:.4f} | fsa auc {:.4f}'.format(
                epoch, FT_fsa['f1'], FT_fsa['accuracy'], FT_fsa['roc_auc']))
            FT_fsa_val, fsa_loss, fsa_roc, fsa_pr = train.run_a_val_fine_tuning_epoch(model, FSA_val_loader, task='FSA')
            print('val | fsa loss {} | fsa f1 {:.4f} | fsa acc {:.4f} | fsa auc {:.4f}'.format(
                fsa_loss, FT_fsa_val['f1'], FT_fsa_val['accuracy'], FT_fsa_val['roc_auc']))

            if early_stopping(1-FT_fsa_val['f1'], model, FT_fsa_val, fsa_roc, fsa_pr):
                print("FSA fine tuning stopped.")
                break
        early_stopping.load_best_model(model)
        train.save_metrics_and_curves(metrics=early_stopping.metrix,
                                      roc_curves=early_stopping.roc,
                                      pr_curves=early_stopping.pr,
                                      output_dir='./cv/' + str(folds) + 'fsa_test')

        print('initializing PV fine tuning...')
        early_stopping = PANDA.EarlyStopping(patience=5, verbose=True, delta=0.001, path='weights/test/Panda.pth')
        for epoch in range(args['epochs']):
            print('Folds: {}'.format(folds))
            FT_pv = train.run_a_fine_tuning_epoch(model, PV_loader, Epoch=epoch, TEpoch=args['epochs'], task='PV')
            print('epoch {:d} | pv f1 {:.4f} | pv acc {:.4f} | pv auc {:.4f}'.format(
                epoch, FT_pv['f1'], FT_pv['accuracy'], FT_pv['roc_auc']))
            FT_pv_val, pv_loss, pv_roc, pv_pr = train.run_a_val_fine_tuning_epoch(model, PV_val_loader, task='PV')
            print('val | pv loss {} | pv f1 {:.4f} | pv acc {:.4f} | pv auc {:.4f}'.format(
                pv_loss, FT_pv_val['f1'], FT_pv_val['accuracy'], FT_pv_val['roc_auc']))

            if early_stopping(1-FT_pv_val['f1'], model, FT_pv_val, pv_roc, pv_pr):
                print("PV fine tuning stopped.")
                break
        early_stopping.load_best_model(model)

        train.save_metrics_and_curves(metrics=early_stopping.metrix,
                                      roc_curves=early_stopping.roc,
                                      pr_curves=early_stopping.pr,
                                      output_dir='./cv/' + str(folds) + 'pv_test')

        torch.save(model.state_dict(), 'weights/cv/' + str(folds) + '.pth')

    # python CV.py  -b 64 -e 300
