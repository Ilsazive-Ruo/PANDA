import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score, f1_score,
    average_precision_score, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
from dgllife.utils import CanonicalBondFeaturizer, CanonicalAtomFeaturizer
from dgl import load_graphs
from dgl.data.utils import load_info
from dgllife.utils import RandomSplitter
import pandas as pd
import tqdm
import numpy as np
import dgl
import os
import PANDA


class PandaDataset(Dataset):
    def __init__(self, graphs_a, graphs_e, labels_task1, labels_task2, labels_task3, extra_feats, task_ids):
        """
        - graphs: list of DGLGraphs
        - labels_task1, labels_task2, labels_task3: list of labels or placeholders
        - extra_feats: list of 4D float tensors or placeholder (0 vector)
        - task_ids: list of integers (0, 1, or 2) indicating the task type
        """
        self.graphs_a = graphs_a
        self.graphs_e = graphs_e
        self.labels_task1 = labels_task1
        self.labels_task2 = labels_task2
        self.labels_task3 = labels_task3
        self.extra_feats = extra_feats
        self.task_ids = task_ids

    def __len__(self):
        return len(self.graphs_a)

    def __getitem__(self, idx):
        g_a = self.graphs_a[idx]
        g_e = self.graphs_e[idx]

        y1 = torch.tensor(self.labels_task1[idx]).squeeze()
        y2 = torch.tensor(self.labels_task2[idx]).squeeze()
        y3 = torch.tensor(self.labels_task3[idx]).squeeze()
        m1 = torch.tensor(1.0 if self.task_ids[idx] == 0 else 0.0)
        m2 = torch.tensor(1.0 if self.task_ids[idx] == 1 else 0.0)
        m3 = torch.tensor(1.0 if self.task_ids[idx] == 2 else 0.0)

        extra = self.extra_feats[idx]

        return {
            "graph_a": g_a,
            "graph_e": g_e,
            "labels": {
                "task1": y1,
                "task2": y2,
                "task3": y3
            },
            "masks": {
                "task1": m1,
                "task2": m2,
                "task3": m3
            },
            "extra_feat": extra
        }


def collate_fn(samples):
    graphs_a = [s["graph_a"] for s in samples]
    graphs_e = [s["graph_e"] for s in samples]
    batched_a = dgl.batch(graphs_a)
    batched_e = dgl.batch(graphs_e)

    labels = {
        "task1": torch.stack([s["labels"]["task1"] for s in samples]),
        "task2": torch.stack([s["labels"]["task2"] for s in samples]),
        "task3": torch.stack([s["labels"]["task3"] for s in samples])
    }
    masks = {
        "task1": torch.stack([s["masks"]["task1"] for s in samples]),
        "task2": torch.stack([s["masks"]["task2"] for s in samples]),
        "task3": torch.stack([s["masks"]["task3"] for s in samples])
    }

    extra_feats = torch.stack([s["extra_feat"] for s in samples])

    return {
        "graph_a": batched_a,
        "graph_e": batched_e,
        "labels": labels,
        "masks": masks,
        "extra_feat": extra_feats
    }


def load_gdata(g_file, d_file):
    graphs = load_graphs(g_file)

    g_names = load_info(d_file)
    g_dicts = {}
    for i in range(len(g_names)):
        g_dicts[g_names[i]] = dgl.add_self_loop(graphs[0][i])

    return g_dicts


def load_syn_sa_dataset(file, g_dict):
    print('loading datas...')
    dataset_file = pd.read_csv(file)
    print(dataset_file.head())
    graphs1, graphs2, labels = [], [], []
    for index in dataset_file.index:
        if dataset_file['name-1'][index] in g_dict.keys() and dataset_file['name-2'][index] in g_dict.keys():
            graphs1.append(g_dict[dataset_file['name-1'][index]])
            graphs2.append(g_dict[dataset_file['name-2'][index]])
            labels.append(dataset_file['class'][index])
    print('total samples:', len(labels))
    labels = np.array(labels).reshape(-1, 1)

    return graphs1, graphs2, labels


def load_fsa_dataset(file, g_dict):
    print('loading datas...')
    dataset_file = pd.read_csv(file)
    print(dataset_file.head())
    graphs1, graphs2, labels, mps = [], [], [], []
    for index in dataset_file.index:
        if dataset_file['name-1'][index] in g_dict.keys() and dataset_file['name-2'][index] in g_dict.keys():
            graphs1.append(g_dict[dataset_file['name-1'][index]])
            graphs2.append(g_dict[dataset_file['name-2'][index]])
            mps.append([dataset_file['Ca'][index], dataset_file['Qa'][index],
                        dataset_file['Cb'][index], dataset_file['Qb'][index]])
            labels.append(dataset_file['class'][index])
    print('total samples:', len(labels))
    labels = np.array(labels).reshape(-1, 1)

    return graphs1, graphs2, labels, mps


def freeze_eval(module):
    for p in module.parameters():
        p.requires_grad = False
    module.eval()


def activate_train(module):
    for p in module.parameters():
        p.requires_grad = True
    module.train()


def compute_loss(outputs, labels, masks, weights):
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    loss1 = loss_fn(outputs[0], labels[0].long()) * masks[0]
    loss2 = loss_fn(outputs[1], labels[1].long()) * masks[1]
    loss3 = loss_fn(outputs[2], labels[2].long()) * masks[2]

    loss = (
        weights[0] * loss1.sum() / masks[0].sum() +
        weights[1] * loss2.sum() / masks[1].sum() +
        weights[2] * loss3.sum() / masks[2].sum()
    )
    return loss


def compute_metrics(y_true, y_pred, task_type='binary', num_classes=4):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    results = {}
    roc_curves = {}
    pr_curves = {}

    if task_type == 'binary':
        pred_label = (y_pred > 0.5).astype(int)
        results['accuracy'] = accuracy_score(y_true, pred_label)
        results['precision'] = precision_score(y_true, pred_label, zero_division=0)
        results['recall'] = recall_score(y_true, pred_label, zero_division=0)
        results['roc_auc'] = roc_auc_score(y_true, y_pred)
        results['pr_auc'] = average_precision_score(y_true, y_pred)
        results['f1'] = f1_score(y_true, pred_label, zero_division=0)

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        roc_curves['binary'] = (fpr, tpr)
        pr_curves['binary'] = (recall, precision)

    elif task_type == 'multiclass':
        assert num_classes is not None, "num_classes must be specified for multiclass"

        pred_label = np.argmax(y_pred, axis=1)
        results['accuracy'] = accuracy_score(y_true, pred_label)
        results['precision'] = precision_score(y_true, pred_label, average='macro', zero_division=0)
        results['f1'] = f1_score(y_true, pred_label, average='macro', zero_division=0)
        results['recall'] = recall_score(y_true, pred_label, average='macro', zero_division=0)

        # One-hot coding
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

        # multi-class ROC-AUC & PR-AUC（macro）
        try:
            results['roc_auc'] = roc_auc_score(y_true_bin, y_pred, average='macro', multi_class='ovr')
        except ValueError:
            results['roc_auc'] = 0

        try:
            results['pr_auc'] = average_precision_score(y_true_bin, y_pred, average='macro')
        except ValueError:
            results['pr_auc_ovr'] = 0

        # ROC/PR curves
        for i in range(num_classes):
            try:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
                roc_curves[i] = (fpr, tpr)

                prec, rec, _ = precision_recall_curve(y_true_bin[:, i], y_pred[:, i])
                pr_curves[i] = (rec, prec)
            except ValueError:
                roc_curves[i] = ([], [])
                pr_curves[i] = ([], [])

        report = classification_report(y_true, pred_label, output_dict=True)
        for k, v in report.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    results[f"{k}_{sub_k}"] = sub_v
            else:
                results[k] = v

    return results, roc_curves, pr_curves


def save_metrics_and_curves(
    metrics: dict,
    roc_curves: dict,
    pr_curves: dict,
    output_dir: str = "./eval_results"
):
    os.makedirs(output_dir, exist_ok=True)

    # save metrics
    metrics_path = os.path.join(output_dir, "metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    # save ROC
    if roc_curves:
        for class_id, (fpr, tpr) in roc_curves.items():
            roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
            roc_df.to_csv(os.path.join(output_dir, f"roc_curve_class_{class_id}.csv"), index=False)

    # save PR
    if pr_curves:
        for class_id, (recall, precision) in pr_curves.items():
            pr_df = pd.DataFrame({'recall': recall, 'precision': precision})
            pr_df.to_csv(os.path.join(output_dir, f"pr_curve_class_{class_id}.csv"), index=False)

    print(f"Evaluation results saved to '{output_dir}'")


def evaluate(model, dataloader, device):
    model.eval()
    task1_y, task1_pred = [], []
    task2_y, task2_pred = [], []
    task3_y, task3_pred = [], []

    with torch.no_grad():
        for batch in dataloader:
            graph_a = batch['graph_a'].to(device)
            graph_e = batch['graph_e'].to(device)
            extra_feat = batch['extra_feat'].to(device)
            y1 = batch['labels']['task1'].to(device)
            y2 = batch['labels']['task2'].to(device)
            y3 = batch['labels']['task3'].to(device)
            mask1 = batch['masks']['task1'].bool().to(device)
            mask2 = batch['masks']['task2'].bool().to(device)
            mask3 = batch['masks']['task3'].bool().to(device)

            out1, out2, out3 = model(graph_a, graph_e, extra_feat)

            task1_y += y1[mask1].tolist()
            task1_pred += out1[mask1].softmax(dim=-1).cpu().tolist()

            task2_y += y2[mask2].tolist()
            task2_pred += torch.argmax(out2[mask2.bool()], dim=1).cpu().tolist()

            task3_y += y3[mask3].tolist()
            task3_pred += torch.argmax(out3[mask3.bool()], dim=1).cpu().tolist()

    metrics1, roc1, pr1 = compute_metrics(task1_y, task1_pred, task_type='multiclass')
    metrics2, roc2, pr2 = compute_metrics(task2_y, task2_pred, task_type='binary')
    metrics3, roc3, pr3 = compute_metrics(task3_y, task3_pred, task_type='binary')

    return {'syn': {'metrics': metrics1, 'roc': roc1, 'pr': pr1},
            'sa': {'metrics': metrics2, 'roc': roc2, 'pr': pr2},
            'fsa': {'metrics': metrics3, 'roc': roc3, 'pr': pr3}}


class PCGrad:
    def __init__(self, optimizer):
        self._optim = optimizer

    def zero_grad(self):
        self._optim.zero_grad()

    def step(self):
        self._optim.step()

    def pc_backward(self, objectives):

        grads = []
        for obj in objectives:
            self._optim.zero_grad()
            obj.backward(retain_graph=True)
            single_grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in self._optim.param_groups[0]['params']]
            grads.append(single_grads)

        final_grads = grads[0]
        for i in range(1, len(grads)):
            for j, (g, f) in enumerate(zip(grads[i], final_grads)):
                dot = torch.dot(f.flatten(), g.flatten())
                if dot < 0:
                    g = g - (dot / (g.norm() ** 2 + 1e-6)) * g
                final_grads[j] += g

        for p, g in zip(self._optim.param_groups[0]['params'], final_grads):
            p.grad = g


def run_a_train_epoch(model, data_loader, pcgrad, Epoch, TEpoch, device='cuda:0'):
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='none')

    running_loss = 0.0
    loop = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))

    task1_y, task1_pred = [], []
    task2_y, task2_pred = [], []
    task3_y, task3_pred = [], []
    for step, batch_data in loop:
        graph_a = batch_data['graph_a'].to(device)
        graph_e = batch_data['graph_e'].to(device)
        extra_feat = batch_data['extra_feat'].to(device)
        y1 = batch_data['labels']['task1'].to(device)
        y2 = batch_data['labels']['task2'].to(device)
        y3 = batch_data['labels']['task3'].to(device)
        mask1 = batch_data['masks']['task1'].to(device)
        mask2 = batch_data['masks']['task2'].to(device)
        mask3 = batch_data['masks']['task3'].to(device)

        out1, out2, out3 = model(graph_a, graph_e, extra_feat)

        if mask1.sum() > 0:
            loss1 = criterion(out1, y1)
            loss1 = loss1[mask1.bool()].mean() if mask1.any() else 0
        else:
            loss1 = torch.tensor(0.0, requires_grad=True).to(device)

        if mask2.sum() > 0:
            loss2 = criterion(out2, y2)
            loss2 = loss2[mask2.bool()].mean() if mask2.any() else 0
        else:
            loss2 = torch.tensor(0.0, requires_grad=True).to(device)

        if mask3.sum() > 0:
            loss3 = criterion(out3, y3)
            loss3 = loss3[mask3.bool()].mean() if mask3.any() else 0
        else:
            loss3 = torch.tensor(0.0, requires_grad=True).to(device)

        loss = loss1 + loss2 + loss3

        task1_y.append(y1[mask1.bool()].detach().cpu())
        task1_pred.append(out1[mask1.bool()].softmax(dim=-1).detach().cpu())

        task2_y.append(y2[mask2.bool()].detach().cpu())
        task2_pred.append(torch.argmax(out2[mask2.bool()], dim=1).detach().cpu())

        task3_y.append(y3[mask3.bool()].detach().cpu())
        task3_pred.append(torch.argmax(out3[mask3.bool()], dim=1).detach().cpu())

        pcgrad.zero_grad()
        pcgrad.pc_backward([loss1, loss2, loss3])
        pcgrad.step()

        running_loss += loss.item()

        loop.set_description(f'Epoch [{Epoch}/{TEpoch}]')
        loop.set_postfix(loss=running_loss / (step + 1))

    task1_pred = torch.cat(task1_pred, dim=0)
    task1_y = torch.cat(task1_y, dim=0)
    task2_pred = torch.cat(task2_pred, dim=0)
    task2_y = torch.cat(task2_y, dim=0)
    task3_pred = torch.cat(task3_pred, dim=0)
    task3_y = torch.cat(task3_y, dim=0)

    metrics1, _, _ = compute_metrics(task1_y.numpy(), task1_pred.numpy(), task_type='multiclass')
    metrics2, _, _ = compute_metrics(task2_y.numpy(), task2_pred.numpy(), task_type='binary')
    metrics3, _, _ = compute_metrics(task3_y.numpy(), task3_pred.numpy(), task_type='binary')

    return metrics1, metrics2, metrics3


def run_a_fine_tuning_epoch(model, data_loader, Epoch, TEpoch, task='PV', stage='FT', device='cuda:0'):
    criterion = nn.CrossEntropyLoss(reduction='mean')

    running_loss = 0.0
    loop = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))

    freeze_eval(model.predictor_syn)
    freeze_eval(model.fusion_layer_syn)
    freeze_eval(model.predictor_sa)
    freeze_eval(model.fusion_layer_sa)
    freeze_eval(model.predictor_fsa)
    freeze_eval(model.fusion_layer_fsa)
    freeze_eval(model.archit1)
    freeze_eval(model.archit2)

    if task == "SA":
        activate_train(model.archit1)
        activate_train(model.archit2)
        activate_train(model.fusion_layer_sa)
        activate_train(model.predictor_sa)

    elif task == "FSA":
        activate_train(model.fusion_layer_fsa)
        activate_train(model.predictor_fsa)

    elif task == "PV" and stage == "FT":
        activate_train(model.fusion_layer_syn)
        activate_train(model.predictor_syn)

    elif task == "PV" and stage == "WU":
        activate_train(model.archit1)
        activate_train(model.archit2)
        activate_train(model.fusion_layer_syn)
        activate_train(model.predictor_syn)

    else:
        raise ValueError(f"Unknown mode: {task}")

    adam = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
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

        adam.zero_grad()
        loss.backward()
        adam.step()

        running_loss += loss.item()

        loop.set_description(f'Epoch [{Epoch}/{TEpoch}]')
        loop.set_postfix(loss=running_loss / (step + 1))

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    metrics, _, _ = compute_metrics(y_true.numpy(), y_pred.numpy(), task_type=task_type)

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
        metrics, roc, pr = compute_metrics(y_true.numpy(), y_pred.numpy(), task_type=task_type)

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
    g_syn = load_gdata('data/gra/SYN/SYN_dgl_graph.bin',
                       "data/gra/SYN/SYN_info.pkl")
    syn_gra_a, syn_gra_e, syn_labels = load_syn_sa_dataset('data/syn_dataset.csv', g_syn)

    print('loading sa data')
    g_sa = load_gdata('data/gra/SA/SA_dgl_graph.bin',
                      "data/gra/SA/SA_info.pkl")
    sa_gra_a, sa_gra_e, sa_labels = load_syn_sa_dataset('data/sa_dataset.csv', g_sa)

    print('loading fsa data')
    g_fsa = load_gdata('data/gra/SA/SA_dgl_graph.bin',
                       "data/gra/SA/SA_info.pkl")
    fsa_gra_a, fsa_gra_e, fsa_labels, mcs = load_fsa_dataset('data/fsa_dataset.csv', g_fsa)

    syn_set = PandaDataset(graphs_a=syn_gra_a,
                           graphs_e=syn_gra_e,
                           labels_task1=syn_labels,
                           labels_task2=[0] * len(syn_gra_a),
                           labels_task3=[0] * len(syn_gra_a),
                           extra_feats=torch.zeros(len(syn_gra_a), 4),
                           task_ids=[0] * len(syn_gra_a))
    sa_set = PandaDataset(graphs_a=sa_gra_a,
                          graphs_e=sa_gra_e,
                          labels_task1=[0] * len(sa_gra_a),
                          labels_task2=sa_labels,
                          labels_task3=[0] * len(sa_gra_a),
                          extra_feats=torch.zeros(len(sa_gra_a), 4),
                          task_ids=[1] * len(sa_gra_a))
    fsa_set = PandaDataset(graphs_a=fsa_gra_a,
                           graphs_e=fsa_gra_e,
                           labels_task1=[0] * len(fsa_gra_a),
                           labels_task2=[0] * len(fsa_gra_a),
                           labels_task3=fsa_labels,
                           extra_feats=torch.tensor(mcs),
                           task_ids=[2] * len(fsa_gra_a))

    train_sa, val_sa, test_sa = RandomSplitter.train_val_test_split(
        sa_set, frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=42)
    train_fsa, val_fsa, test_fsa = RandomSplitter.train_val_test_split(
        fsa_set, frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=42)
    train_syn, val_syn, test_syn = RandomSplitter.train_val_test_split(
        syn_set, frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=42)

    ######

    train_sa = ConcatDataset([train_sa] * 15)
    train_fsa = ConcatDataset([train_fsa] * 50)

    train_set = ConcatDataset([train_syn, train_sa, train_fsa])
    val_set = ConcatDataset([val_syn, val_fsa, val_sa])

    stage1_loader = DataLoader(dataset=train_set,
                               batch_size=128,
                               shuffle=True,
                               num_workers=12,
                               pin_memory=True,
                               collate_fn=collate_fn,
                               persistent_workers=True,
                               drop_last=True)
    stage1_val_loader = DataLoader(dataset=val_set,
                                   batch_size=128,
                                   shuffle=True,
                                   num_workers=12,
                                   pin_memory=True,
                                   collate_fn=collate_fn,
                                   persistent_workers=True)

    SA_loader = DataLoader(dataset=train_sa,
                           batch_size=args['batch_size'],
                           shuffle=True,
                           num_workers=12,
                           pin_memory=True,
                           collate_fn=collate_fn,
                           persistent_workers=True)
    SA_val_loader = DataLoader(dataset=val_sa,
                               batch_size=args['batch_size'],
                               shuffle=True,
                               num_workers=12,
                               pin_memory=True,
                               collate_fn=collate_fn,
                               persistent_workers=True)
    SA_test_loader = DataLoader(dataset=test_sa,
                                batch_size=args['batch_size'],
                                shuffle=True,
                                num_workers=12,
                                pin_memory=True,
                                collate_fn=collate_fn,
                                persistent_workers=True)

    FSA_loader = DataLoader(dataset=train_fsa,
                            batch_size=args['batch_size'],
                            shuffle=True,
                            num_workers=12,
                            pin_memory=True,
                            collate_fn=collate_fn,
                            persistent_workers=True)
    FSA_val_loader = DataLoader(dataset=val_fsa,
                                batch_size=args['batch_size'],
                                shuffle=True,
                                num_workers=12,
                                pin_memory=True,
                                collate_fn=collate_fn,
                                persistent_workers=True)
    FSA_test_loader = DataLoader(dataset=test_fsa,
                                 batch_size=args['batch_size'],
                                 shuffle=True,
                                 num_workers=12,
                                 pin_memory=True,
                                 collate_fn=collate_fn,
                                 persistent_workers=True)

    PV_loader = DataLoader(dataset=train_syn,
                           batch_size=args['batch_size'],
                           shuffle=True,
                           num_workers=12,
                           pin_memory=True,
                           collate_fn=collate_fn,
                           persistent_workers=True)
    PV_val_loader = DataLoader(dataset=val_syn,
                               batch_size=args['batch_size'],
                               shuffle=True,
                               num_workers=12,
                               pin_memory=True,
                               collate_fn=collate_fn,
                               persistent_workers=True)
    PV_test_loader = DataLoader(dataset=test_syn,
                                batch_size=args['batch_size'],
                                shuffle=True,
                                num_workers=12,
                                pin_memory=True,
                                collate_fn=collate_fn,
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
        FT_pv = run_a_fine_tuning_epoch(model, PV_loader,
                                        Epoch=epoch, TEpoch=args['epochs'], task='PV', stage='WU')
        print('epoch {:d} | pv f1 {:.4f} | pv acc {:.4f} | pv auc {:.4f}'.format(
            epoch, FT_pv['f1'], FT_pv['accuracy'], FT_pv['roc_auc']))
        FT_pv_val, pv_loss, _, _ = run_a_val_fine_tuning_epoch(model, PV_val_loader, task='PV')
        print('val | pv loss {} | pv f1 {:.4f} | pv acc {:.4f} | pv auc {:.4f}'.format(
            pv_loss, FT_pv_val['f1'], FT_pv_val['accuracy'], FT_pv_val['roc_auc']))

        if early_stopping(pv_loss, model):
            print("warmup stopped.")
            break
    early_stopping.load_best_model(model)

    early_stopping = PANDA.EarlyStopping(patience=5, verbose=True, delta=0.005, path='weights/test/Panda.pth')
    pcgrad = PCGrad(torch.optim.Adam(model.parameters(), lr=0.001))
    for epoch in range(50):
        train_acc_syn, train_acc_sa, train_acc_fsa = run_a_train_epoch(model, stage1_loader, pcgrad,
                                                                             Epoch=epoch, TEpoch=args['epochs'])

        print('epoch {:d} | train f1 syn {:.4f} | train f1 sa {:.4f} | train f1 fsa {:.4f}'.format(
            epoch, train_acc_syn['f1'], train_acc_sa['f1'], train_acc_fsa['f1']))

        val_res = evaluate(model, stage1_val_loader, 'cuda:0')
        print('          val   f1 syn {:.4f} | val   f1 sa {:.4f} | val   f1 fsa {:.4f}'.format(
            val_res['syn']['metrics']['f1'], val_res['sa']['metrics']['f1'],
            val_res['fsa']['metrics']['f1']))

        ave_val_acc = (val_res['syn']['metrics']['f1'] + val_res['sa']['metrics']['f1'] * 2 +
                       val_res['fsa']['metrics']['f1']) / 4
        if early_stopping(1 - ave_val_acc, model):
            print("Training stopped early.")
            break

    early_stopping.load_best_model(model)

    print('initializing SA fine tuning...')
    early_stopping = PANDA.EarlyStopping(patience=5, verbose=True, delta=0.001, path='weights/test/Panda.pth')
    for epoch in range(args['epochs']):
        FT_sa = run_a_fine_tuning_epoch(model, SA_loader, Epoch=epoch, TEpoch=args['epochs'], task='SA')
        print('epoch {:d} | sa f1 {:.4f} | sa acc {:.4f} | sa auc {:.4f}'.format(
            epoch, FT_sa['f1'], FT_sa['accuracy'], FT_sa['roc_auc']))
        FT_sa_val, sa_loss, sa_roc, sa_pr = run_a_val_fine_tuning_epoch(model, SA_val_loader, task='SA')
        print('val | sa loss {} | sa f1 {:.4f} | sa acc {:.4f} | sa auc {:.4f}'.format(
            sa_loss, FT_sa_val['f1'], FT_sa_val['accuracy'], FT_sa_val['roc_auc']))
        test_sa_metrix, test_sa_loss, test_sa_roc, test_sa_pr = run_a_val_fine_tuning_epoch(
            model, SA_test_loader, task='SA')

        if early_stopping(1-(FT_sa_val['f1'] + test_sa_metrix['f1'])/2, model, FT_sa_val, sa_roc, sa_pr):
            print("SA fine tuning stopped.")
            break
    early_stopping.load_best_model(model)
    test_sa_metrix, test_sa_loss, test_sa_roc, test_sa_pr = run_a_val_fine_tuning_epoch(
        model, SA_test_loader, task='SA')
    save_metrics_and_curves(metrics=test_sa_metrix,
                            roc_curves=test_sa_roc,
                            pr_curves=test_sa_pr,
                            output_dir='./final/' + 'sa_test')
    save_metrics_and_curves(metrics=early_stopping.metrix,
                            roc_curves=early_stopping.roc,
                            pr_curves=early_stopping.pr,
                            output_dir='./final/' + 'sa_val')

    print('initializing FSA fine tuning...')
    early_stopping = PANDA.EarlyStopping(patience=5, verbose=True, delta=0.001, path='weights/test/Panda.pth')

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    for epoch in range(args['epochs']):
        FT_fsa = run_a_fine_tuning_epoch(model, FSA_loader, Epoch=epoch, TEpoch=args['epochs'], task='FSA')
        print('epoch {:d} | fsa f1 {:.4f} | fsa acc {:.4f} | fsa auc {:.4f}'.format(
            epoch, FT_fsa['f1'], FT_fsa['accuracy'], FT_fsa['roc_auc']))
        FT_fsa_val, fsa_loss, fsa_roc, fsa_pr = run_a_val_fine_tuning_epoch(model, FSA_val_loader, task='FSA')
        print('val | fsa loss {} | fsa f1 {:.4f} | fsa acc {:.4f} | fsa auc {:.4f}'.format(
            fsa_loss, FT_fsa_val['f1'], FT_fsa_val['accuracy'], FT_fsa_val['roc_auc']))

        if early_stopping(1-FT_fsa_val['f1'], model, FT_fsa_val, fsa_roc, fsa_pr):
            print("FSA fine tuning stopped.")
            break
    early_stopping.load_best_model(model)
    test_fsa_metrix, test_fsa_loss, test_fsa_roc, test_fsa_pr = run_a_val_fine_tuning_epoch(
        model, FSA_test_loader, task='FSA')
    save_metrics_and_curves(metrics=test_fsa_metrix,
                            roc_curves=test_fsa_roc,
                            pr_curves=test_fsa_pr,
                            output_dir='./final/' + 'fsa_test')
    save_metrics_and_curves(metrics=early_stopping.metrix,
                            roc_curves=early_stopping.roc,
                            pr_curves=early_stopping.pr,
                            output_dir='./final/' + 'fsa_val')

    print('initializing PV fine tuning...')
    early_stopping = PANDA.EarlyStopping(patience=5, verbose=True, delta=0.001, path='weights/test/Panda.pth')

    for epoch in range(args['epochs']):
        FT_pv = run_a_fine_tuning_epoch(model, PV_loader, Epoch=epoch, TEpoch=args['epochs'], task='PV')
        print('epoch {:d} | pv f1 {:.4f} | pv acc {:.4f} | pv auc {:.4f}'.format(
            epoch, FT_pv['f1'], FT_pv['accuracy'], FT_pv['roc_auc']))
        FT_pv_val, pv_loss, pv_roc, pv_pr = run_a_val_fine_tuning_epoch(model, PV_val_loader, task='PV')
        print('val | pv loss {} | pv f1 {:.4f} | pv acc {:.4f} | pv auc {:.4f}'.format(
            pv_loss, FT_pv_val['f1'], FT_pv_val['accuracy'], FT_pv_val['roc_auc']))

        if early_stopping(1-FT_pv_val['f1'], model, FT_pv_val, pv_roc, pv_pr):
            print("PV fine tuning stopped.")
            break
    early_stopping.load_best_model(model)
    test_pv_metrix, test_pv_loss, test_pv_roc, test_pv_pr = run_a_val_fine_tuning_epoch(
        model, PV_test_loader, task='PV')
    save_metrics_and_curves(metrics=test_pv_metrix,
                            roc_curves=test_pv_roc,
                            pr_curves=test_pv_pr,
                            output_dir='./final/' + 'pv_test')

    save_metrics_and_curves(metrics=early_stopping.metrix,
                            roc_curves=early_stopping.roc,
                            pr_curves=early_stopping.pr,
                            output_dir='./final/' + 'pv_val')

    torch.save(model.state_dict(), 'final/' + 'PANDA.pth')

    # python train.py  -b 64 -e 300
