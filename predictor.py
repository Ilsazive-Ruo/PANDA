import pandas as pd
from rdkit import Chem
import numpy as np
import os
import dgl
import torch
import random
import sys
import shutil
import tqdm
import PANDA
from dgllife.utils import CanonicalBondFeaturizer, CanonicalAtomFeaturizer, MolToBigraph
from torch.utils.data import Dataset, DataLoader


class PredictionDataset(Dataset):
    def __init__(self, graphs_a, graphs_e, extra_feats):
        """
        - graphs: list of DGLGraphs
        - extra_feats: list of 4D float tensors or placeholder (0 vector)
        """
        self.graphs_a = graphs_a
        self.graphs_e = graphs_e
        self.extra_feats = extra_feats

    def __len__(self):
        return len(self.graphs_a)

    def __getitem__(self, idx):
        g_a = self.graphs_a[idx]
        g_e = self.graphs_e[idx]
        extra = self.extra_feats[idx]

        return {
            "graph_a": g_a,
            "graph_e": g_e,
            "extra_feat": extra
        }


def smi2graph(smi):
    mol = Chem.MolFromSmiles(smi)
    print(mol)
    if mol is not None:
        mol_to_graph = MolToBigraph(node_featurizer=CanonicalAtomFeaturizer(),
                                    edge_featurizer=CanonicalBondFeaturizer())
        g = mol_to_graph(mol)
        g = dgl.add_self_loop(g)

        return g

    else:
        raise ValueError(f"SMILES is None")


def read_input_file(file):
    df = pd.read_csv(file)
    print(df)
    APIs = df['API'].values.tolist()
    ga = [smi2graph(smi) for smi in df['API_smi'].values.tolist()]
    excs = df['excipient'].values.tolist()
    ge = [smi2graph(smi) for smi in df['excipient_smi'].values.tolist()]

    return APIs, ga, excs, ge


def random_search_mc(counts=1000):
    mc = set()

    while len(mc) < counts:  # Randomly initialize the candidate preparation condition list
        m = random.randint(1, 10)
        n = random.randint(1, 10)
        i = random.randint(1, 29)
        j = random.randint(i + 1, 30)
        mc.add((m, i, n, j))

    print(np.array(list(mc)))

    return np.array(list(mc))


def list2set(APIs, ga, excs, ge, task='all', num_mc=1000):
    PDSets = {}
    frames = {}
    if task == 'all' or task == 'FSA':

        for i in range(len(ga)):
            pairs = {}
            mc = random_search_mc(num_mc)
            pairs['ga'] = [ga[i]] * mc.shape[0]
            pairs['ge'] = [ge[i]] * mc.shape[0]
            PDSets[APIs[i] + '&' + excs[i]] = PredictionDataset(graphs_a=pairs['ga'],
                                                                graphs_e=pairs['ge'],
                                                                extra_feats=torch.tensor(mc))
            df = pd.DataFrame({
                'APIs': [APIs[i]] * mc.shape[0],
                'excipients': [excs[i]] * mc.shape[0],
            })

            mc_df = pd.DataFrame(mc, columns=['C_API', 'V_API', 'C_E', 'V_E'])
            mc_df['A/E'] = mc_df['C_API'] * mc_df['V_API'] / (mc_df['V_E'] * mc_df['C_E'])

            frames[APIs[i] + '&' + excs[i]] = pd.concat([df, mc_df], axis=1)

    elif task == 'SA' or task == 'PV':
        PDSets[task] = PredictionDataset(graphs_a=ga, graphs_e=ge, extra_feats=torch.zeros(len(ga), 4))
        frames[task] = pd.DataFrame({'APIs': APIs, 'excipients': excs})

    else:
        raise ValueError(f"task {task} not supported")

    return PDSets, frames


def collate_prediction(samples):
    graphs_a = [s["graph_a"] for s in samples]
    graphs_e = [s["graph_e"] for s in samples]
    batched_a = dgl.batch(graphs_a)
    batched_e = dgl.batch(graphs_e)

    extra_feats = torch.stack([s["extra_feat"] for s in samples])  # (B, 4)

    return {
        "graph_a": batched_a,
        "graph_e": batched_e,
        "extra_feat": extra_feats
    }


def prediction(model, dataset, sample_frame, path, set_name, task='all', device='cuda:0'):
    dataloader = DataLoader(dataset=dataset,
                            batch_size=128,
                            shuffle=False,
                            num_workers=12,
                            pin_memory=True,
                            collate_fn=collate_prediction,
                            persistent_workers=True)
    with torch.no_grad():
        model.eval()
        loop = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))

        y_pred = []
        for step, batch_data in loop:
            graph_a = batch_data['graph_a'].to(device)
            graph_e = batch_data['graph_e'].to(device)
            extra_feat = batch_data['extra_feat'].to(device)

            out1, out2, out3 = model(graph_a, graph_e, extra_feat)
            pv_prob = out1.softmax(dim=-1).detach().cpu()
            pv_pred = out1.argmax(dim=1).detach().cpu()
            pv_score = pv_prob[torch.arange(out1.size(0)), pv_pred]
            sa_prob = out2.softmax(dim=-1).detach().cpu()
            sa_pred = out2.argmax(dim=1).detach().cpu()
            sa_score = sa_prob[torch.arange(out2.size(0)), sa_pred]
            fsa_prob = out3.softmax(dim=-1).detach().cpu()
            fsa_pred = out3.argmax(dim=1).detach().cpu()
            fsa_score = fsa_prob[torch.arange(out3.size(0)), fsa_pred]

            if task == 'all' or task == 'FSA':
                y_pred.append([pv_pred, pv_score, sa_pred, sa_score, fsa_pred, fsa_score])

            elif task == 'SA' or task == 'PV':
                y_pred.append([sa_pred, sa_score, pv_pred, pv_score])

            else:
                raise ValueError(f"task {task} not supported")

            loop.set_description(f'val')

        transposed = list(zip(*y_pred))
        merged_tensors = [torch.cat(t, dim=0) for t in transposed]

        np_matrix = np.stack([t.numpy() for t in merged_tensors], axis=1)
        if task == 'all' or task == 'FSA':
            score_frame = pd.DataFrame(
                np_matrix, columns=['pv_label', 'pv_score', 'sa_label', 'sa_score', 'fsa_label', 'fsa_score'])

        elif task == 'SA' or task == 'PV':
            score_frame = pd.DataFrame(np_matrix, columns=['sa_label', 'sa_score', 'pv_label', 'pv_score'])

        df = pd.concat([sample_frame, score_frame], axis=1)
        df.to_csv(path + '/' + set_name + '.csv', index=False)
        print('writing ' + str(path + '/' + set_name + '.csv'))


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser('CV')
    parser.add_argument('-i', '--input', required=True,
                        help='input file path')
    parser.add_argument('-n', '--num_mc', type=int, default=1000,
                        help='number of MC to generate')
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='output file path')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='model path')
    parser.add_argument('-t', '--task', type=str, required=True,
                        help='task type')
    args = parser.parse_args().__dict__
    # python predictor.py -i data/Condition_predict.csv -p prediction/Conditions -m final/PANDA.pth -t FSA -n 1000

    if os.path.exists(args['path']):
        print('Path exists! Cover? (y/n)')
        if input().lower() == 'y':
            shutil.rmtree(args['path'])
        else:
            sys.exit()
    os.mkdir(args['path'])

    print('reading data...')
    APIs, ga, excs, ge = read_input_file(args['input'])
    datasets, frames = list2set(APIs, ga, excs, ge, task=args['task'], num_mc=args['num_mc'])

    print('initializing model...')
    node_feat_size = CanonicalAtomFeaturizer().feat_size('h')
    bond_feat_size = CanonicalBondFeaturizer().feat_size('e')
    encoder = PANDA.ARCHIT(in_features=node_feat_size,
                           edge_dim=bond_feat_size,
                           hidden_features=node_feat_size,
                           out_features=node_feat_size)
    model = PANDA.Panda(encoder=encoder,
                        node_size=node_feat_size,
                        hidden_size=1024)
    model.load_state_dict(torch.load(args['model']))
    model.to('cuda:0')
    print('starting prediction...')
    for key in datasets.keys():
        print(key)
        prediction(model, datasets[key], frames[key], args['path'], set_name=key, task=args['task'])
    print('done!')

