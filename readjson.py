import json
import ast
import re
from rdkit import Chem
from graphing import mol2dgl
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
from dgllife.utils import CanonicalBondFeaturizer, CanonicalAtomFeaturizer, SMILESToBigraph
from dgllife.utils import RandomSplitter
from dgllife.data import ESOL
from torch.utils.data import DataLoader


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


def collate_graph_pairs(data):
    # 从批次中提取图对
    g1_list, g2_list = zip(*data)  # zip 解压批次

    # 使用 DGL 的 batch 函数将多个图合并成一个图
    batched_g1 = dgl.batch(g1_list)  # 合并图 1
    batched_g2 = dgl.batch(g2_list)  # 合并图 2

    # 返回合并后的图对
    return batched_g1, batched_g2


if __name__ == '__main__':
    file_path = r'C:\Users\ilsazive\Desktop\foodb_2020_04_07_json\Compound.json'

    # 打开并加载 JSON 文件
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.readlines()  # 读取文件内容
        print(content[0])
        print(re.split('[":]', content[0]))


    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'


    gdict = load_gdata('data/gra/SYN/SYN_dgl_graph.bin', 'data/gra/SYN/SYN_info.pkl')
    print(len(gdict))


    mols = {}
    for line in content:
        line = json.loads(line)
        if line['cas_number'] != None and line['moldb_smiles'] != None:
            mols[line['name']] = [line['cas_number'], line['moldb_smiles'], line['klass']]

    # mols = pd.DataFrame.from_dict(mols, orient="index", columns=['cas_number', 'moldb_smiles', 'klass'])


    model = torch.load('models/SA/model.pth')
    model = model.to('cuda:0')
    model.eval()

    gp = []
    gpn = []

    random_integers = np.random.randint(0, 15001, size=1700)
    seed = 0
    for mol in mols:
        if seed in random_integers:
        # if seed == 1 or seed == 2:
            print(mol)
            mole = Chem.MolFromSmiles(mols[mol][1])
            try:
                ge = mol2dgl(mole)
                for ga in gdict:
                    if ge.number_of_nodes() > 1 and gdict[ga].number_of_nodes() > 1:
                        gp.append([gdict[ga], ge])
                        gpn.append([ga, mol])

            except AttributeError as e:
                print(mols[mol][1])
                continue
        seed += 1

    pred = []
    ds = module2.HSDataset(gp)
    HS_loader = DataLoader(dataset=ds, batch_size=1, num_workers=20, pin_memory=True,
                           collate_fn=collate_graph_pairs)
    loop = tqdm.tqdm(enumerate(HS_loader), total=len(HS_loader))
    for step, batch_data in loop:
        bg1, bg2 = batch_data
        bg1 = bg1.to(device)
        # print('bg1', bg1)
        bg2 = bg2.to(device)
        try:
            pre = model(bg1, bg2)
            pred.append(pre.cpu().detach()[0].tolist())
        except:
            continue

    # pred = np.array(pred)
    # gpn = np.array(gpn)

    labels = []
    f = 0
    for row in pred:

        max_value = max(row)  # 找到行的最大值
        max_index = row.index(max_value)  # 获取最大值的索引
        labels.append([gpn[f][0], gpn[f][1], max_index])  # 将索引添加到列表
        f += 1

    # labels = np.array(labels).reshape(-1, 1)
    # res = np.vstack([gpn, labels])
    df = pd.DataFrame(labels)
    df.to_csv('data/HS.csv', index=False)


