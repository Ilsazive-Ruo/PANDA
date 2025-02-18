import torch
import os
import dgl
import sys
import shutil
from dgl.data.utils import save_info
from dgllife.utils import CanonicalBondFeaturizer, CanonicalAtomFeaturizer, MolToBigraph
from rdkit import Chem
import pandas as pd

# ========================
# 1. 初始化特征提取器
# ========================
atom_featurizer = CanonicalAtomFeaturizer()
bond_featurizer = CanonicalBondFeaturizer()


# ========================
# 2. 将SMILES转为DGL图并提取特征
# ========================
def mol2dgl(mole):
    """
    将SMILES字符串转为DGL图，并提取节点和边特征
    """
    mol_to_graph = MolToBigraph(node_featurizer=CanonicalAtomFeaturizer(),
                                edge_featurizer=CanonicalBondFeaturizer())
    g = mol_to_graph(mole)
    g = dgl.add_self_loop(g)

    return g


def path_check(exp):
    if os.path.exists('data/gra/' + exp):
        print('Path exists! Cover? (y/n)')
        if input().lower() == 'y':
            shutil.rmtree('data/gra/' + exp)
            os.mkdir('data/gra/' + exp)
        else:
            sys.exit()


def read_mol_dir(path, rename):  # 读取路径下所有分子文件
    filename = os.listdir(path)
    name_list = []
    mol_list = []
    for name in filename:
        if rename:
            name = name.lower()
        name_list.append(name.replace('.mol', ''))
        mol_list.append(Chem.MolFromMolFile(path + name))

    mol_dict = dict(zip(name_list, mol_list))
    return mol_dict, name_list


def read_smi_csv(file):
    smi_df = pd.read_csv(file)
    smi_dict = {}
    for index in smi_df.index:
        print('Graphing:', smi_df.loc[index, 'name'])
        mol = Chem.MolFromSmiles(smi_df.loc[index, 'smiles'])
        g = mol2dgl(mol)
        if g.number_of_edges() == 0:
            print("单原子图，不予考虑")
        else:
            smi_dict[smi_df.loc[index, 'name']] = [smi_df.loc[index, 'smiles'], g]
    g_df = pd.DataFrame.from_dict(smi_dict, orient='index', columns=['smiles', 'graph'])
    return g_df


# def load_data(mol_path, data_file):
#     mol_dict, mol_list = read_mol_dir(path=mol_path, rename=False)
#     g_dict = {}
#     for name in mol_list:
#         g_dict[name] = mol2dgl(mol_dict[name])
#     data_infos = pd.read_csv(data_file)
#     temp = []
#     for i in range(len(data_infos.index)):
#         drug_name = data_infos.iloc[i, 0]
#         ligand_name = data_infos.iloc[i, 1]
#         label = data_infos.iloc[i, 2]
#         g_drug = g_dict[drug_name]
#         g_ligand = g_dict[ligand_name]
#         temp.append([drug_name, g_drug, ligand_name,
#                      g_ligand, dgl.batch([g_drug, g_ligand]), label])
#
#     df = pd.DataFrame(temp,
#                       columns=['Drug', 'Drug Graph',
#                                'Ligand', 'Ligand Graph',
#                                'Graph', 'Label'])
#
#     return df


def save_data(path, data_name, df):
    graph_path = os.path.join(path, data_name + '_dgl_graph.bin')
    print(type(df['graph'].tolist()[0]))
    dgl.save_graphs(graph_path, df['graph'].tolist())
    info_path = os.path.join(path, data_name + '_info.pkl')
    save_info(info_path, df.index.to_list())
# ========================


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser('Read data to graph')
    parser.add_argument('-out', '--output_path', type=str, required=True,
                        help='path of data to save')
    parser.add_argument('-in', '--smi_file', type=str, required=True,
                        help='name of smi source file')
    parser.add_argument('-exp', '--experiment_name', type=str, required=True,
                        help='name of saved file')
    args = parser.parse_args().__dict__

    path_check(args['experiment_name'])

    smis = read_smi_csv(args['smi_file'])

    save_data(path=args['output_path'], data_name=args['experiment_name'], df=smis)

    print('data have been saved to ' + args['output_path'])
    # python graphing.py -out data/gra/SA -in data/smi_dict.csv -exp SA

