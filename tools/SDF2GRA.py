from rdkit import Chem
from rdkit.Chem import QED
import pandas as pd
from rdkit.Chem import PandasTools
import torch
import os
import dgl
import sys
import shutil
from dgl.data.utils import save_info
from dgllife.utils import CanonicalBondFeaturizer, CanonicalAtomFeaturizer, MolToBigraph


atom_featurizer = CanonicalAtomFeaturizer()
bond_featurizer = CanonicalBondFeaturizer()


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


# 检查字段是否存在
def extract_fields_from_sdf(sdf_path, fields):
    records = []
    with open(sdf_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 每个分子以 $$$$ 结尾
    molecules = content.strip().split("$$$$")

    for mol_block in molecules:
        mol_data = {}
        for field in fields:
            marker = f">  <{field}>"
            if marker in mol_block:
                # 提取字段值
                after_marker = mol_block.split(marker)[1]
                value = after_marker.strip().splitlines()[0].strip()
                mol_data[field] = value
            else:
                mol_data[field] = None  # 缺失字段
        if any(mol_data.values()):  # 至少一个字段存在才记录
            records.append(mol_data)

    return records


def smi2g(smi_set):
    smi_dict = {}
    prop_dict = {}
    for index in smi_set.index:
        print('Graphing:', smi_set.loc[index, 'SMILES'])
        mol = Chem.MolFromSmiles(smi_set.loc[index, 'SMILES'])
        g = mol2dgl(mol)
        if g.number_of_edges() == 0:
            print("单原子图，不予考虑")
        else:
            smi_dict[smi_set.loc[index, 'SMILES']] = g
            prop_dict[smi_set.loc[index, 'SMILES']] = smi_set.iloc[index, 1:]
    gra_df = pd.DataFrame.from_dict(smi_dict, orient='index', columns=['graph'])
    prop_df = pd.DataFrame.from_dict(prop_dict, orient='index', columns=["QED", "logP", "SAS"])
    return gra_df, prop_df


def save_data(path, data_name, gra_df, prop_df):
    graph_path = os.path.join(path, data_name + '_dgl_graph.bin')
    print(type(gra_df['graph'].tolist()[0]))
    dgl.save_graphs(graph_path, gra_df['graph'].tolist())
    info_path = os.path.join(path, data_name + '_info.pkl')
    save_info(info_path, gra_df.index.to_list())
    prop_path = os.path.join(path, data_name + '_prop.csv')
    prop_df.to_csv(prop_path, index=True, index_label='SMILES')
# ========================


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser('Read data to graph')
    parser.add_argument('-out', '--output_path', type=str, required=True,
                        help='path of data to save')
    parser.add_argument('-in', '--sdf_file', type=str, required=True,
                        help='name of smi source file')
    parser.add_argument('-exp', '--experiment_name', type=str, required=True,
                        help='name of saved file')
    args = parser.parse_args().__dict__

    path_check(args['experiment_name'])
    smis = extract_fields_from_sdf(sdf_path=args['sdf_file'], fields=["SMILES", "QED", "logP", "SAS"])
    print("sdf file loaded")
    smis = pd.DataFrame(smis)
    g_df, p_df = smi2g(smis)

    save_data(path=args['output_path'], data_name=args['experiment_name'], gra_df=g_df, prop_df=p_df)

    print('data have been saved to ' + args['output_path'])
    # python SDF2GRA.py -out data/gra/preval -in data/zinc_250k_std_validation.sdf -exp preval

