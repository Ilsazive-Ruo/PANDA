import sys
import Functions
import pandas as pd
from rdkit import Chem
import joblib
import shutil
import xgboost
import os


def read_dir(path, rename):  # 读取路径下所有分子文件
    filename = os.listdir(path)
    mol_dict = {}

    for name in filename:
        print(name)
        if rename:
            name = name.lower()
        mol = Chem.MolFromMolFile(path + name)
        name = name.replace('.mol', '')

        mol_dict[name] = mol

    return mol_dict


mol_dict = read_dir('mol/all/', rename=False)
smi_dict = {}

for mol in mol_dict.keys():
    print('name:', mol_dict[mol])

    smi_dict[mol] = Chem.MolToSmiles(mol_dict[mol])

print(smi_dict)
smi_df = pd.DataFrame(smi_dict, index=['SMILES']).T
print(smi_df.head())
smi_df.to_csv('data/smi_dict.csv')
