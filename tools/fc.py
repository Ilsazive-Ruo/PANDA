import pandas as pd
import os


def gene_fc(gene_list, df1, df2):
    fc = {}
    for gene in gene_list:
        if gene in df1.index:
            fc[gene] = df1.loc[gene].sum()/df2.loc[gene].sum()
        else:
            print('gene %s not found in df1' % gene)

    return fc


path = r"data"
file_dict = {}

for files in os.listdir(r"data"):
    print(files)
    file_path = path + "/" + files
    df = pd.read_csv(file_path)
    df.set_index('Name', inplace=True)
    file_dict[files] = df

fc = {}
gene_list = ['Tnf', 'Cxcl2', 'Socs3', 'Icam1', 'Bcl3', 'Akt1',
             'Cx3cl1', 'Tnfrsf1a', 'Pik3r2', 'Cebpb', 'Nod2', 'Mmp14', 'Rela']
for i in file_dict.keys():
    fc[i] = gene_fc(gene_list, file_dict[i], file_dict['Model.csv'])
    print(fc[i])

res = pd.DataFrame(fc)
res.to_csv('fc.csv')