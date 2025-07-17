import pandas as pd
import numpy as np
from scipy.stats import ttest_ind


def fc_p_calc(df1, df2):
    df1 = df1.set_index('Name')
    df1 = df1[~df1.index.duplicated(keep='first')]
    df2 = df2.set_index('Name')
    df2 = df2[~df2.index.duplicated(keep='first')]
    FoldChange = []
    p_val = []
    exp_mean = []
    for gene in df1.index:
        expr1 = df1.loc[gene, :]
        expr2 = df2.loc[gene, :]
        stat, p_value = ttest_ind(expr1, expr2)
        if expr1.mean() < 0 and expr2.mean() < 0:
            FoldChange.append('NS')
        else:
            if expr1.mean() == 0:
                mean1 = 0.1
            else:
                mean1 = expr1.mean()
            if expr2.mean() == 0:
                mean2 = 0.1
            else:
                mean2 = expr2.mean()
            FoldChange.append(mean1 / mean2)
        if np.isnan(p_value):
            p_val.append(0)
        else:
            p_val.append(p_value)
        exp_mean.append(expr1.mean())
    return FoldChange, p_val, exp_mean, df1.index


exp_group = pd.read_csv(r'data/tpm1.csv')

con_group = pd.read_csv(r'data/tpm2.csv')

res = pd.DataFrame()

FC, p_val, mean_exp, name = fc_p_calc(exp_group, con_group)
res['FC'] = FC
res['p_value'] = p_val
res['mean_exp'] = mean_exp

res.set_index(name, inplace=True)
res.to_csv(r'data/tpm1 vs tpm2.csv')


