import numpy as np
import pandas as pd
from scipy.stats import entropy
import os


def compute_js_divergence_from_scatter(x1, y1, x2, y2, bins=50):
    """
    计算基于散点图的二维分布 JS 散度
    参数:
    x1, y1: 第一组散点数据
    x2, y2: 第二组散点数据
    bins: 直方图网格分辨率（默认50x50）
    返回:
    JS 散度值
    """
    # 确定二维直方图的范围（取两个分布的联合范围）
    range_x = (min(x1.min(), x2.min()), max(x1.max(), x2.max()))
    range_y = (min(y1.min(), y2.min()), max(y1.max(), y2.max()))

    # 计算二维直方图
    hist1, _, _ = np.histogram2d(x1, y1, bins=bins, range=[range_x, range_y])
    hist2, _, _ = np.histogram2d(x2, y2, bins=bins, range=[range_x, range_y])

    # 归一化直方图为概率分布
    p = hist1 / np.sum(hist1)
    q = hist2 / np.sum(hist2)

    # 避免零值对数运算问题，添加小偏移量 epsilon
    epsilon = 1e-10
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)
    m = 0.5 * (p + q)

    # 计算 JS 散度
    kl_p_m = np.sum(p * np.log(p / m))
    kl_q_m = np.sum(q * np.log(q / m))
    js_div = 0.5 * kl_p_m + 0.5 * kl_q_m

    return js_div


dir_path = 'data/js'
files = os.listdir(dir_path)
x, y = {}, {}
for file in files:
    data_set = pd.read_csv(dir_path + '/' + file)
    df_P = data_set[data_set['label'] == 1]
    x['P' + file] = df_P['Qa']
    y['P' + file] = df_P['Qb']
    df_N = data_set[data_set['label'] == 0]
    x['N' + file] = df_N['Qa']
    y['N' + file] = df_N['Qb']

# 计算 JS 散度
res = {}
for g1 in x.keys():
    for g2 in x.keys():
        js_result = compute_js_divergence_from_scatter(x[g1], y[g1], x[g2], y[g2])
        res[g1, g2] = js_result
        print(f"基于散点图的 JS 散度: {js_result:.4f}")

df_ToSave = pd.DataFrame(res, index=[0])
df_ToSave.to_csv('data/result_V.csv', index=False)
