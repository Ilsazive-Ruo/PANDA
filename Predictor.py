import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit

# 示例数据：12个点（你可以替换成你自己的数据）
x = np.array([1, 2, 3, 10, 15, 19, 25, 32, 33, 36, 50, 75, 100, 200, 300, 400, 500, 700, 1000, 1400, 1650])
y = np.array([2341, 2204, 1602, 1353, 1196, 1171, 1019, 788, 759, 588, 527, 486, 473, 412, 358, 336, 200, 100, 50, 25, 0])  # 例如：y = sin(x)的数据点

# 创建三次样条插值
cs = CubicSpline(x, y)

# 生成 1650 个插值点
x_new = np.linspace(0, 1650, 1650)  # 1650个数据点
y_new = cs(x_new)  # 插值后的y值

res = np.column_stack((y_new, x_new))
df = pd.DataFrame(res)
df.to_csv('data/temp_pred.csv')