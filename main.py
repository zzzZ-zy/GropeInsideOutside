import math
# 关于gamma函数，scipy和math库中均可以调用，此处先选择调用math
import numpy as np
from scipy.stats import chi2
from scipy import stats

# 算例输入
data = np.array([[1, 50.3175, 50.1432, 50.4081],
                 [2, 49.5077, 50.2715, 49.4091],
                 [3, 49.3587, 49.1436, 49.5229],
                 [4, 49.5005, 50.4829, 49.7904],
                 [5, 47.9110, 47.9221, 48.6716],
                 [6, 49.8851, 50.1697, 49.8762],
                 [7, 50.4223, 50.7303, 49.9994],
                 [8, 49.7899, 49.8906, 49.5800],
                 [9, 50.4464, 51.6716, 51.1060],
                 [10, 51.4371, 50.7355, 50.7538],
                 [11, 50.2943, 49.7763, 49.7322],
                 [12, 48.6112, 48.0480, 48.5373],
                 [13, 49.3312, 48.9048, 48.6753],
                 [14, 49.4319, 50.1224, 49.2045],
                 [15, 50.3789, 49.7629, 50.5952],
                 [16, 48.6722, 49.6390, 49.1306],
                 [17, 49.5332, 49.5897, 50.0007],
                 [18, 49.4866, 49.5549, 50.4324],
                 [19, 48.6924, 49.8281, 49.8926],
                 [20, 50.8253, 50.6484, 50.8569],
                 [21, 50.3306, 49.4764, 49.5988],
                 [22, 49.9728, 50.2333, 50.7228],
                 [23, 50.5934, 49.9355, 49.3386],
                 [24, 49.5547, 49.6442, 49.7181],
                 [25, 51.3211, 51.9875, 51.7502]])
# d2,d3,d4数值表
d_table = np.array([[2, 1.128, 0.8525, 0.954],
                    [3, 1.693, 0.8884, 1.588],
                    [4, 2.059, 0.8798, 1.978],
                    [5, 2.326, 0.8641, 2.257],
                    [6, 2.534, 0.848, 2.472],
                    [7, 2.704, 0.8332, 2.645],
                    [8, 2.847, 0.8198, 2.791],
                    [9, 2.97, 0.8078, 2.915],
                    [10, 3.078, 0.7971, 3.024],
                    [11, 3.173, 0.7873, 3.121],
                    [12, 3.258, 0.7785, 3.207],
                    [13, 3.336, 0.7704, 3.285],
                    [14, 3.407, 0.763, 3.356],
                    [15, 3.472, 0.7562, 3.422],
                    [16, 3.532, 0.7499, 3.482],
                    [17, 3.588, 0.7441, 3.538],
                    [18, 3.64, 0.7386, 3.591],
                    [19, 3.689, 0.7335, 3.64],
                    [20, 3.735, 0.7287, 3.686],
                    [21, 3.778, 0.7242, 3.73],
                    [22, 3.819, 0.7199, 3.771],
                    [23, 3.858, 0.7159, 3.811],
                    [24, 3.895, 0.7121, 3.847],
                    [25, 3.931, 0.7084, 3.883],
                    [26, 3.964],
                    [27, 3.997],
                    [28, 4.027],
                    [29, 4.057],
                    [30, 4.086],
                    [31, 4.113],
                    [32, 4.139],
                    [33, 4.165],
                    [34, 4.189],
                    [35, 4.213],
                    [36, 4.236],
                    [37, 4.259],
                    [38, 4.28],
                    [39, 4.301],
                    [40, 4.322],
                    [41, 4.341],
                    [42, 4.361],
                    [43, 4.379],
                    [44, 4.398],
                    [45, 4.415],
                    [46, 4.433],
                    [47, 4.45],
                    [48, 4.466],
                    [49, 4.482],
                    [50, 4.498]], dtype=object)
# 全局变量
n = data.shape[0] * (data.shape[1] - 1)  # 总数据量
v = n - 1  # 自由度
x_bar = np.average(np.delete(data, 0, axis=1))  # 全过程均值
USL = 53
LSL = 47

# 已解决(需要模块化处理，此处先以第二种情况编写)
# ****问题1：如何理解下面这段话？****
# 若数据位于一列，且子组大小相同，输入表示子组大小的数字
# 若数据位于多列，则每行为一个子组，且子组大小应该相同

# ****问题2：子组是否可以个数不同？****


def c4(num):
    return math.sqrt(2 / (num - 1)) * math.gamma(num * 0.5) / math.gamma((num - 1) * 0.5)


# 计算子组内部标准差,先以合并标准差为例
# 函数输入data_origin为带有子组编号的原始数据
def sigma_within(data_origin):
    d = data_origin.shape[0] * (data_origin.shape[1] - 2)
    c4_d_plus_1 = c4(d + 1)
    data_pure = np.delete(data_origin, 0, axis=1)
    data_average = np.average(data_pure, axis=1)
    data_average = data_average.reshape((25, 1))
    s_p = math.sqrt(np.average(np.power(data_pure - data_average, 2)) / d)
    return s_p / c4_d_plus_1

# ****问题3：c_4_N应该怎么算？****
# 下面的计算方法是仿照上面的c4_d_plus_1,代换成了计算c4的方法


# 计算整体标准差
def sigma_overall(data_origin):
    data_pure = np.delete(data_origin, 0, axis=1)
    s = math.sqrt(np.sum(np.power(data_pure - np.average(data_pure), 2)) / (n - 1))
    c4_n = c4(n)
    return s / c4_n


# 计算子组间标准差
def sigma_between():
    pass


# 以下是整体能力度量
def pp(usl, lsl, sigmaoverall, toler=6):
    return (usl - lsl) / toler / sigmaoverall


# 置信区间上限，需要调用上面的pp函数的返回值，应该可以写成继承的形式，此处先直接调用结果

# ****问题4：文档中没写示例中显著性水平取值，测试取0.025基本对应示例结果

def pp_upper_and_lower(p_p, alpha):
    return [p_p * math.sqrt(chi2.ppf(alpha, n) / n), p_p * math.sqrt(chi2.isf(alpha, n) / n)]


# ****问题5：可能由于计算过程中四舍五入的原因，计算的结果小数点后两位会和示例结果有出入

def ppl(sigmaoverall, toler=6):
    return (x_bar - LSL) / (toler * 0.5) / sigmaoverall


def ppu(sigmaoverall, toler=6):
    return (USL - x_bar) / (toler * 0.5) / sigmaoverall


# ****问题6：文档中Ppk公式好像写错了****
def ppk(pp_l, pp_u):
    return min(pp_l, pp_u)


# ****问题7：此处文档中同样没写示例中显著性水平取值，测试取0.025基本对应示例结果,同样存在小数点后两位与结果有差别的情况
def ppk_upper_and_lower(pp_k, alpha, toler=6):
    return [pp_k + stats.norm.ppf(alpha) * math.sqrt(1 / (toler / 2) / (toler / 2) / n + pp_k * pp_k / 2 / v),
            pp_k + stats.norm.isf(alpha) * math.sqrt(1 / (toler / 2) / (toler / 2) / n + pp_k * pp_k / 2 / v)]


if __name__ == '__main__':
    Pp = pp(USL, LSL, sigma_overall(data))
    print(x_bar)
    print(pp_upper_and_lower(Pp, 0.025))
    print(sigma_overall(data))
    print(ppl(sigma_overall(data)))
    print(ppu(sigma_overall(data)))
    PPL = ppl((sigma_overall(data)))
    PPU = ppu((sigma_overall(data)))
    print("="*10)
    Ppk = ppk(PPU, PPL)
    print(ppk_upper_and_lower(Ppk, 0.025))
