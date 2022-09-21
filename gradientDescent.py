# coding:UTF-8
# @Author:Chrazqee_
import copy
import math
import numpy as np

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])


# 损失函数
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i]) ** 2
    total_cost = 1 / (2 * m) * cost
    return total_cost


def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    w = copy.deepcopy(w_in)
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        # 计算梯度
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # 更新成本函数的 w and b
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        # 保存每一次迭代的成本函数的值
        if i < 10000:
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])
        if i % math.ceil(num_iters / 10) == 0:  # 控制打印的数量
            # i: 5 => 显示为五位  0.2e => 使用科学记数法并保留两位小数
            print(f'Iteration{i: 5}: Cost {J_history[-1]: 0.2e}',
                  f'dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}',
                  f'w: {w: 0.3e}, b:{b: 0.5e}')
    return w, b, J_history, p_history  # return for calculating calculated data


# 初始化参数
w_init = 0
b_init = 0
# 梯度下降设置
iterations = 10000
tmp_alpha = 1.0e-2
# 运行梯度下降
w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, iterations,
                                                    compute_cost, compute_gradient)
# 8.4f => 保留 8 位数字，小数点后 4 位
print(f'(w, b) found by gradient descent: ({w_final: 8.4f},{b_final: 8.4f})')
print(f"1200 sqft house prediction {w_final * 1.2 + b_final: 0.10f} Thousand dollars")
