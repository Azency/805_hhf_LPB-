import numpy as np 

def trapezoidal_rule(t, W):
    """
    使用梯形法则计算积分
    t : 时间数组
    W : 位置数组
    """
    if len(t) != len(W):
        raise Exception("Dim not mathces!")
    dt = np.diff(t)
    integral = np.sum(0.5 * dt * (W[:-1] + W[1:]))
    return integral


