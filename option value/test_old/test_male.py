ax_path = "./data/lc_ax_male.csv"
bx_path = "./data/lc_bx_male.csv"
kt_path = "./data/lc_kt_male.csv"
path = [ax_path, bx_path, kt_path]

import Prob 
death_Prob = Prob.D_Prob()
death_Prob.read_abk(path)

import ST
import numpy as np
from statistics import mean

np.random.seed(8)  # 设置固定的随机数种子

import time
import pandas as pd
import os
import multiprocessing



def load_data(x0, p, lifespan, T0):
    terminal_live_cache_file = f'./data/terminal_live_cache_male_{x0}_{p}.csv'
    interval_P_cache_file = f'./data/interval_P_cache_male_{x0}_{p}.csv'

    # 尝试从CSV文件读取数据
    if os.path.exists(terminal_live_cache_file):
        terminal_live_cache = pd.read_csv(terminal_live_cache_file, index_col=0).to_dict()['Value']
    else:
        terminal_live_cache = {}

    if os.path.exists(interval_P_cache_file):
        interval_P_cache_df = pd.read_csv(interval_P_cache_file)
        interval_P_cache = {(row['Start'], row['End']): row['Value'] for index, row in interval_P_cache_df.iterrows()}
    else:
        interval_P_cache = {}

    for i in range(0, (lifespan - x0) * p):
        if i == 0:
            terminal_live_cache[i] = death_Prob.accu_live(x0, T0, lifespan - x0, 0)
        else:
            terminal_live_cache[i] = death_Prob.interval_terminal_live(x0, T0, i, lifespan - x0, p)
        for j in range(i + 1, (lifespan - x0) * p + 1):
            if i == 0:
                interval_P_cache[(i, j)] = death_Prob.unit_P(x0, T0, j, p)
            else:
                interval_P_cache[(i, j)] = death_Prob.interval_death_P(x0, T0, i, j, p)

    pd.DataFrame.from_dict(terminal_live_cache, orient='index', columns=['Value']).to_csv(terminal_live_cache_file)
    interval_P_cache_df = pd.DataFrame([(key[0], key[1], value) for key, value in interval_P_cache.items()], columns=['Start', 'End', 'Value'])
    interval_P_cache_df.to_csv(interval_P_cache_file, index=False)
    return interval_P_cache, terminal_live_cache

import concurrent.futures
def MC(M, k, p, buff_day, S0, r, g, x0, T0, lifespan, l, interval_P_cache, terminal_live_cache):
    T = (lifespan - x0) * p - k
    print('T', T)
    N = buff_day * T
    print('N', N)
    sigma = 0.2

    # 你现有的MC函数代码继续
    result = []
    avg_res = []
    ave = []

    t_all, int_item = ST.intg_item(N, p, T, S0, r, l, g, sigma)
    def parallely_mc():
        t, int_item_temp = ST.intg_item(N, p, T, S0, r, l, g, sigma)
        # print("期权序列", int_item)
        # print("时间节点", t_all)
        # print("最后一个时间节点", t_all[-1])

        intern_res = 0  # 内部的离散积分值，这里以年为单位

        for w in range(T):

            intern_res += int_item_temp[int((w + 1) * buff_day) - 1] * interval_P_cache[(k, w+k+1)]

        intern_res = intern_res + int_item_temp[-1] * terminal_live_cache[k]
        print("小mc一次")
        return intern_res
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 使用executor.map来并行计算每个n次方的和
        results = list(
            executor.map(
                lambda n: parallely_mc(), range(M)
                )
            )

    int_death_payoff = 0
    for w in range(T):
        int_death_payoff += np.exp(-l*t_all[int((w+1)*buff_day) - 1])*interval_P_cache[(k, w+k+1)]
    fee = 1 - int_death_payoff - terminal_live_cache[k]*np.exp(-l*T/p)


    avg_res = np.mean(results)
    fee_avg_res = np.mean(fee) 

    print("MC一次")

    return avg_res, fee_avg_res, ave


def preliminary_search_per_i(initial_l, k, step_size, M, p, buff_day, S0, r, g, x0, T0, lifespan, interval_P_cache, terminal_live_cache, max_iter=100, ):
    T = lifespan - x0 - k
    # l = 0.000129 * T**2 - 0.010775 * T + 0.200646
    l = initial_l
    lower_l, upper_l = None, None  # 初始化 lower_l 和 upper_l

    # 首先减小 l 直到 fee_avg_res - avg_res[-1] 变为负值
    for _ in range(max_iter):
        print("l: ", l)
        avg_res, fee_avg_res, _ = MC(M, k, p, buff_day, S0, r, g, x0, T0, lifespan, l, interval_P_cache, terminal_live_cache)
        print("max(exp(g*t), S_t): ", avg_res)
        print("fee_avg_res: ", fee_avg_res)
        if fee_avg_res - avg_res < 0:
            upper_l = l + step_size  # 记录这个点的前一个点为 upper_l
            break
        l -= step_size

    # 从取负值的 l 开始，反转方向，以较小的步长增加 l
    step_size /= 3  # 减小步长
    for _ in range(max_iter):
        l += step_size
        print("l: ", l)
        avg_res, fee_avg_res, _ = MC(M, k, p, buff_day, S0, r, g, x0, T0, lifespan, l, interval_P_cache, terminal_live_cache)
        print("max(exp(g*t), S_t): ", avg_res)
        print("fee_avg_res: ", fee_avg_res)
        if fee_avg_res - avg_res > 0:
            lower_l = l - step_size  # 记录这个点的前一个点为 lower_l
            upper_l = l
            break

    print("Before lower_l, upper_l:", lower_l, upper_l)
    # 第二轮搜索：缩小搜索范围
    new_lower_l = lower_l  # 初始化新的搜索范围
    new_upper_l = upper_l

    l_mid = (lower_l + upper_l) / 2  # 找到当前范围的中点
    print("l_mid: ", l_mid)
    avg_res_mid, fee_avg_res_mid, _ = MC(M, k, p, buff_day, S0, r, g, x0, T0, lifespan, l_mid, interval_P_cache, terminal_live_cache)
    print("mid_max(exp(g*t), S_t): ", avg_res_mid)
    print("mid_fee_avg_res: ", fee_avg_res_mid)
    # 根据中点处的差值调整搜索范围
    if fee_avg_res_mid - avg_res_mid > 0:
        new_upper_l = l_mid
    else:
        new_lower_l = l_mid
        
    return new_lower_l, new_upper_l



def fine_search_per_i(M, k, p, buff_day, S0, r, g, x0, T0, lifespan, lower_l, upper_l, fine_step_size, interval_P_cache, terminal_live_cache):
    best_l = lower_l + fine_step_size #lower_l时diff<0,所以从lower_l+fine_step_size查看diff和0的区别
    min_difference = np.inf
    found_small_diff = False  # 标记是否找到小于0.0003的差异
    
    l = lower_l + fine_step_size
    while l <= upper_l - fine_step_size:
        print("l: ", l)
        avg_res, fee_avg_res, _ = MC(M, k, p, buff_day, S0, r, g, x0, T0, lifespan, l, interval_P_cache, terminal_live_cache)
        print("max(exp(g*t), S_t): ", avg_res)
        print("fee_avg_res: ", fee_avg_res)
        difference = abs(fee_avg_res - avg_res)  # 计算差值的绝对值
        # 如果找到了小于0.0003的差异，标记为True
        if difference <0.000003:
            break
    
        if difference < 0.0003:
            found_small_diff = True
        
        # 在找到小于0.0003的差异后，如果差异又大于0.0025，结束搜索
        if found_small_diff and difference > 0.002:
            print(f"Ending search: difference increased beyond 0.0025 after finding a smaller difference.")
            break  # 结束循环
        
        # 更新最小差异和最优l
        if difference < min_difference:
            min_difference = difference
            best_l = l
            
        l += fine_step_size  # 使用非常小的步长
    
    print(f"Best l before stopping: {best_l}, Min difference before stopping: {min_difference}")
    return best_l, min_difference


def generate_p_trade(p):

    init_l_list = [
        0.001199,
        0.001313,
        0.001459,
        0.001567,
        0.001803,
        0.00201,
        0.002264,
        0.0025,
        0.002777,
        0.003156,
        0.003646,
        0.004171,
        0.004733,
        0.005316,
        0.006224,
        0.007382,
        0.008836,
        0.010734,
        0.012753,
        0.015821,
        0.020291,
        0.027767,
        0.037762,
        0.059695,
        0.113971,
    ]

    all_division = len(init_l_list)


    real_list = []
    for i in range(all_division):
        for j in range(p):
            real_list.append({'k':i*p+j, 'init_l':init_l_list[i]})   # 

    return real_list


def generate_log_dir(p):

    year,mon,day,hour, min, sec, _, _,_ = time.gmtime()

    path_dir = f"./logs/p={p}/".format(p)
    path_save = path_dir + f"{mon}_{day}-{hour}:{min}".format(mon,day,hour,min)

    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
        print("Folder created")
    else:
        print("Folder already exists")


    return path_dir, path_save



def run():
    # 初始化参数
    M = 7000
    x0 = 50
    T0 = 2020
    lifespan = 75


    S0 = 1.0
    p = 4
    buff_p = int(12/p)
    buff_day = 10*21*buff_p
    r = 0.05
    g = 0
    tolerance = 1e-6 
    step_size = 0.00012 #preliminary_search_per_i
    initial_l = 0.15

    best_ls = []  # 用于存储每个索引i对应的最优l值
    min_differences = []  # 用于存储每个索引i对应的最小差值

    # 读入pandas，返回csv
    interval_P_cache, terminal_live_cache = load_data(x0, p, lifespan, T0)

    # 设置存储的路径
    path_dir, path_save = generate_log_dir(p)

    # 根据颗粒度生成计算的k，init_l列表
    real_list = generate_p_trade(p)



    

    for iter in real_list:
        with open(path_save,'w+') as save_file:
            print("k:", iter['k'], "  init_l:", iter['init_l'])
            
            # lower_l, upper_l = 0.1, 0.2
            lower_l, upper_l = preliminary_search_per_i(iter['init_l'], iter['k'], step_size, M, p, buff_day, S0, r, g, x0, T0, lifespan, interval_P_cache, terminal_live_cache, max_iter=100)
            print(f"索引{iter['k']}, new lower_l, upper_l: {lower_l}, {upper_l}")

            save_file.write(f"索引{iter['k']}, new lower_l, upper_l: {lower_l}, {upper_l}\n")


            if lower_l is not None and upper_l is not None:
                fine_step_size = 0.000001 # 精细搜索步长 
                best_l, min_difference = fine_search_per_i(M, iter['k'], p, buff_day, S0, r, g, x0, T0, lifespan, lower_l, upper_l, fine_step_size, interval_P_cache, terminal_live_cache)
                best_ls.append(best_l)
                min_differences.append(min_difference)
                print(f"Index {iter['k']}: Best l value: {best_l} with minimum difference: {min_difference}")
                save_file.write(f"Index {iter['k']}: Best l value: {best_l} with minimum difference: {min_difference}")
                save_file.write("case_1")
            else:
                best_ls.append(None)
                min_differences.append(None)
                print(f"Index {iter['k']}: Unable to find suitable l interval during preliminary search.")
                save_file.write(f"Index {iter['k']}: Unable to find suitable l interval during preliminary search.")
                save_file.write("case_2")

            save_file.write("\n##########################################################################\n\n\n")
        save_file.close()

# 执行代码
run()




