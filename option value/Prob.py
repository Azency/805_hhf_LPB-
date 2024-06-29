import pandas as pd
import numpy as np
import math

class D_Prob():
    def __init__(self) -> None:
        self.ax = None
        self.bx = None
        self.kt = None

    def read_abk(self, path_list:list):
        '''
        path_list : ax, bx, kt存储的路径
        
        '''
        if len(path_list) != 3:
            raise Exception("至少要三个路径！")
        self.ax = pd.read_csv(path_list[0])['x'].tolist()
        self.bx = pd.read_csv(path_list[1])['x'].tolist()
        self.kt = pd.read_csv(path_list[2])["Point.Forecast"].tolist()

#np.exp(-self._gen_mxt(x0+i, T0+i, T0)），x0+i岁的人在当年的存活率 ，i=1,2,3,...,t-1,T
    def _gen_mxt(self, x0, T, T0) -> float:
        '''
        需要mxt的调用这个值
        x0 : 年龄
        T : 公元年份
        T0:投保的公元年份
        '''
        a = np.exp(self.ax[x0] + self.bx[x0]*self.kt[T-T0-1])
        #真实的年龄，没有加1，因为ax本身序列索引也是从0开始（0岁，1岁，...）
        # print(f"self.ax[x0]:{self.ax[x0]}且x0:{x0}")
        # print(f"self.bx[x0]:{self.bx[x0]}且x0:{x0}")
        # print(f"self.kt[T-T0-1]:{self.kt[T-T0-1]}且序列是:{T-T0-1}")
        # print(a)

        return a
    
#     def single_live(self, x0, T0, t, p):#计算在第i年从第i*p天活到第t天的概率，其中第i*p天被默认是第i年的第一天，且t整除p=i， t>=0
#         i = t//p
#         k = i+1
#         P = np.exp(-self._gen_mxt(x0+k, T0+k, T0)*(t-i*p)/p)
#         # print(f"整数为{i}年, m_x,t:{self._gen_mxt(x0+i, T0+i, T0)}，样本点{t}对应的{t-i*p}天的零散存活概率q:{P}")

#         return P
    
#     def interval_single_live(self, x0, T0, t, p):#计算在第t天活着，直到到第(i+1)*p天仍活着的概率
#         i = t//p
#         k = i+1
#         P = np.exp(-self._gen_mxt(x0+k, T0+k, T0)*(k*p-t-1)/p)

#         return P
    
#     def single_death(self, x0, T0, t, p):#计算在t//p+1年中，任意一颗粒度的死亡概率
#         i = t//p
#         k = i + 1
#         P = 1-np.exp(-self._gen_mxt(x0+k, T0+k, T0)/p) #最后一天的死亡率

#         return P

#     def gen_P(self, x0, T0, t):#计算从0时刻起到第t整数年死亡的条件概率
#         '''
#         生成对应的概率
#         x0 : 投保的年龄 
#         T0 : 投保的公元年份,eg2020
#         t : 死亡的时间
#         '''
#         P = 1-np.exp(-self._gen_mxt(x0+t, T0+t, T0)) #最后一年的死亡率
#         for i in range(t-1, 0, -1):#倒退计算，最后一年之前的累计存活概率  ##############是否需要计算到-1年，否，因为kt序列从0开始索引
#             P *= np.exp(-1 * self._gen_mxt(x0+i, T0+i, T0))

#         # P = 2*self._gen_mxt(x0+t, T0+t, T0)/(2 + self._gen_mxt(x0+t, T0+t, T0)) #最后一年的死亡概率
#         # for i in range(t-1, 0, -1):#倒退计算，最后一年之前的累计存活概率
#         #     P *= 1 - 2*self._gen_mxt(x0+i, T0+i, T0)/(2 + self._gen_mxt(x0+i, T0+i, T0))

#         return P

    
#     def accu_live(self, x0, T0, t, s):#计算从s年起到第t年的累计存活概率,其中t最多是T,s>=0.当s= 0，序列取值是：t,t-1,t-2,...,1
#         P_a = 1#第一年的存活概率
#         for i in range(t, s, -1):#倒退计算，最后一年之前的累计存活概率
#             P_a *= np.exp(-1 * self._gen_mxt(x0+i, T0+i, T0))  #x0+i岁的人在当年的存活率 ，i=t,t-1,...1 
#             # print(f"第{i}年存活率：{np.exp(-1 * self._gen_mxt(x0+i, T0+i, T0))}")    

#         return P_a
    
#     def unit_P(self, x0, T0, t, p):#计算从0时前一天（购买合同）起，在第t个计数单元刚好死亡的条件概率，t>=0
#         P = self.single_death(x0, T0, t, p)
#         P *=self.single_live(x0, T0, t, p)
#         P *= self.accu_live(x0, T0, t//p, 0)

#         return P
    
# #代码中的计算逻辑和文章中写的不一致，均是从0开始索引    
# #interval_death_P和unit_P不同，
# #interval_death_P是默认为PH在t时刻是活着的，t>=0，当t = 0，表示签完合同的第一天还活着，签合同之前的日子，我们默认PH是活着的
# #unit_P(self, x0, T0, t, p)则是从签合同的前一天开始算起，t = 0表示，签完合同后的一天
#     def interval_death_P(self, x0, T0, t, s, p):#计算从t个计数单元起到第s个计数单元时死亡的条件概率,s>t>=0严格
#         j = t//p
#         i = s//p
#         if j == i:
#             P = self.single_death(x0, T0, s, p)*np.exp(-self._gen_mxt(x0+i+1, T0+i+1, T0)*(s-t-1)/p)
#         else:
#             P = self.single_death(x0, T0, s, p)
#             # print("single_death",P)
#             P *=self.single_live(x0, T0, s, p)
#             # print("single_live", self.single_live(x0, T0, s, p), P)
#             P *= self.accu_live(x0, T0, i, j+1)
#             # print("accu_live",self.accu_live(x0, T0, i, j+1), P)
#             P *= self.interval_single_live(x0, T0, t, p)
#             # print("interval_single_live",self.interval_single_live(x0, T0, t, p), P)

#         return P
    
#     def interval_live_P(self, x0, T0, t, s, p):#计算从t时刻起到第s个计数单元时累计存活概率
#         j = t//p
#         i = s//p
#         if j == i:
#             P = np.exp(-self._gen_mxt(x0+i+1, T0+i+1, T0)*(s-t)/p)
#         else:
#             P = 1 - self.single_death(x0, T0, s, p)
#             # print("最后一天存活的概率",P)
#             P *=self.single_live(x0, T0, s, p)
#             # print("最后一天之前非整年的存活概率", self.single_live(x0, T0, i, p))
#             P *= self.accu_live(x0, T0, i, j+1)
#             # print("整数的前j+1-i年的存活概率", self.accu_live(x0, T0, k))
#             P *= self.interval_single_live(x0, T0, t, p)

#         return P

#     def interval_terminal_live(self, x0, T0, t, T, p):#计算从t时刻起到最后时刻的累计存活概率,t>=0,当取t = 0,interval_terminal_live = accu_live( x0, T0, T, 0)
#         j = t//p
#         P = self.accu_live(x0, T0, T, j)##accu_live函数，T，T-1，...,j+1年存活率累积
#         # print(f"j+1到T的存活率{P}")
#         P *= self.interval_single_live(x0, T0, t, p)
#         # print(f"零散的第一年存活率：{self.interval_single_live(x0, T0, t, p)}")

#         return P


    def first_single_live(self, x0, T0, t, p):#计算在第i年从第t天活到该年最后一天i*p天的概率，t>=0,t = 0时，P = 1
        i= math.ceil(t/p)
        P = np.exp(-self._gen_mxt(x0+i, T0+i, T0)*(i*p - t)/p)
        # print(f"整数为{i}年, m_x,t:{self._gen_mxt(x0+i, T0+i, T0)}，样本点{t}对应的{t-i*p}天的零散存活概率q:{P}")

        return P
    
    def last_single_live(self, x0, T0, t, p):#计算在第i年的第一天:(i-1)*p+1天，活到第t-1天的概率
        i= math.ceil(t/p)
        if i == 0:
            P = 1
        else:
            P = np.exp(-self._gen_mxt(x0+i, T0+i, T0)*(t-1-(i-1)*p)/p)

        return P
    
    def last_single_death(self, x0, T0, t, p):#计算在i年中，任意一颗粒度的死亡概率
        i= math.ceil(t/p)
        if i == 0:
            P = 0
        else:
            P = 1-np.exp(-self._gen_mxt(x0+i, T0+i, T0)/p) #最后一天的死亡率

        return P

    def gen_P(self, x0, T0, t):#计算从0时刻起到第t整数年死亡的条件概率
        '''
        生成对应的概率
        x0 : 投保的年龄 
        T0 : 投保的公元年份,eg2020
        t : 死亡的时间
        '''
        P = 1-np.exp(-self._gen_mxt(x0+t, T0+t, T0)) #最后一年的死亡率
        for i in range(t-1, 0, -1):#倒退计算，最后一年之前的累计存活概率  ##############是否需要计算到-1年，否，因为kt序列从0开始索引
            P *= np.exp(-1 * self._gen_mxt(x0+i, T0+i, T0))

        # P = 2*self._gen_mxt(x0+t, T0+t, T0)/(2 + self._gen_mxt(x0+t, T0+t, T0)) #最后一年的死亡概率
        # for i in range(t-1, 0, -1):#倒退计算，最后一年之前的累计存活概率
        #     P *= 1 - 2*self._gen_mxt(x0+i, T0+i, T0)/(2 + self._gen_mxt(x0+i, T0+i, T0))

        return P

    
    def accu_live(self, x0, T0, t, s):#计算从s+1年起到第t年的累计存活概率,其中t最多是T,s>=0.当s= 0，序列取值是：t,t-1,t-2,...,1
        if t == 0:
            P_a = 1
        else:
            P_a = 1#投保前存活概率
            for i in range(t, s, -1):#倒退计算，t,t-1,t-2,...,1
                P_a *= np.exp(-1 * self._gen_mxt(x0+i, T0+i, T0))  #x0+i岁的人在当年的存活率 ，i=t,t-1,...1 
                # print(f"第{i}年存活率：{np.exp(-1 * self._gen_mxt(x0+i, T0+i, T0))}")    

        return P_a
    
    def unit_P(self, x0, T0, t, p):#计算从购买合同 0时刻起，在第t个计数单元刚好死亡的条件概率，t>=1
        P = self.last_single_death(x0, T0, t, p)
        # print("last_single_death",P)
        P *= self.last_single_live(x0, T0, t, p)#t天前这一年内活着
        # print("last_single_live",self.last_single_live(x0, T0, t, p))
        P *= self.accu_live(x0, T0, math.ceil(t/p)-1, 0)
        # print("accu_live",self.accu_live(x0, T0, math.ceil(t/p)-1, 0))


        return P
    
#代码中的计算逻辑和文章中写的一致，interval_death_P和unit_P不同
#interval_death_P 从第t（t>0）个计数单元到第s（s>1)个计数单元条件死亡概率
#unit_P 从第t=0个计数单元到第s（s>=0)个计数单元条件死亡概率
    def interval_death_P(self, x0, T0, t, s, p):#计算从t个计数单元起到第s个计数单元时死亡的条件概率,s>t>0严格
        i = math.ceil(s/p)
        j = math.ceil(t/p)

        if j == i:
            P = self.last_single_death(x0, T0, s, p)*np.exp(-self._gen_mxt(x0+i, T0+i, T0)*(s-1-t)/p)
        else:
            P = self.last_single_death(x0, T0, s, p)
            # print("single_death",P)
            P *=self.last_single_live(x0, T0, s, p)
            # print("single_live", self.last_single_live(x0, T0, s, p), P)
            P *= self.accu_live(x0, T0, i-1, j)
            # print("accu_live",self.accu_live(x0, T0, i-1, j), P)
            P *= self.first_single_live(x0, T0, t, p)
            # print("interval_single_live",self.first_single_live(x0, T0, t, p), P)

        return P
    
    def interval_live_P(self, x0, T0, t, s, p):#计算从t时刻起到第s个计数单元时累计存活概率
        i = math.ceil(s/p)
        j = math.ceil(t/p)

        if j == i:
            P = np.exp(-self._gen_mxt(x0+i, T0+i, T0)*(s-t)/p)
        else:
            P = self.last_single_live(x0, T0, s+1, p)
            # print("最后年份中非整年的存活概率", self.last_single_live(x0, T0, s, p))
            P *= self.accu_live(x0, T0, i-1, j)
            # print("整数的前j+1到i-1年的存活概率", self.accu_live(x0, T0, i-1, j))
            P *= self.first_single_live(x0, T0, t, p)
            # print("从第t+1天到完整第j年的存活概率", self.first_single_live(x0, T0, t, p))

        return P

    def interval_terminal_live(self, x0, T0, t, T, p):#计算从t时刻起到最后时刻的累计存活概率,t>=1
        j = math.ceil(t/p)

        P = self.accu_live(x0, T0, T, j)##accu_live函数，T，T-1，...,j+1年存活率累积
        # print(f"j+1到T的整数年存活率{P}")
        P *= self.first_single_live(x0, T0, t, p)
        # print(f"第一年零散的存活率：{self.first_single_live(x0, T0, t, p)}")

        return P