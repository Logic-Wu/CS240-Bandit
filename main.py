import numpy as np
import matplotlib.pyplot as plt
import math
import random


# --------------- Bandit Class -----------------
class Bandits:
    def __init__(self, m, D, t):
        # 该杆被拉过的次数
        self.N = 0
        # mean and var
        self.m = m
        self.D = D
        self.estimated_mean = 0

        # parameters for soft max
        self.para = math.exp(self.estimated_mean / t)
        self.weight = 0

    # 根据概率分布随机获得reward
    def pull(self):
        return self.D * np.random.randn() + self.m

    def greedy_mix_init(self):
        self.N = 0
        self.estimated_mean = 0

    def soft_max_init(self, t_para):
        self.N = 0
        self.estimated_mean = 0
        self.weight = self.para / t_para

    # 更新epsilon greedy算法下的估计期望
    def epsilon_greedy_update(self, x):
        self.N += 1
        self.estimated_mean = (1 - 1.0 / self.N) * self.estimated_mean + 1.0 / self.N * x

    def softmax_update(self, x):
        self.N += 1  # 当前轮选择index臂，index拉取次数+1
        self.estimated_mean = (1 - 1.0 / self.N) * self.estimated_mean + 1.0 / self.N * x


# ------------------- epsilon greedy algorithm -----------------------
def run_epsilon_greedy_experiment(k_num, m_bandits, epsilon, N):
    player_choice = np.empty(N)

    for i in range(N):
        # epsilon greedy
        p = np.random.random()
        if p < epsilon:  # exploration exploitation trade-off
            # 在epsilon概率中选择探索
            # 在所有拉杆中随机选择一个（探索）
            j = np.random.choice(k_num)
        else:
            # 在（1-epsilon）概率中选择开发
            # 体现greedy思想，选择目前所知回报期望最大的杆
            j = np.argmax([a.estimated_mean for a in m_bandits])
        # 拉动所选杆，并更新该杆的期望
        reward = m_bandits[j].pull()
        # 更新估计期望
        m_bandits[j].epsilon_greedy_update(reward)
        # 记录本轮选择的奖励
        player_choice[i] = reward

    # 可视化选择过程
    cumulative_average = np.cumsum(player_choice) / (np.arange(N) + 1)

    # 绘制每一步的回报平均值表
    plt.plot(cumulative_average)
    for i in range(k_num):
        plt.plot(np.ones(N) * bandits[i].m)
    plt.xscale('log')
    plt.show()
    print("Estimated mean for each bandit in epsilon greedy")
    for x in m_bandits:
        print(x.estimated_mean)

    return cumulative_average


# --------------------------- GreedyMix ---------------------------------
def run_greedy_mix_experiment(k_num, m_bandits, epsilon, N):
    player_choice = np.empty(N)

    for i in range(N):
        # epsilon greedy
        p = np.random.random()
        if p < (epsilon * math.log(i + 1) / (i + 1)):  # exploration exploitation trade-off
            # 在epsilon概率中选择探索
            # 在所有拉杆中随机选择一个（探索）
            j = np.random.choice(k_num)
        else:
            # 在（1-epsilon）概率中选择开发
            # 体现greedy思想，选择目前所知回报期望最大的杆
            j = np.argmax([a.estimated_mean for a in m_bandits])
        # 拉动所选杆，并更新该杆的期望
        reward = m_bandits[j].pull()
        # 更新估计期望
        m_bandits[j].epsilon_greedy_update(reward)
        # 记录本轮选择的奖励
        player_choice[i] = reward

    # 可视化选择过程
    cumulative_average = np.cumsum(player_choice) / (np.arange(N) + 1)

    # 绘制每一步的回报平均值表
    plt.plot(cumulative_average)
    for i in range(k_num):
        plt.plot(np.ones(N) * bandits[i].m)
    plt.xscale('log')
    plt.show()
    print("Estimated mean for each bandit in epsilon greedy")
    for x in m_bandits:
        print(x.estimated_mean)

    return cumulative_average


# --------------------------- soft max algorithm --------------------------
def run_soft_max(k_num, m_bandits, temp, N):
    player_choice = np.empty(N)

    for i in range(N):
        choice = particle_choose(k_num, m_bandits)  # 粒子模型模拟带权选择
        reward = m_bandits[choice].pull()  # 拉下拉杆并获取奖励
        m_bandits[choice].softmax_update(reward)  # 更新估计平均值参数
        total_para = 0
        for j in range(k_num):  # 遍历臂更新softmax参数
            m_bandits[j].para = math.exp(m_bandits[j].estimated_mean / temp)  # 计算每个臂的softmax参数
            total_para += m_bandits[j].para  # 计算参数总和
        for j in range(k_num):
            m_bandits[j].weight = m_bandits[j].para / total_para  # 通过Softmax函数更新选取权重
        player_choice[i] = reward

    # 可视化选择过程
    cumulative_average = np.cumsum(player_choice) / (np.arange(N) + 1)

    # 绘制每一步的回报平均值表
    plt.plot(cumulative_average)
    for i in range(k_num):
        plt.plot(np.ones(N) * bandits[i].m)
    plt.xscale('log')
    plt.show()
    print("Estimated mean for each bandit in Soft max")
    for x in m_bandits:
        print(x.estimated_mean)

    return cumulative_average


# ----------------------- soft mix ---------------------------
def run_soft_mix(k_num, m_bandits, temp, N):  # 类似于softmax
    player_choice = np.empty(N)

    for i in range(N):
        choice = particle_choose(k_num, m_bandits)
        reward = m_bandits[choice].pull()
        m_bandits[choice].softmax_update(reward)
        total_para = 0
        for j in range(k_num):
            m_bandits[j].para = math.exp(
                m_bandits[j].estimated_mean / (temp * math.log(i + 2) / (i + 2)))  # 更新的时候乘以一个 log(t)/t 的时间乘子
            total_para += m_bandits[j].para
        for j in range(k_num):
            m_bandits[j].weight = m_bandits[j].para / total_para
        player_choice[i] = reward

    # 可视化选择过程
    cumulative_average = np.cumsum(player_choice) / (np.arange(N) + 1)

    # 绘制每一步的回报平均值表
    plt.plot(cumulative_average)
    for i in range(k_num):
        plt.plot(np.ones(N) * bandits[i].m)
    plt.xscale('log')
    plt.show()
    print("Estimated mean for each bandit in Soft mix")
    for x in m_bandits:
        print(x.estimated_mean)

    return cumulative_average


# 按权重选杆
def particle_choose(k_num, bandit):  # 粒子模型模拟权重
    particles = []  # 初始化粒子库
    for i in range(k_num):
        n = math.floor(bandit[i].weight * 1000)  # 将权重化为权重*1k个粒子
        for j in range(n):
            particles.append(i)  # 写入粒子库
    choose = random.choice(particles)  # 在粒子库中选择一个，概率等于各臂的权重
    return choose  # 返回选项


if __name__ == '__main__':
    # epsilon
    eps = 0.01
    # epsilon mix
    eps_mix = 100
    # temperature for soft_max
    temperature = 0.1
    # temperature for soft_mix
    temperature_mix = 20
    # 测试数量
    horizon = 10000
    # 臂数
    k = 10
    # 臂组
    bandits = []
    # 分别生成十个回报符合正态分布的拉杆，他们的均值为1到10，方差为4
    for i in range(k):
        bandits.append(Bandits(i + 1, 4, temperature))
        print("Bandit ", i + 1, " mean ", bandits[i].m, " var ", bandits[i].D)

    # epsilon greedy 实验
    experiment_greedy = run_epsilon_greedy_experiment(k, bandits, eps, horizon)

    # 重置杆属性，为greedy mix初始化
    for i in range(k):
        bandits[i].greedy_mix_init()

    # greedy mix 实验
    experiment_greedy_mix = run_greedy_mix_experiment(k, bandits, eps_mix, horizon)

    # 重置杆属性，为soft max初始化
    total = 0
    for i in range(k):
        total += bandits[i].para
    for i in range(k):
        bandits[i].soft_max_init(total)

    # soft max 实验
    experiment_softmax = run_soft_max(k, bandits, temperature, horizon)

    # 重置杆属性，为soft max初始化
    total = 0
    for i in range(k):
        bandits[i].para = math.exp(0)
        total += bandits[i].para
    for i in range(k):
        bandits[i].soft_max_init(total)

    # soft mix 实验
    experiment_soft_mix = run_soft_mix(k, bandits, temperature_mix, horizon)
