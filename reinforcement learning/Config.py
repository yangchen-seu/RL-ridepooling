'''
Author: your name
Date: 2021-12-07 21:36:06
LastEditTime: 2022-11-27 07:07:00
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \强化学习网约车\Reinforcementlearning\Config.py
'''
import torch
import time

class Config:

    def __init__(self) -> None:
        self.vehicle_num = 100
        self.n_states = 2
        
        # files
        self.input_path = 'input\\'
        self.output_path = 'output\\'
        
        self.order_file_name = 'order.csv'
        self.network_file = 'network.csv'

        # 与网络相关的参数
        self.R = 2 # 搜索半径
        self.unit_distance_value = 10 # 平台收益一公里5块钱
        self.unit_distance_cost = 2 # 司机消耗一公里2块钱
        
        self.date = '2017-05-01'
        self.simulation_begin_time = ' 08:00:00' # 仿真开始时间
        self.simulation_end_time = ' 09:00:00' # 仿真结束时间
        self.begin_timestamp  = time.mktime(time.strptime(self.date +  self.simulation_begin_time, "%Y-%m-%d %H:%M:%S"))
        self.end_timestamp  = time.mktime(time.strptime(self.date +  self.simulation_end_time, "%Y-%m-%d %H:%M:%S"))

        self.unit_driving_time = 120/1000 # 行驶速度
        self.unit_time_value = 1.5/120 # 每秒的行驶费用
        self.demand_ratio = 1

        # matching condition
        self.pickup_distance_threshold = 2000
        self.detour_distance_threshold = 3000

        self.algo = "DQN"  # 算法名称
        self.env_name = 'Ridesharing_env-v0' # 环境名称
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.train_eps = 50 # 训练的回合数
        self.eval_eps = 30 # 测试的回合数

        # 超参数
        self.gamma = 0.95 # 强化学习中的折扣因子
        self.epsilon_start = 0.90 # e-greedy策略中初始epsilon
        self.epsilon_end = 0.01 # e-greedy策略中的终止epsilon
        self.epsilon_decay = 500 # e-greedy策略中epsilon的衰减率
        self.lr = 0.0001  # 学习率
        self.memory_capacity = 100000  # 经验回放的容量
        self.batch_size = 64 # mini-batch SGD中的批量大小
        self.target_update = 4 # 目标网络的更新频率
        self.hidden_dim = 128  # 网络隐藏层

        self.optimazition_target = 'expected_shared_distance' # platform_income, expected_shared_distance
        self.matching_condition = True

         # matching condition
        self.pickup_distance_threshold = 2000
        self.extra_distance_threshold = 3000