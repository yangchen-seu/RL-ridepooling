'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-07-03 09:04:17
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-07-06 06:28:47
FilePath: /matching/reinforcement learning/temp.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pickle
import numpy as np
file = open('./output/platform_income/system_metric.pkl','rb')
metric = pickle.load(file)
res = {}
res['extra_distance'] = np.mean(metric['extra_distance'])
res['saved_distance'] = np.mean(metric['saved_distance'])
res['taker_pickup_time'] = np.mean(metric['taker_pickup_time'])
res['waiting_time'] = np.mean(metric['waiting_time'])
res['platflorm_income'] = np.mean(metric['platflorm_income'])
res['shared_distance'] = np.mean(metric['shared_distance'])
res['dispatch_time'] = np.mean(metric['dispatch_time'])
res['success_rate'] = metric['success_rate']
print(res)