
import sys, os
import numpy as np

import tqdm
import datetime
import Config
import Ridesharing_env
import Vehicle
import random
import torch
from common.model import critic
import Critic as cr
from common.memory import ReplayBuffer

cfg = Config.Config()

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间

def __init__(self,):
    pass


def env_agent_config(cfg, seed = 1):

    vehicle_num = cfg.vehicle_num
    vehicle_lis = []
    # 初始化critic
    critic = cr.Critic(cfg.n_states, cfg)
    # 初始化多智能体
    for driver_id in range(vehicle_num):
        agent = Vehicle.Vehicle(driver_id,cfg)
        vehicle_lis.append(agent)

    # 初始化环境
    env = Ridesharing_env.Ridesharing_env(vehicle_lis, critic , cfg)
    print('环境初始化完毕')
    return env, vehicle_lis, critic

def train(cfg, env, vehicle_lis, critic):
    memory = ReplayBuffer(cfg.memory_capacity) # 经验回放
    res = {} # 存储系统指标
    # 训练

    print('begin training')
    print(f'env:{cfg.env_name}, algo:{cfg.algo}, device:{cfg.device}')
    rewards = [] # 记录所有回合的奖励
    smooth_rewards = [] # 平滑奖励
    for i_ep in tqdm.trange(cfg.train_eps):
        # print('episode:', i_ep)
        ep_reward = 0 # 一个回合的奖励
        time_slot = env.reset(vehicle_lis, critic, cfg) # 重置环境，返回初始状态
        while True:
            memory_agents = []
            for agent in vehicle_lis:
                if agent.state == 1:
                    # print('agent{} action'.format(agent.id))   
                    state = torch.tensor([agent.location, time_slot], device=cfg.device, dtype=torch.float) 
                    agent.value = critic.value_net(state)
                    agent.target_value = critic.target_net(state)
                    memory_agents.append(agent)



            reward, time_slot, done = env.step() # 更新环境，返回transition

            # 存储经验
            for agent in memory_agents:
                state = [agent.origin_location, time_slot]
                next_state = [agent.location, (agent.activate_time - cfg.begin_timestamp)//10]
                memory.push(state, reward / len(memory_agents), \
                    next_state, done) # 存入memory中


            critic.update(memory)  # 智能体更新

            ep_reward += reward # 累加奖励

            if done:
                break

        # env.render()
        if (i_ep + 1)% cfg.target_update == 0: # 更新目标网络
            critic.target_net.load_state_dict(critic.value_net.state_dict())
        
        # if (i_ep + 1) % 10 == 0:
        # print('回合:{}/{}, 奖励：{}'.format(i_ep + 1, cfg.train_eps, ep_reward))
        rewards.append(ep_reward)
        # 存储最好的奖励
        if ep_reward >= max(rewards):
            env.simu.save_metric(path = "./output/{}/system_metric.pkl".format(cfg.optimazition_target))
            res = env.simu.res

        if smooth_rewards:
            smooth_rewards.append(0.9 * smooth_rewards[-1] + 0.1 * ep_reward)
        else:
            smooth_rewards.append(ep_reward)

    print('finish training')

     
    plot_metrics(metric = env.simu.detour_distance, metric_name = 'detour_distance')
    plot_metrics(metric = env.simu.traveltime, metric_name = 'traveltime')
    plot_metrics(metric = env.simu.waitingtime, metric_name = 'waiting_time')
    plot_metrics(metric = env.simu.pickup_time, metric_name = 'pickup_time')
    plot_metrics(metric = env.simu.platflorm_income, metric_name = 'platflorm_income')
    plot_metrics(metric = env.simu.shared_distance, metric_name = 'shared_distance')
    plot_metrics(metric = env.simu.dispatch_time, metric_name = 'dispatch_time')

    print(res)
    return rewards, smooth_rewards,res



def train_plot(cfg, agent, rewards, smoothed_rewards):
    os.makedirs(cfg.result_path,exist_ok=True)
    os.makedirs(cfg.model_path,exist_ok=True)
    agent.save(path = cfg.model_path) 
    save_results(rewards, smoothed_rewards, tag='train', path = cfg.result_path)
    plot_rewards(rewards, smoothed_rewards, device = cfg.device, \
        algo = cfg.algo, env_name = cfg.env, tag = 'train'
        )

def test_plot(cfg, agent, rewards, smoothed_rewards):
    os.makedirs(cfg.result_path,exist_ok=True)
    os.makedirs(cfg.model_path,exist_ok=True)
    agent.save(path = cfg.model_path) 
    save_results(rewards, smoothed_rewards, tag='test', path = cfg.result_path)
    plot_rewards(rewards, smoothed_rewards, device = cfg.device, \
        algo = cfg.algo, env_name = cfg.env, tag = 'test'
        )

def plot_rewards(rewards,smoothed_rewards,device, algo,env_name,tag='train'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set() 
    plt.figure() # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve on {} of {} for {}".format(device, algo,env_name))
    plt.xlabel('epsiodes')
    plt.plot(rewards,label='rewards')
    plt.plot(smoothed_rewards,label='smoothed rewards')
    plt.legend()
    plt.savefig('./output/reward.png')   

     # 绘图
def plot_metrics(metric,  metric_name, algo = 'Reinforecement Learning', env_name='ridesharing'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    plt.figure() # 创建一个图形实例，方便同时多画几个图
    plt.title("{} of {} for {}".format(metric_name, algo,env_name))
    plt.xlabel('orders')
    plt.plot(metric,label=metric_name)
    plt.legend()
    plt.savefig('./output/{}/{}.png'.format(cfg.optimazition_target, metric_name))   

def save_results(rewards,smooth_rewards,tag='train',path = '\\outputs\\results\\'):
    '''save rewards and smooth_rewards
    '''
    np.save(path +'{}_rewards.npy'.format(tag), rewards)
    np.save( path+'{}_smooth_rewards.npy'.format(tag), smooth_rewards)
    print('结果保存完毕！')


if __name__ == '__main__':
    cfg = Config.Config()
    env, vehicle_lis, critic = env_agent_config(cfg, seed = 1)
    # training
    rewards, smooth_rewards , res = train(cfg, env, vehicle_lis,critic)
    plot_rewards(rewards,smooth_rewards,cfg.device, cfg.algo,cfg.env_name,tag='train')


    path = './output/critic.pickle'
    def save(path):
        import pickle
        with open(path, 'wb') as handle:
            pickle.dump(critic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    save(path)
