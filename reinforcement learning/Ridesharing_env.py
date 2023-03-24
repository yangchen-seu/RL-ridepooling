'''
Author: your name
Date: 2022-02-21 14:36:55
LastEditTime: 2022-07-05 09:08:37
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \强化学习网约车\random strategy\Ridesharing_env.py
'''

import gym
import numpy as np
import Simulation as simu
# from Reinforcementlearning.Config import Config


class Ridesharing_env(gym.Env):
    """
    The training environment for vehicle dispatching.
    """
    def __init__(self, agent_lis,critic, cfg):
        for agent in agent_lis:
            agent.reset()

        self.simu = simu.Simulation(agent_lis, critic,cfg)
        self.render = False
         # timesteps
        self.episode_timestep = 0
        self.n_episode        = 0

    def step(self):
        # print('CustomEnv Step successful!')
        reward, time_slot, done = self.simu.step()
        if self.render:
            self.show()
        return reward,time_slot, done

    def reset(self, agent_lis, critic, cfg):
        self.__init__(agent_lis, critic, cfg)
        time_slot = 0
        return time_slot


    def show(self):
        pass

