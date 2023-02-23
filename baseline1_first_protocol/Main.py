
import Simulation as sm
import Config



class Main():

    def __init__(self,cfg ) -> None:
        self.simu = sm.Simulation(cfg)
        self.test_episodes = 1
 
 
    def run(self):
        rewards = []
        for i_ep in range(self.test_episodes):
            print('episode:', i_ep)
            ep_reward = 0 # 一个回合的奖励
            self.simu.reset()
            while True:
                reward, done = self.simu.step()
                # print(self.simu.time)
                # print('reward', reward)
                ep_reward += reward
                if done:
                    break
            rewards.append(ep_reward)

            # print('rewards',rewards,smooth_rewards,'smooth_rewards')
        self.simu.save_metric(path = "output/{}/system_metric.pkl".format(cfg.optimazition_target))
        res = self.simu.res
        self.plot_metrics(cfg, metric = self.simu.detour_distance, metric_name = 'detour_distance')
        self.plot_metrics(cfg, metric =self.simu.traveltime, metric_name = 'traveltime')
        self.plot_metrics(cfg, metric = self.simu.waitingtime, metric_name = 'waiting_time')
        self.plot_metrics(cfg, metric = self.simu.pickup_time, metric_name = 'pickup_time')
        self.plot_metrics(cfg, metric = self.simu.platform_income, metric_name = 'platform_income')
        self.plot_metrics(cfg, metric = self.simu.shared_distance, metric_name = 'shared_distance')
        self.plot_metrics(cfg, metric = self.simu.dispatch_time, metric_name = 'dispatch_time')

        print(res)
        
        return res

    # 绘图
    def test_plot(self,rewards, smoothed_rewards, algo, env_name):
 
        self.plot_rewards(rewards, smoothed_rewards,  algo, env_name)
 
 
    # 绘图
    def plot_rewards(self, cfg, rewards,smoothed_rewards, algo = 'batchmatching', env_name='ridesharing'):
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set()
        plt.figure() # 创建一个图形实例，方便同时多画几个图
        plt.title("reward of {} for {}".format(algo,env_name))
        plt.xlabel('epsiodes')
        plt.plot(rewards,label='rewards')
        plt.plot(smoothed_rewards,label='smoothed rewards')
        plt.legend()
        # plt.savefig('output\\rewards.png')
        plt.savefig('output\\{}/rewards.png'.format(cfg.optimazition_target))      

     # 绘图
    def plot_metrics(self,cfg, metric,  metric_name, algo = 'batchmatching', env_name='ridesharing'):
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set()
        plt.figure() # 创建一个图形实例，方便同时多画几个图
        plt.title("{} of {} for {}".format(metric_name, algo,env_name))
        plt.xlabel('orders')
        plt.plot(metric,label=metric_name)
        plt.legend()
        plt.savefig('output\\{}/{}.png'.format(cfg.optimazition_target, metric_name))       

# platform_income, or expected_shared_distance or passenger_experience
cfg = Config.Config()
import time
start = time.time()
ma = Main(cfg).run() 
end = time.time()
print('执行时间',end - start)
