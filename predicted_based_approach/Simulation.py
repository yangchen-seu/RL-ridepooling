

import time
import pandas as pd
import numpy as np
import Network as net
import Config
import Seeker
import random
from common import Hungarian, KM_method
import Vehicle


Config = Config.Config()


class Simulation():

    def __init__(self, cfg) -> None:
        self.date = cfg.date
        self.order_list = pd.read_csv(
            './input/order.csv').sample(frac=cfg.demand_ratio, random_state=1)
        self.order_list['beginTime_stamp'] = self.order_list['dwv_order_make_haikou_1.arrive_time'].apply(
            lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))
        self.begin_time = time.mktime(time.strptime(
            cfg.date + cfg.simulation_begin_time, "%Y-%m-%d %H:%M:%S"))
        self.end_time = time.mktime(time.strptime(
            cfg.date + cfg.simulation_end_time, "%Y-%m-%d %H:%M:%S"))
        self.locations = self.order_list['O_location'].unique()
        self.network = net.Network()
        self.time_unit = 10  # 控制的时间窗,每10s匹配一次
        self.index = 0  # 计数器
        self.device = cfg.device
        self.total_reward = 0
        self.optimazition_target = cfg.optimazition_target  # 仿真的优化目标
        self.matching_condition = cfg.matching_condition  # 匹配时是否有条件限制
        self.pickup_distance_threshold = cfg.pickup_distance_threshold
        self.detour_distance_threshold = cfg.detour_distance_threshold
        self.vehicle_list = []

        self.takers = []
        self.current_seekers = []  # 存储需要匹配的乘客
        self.remain_seekers = []
        self.time_reset()

        for i in range(Config.vehicle_num):
            location = random.choice(self.locations)
            vehicle = Vehicle.Vehicle(i, location)
            self.vehicle_list.append(vehicle)
            # 重置司机的时间
            vehicle.activate_time = self.time

        # system metric
        self.his_order = []  # all orders responsed
        self.waitingtime = []
        self.detour_distance = []
        self.traveltime = []
        self.pickup_time = []
        self.dispatch_time = []
        self.platflorm_income = []
        self.shared_distance = []

    def reset(self):
        self.takers = []
        self.current_seekers = []  # 存储需要匹配的乘客
        self.remain_seekers = []
        self.vehicle_list = []
        self.total_reward = 0
        self.order_list = pd.read_csv(
            './input/order.csv').sample(frac=Config.demand_ratio, random_state=1)
        self.order_list['beginTime_stamp'] = self.order_list['dwv_order_make_haikou_1.arrive_time'].apply(
            lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))
        self.begin_time = time.mktime(time.strptime(
            Config.date + Config.simulation_begin_time, "%Y-%m-%d %H:%M:%S"))
        self.end_time = time.mktime(time.strptime(
            Config.date + Config.simulation_end_time, "%Y-%m-%d %H:%M:%S"))
        self.order_list = self.order_list[self.order_list['beginTime_stamp']
                                          >= self.begin_time]
        self.order_list = self.order_list[self.order_list['beginTime_stamp']
                                          <= self.end_time]

        self.time_reset()

        for i in range(Config.vehicle_num):
            location = random.choice(self.locations)
            vehicle = Vehicle.Vehicle(i, location)
            self.vehicle_list.append(vehicle)
            # 重置司机的时间
            vehicle.activate_time = self.time

        # system metric
        self.his_order = []  # all orders responsed
        self.waitingtime = []
        self.detour_distance = []
        self.traveltime = []
        self.pickup_time = []
        self.dispatch_time = []
        self.platflorm_income = []
        self.shared_distance = []

    def time_reset(self):
        # 转换成时间数组
        self.time = time.strptime(
            Config.date + Config.simulation_begin_time, "%Y-%m-%d %H:%M:%S")
        # 转换成时间戳
        self.time = time.mktime(self.time)
        self.time_slot = 0
        # print('time reset:', self.time)

    def step(self,):
        time_old = self.time
        self.time += self.time_unit
        self.time_slot += 1

        # 筛选时间窗内的订单
        current_time_orders = self.order_list[self.order_list['beginTime_stamp'] >= time_old]
        current_time_orders = current_time_orders[current_time_orders['beginTime_stamp'] <= self.time]
        self.current_seekers = []  # 暂时不考虑等待的订单
        for index, row in current_time_orders.iterrows():
            seeker = Seeker.Seeker(index, row)
            self.current_seekers.append(seeker)
        for seeker in self.remain_seekers:
            self.current_seekers.append(seeker)

        start = time.time()
        reward, done = self.process(self.time)
        end = time.time()
        print('process 用时', end - start )
        return reward,  done

    #

    def process(self, time_, ):
        reward = 0
        takers = []
        vehicles = []
        seekers = self.current_seekers

        if self.time >= time.mktime(time.strptime(Config.date + Config.simulation_end_time, "%Y-%m-%d %H:%M:%S")):
            print('当前episode仿真时间结束,奖励为:', self.total_reward)

            # 计算系统指标
            self.res = {}
            # for seekers
            for order in self.his_order:
                self.waitingtime.append(order.waitingtime)
                self.detour_distance.append(order.detour)
                self.traveltime.append(order.traveltime)

            self.res['waitingTime'] = np.mean(self.waitingtime)
            self.res['traveltime'] = np.mean(self.traveltime)
            self.res['detour_distance'] = np.mean(self.detour_distance)

            # for vehicle
            self.res['pickup_time'] = np.mean(self.pickup_time)
            self.res['dispatch_time'] = np.mean(self.dispatch_time)
            self.res['shared_distance'] = np.mean(self.shared_distance)

            self.res['platform_income'] = np.mean(self.platflorm_income)
            self.res['response_rate'] = len(
                self.his_order) / len(self.order_list)

            return reward, True
        else:
            # print('当前episode仿真时间:',time_)
            # 判断智能体是否能执行动作
            for vehicle in self.vehicle_list:
                # print('vehicle.activate_time',vehicle.activate_time)
                # print('vehicle.state',vehicle.state)
                vehicle.is_activate(time_)
                vehicle.reset_reposition()
                # if vehicle.target == 0 : # 能执行动作
                #     print('当前episode仿真时间:',time_)
                #     print('id{},vehicle.activate_time{}'.format(vehicle.id, vehicle.activate_time))
                #     print('激活时间{}'.format(vehicle.activate_time - time_))
                if vehicle.state == 1:  # 能执行动作
                    # print('id{},vehicle.target{}'.format(vehicle.id, vehicle.target))
                    if vehicle.target == 0:
                        vehicles.append(vehicle)
                    else:
                        # print('id{},vehicle.target{}'.format(vehicle.id, vehicle.target))
                        takers.append(vehicle)
            # print('len(vehicles)',len(vehicles),'len(takers)', len(takers))
            start = time.time()
            reward = self.batch_matching(takers, vehicles, seekers)
            end = time.time()
            print('匹配用时{}'.format(end- start))
            # for vehicle in self.vehicle_list:
            #     print('vehicle.activate_time',vehicle.activate_time)
            #     print('vehicle.target',vehicle.target)
            self.total_reward += reward

            return reward,  False

    # 匹配算法
    def batch_matching(self, takers, vehicles, seekers):
        import time
        start = time.time()
        step_rewrad = 0
        # 构造权重矩阵
        demand = len(seekers)
        supply = len(takers) + len(vehicles)
        row_nums = demand + supply  # 加入乘客选择wait
        column_nums = demand + supply  # 加入司机选择wait
        # print('row_nums,column_nums ',row_nums,column_nums )
        dim = max(row_nums, column_nums)
        matrix = np.zeros((dim, dim))

        # 从司机角度计算响应乘客的权重
        for row in range(supply):
            for column in range(demand):

                if row < len(takers):
                    matrix[row, column] = self.calTakersWeights(takers[row], seekers[column],
                                                                optimazition_target=self.optimazition_target,
                                                                matching_condition=self.matching_condition)

                else:
                    matrix[row, column] = self.calVehiclesWeights(vehicles[row - len(takers)], seekers[column],
                                                                  optimazition_target=self.optimazition_target,
                                                                  matching_condition=self.matching_condition)

        # 计算司机选择调度的权重
        for row in range((row_nums - 1)):
            matrix[row, column_nums - 1] = 0

        # 计算乘客选择等待的权重
        for column in range(len(seekers)):
            for row in range(len(takers) + len(vehicles), supply):
                matrix[row, column] = \
                    self.calSeekerWaitingWeights(seekers[column],
                                                 optimazition_target=self.optimazition_target)

        end = time.time()
        print('构造矩阵用时',end-start)
        # print(matrix)

        # 匹配
        if demand == 0 or supply == 0:
            self.remain_seekers = []
            for seeker in seekers:
                if seeker.response_target == 0 and (self.time - seeker.begin_time_stamp) < 600:
                    seeker.set_delay(self.time)
                    self.remain_seekers.append(seeker)
            return 0

        import time
        start = time.time()
        matcher = KM_method.KM_method(matrix)
        res, weights = matcher.run()
        end = time.time()

        # print(res)
        failed = 0
        successed = 0
        for i in range(len(takers)):
            #  第i个taker响应第res[1][i]个订单
            if res[i] >= len(seekers):
                # 接到了虚拟订单，taker应该进入下一个匹配池
                takers[i].reposition_target = 1
                # print('taker{}拼车失败，进入匹配池'.format(
                #     takers[i].id))
                failed += 1
            else:
                takers[i].order_list.append(seekers[res[i]])
                self.his_order.append(seekers[res[i]])
                takers[i].reward += matrix[i, res[i]]
                step_rewrad += matrix[i, res[i]]
                print('taker{}拼车成功，权重为{}'.format(
                    takers[i].id, matrix[i, res[i]]))
                successed += 1
                # 记录seeker等待时间
                seekers[res[i]].set_waitingtime(
                    self.time - seekers[res[i]].begin_time_stamp)
                seekers[res[i]].response_target = 1

        for i in range(len(vehicles)):
            #  第i个vehicle响应第res[1][i]个订单
            if res[i + len(takers)] >= len(seekers):
                # 接到了虚拟订单，taker应该进入下一个匹配池
                vehicles[i].reposition_target = 1
                # print('vehicle id{}'.format(vehicles[i].id))
                # print('vehicle{}拼车失败，进入匹配池'.format(
                #     vehicles[i].id))
                failed += 1

            else:
                # print('vehicle id{},order id{}'.format(vehicles[i].id, seekers[res[i + len(takers)]].id))
                vehicles[i].order_list.append(seekers[res[i + len(takers)]])
                self.his_order.append(seekers[res[i + len(takers)]])
                vehicles[i].reward += matrix[i +
                                             len(takers), res[i + len(takers)]]
                print('vehicles{}拼车成功，权重为{}'.
                      format(vehicles[i].id, matrix[i + len(takers), res[i + len(takers)]]))
                successed += 1
                step_rewrad += matrix[i, res[i + len(takers)]]
                # 记录seeker等待时间
                seekers[res[i + len(takers)]].set_waitingtime(self.time -
                                                              seekers[res[i + len(takers)]].begin_time_stamp)
                seekers[res[i + len(takers)]].response_target = 1

        print('匹配时间{},匹配成功{},匹配失败{},takers{},vehicles{},demand{},time{}'.
              format(end-start, successed, failed, len(takers) , len(vehicles), len(seekers), self.time_slot))
        start = time.time()
        # 更新位置
        for taker in takers:
            if taker.reposition_target == 1:
                # print('taker repository了')
                continue

            if len(taker.order_list) > 1:

                # 接驾时间
                pickup_distance = self.network.get_path(
                    taker.order_list[1].O_location, taker.location)
                pickup_time = Config.unit_driving_time * pickup_distance

                self.pickup_time.append(pickup_time)

                # 决定派送顺序，是否fifo
                fifo, distance = self.is_fifo(
                    taker.order_list[0], taker.order_list[1])
                if fifo:
                    # 先上先下
                    p0_invehicle = pickup_distance + distance[0]
                    p0_expected_distance = taker.order_list[0].shortest_distance
                    # 绕行
                    taker.order_list[0].set_detour(
                        p0_invehicle - p0_expected_distance)
                    p1_invehicle = sum(distance)
                    p1_expected_distance = taker.order_list[1].shortest_distance
                    taker.order_list[1].set_detour(
                        p1_invehicle - p1_expected_distance)
                    # travel time
                    taker.order_list[0].set_traveltime(
                        Config.unit_driving_time * p0_invehicle)
                    taker.order_list[1].set_traveltime(
                        Config.unit_driving_time * p1_invehicle)

                else:
                    # 先上后下
                    po_invehicle = pickup_distance + sum(distance)
                    po_expected_distance = taker.order_list[0].shortest_distance
                    taker.order_list[0].set_detour(
                        po_invehicle - po_expected_distance)
                    taker.order_list[1].set_detour(0)

                # 计算司机完成两个订单需要的时间
                dispatching_time = pickup_time + \
                    Config.unit_driving_time * sum(distance)
                self.dispatch_time.append(dispatching_time)

                # 计算平台收益
                self.platflorm_income.append(
                    Config.unit_distance_value/1000 * sum(distance))

                # 计算拼车距离
                self.shared_distance.append(distance[0])

                # 完成该拼车过程所花费的时间
                time_consume = dispatching_time
                # 更新智能体可以采取动作的时间
                taker.activate_time += time_consume
                # print('拼车完成，activate_time:{}'.format(taker.activate_time - self.time))
                # 更新智能体的位置
                taker.location = taker.order_list[1].D_location
                # 完成订单
                taker.order_list = []
                taker.target = 0  # 变成vehicle
                taker.reward = 0

            else:
                # 派送时间
                dispatching_time = Config.unit_driving_time * \
                    (self.network.get_path(
                        taker.order_list[0].O_location, taker.order_list[0].D_location))
                if self.time >= taker.activate_time + dispatching_time:
                    # 全程没拼到车，单独走到了终点
                    self.dispatch_time.append(dispatching_time)
                    self.platflorm_income.append(Config.unit_distance_value/1000 *
                                                 self.network.get_path(taker.order_list[0].O_location, taker.order_list[0].D_location))
                    self.shared_distance.append(0)

                    # 更新智能体可以采取动作的时间
                    taker.activate_time += dispatching_time
                    # print('没有拼到车，activate_time:{}'.format(taker.activate_time - self.time))
                    # 更新智能体的位置
                    taker.location = taker.order_list[0].D_location
                    # 完成订单
                    taker.order_list = []
                    taker.target = 0  # 变成vehicle
                    taker.reward = 0

        for vehicle in vehicles:
            if vehicle.reposition_target == 1:
                # print('vehicle repository了')
                continue

            vehicle.target = 1  # 变成taker
            pickup_time = Config.unit_driving_time * \
                self.network.get_path(
                    vehicle.location, vehicle.order_list[0].O_location)
            vehicle.origin_location = vehicle.location

            vehicle.location = vehicle.order_list[0].O_location
            self.pickup_time.append(pickup_time)
            vehicle.activate_time += pickup_time

        end = time.time()
        print('派送用时{},takers{},vehicles{}'.format(end-start, len(takers),len(vehicles)))

        self.remain_seekers = []
        for seeker in seekers:
            if seeker.response_target == 0 and (self.time - seeker.begin_time_stamp) < 600:
                seeker.set_delay(self.time)
                self.remain_seekers.append(seeker)

        return step_rewrad

    def calTakersWeights(self, taker, seeker,  optimazition_target, matching_condition):
        if optimazition_target == 'platform_income':
            dispatch_distance = self.network.get_path(
                seeker.O_location, seeker.D_location)
            pick_up_distance = self.network.get_path(
                seeker.O_location, taker.location)
            if matching_condition and (pick_up_distance > self.pickup_distance_threshold):
                # print('taker pick_up_distance not pass', pick_up_distance)
                return -1000
            else:
                # print('taker pick_up_distance', pick_up_distance,'dispatch_distance',dispatch_distance)
                reward = Config.unit_distance_value/1000 * dispatch_distance * seeker.delay
                return reward

        else:  # expected shared distance
            pick_up_distance = self.network.get_path(
                seeker.O_location, taker.location)
            fifo, distance = self.is_fifo(taker.order_list[0], seeker)
            if fifo:
                shared_distance = self.network.get_path(
                    seeker.O_location, taker.order_list[0].D_location)
            else:
                shared_distance = seeker.shortest_distance
            detour_distance = sum(distance) - seeker.shortest_distance
            if matching_condition and (pick_up_distance > self.pickup_distance_threshold or
                                       detour_distance > self.detour_distance_threshold):
                # print('detour_distance not pass', detour_distance)
                return -1000
            else:
                reward = shared_distance * seeker.delay

                return reward

    def calVehiclesWeights(self, vehicle, seeker,  optimazition_target, matching_condition):
        if optimazition_target == 'platform_income':
            dispatch_distance = self.network.get_path(
                seeker.O_location, seeker.D_location)
            pick_up_distance = self.network.get_path(
                seeker.O_location, vehicle.location)
            if matching_condition and (pick_up_distance > Config.pickup_distance_threshold
                                       or dispatch_distance - pick_up_distance < 0):
                # print('vehicle pick_up_distance not pass', pick_up_distance)
                return 0
            else:
                reward = Config.unit_distance_value/1000 * \
                    (dispatch_distance - pick_up_distance) * seeker.delay
                return reward

        else:  # expected shared distance
            start = time.time()
            pick_up_distance = self.network.get_path(
                seeker.O_location, vehicle.location)
            end = time.time()
            print('计算最短路时间',end -  start)
            if matching_condition and pick_up_distance > Config.pickup_distance_threshold:
                return -1000
            else:
                reward = seeker.est
                return reward

    # 计算乘客选择等待的权重
    def calSeekerWaitingWeights(self, seeker,  optimazition_target):
        if optimazition_target == 'platform_income':
            # 不可行
            return seeker.delay

        else:  # expected shared distance
            reward = seeker.es - 10 * seeker.delay
            return reward

    def is_fifo(self, p0, p1):
        fifo = [self.network.get_path(p1.O_location, p0.D_location),
                self.network.get_path(p0.D_location, p1.D_location)]
        lifo = [self.network.get_path(p1.O_location, p1.D_location),
                self.network.get_path(p1.D_location, p0.D_location)]
        if fifo < lifo:
            return True, fifo
        else:
            return False, lifo

    def save_metric(self, path="output/system_metric.pkl"):
        dic = {}
        dic['pickup_time'] = self.pickup_time
        dic['detour_distance'] = self.detour_distance
        dic['traveltime'] = self.traveltime
        dic['waiting_time'] = self.waitingtime

        dic['dispatch_time'] = self.dispatch_time
        dic['platflorm_income'] = self.platflorm_income
        dic['shared_distance'] = self.shared_distance
        dic['response_rate'] = self.res['response_rate']

        import pickle

        with open(path, "wb") as tf:
            pickle.dump(dic, tf)
