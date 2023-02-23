'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-07-03 09:04:17
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-07-16 17:11:57
FilePath: /matching/reinforcement learning/Seeker.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import time

class Seeker():

    def __init__(self,index, row ) -> None:

        self.id = index
        self.begin_time = row['dwv_order_make_haikou_1.arrive_time']
        self.O_location = row.O_location
        self.D_location = row.D_location
        self.begin_time_stamp = time.mktime(time.strptime\
            (self.begin_time, "%Y-%m-%d %H:%M:%S"))
        self.matching_prob = row.matching_prob
        self.ls = row.ride_distance
        self.detour_distance = row.detour_distance
        self.es = row.shared_distance
        self.lst = row.ride_distance_for_taker
        self.detour_distance_for_taker = row.detour_distance_for_taker
        self.est = row.shared_distance_for_taker
        self.service_target = 0
        self.detour = 0
        self.shortest_distance = 0
        self.traveltime = 0
        self.waitingtime = 0
        self.delay = 1
        self.response_target = 0
        
    def set_delay(self, time):
        tmp = (time - self.begin_time_stamp) % 60 
        self.delay = 0.9 ** tmp

        
    def set_shortest_path(self,distance):
        self.shortest_distance = distance

    def set_waitingtime(self, waitingtime):
        self.waitingtime = waitingtime

    def set_traveltime(self,traveltime):
        self.traveltime = traveltime

    def set_detour(self,detour):
        self.detour = detour

    def cal_expected_ride_distance_for_wait(self, gamma):
        self.shared_distance  =self.shared_distance * gamma
