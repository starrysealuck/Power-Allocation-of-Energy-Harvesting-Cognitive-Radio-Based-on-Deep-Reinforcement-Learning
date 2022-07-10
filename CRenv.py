import numpy as np
import gym
from gym import spaces
#建立CR环境
########################################
class CR(gym.Env):
    def __init__(self):
        #环境参数初始化
        self.max_time=20
        self.occupy_time=17
        self.fixed_power=0.1
        self.threshold_power=0.01
        self.max_battery=0.2
        self.count=0
        self.v=1e-3
        self.penalization=-10
        self.done=0
        #状态空间
        self.last_battery=0
        self.now_battery=0
        self.last_energy=0
        self.now_energy=0
        self.judge_para=0
        self.power_gains=np.array([0,0,0])
        #动作空间
        self.choose_para=0
        self.power=0
        self.action_space=spaces.Dict({'choose':spaces.Discrete(2),'power':spaces.Discrete(11)})
        self.action_space.seed(6)
    def disposal_action(self):
        action=self.action_space.sample()
        action=list(action.values())
        action[1]*=0.01
        return np.array(action)
    def __get_power_gains__(self):
        power_gain=np.random.exponential(0.1,size=(1,))
        power_gain = np.append(power_gain, power_gain)
        power_gain=np.append(power_gain,np.random.exponential(0.2,size=(1,)))#展平
        return power_gain
    #如果选择收集能量t时刻收集到的能量
    def __energy_harvest__(self):
        return np.random.uniform(0,self.max_battery)
    def __updata_battery__(self):
        return min(max(self.last_battery+self.choose_para*self.now_energy-(1-self.choose_para)*self.power,0),self.max_battery)
    #初始化化环境
    def reset(self):
        #np.random.seed(1029)
        self.count=1.0
        self.done=0
        self.judge_para=0
        self.last_battery,self.now_battery=0.0,0.0
        self.last_energy,self.now_energy=0.0,0.0
        self.power_gains=self.__get_power_gains__()
        obs=np.array([self.now_battery,self.last_energy,self.judge_para])
        obs=np.append(obs,self.power_gains)
        return obs.astype('float32')
    def get_reward(self):
        if self.choose_para==0 and self.judge_para==1 and self.power<=self.last_battery and self.power*self.power_gains[1]<=self.threshold_power and self.power>=0:
            return  np.log2(1 + (self.power * self.power_gains[2]) / (self.fixed_power * self.power_gains[0] + self.v**2))
        elif self.choose_para==0 and self.judge_para==0 and self.power<=self.last_battery and self.power>=0:
            return  np.log2(1+self.power*self.power_gains[2]/self.v**2)
        elif self.choose_para==1 and self.power>self.last_battery:
            return 0.0
        else:
            return self.penalization
    def render(self, mode="human"):
        return None
    def close(self):
        return None


    def step(self,action):
        #print(action)
        if self.count>self.occupy_time:
            self.judge_para=1
        self.choose_para=action[0]
        self.power=action[1]
        #进行动作
        reward=self.get_reward()
        if self.choose_para==1:
            self.now_energy=self.__energy_harvest__()
        else:
            self.now_energy=0
        #进入下一个状态
        #print('可用能量',self.last_battery)
        #print('能量收集',self.now_energy)
        self.last_energy=self.now_energy
        self.now_battery = self.__updata_battery__()
        self.last_battery=self.now_battery
        #print('电池', self.now_battery)
        self.power_gains=self.__get_power_gains__()
        obs = np.array([self.now_battery, self.last_energy,self.judge_para])
        obs = np.append(obs, self.power_gains)
        self.count+=1
        if self.count>self.max_time:
            self.done=1
        return obs.astype('float32'),reward,self.done







