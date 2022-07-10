import torch
import torch.nn as nn
import numpy as np
import random
from torch.autograd import Function
import os
#DDPG超参数
class config():
    def __init__(self):
        # 模型中的超参数
        self.repay_buff_size = 1000000
        self.gamma = 0.99
        self.batch_size = 128
        self.actor_input_size = 6
        self.critic_input_size = 8
        self.actor_hidden = 256
        self.critic_hidden=128
        self.actor_output = 1
        self.critic_output=1
        self.soft_updata = 0.01
        self.actor_lr =0.00008
        self.critic_lr = 0.0002
        self.slow_update=5
        # 高斯噪声的超参数
        self.std = 0.1
        self.mu = 0
        #self.decay_rate = 1e-6#噪声衰减
        #偏置初始化
        self.init_w = 1e-3
#自定义自动求导，解决torch.where梯度不能反向传播的问题
'''
class my_backward(Function):
    @staticmethod
    def forward(ctx,primitive_action):
        ctx.save_for_backward(primitive_action)
        true_anction=torch.where(primitive_action>=0.5,1.0,0.0)
        return true_anction
    @staticmethod
    def backward(ctx,grad_output):
        return grad_output
#policy_net反向传播
'''
#注册钩子

def grad_hook(grad):
    grad+=1
    return grad

#搭建Actor
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.config = config()
        self.linear1=nn.Linear(self.config.actor_input_size,self.config.actor_hidden)
        self.linear2=nn.Linear(self.config.actor_hidden,self.config.actor_hidden)
        self.linear3=nn.Linear(self.config.actor_hidden,self.config.actor_output)
        self.linear4=nn.Linear(self.config.actor_hidden,self.config.actor_output)
        self.linear4.weight.data=nn.init.xavier_normal_(self.linear4.weight.data,gain=nn.init.calculate_gain('sigmoid'))
        self.linear4.bias.data.uniform_(-0.01, 0.01)
        self.linear4.weight.requires_grad=True
        self.linear4.weight.retain_grad()
        self.action1=0
        self.action2=0
        self.a0=0
        self.f1=nn.ReLU()
        self.f2=nn.Tanh()
        self.f3=nn.Sigmoid()
        self.count=0
        self.x=0
        for m in self.modules():
            if isinstance(m,nn.Linear)  and self.count<2:
                m.weight.data=nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.bias.data.uniform_(-0.001,0.001)

            elif self.count==2:
                m.weight.data=nn.init.xavier_normal_(m.weight.data,gain=nn.init.calculate_gain('tanh'))
                m.bias.data.uniform_(-0.001,0.001)
            self.count+=1

    def forward(self,x):
        x=self.f1(self.linear1(x))
        x=self.f1(self.linear2(x))
        self.a0=self.f3(self.linear4(x))
        a1=(self.f2(self.linear3(x))+1)*0.01/2
        #   self.a0.retain_grad()
        #print(a0.data)
        #print(self.a0.data,a1.data)
        #print(x.data)
        #print(a0.is_leaf)
        if len(self.a0.shape)==2:
            self.action1= torch.clamp(a1, 0, 0.01)#.view(self.config.batch_size,1)
            self.action2=self.a0.clone()#torch.where(a0>=0.5,1.0,0.0)#.view(self.config.batch_size,1)#torch.where把action2变为叶子节点,使梯度计算错误
            for i in range(len(self.a0)):
                if self.a0[i,0]>=0.5:
                    self.action2[i,0]=1.0
                else :
                    self.action2[i,0]=0.0
            #self.action2.retain_grad()

            #print(self.action2.is_leaf)
            #self.action1.requires_grad = True
            #self.action1.retain_grad()
            #self.action2.requires_grad=True
            #self.action2.retain_grad()
           # '''
           # for i in range(len(x)):#不能直接原地操作改变内存中的值，要进行复制
               # if x[i,0]>=0.5:
                 #  x[i,0]=1
               # else:
                   # x[i,0]=0

        else:
            #x=x.view(1,2)
            self.action1 = torch.clamp(a1, 0, 0.1).view(1,1)
            self.action2 =torch.where(self.a0 >= 0.5, 1.0, 0.0).view(1,1)

            #print(action2,action1)
            '''
           # if x[0]>=0.5:
               # x[0]=1
           # else :
               # x[0]=0
            '''
        #print(x)
        return torch.cat((self.action2, self.action1), 1)
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.config = config()
        self.linear1=nn.Linear(self.config.critic_input_size,self.config.critic_hidden)
        self.linear1.weight.data = nn.init.kaiming_normal_(self.linear1.weight.data, a=0, mode='fan_in',
                                                           nonlinearity='relu')
        self.linear1.weight.data.uniform_(-0.01,0.01)
        self.linear2=nn.Linear(self.config.critic_hidden,self.config.critic_output)
        self.linear2.weight.data.normal_(0,self.config.critic_hidden**0.5)
        self.linear2.weight.data.uniform_(-0.01,0.01)
        self.f1=nn.ReLU()
    def forward(self,x):
        x=self.f1(self.linear1(x))
        x=self.linear2(x)
        return x
class DDPG():
    def __init__(self):
        self.config=config()
        self.policy_net = Actor()
        self.target_policy_net = Actor()
        self.updata_Q = Critic()
        self.target_Q = Critic()
        # 定义误差函数，优化器
        self.loss_fun = nn.MSELoss(reduction='mean')
        self.Q_optim = torch.optim.Adam(self.updata_Q.parameters(), lr=self.config.critic_lr)
        self.policy_optim = torch.optim.Adam(self.policy_net.parameters(), lr=self.config.actor_lr)
        self.repay_buff_size = self.config.repay_buff_size
        self.mu = self.config.mu
        self.std = self.config.std
        #self.decay_rate = config.decay_rate
        self.soft_updata =self.config.soft_updata
        self.buff = []
        self.num = 0
        self.num1 = 0

        # 复制参数到目标网络
        for target_param, param in zip(self.target_Q.parameters(), self.updata_Q.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)
        # 文件保存路径
        #self.path = 'E:\python project\deep reinforce learning trial\checkpoint.pt'


    def noise(self, action):
        #action = action.detach().numpy()
        noises = np.random.normal(self.mu, self.std, 1)
        noises2=np.random.normal(0.3,0.5,1)
        noises2 = np.clip(noises2, -0.1,0.5)
        action0=action[0]+noises2
        action0=np.where(action0>=0.5,1.0,0.0)
        # self.std=self.std-self.decay_rate
        noises = np.clip(noises, -0.005, 0.005)
        action1=np.clip(action[1] + noises, 0, 0.01)
        return np.append(action0,action1)  # 区间剪裁

    def Buff(self, experience):
        if len(self.buff) < self.repay_buff_size:
            self.buff.append(experience)
        else:
            self.buff.pop(0)  # 在列表0索引处添加
            self.buff.append(experience)
    def convert_action(self,action):
        action=action.detach().numpy()
        for i in range(len(action)):
            if action[i,0]>=0.5:
                action[i,0]=1
            else:
                action[i,0]=0
        return torch.from_numpy(action).float()
    def updata(self):
        with torch.autograd.set_detect_anomaly(True):
            self.num += 1
            self.num1 += 1
            self.Q_optim.zero_grad()
            train_data = random.sample(self.buff, self.config.batch_size)
            state = np.array([exp[0] for exp in train_data])
            noise_action = np.array([exp[1] for exp in train_data])
            reward = np.array([[exp[2]] for exp in train_data])
            next_state = np.array([exp[3] for exp in train_data])
            done = np.array([[exp[4]] for exp in train_data])
            # 变为张量
            state = torch.from_numpy(state).float()
            noise_action = torch.from_numpy(noise_action).float()
            reward = torch.from_numpy(reward).float()
            next_state = torch.from_numpy(next_state).float()
            done = torch.from_numpy(done).float()
            critic_state_input = torch.cat((state, noise_action), 1)  # 相同数组拼接函数，dim=0表示行拼接，1表示列拼接
            next_action = self.target_policy_net(next_state)
            critic_nextstate_input = torch.cat((next_state, next_action), 1)
            true_action = self.policy_net(state)
            #true_action.retain_grad()
            print('动作',true_action.data)
            q_state_value = self.updata_Q(critic_state_input)
            #print(q_state_value.data)
            q_nextstate_value = self.target_Q(critic_nextstate_input)
            target_q = reward + (1 - done) * self.config.gamma * q_nextstate_value
            loss = self.loss_fun(target_q, q_state_value)
            loss.backward()
            #print(self.updata_Q.linear1.weight.grad)
            #print(loss.backward())
            self.Q_optim.step()
            # 更新策略网络
            if self.num1 ==self.config.slow_update:
                self.policy_optim.zero_grad()
                true_q_input = torch.cat((state, true_action), 1)
                true_q_value = self.updata_Q(true_q_input).mean()
                true_q_value = -true_q_value
                #self.policy_net.action2.requires_grad=True
                #self.policy_net.action2.retain_grad()
                # policy_net反向传播
                h = self.policy_net.a0.register_hook(grad_hook)
                true_q_value.backward()
                #print(self.policy_net.a0.grad)
                #print(self.policy_net.action2.grad)
                #print(true_action.grad)
                #self.policy_net.a0.backward_1(true_action.grad[:,0].view(128,1),retain_graph=True)
                #self.policy_net.a0.requires_grad=True
                #self.policy_net.a0.retain_grad()
                #self.policy_net.x.backward_1(self.policy_net.a0.grad,retain_graph=True)
                #print(self.policy_net.linear4.weight.grad)
                #print(self.policy_net.action2.is_leaf)
                #print(self.policy_net.action2.grad)
                #print('动作', true_action.grad)
                self.policy_optim.step()
                #true_action.zero_grad()
                #self.policy_net.a0.zero_grad()
                self.num1 = 0
            # 对网络进行软更新
            if self.num == 10:
                for target_param, param in zip(self.target_Q.parameters(), self.updata_Q.parameters()):
                    target_param.data.copy_(
                        target_param.data * (1.0 - self.config.soft_updata) +
                        param.data * self.config.soft_updata
                    )
                for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
                    target_param.data.copy_(
                        target_param.data * (1.0 - self.config.soft_updata) +
                        param.data * self.config.soft_updata
                    )
                self.num = 0
