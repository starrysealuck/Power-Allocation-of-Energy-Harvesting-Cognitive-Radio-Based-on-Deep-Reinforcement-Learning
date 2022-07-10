import torch
import numpy as np
import torch.nn as nn
import random
import matplotlib.pyplot as plt
class Config():
    def __init__(self):
        self.input_size=6
        self.hidden1_size=128
        self.hidden2_size=64
        self.output_size=22
        self.dicount_rate=0.99
        self.lamba=3e-1
        self.proability_discont_rate=1e-8
        self.slow_update_rate=200
        self.learning_rate=0.0004
        self.batch_size=128
class Q_net(nn.Module):
    def __init__(self):
        super(Q_net, self).__init__()
        self.config=Config()
        self.linear1=nn.Linear(self.config.input_size,self.config.hidden1_size)
        self.linear2=nn.Linear(self.config.hidden1_size,self.config.hidden2_size)
        self.linear3=nn.Linear(self.config.hidden2_size,self.config.hidden2_size)
        self.linear4=nn.Linear(self.config.hidden2_size,self.config.output_size)
        self.f1=nn.ReLU()
        self.f2=nn.Tanh()
        for m in self.modules():
            if isinstance(m,nn.Linear):
                m.weight.data.uniform_(-0.001,0.001)

    def forward(self,x):
        x=self.f2(self.linear1(x))
        x=self.f1(self.linear2(x))
        x=self.f1(self.linear3(x))
        x=self.linear4(x)
        return x
class Double_DQN():
    def __init__(self,env):
        self.config_1=Config()
        self.updata_Q=Q_net()
        self.target_Q=Q_net()
        # 复制参数到目标网络
        for target_param, param in zip(self.target_Q.parameters(), self.updata_Q.parameters()):
            target_param.data.copy_(param.data)
        #定义优化器，损失函数
        self.optim=torch.optim.Adam(self.updata_Q.parameters(),lr=self.config_1.learning_rate)
        self.loss_f=nn.MSELoss()
        self.repay_buff=[]
        self.volume_buff=100000
        self.env=env
    def buff(self,experience):
        if len(self.repay_buff)<100000:
            self.repay_buff.append(experience)
        else:
            self.repay_buff.pop(0)
            self.repay_buff.append(experience)
    def sample_and_transform(self):
        train_data = random.sample(self.repay_buff, 100)
        train_state = np.array([train_[0] for train_ in train_data])  # numpy转为张量速度快
        train_action = np.array([[train_[1]] for train_ in train_data])
        train_reward = np.array([[train_[2]] for train_ in train_data])
        train_nextstate = np.array([train_[3] for train_ in train_data])
        train_done = torch.Tensor([[train_[4]] for train_ in train_data])  # Tensor默认数据类型为float型
        # 处理数据
        train_state = torch.from_numpy(train_state).float()  # 可以这样改变数据类型
        train_action = torch.from_numpy(train_action).float()
        train_reward = torch.from_numpy(train_reward).float()
        train_nextstate = torch.from_numpy(train_nextstate).float()
        return train_state,train_action,train_reward,train_nextstate,train_done
    def transform_action_env(self,action):
        action=action.detach().numpy()
        if action<=10:
            for i in range(11):
                if action==i:
                    action=np.array([0,i*0.01])
                    break
        else:
            action-=11
            for i in range(11):
                if action==i:
                    action=np.array([1,i*0.01])
                    break
        return action
    def transform_action_net(self,action):
        if action[0]==0:
            action_net=action[1]*100
        else:
            action_net=11+action[1]*100
        return action_net

    def greedy_action(self,state):
        if np.random.rand(1)>self.config_1.lamba:
            q_value=self.updata_Q(state)
            action_net=torch.argmax(q_value)#返回最大值索引
            action_env=self.transform_action_env(action_net)
            action_net=action_net.detach().numpy()

        else:
            action_env=self.env.disposal_action()
            action_net=self.transform_action_net(action_env)
        self.config_1.lamba-=self.config_1.proability_discont_rate
        return action_env,action_net
    def train(self):
        num=0
        state=self.env.reset()
        action_env=0
        action_net=0
        epoch=3000
        rewards=0
        ave_Reward=np.array([])
        average_reward=0
        for i in range(epoch):
            while True:
                num += 1
                if len(self.repay_buff)<=10000:
                    action_env=self.env.disposal_action()
                    action_net=self.transform_action_net(action_env)
                else:
                    action_env,action_net=self.greedy_action(torch.tensor(state))
                    #print(action_env)
                next_state,reward,done=self.env.step(action_env)
                rewards+=reward
                #print(action_env,  reward)
                self.buff([state,action_net,reward,next_state,done])
                if len(self.repay_buff)>10000:#开始训练
                    self.optim.zero_grad()
                    train_state,train_action,train_reward,train_nextstate,train_done=self.sample_and_transform()
                    state_q=self.updata_Q(train_state)
                    action_q=torch.gather(state_q,1,train_action.long().data)
                    #print(action_q)
                    next_state_q=self.updata_Q(train_nextstate)
                    action_max_q_index=torch.argmax(next_state_q,dim=1,keepdim=True)
                    true_nextstate_q=self.target_Q(train_nextstate)
                    true_nextstate_q_max=torch.gather(true_nextstate_q,1,action_max_q_index.long().data)
                    predict_q=train_reward+self.config_1.dicount_rate*(1-train_done.data)*true_nextstate_q_max
                    loss=self.loss_f(predict_q,action_q)
                    loss.backward()
                    self.optim.step()
                if num==self.config_1.slow_update_rate:
                    for target_param, param in zip(self.target_Q.parameters(),self.updata_Q.parameters()):  # 复制参数到目标网路targe_net
                        target_param.data.copy_(param.data)
                    num=0
                if done:
                    state=self.env.reset()
                    average_reward=0.99*average_reward+0.01*rewards
                    ave_Reward=np.append(ave_Reward,average_reward)
                    print('第%d回合  总奖励%.4f' %(i,rewards))
                    print('滑动平均奖励%.4f' %(average_reward))
                    rewards=0
                    break
                state=next_state
        plt.plot(range(epoch), ave_Reward)
        plt.xlabel('epoch')
        plt.ylabel('average reward')
        plt.show()




