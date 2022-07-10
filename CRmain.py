import torch
import numpy as np
import random
import os
from CRenv import CR
from DDPG_opt import DDPG

from Double_DQN_opt import Double_DQN
import matplotlib.pyplot as plt
def seed_fixed(seed):
    # 要固定这些随机种子结果才能相同
    # self.env.reset(seed=1029)  # 初始化环境时的随机种子
    # self.env.action_space.seed(1029)  # 动作采样的随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

'''   
def train(env,ddpg):
    epoch=10000
    rewards=0
    Reward=[]
    average_reward=0
    state=env.reset()
    state=state.astype('float32')
    for u in range(epoch):
        while True:
            action=ddpg.policy_net(torch.tensor(state))
            #print(ddpg.policy_net.state_dict()['linear1.weight'].data)
            #print(action)
            action = action.detach().numpy()
            action=action.flatten()
            #print(action)
            #print(type(action[1]))
            action=ddpg.noise(action)

            #print(action)
            next_state,reward,done=env.step(action)
            rewards+=reward
            #print('奖励',reward)
            ddpg.Buff([state, action, reward, next_state, done])
            if len(ddpg.buff)>=5000:#更新网络
                ddpg.updata()
            if done:
                state=env.reset()
                average_reward=0.9*average_reward+0.1*rewards
                Reward.append(rewards)
                print('第%d训练回合   总奖励为%.4f' %(u,rewards))
                rewards=0
                break
            state=next_state
    plt.plot(range(epoch), Reward)
    plt.xlabel('epoch')
    plt.ylabel('Reward')
    plt.show()



'''
if __name__=='__main__':

    seed_fixed(6)
    env=CR()
    double_dqn=Double_DQN(env)
    double_dqn.train()

