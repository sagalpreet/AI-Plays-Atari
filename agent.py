import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import copy

class approximator(nn.Module):
    def __init__(self):
        super().__init__()
        self.act1 = F.leaky_relu
        self.layer1 = nn.Linear(129, 32) # hidden layer 1
        self.act2 = F.leaky_relu
        self.layer2 = nn.Linear(32, 32)  # hidden layer 2
        self.layer3 = nn.Linear(32, 1)   # output layer
        self.loss = F.mse_loss           # loss function
    
    def forward(self, inputs):
        out = self.act1(self.layer1(inputs))
        out = self.act2(self.layer2(out))
        out = self.layer3(out)
        return out
    
    def train(self, inputs, targets): 
        self.opt = torch.optim.SGD(self.parameters(), lr = 1e-8)
        num_epochs = 1
        for epoch in range(num_epochs):
            out = self(inputs)
            targets.reshape(1, 1)
            error = self.loss(out, targets)
            error.backward()
            self.opt.step()
            self.opt.zero_grad()      

class Agent:
    def __init__(self, env):
        self.experience = deque([], 1000)
        
        self.q_values = approximator()
        self.q_values_ = approximator()
    
        self.cnt = 0
        self.env = env
        self.actions = [action for action in range(env.env.action_space.n)]
        print(f'actions: {self.actions}')
        self.epsilon = 1
        self.decay = 1e-5
        self.gamma = 1
        
    def select_action(self, state):
        action = -1
        
        if (np.random.random() < self.epsilon):
            action = np.random.choice(self.actions)
            return action
        
        values = []
        for action in self.actions:
            inp = torch.tensor(np.append(state, [action])).float().reshape(1, -1)
            value = float(self.q_values(inp))
            values.append((value, action))
        values.sort(reverse = True)
        return values[0][1]
    
    def max_value(self, state):
        ans = float('-inf')
        for action in self.actions:
            inp = torch.tensor(np.append(state, [action])).float().reshape(1, -1)
            ans = max(ans, float(self.q_values_(inp)))
        return ans
    
    def take_action(self, state):
        self.cnt += 1
        action = self.select_action(state)
        next_state, reward, done, info = self.env.step(action)
        rtn = (next_state, reward, done)
        self.experience.append([(state, action), next_state, reward, done])
        
        if (len(self.experience) > 100):
            for (state, action), next_state, reward, done in self.experience:
                if (np.random.random() > 0.1):
                    continue
                y = reward if done else (reward + self.gamma * self.max_value(next_state))
                
                inp = torch.tensor(np.append(state, action)).float().reshape(1, -1)
                tar = torch.tensor([[y]]).float()
                
                self.q_values.train(inp, tar)
        
        if (self.cnt == 100):
            self.cnt = 0
            self.q_values_ = copy.deepcopy(self.q_values)
            self.epsilon -= self.decay
            self.epsilon = max(1e-4, self.epsilon)
        
        return rtn # state, reward, done
