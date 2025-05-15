import torch
import random
import numpy as np
from collections import deque
from model import DuelingDQN

class Agent:
    def __init__(self, state_dim, action_dim, lr=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.eps = 1.0
        self.eps_decay = 0.995
        self.eps_min = 0.1

    def select_action(self, state):
        if random.random() < self.eps:
            return random.randint(0, 4)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.q_net(state).argmax().item()

    def store(self, s, a, r, s_, d):
        self.replay_buffer.append((s, a, r, s_, d))

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        s, a, r, s_, d = zip(*batch)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        s_ = torch.FloatTensor(s_).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        q = self.q_net(s).gather(1, a)
        a_prime = self.q_net(s_).argmax(1).unsqueeze(1)
        q_target = r + self.gamma * self.target_net(s_).gather(1, a_prime) * (1 - d)
        loss = (q - q_target.detach()).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self, tau=0.01):
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
