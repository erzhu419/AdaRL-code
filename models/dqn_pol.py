import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

import gym_cartpole_world
import gym_pong


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, theta_size):
        super(QNetwork, self).__init__()
        input_dim = state_size + theta_size
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, action_size)

        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc4.weight, nonlinearity='linear')

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# Double DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DoubleDQNAgent:
    def __init__(self, state_size, action_size, theta_size, is_load=False, load_p=None):
        # if you want to see learning, then change to True
        self.render = False
        self.load_model = is_load
        if self.load_model:
            self.load_path = load_p
        # get size of state and action and also theta
        self.state_size = state_size
        self.action_size = action_size
        self.theta_size = theta_size

        # these is hyper parameters for the Double DQN
        self.discount_factor = 1
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.5
        self.batch_size = 640
        self.train_start = 5000
        self.update_target_steps = 10000
        # create replay memory using deque
        self.memory = deque(maxlen=200000)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # create main model and target model
        self.model = self.build_model().to(self.device)
        self.target_model = self.build_model().to(self.device)

        # optimizer and loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        # initialize target model
        self.update_target_model()

        if self.load_model:
            self.model.load_state_dict(torch.load(self.load_path, map_location=self.device))
            self.model.eval() # Optional depending on if this is pure eval, but typically okay

    def build_model(self):
        return QNetwork(self.state_size, self.action_size, self.theta_size)

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # get action from model using epsilon-greedy policy
    def get_action(self, state, theta, is_training=True):
        if is_training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state_all = np.concatenate((state, theta), axis=1)
            state_tensor = torch.FloatTensor(state_all).to(self.device)
            with torch.no_grad():
                q_value = self.model(state_tensor)
            return torch.argmax(q_value[0]).item()

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, theta, done, score):
        self.memory.append((state, action, reward, next_state, theta, done))
        if score >= 50 and score < 100:
            self.epsilon_min = 0.3
        if score >= 100 and score < 200:
            self.epsilon_min = 0.2
        if score >= 200 and score < 300:
            self.epsilon_min = 0.1
        if score >= 300 and score < 450:
            self.epsilon_min = 0.01
        if score >= 450:
            self.epsilon_min = 0.001
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        if len(self.memory) < self.train_start:
            return 0.0

        mini_batch = random.sample(self.memory, self.batch_size)

        update_input = np.zeros((self.batch_size, self.state_size + self.theta_size))
        update_target = np.zeros((self.batch_size, self.state_size + self.theta_size))
        actions = np.zeros(self.batch_size, dtype=int)
        rewards = np.zeros(self.batch_size)
        dones = np.zeros(self.batch_size, dtype=bool)

        for i in range(self.batch_size):
            update_input[i] = np.concatenate((mini_batch[i][0], mini_batch[i][4]), axis=1)
            actions[i] = mini_batch[i][1]
            rewards[i] = mini_batch[i][2]
            update_target[i] = np.concatenate((mini_batch[i][3], mini_batch[i][4]), axis=1)
            dones[i] = mini_batch[i][5]

        update_input_t = torch.FloatTensor(update_input).to(self.device)
        update_target_t = torch.FloatTensor(update_target).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device).unsqueeze(1) # shape: (batch_size, 1)
        rewards_t = torch.FloatTensor(rewards).to(self.device) # shape: (batch_size,)
        dones_t = torch.FloatTensor(dones).to(self.device) # shape: (batch_size,)

        # Current Q values
        self.model.train()
        q_values = self.model(update_input_t).gather(1, actions_t).squeeze(1)

        # Target Q values using Double DQN logic
        with torch.no_grad():
            next_actions = self.model(update_target_t).argmax(1).unsqueeze(1)
            target_val = self.target_model(update_target_t).gather(1, next_actions).squeeze(1)
            
            target_q_values = rewards_t + self.discount_factor * target_val * (1 - dones_t)

        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()
