# Based in part on the DDPG algorithm as provided by Udacity's DRL course.

import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

from noise import OUNoise, GaussianNoise, GeometricBrownianNoise

from memory import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 5e-4         # learning rate of the actor
LR_CRITIC = 5e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0      # L2 weight decay
UPDATE_RATE = 1         # learn after #UPDATE_RATE steps

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, number_agents, random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            number_agents (int): number of agents
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.number_agents = number_agents

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise processes
        self.noise = OUNoise((number_agents, action_size), random_seed)
        #self.noise = GaussianNoise(size=[number_agents,action_size], seed = 0,sigma=2e-1)
        #self.noise = GeometricBrownianNoise(size=[number_agents,action_size], seed = 0,sigma=2e-1)
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, state, action, reward, next_state, done):
        """Save experiences in replay memory, and use random sample from buffer to learn."""
        
        # We save experience tuples in the memory for each agent.
        for i in range(self.number_agents):
            self.memory.add(state[i, :], action[i, :], reward[i], next_state[i, :], done[i])
        
        # Learn, if enough samples are available in memory (threshold value: BATCH_SIZE) and at learning interval settings
        if len(self.memory) > BATCH_SIZE:
            for _ in range(UPDATE_RATE):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

#     def act(self, states, add_noise=True):
#         """Returns actions for given state as per current policy."""
#                                                                   # The code has been adapted to implement batch normalization.
#         actions = np.zeros((self.number_agents, self.action_size))
#         self.actor_local.eval()
#         with torch.no_grad():
#             for agent_number, state in enumerate(states): 
#                 state = torch.from_numpy(state).float().unsqueeze(0).to(device)   # The code has been adapted to implement batch normalization.
#                 action = self.actor_local(state).cpu().data.numpy()
#                 actions[agent_number, :] = action
#         self.actor_local.train()
#         if add_noise:
#             actions += self.noise.sample()
#         return np.clip(actions, -1, 1)

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)
        actions = np.zeros((self.number_agents, self.action_size))
        self.actor_local.eval()
        with torch.no_grad():
            for agent_number, state in enumerate(states):
                action = self.actor_local(state).cpu().data.numpy()
                actions[agent_number, :] = action
        self.actor_local.train()
        if add_noise:
            actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
