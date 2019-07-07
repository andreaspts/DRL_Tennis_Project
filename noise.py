# Based in part on the DDPG algorithm as provided by Udacity's DRL course.
# classes for different noises

import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class OUNoise:
    """Ornstein-Uhlenbeck process. In financial mathematics this is also known as the Vasicek model of interest rates."""
    

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()
        print("\n\n~~~OU NOISE RUNNING~~~\n\n")

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
    
class GaussianNoise():
    """Statistical noise generated through a Gaussian distribution: https://en.wikipedia.org/wiki/Gaussian_noise."""
    def __init__(self,size,seed,sigma=0.05,mu=0):
        self.size = size
        self.sigma=sigma
        self.mu=mu
        self.seed = random.seed(seed)
        print("\n\n~~~GAUSSIAN NOISE RUNNING~~~\n\n")
     
    def reset(self):
        pass
    
    def sample(self):
        return np.random.normal(self.mu,self.sigma,self.size)

    
class GeometricBrownianNoise():
    """Noise generated according to a geometric brownian motion: https://en.wikipedia.org/wiki/Geometric_Brownian_motion."""
    def __init__(self, size, seed=0,sigma=0.2, mu=0):
        self.mu = mu * np.ones(size)
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()
        print("\n\n~~~GEOMETRIC BROWNIAN NOISE RUNNING~~~\n\n")
        
    def reset(self):
        """Update internal state and return it as a noise sample. Include arbitrary initial condition."""
        self.state = copy.copy(self.mu) #+ 1e-9 * np.random.normal(self.mu,self.sigma,self.size)
    
    def sample(self):
        x = self.state
        dx = self.mu * x + self.sigma * self.state * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state