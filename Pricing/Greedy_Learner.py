#Greedy Algorithm

from turtle import pu
from Learner import * 
import numpy as np

class Greedy_Learner(Learner):
  """
  The Greedy Learner selects the arm to pull by maximizing the expected reward array
  """
  def __init__(self, n_arms):
    super().__init__(n_arms) #using Learner class' attribute (n_arms)
    self.expected_rewards = np.zeros(n_arms) #return array filled with zeros with the shape of n_arms

  def pull_arm(self):
    if(self.t < self.n_arms): #to make sure every arm pulled at least once
      return self.t
    idxs = np.argwhere(self.expected_rewards == self.expected_rewards.max()).reshape(-1) #the numpy's argwhere and max for maximized reward  
    pulled_arm = np.random.choice(idxs) #because idxs can return more than one value, then np.random is used to pick the random value from the selection
    return pulled_arm

  def update(self, pulled_arm, reward, margin):
    self.t += 1 #update round
    self.update_observations(pulled_arm, reward, margin) #superclass learner's method
    self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm] * (self.t - 1) + reward) / self.t #update the reward array