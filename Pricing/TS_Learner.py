#Bandit Algorithm
#Thompson Sampling

from turtle import pu
from Learner import * 
import numpy as np

class TS_Learner(Learner):
  """ 
  the TS Algorithm selects the arm to pull by sampling a value for each arm from a beta distribution, 
  then select the arm associated to the beta distribution that generated the sample with the maximum value.
  """
  def __init__(self, n_arms):
    """
    size of np.array = n_arms
    beta parameter defines by (size of np.array*2)
    """
    super().__init__(n_arms) #using Learner class' attribute (n_arms)
    self.beta_parameters = np.ones((n_arms, 2)) #value to store beta paramaters, see thompsonn technicalities
    

  def pull_arm(self):
    #method to select which arm at each round t.
    #argmax to select the index of the maximum value
    #alpha and beta of beta distribution
    idx = np.argmax(np.random.beta(self.beta_parameters[:,0], self.beta_parameters[:,1]))
    return idx

  def update(self, pulled_arm, reward, margin):
    self.t += 1 #update the round
    self.update_observations(pulled_arm, reward, margin) #update the list value of the collected reward
    self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward #success status, update the alpha beta parameters
    self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + (1.0 - reward) #fail status, update the alpha beta parameters