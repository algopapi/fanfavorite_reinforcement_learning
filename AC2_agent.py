import os
import gym
import numpy as np
import tensorflow as ts
import random
import pylab
import cv2

from keras.models import Model, load_model
from keras.layers import Flatten, Dense, Input
from keras.optimizers import Adam, RMSprop


# Policy
def Actor(action_space, observation_space, learning_rate):
  X_Input = Input(observation_space)
  X = Dense(512, activation = "relu", kernel_initializer="he_uniform")(X_Input)
  X = Dense(256, activation = "relu", kernel_initializer="he_uniform")(X)
  X = Dense(64,  activation = "relu", kernel_initializer="he_uniform")(X)

  value = Dense(1, activation = None)(X)
  actions = Dense(action_space, activation = "softmax", kernel_initializer="he_uniform")

  Actor = Model(inputs = X_Input, outputs = actions)
  Actor.compile(optimizer = RMSprop(lr = learning_rate), loss = "categorical_crossentropy")

  return Actor
  
def Critic(observation_space, learning_rate):
  X_Input = Input(observation_space)
  X = Dense(512, activation = "relu", kernel_initializer="he_uniform")(X_Input)
  X = Dense(256, activation = "relu", kernel_initializer="he_uniform")(X)
  X = Dense(64,  activation = "relu", kernel_initializer="he_uniform")(X)

  value = Dense(1, activation = None)(X)

  Critic = Model(inputs = X_Input, outputs = value)
  Critic.compile(optimizer = RMSprop(lr = learning_rate), loss = "mse")

  return Critic
  

class AC2_agent():
  def __init__(self, env_name):
    self.env_name = env_name
    self.env = gym.make(env_name)
    self.action_space = self.env.action_space.n
    self.observation_space = self.env.observation_space.shape[0]

    # Hyper-parameters
    self.gamma = 0.99
    self.learning_rate = 0.95
    self.EPISODES = 10000

    self.Actor = Actor(self.action_space, self.observation_space, self.learning_rate)
    self.Critic = Critic(self.action_space, self.observation_space, self.learning_rate)
    self.Observations, self.next_Observations, self.Actions, self.Rewards,  = [], [], [], []

    
  def act(self, observation):
    prob = self.Actor.predict(observation)[0]
    action = np.random.choice(self.action_space, p = prob)
    return action

  def remember(self, observation, next_observation,  action, reward, done):
    self.Observations.append(observation)
    self.next_Observations.append(next_observation)
    self.Actions.append(action)
    self.Rewards.append(reward)
  
  def forget(self):
    self.Observations, self.next_Observations, self.Actions, self.Rewards = [], [], [], []

  def discounted_rewards(self, rewards):
    discounted_rewards = np.zeros_like(rewards)
    gamma = self.gamma
    sum_r = 0
    for i in reversed(range(0, len(rewards))):
      sum_r = sum_r * gamma * rewards[i]
      discounted_rewards[i] = sum_r

  def replay(self):
    observations = self.Observations
    next_observations = self.next_Observations
    actions = self.Actions
    rewards = self.Rewards

    discounted_rewards = self.discounted_rewards(rewards)

    V_pi_current = self.Critic.predict(observations)
    V_pi_next = self.Critic.predict(next_observations)
    
    
    critic_actions = self.Critic.predict(observations)
    
    advantage = discounted_rewards 

    self.Actor.fit(observations, discounted_rewards, verbose =1, epochs=1)

    pass

  def train(self):
    for e in range(self.EPISODES):
      done = False
      observation = self.env.reset()
      while not done:
        self.env.render()
        
        action = self.act(observation)

        next_observation, reward, done, info = self.env.step(action)
        next_observation = np.reshape([1, observation])

        self.remember(observation, next_observation, action, reward, done)
        
        if done:
          # do something when done
          self.replay()
        
        observation = next_observation


