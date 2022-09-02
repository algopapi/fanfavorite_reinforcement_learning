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
def Actor_Critic(action_space, observation_space, learning_rate):
  X_Input = Input(observation_space)
  X = Dense(512, activation = "relu", kernel_initializer="he_uniform")(X_Input)
  X = Dense(256, activation = "relu", kernel_initializer="he_uniform")(X)
  X = Dense(64,  activation = "relu", kernel_initializer="he_uniform")(X)

  value = Dense(1, activation = None)(X)
  actions = Dense(action_space, activation = "softmax", kernel_initializer="he_uniform")

  Actor = Model(inputs = X_Input, outputs = actions)
  Actor.compile(optimizer = RMSprop(lr = learning_rate), loss = "categorical_crossentropy")

  Critic = Model(inputs = X_Input, outputs = value)
  Critic.compile(optimizer = RMSprop(lr = learning_rate), loss = 'mse')

  return Actor, Critic


class AC2_agent():
  def __init__(self, env_name):
    self.env_name = env_name
    self.env = gym.make(env_name)
    self.action_space = self.env.action_space.n
    self.obvervation_space = self.env.observation_space.shape[0]

    self.EPISODES = 10000

    self.Actor, self.Critic = Actor_Critic(self.action_space, self.observation_space, self.learning_rate)

    self.Observations, self.Actions, self.Rewards = [], [], []

    
  def act(self, observation):
    prob = self.Actor.predict(observation)[0]
    action = np.random.choice(self.action_space, p = prob)
    return action

  def remember(self, observation, action, reward, done):
    self.Observations.append(observation)
    self.Actions.append(action)
    self.Rewards.append(reward)
  
  def forget(self):
    self.Observations, self.Actions, self.Rewards = [], [], []

  def replay():
    pass

  def train(self):
    for e in range(self.EPISODES):
      done = False
      observation = self.env.reset()
      while not done:
        self.env.render()
        
        action = self.act(observation)

        observation, reward, done, info = self.env.step(action)
        observation = np.reshape([1, observation])
        
        if done:
          # do something when done
          pass




