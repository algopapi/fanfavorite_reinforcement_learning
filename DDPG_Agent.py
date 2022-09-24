
import os
import gym
import datetime

import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.losses import Huber, MeanSquaredError
from tensorflow.keras.optimizers import Adam

from matplotlib import pyplot
from utils.plotmodel import PlotModel

""" DDPG Policy Netowrk """
class Q_Network(tf.keras.Model):
    def __init__(
        self, 
        action_space, 
        observation_space
    ):
        super(Q_Network, self).__init__()
        self.obvervation_input = Dense(128, activation="relu", input_shape=(observation_space,), kernel_initializer="he_uniform")
        self.dense2 = Dense(128, activation="relu", kernel_initializer="he_uniform")
        self.actor = Dense(action_space, activation="softmax",kernel_initializer="he_uniform")

        self.action_input = Dense(128, )

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(inputs)
        return self.actor(x)

""" DDPG Q Function Network """
class Policy_Network(tf.keras.Model):

    def __init__(
        self, 
        observation_space, 
        action_space, 
    ):
        super(Policy_Network, self).__init__()
        self.dense1 = Dense(128, activation="relu", input_shape=(observation_space,), kernel_initializer="he_uniform")
        self.dense2 = Dense(128, activation="relu", kernel_initializer="he_uniform")
        self.policy = Dense(action_space, activation="softmax", kernel_initializer="he_uniform")

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.policy(x)


class DDPG_Agent():
    def __init__(self, env_name):
        self.env_name = env_name
        self.agent_name = "DDPG_Agent"
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space
        print("action space", self.action_space)
        self.state_space = self.env.observation_space.shape[0]


if __name__== "__main__":
    env_name = "Pendulum-v1"
    agent = DDPG_Agent(env_name)
