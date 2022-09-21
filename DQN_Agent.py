import os
import queue
import random
from sys import maxsize
import pylab
import gym

import tensorflow as tf
import numpy as np

import keras
from keras.models import Model, load_model
from keras.layers import Flatten, Dense, Input
from keras.losses import mean_squared_error, SparseCategoricalCrossentropy, CategoricalCrossentropy

from collections import deque

class DQN_Model(Model):
  def __init__(self, observation_space, action_space):
    super().__init__()
    self.d1 = Dense(512, activation="relu", input_shape=(observation_space,), kernel_initializer="he_uniform")
    self.d2 = Dense(256, activation="relu", kernel_initializer="he_uniform")
    self.d3 = Dense(64, activation="relu", kernel_initializer="he_uniform")
    self.output_layer = Dense(action_space, activation="linear", kernel_initializer="he_uniform")

  def call(self, input):
    x = self.d1(input)
    x = self.d2(x)
    x = self.d3(x)
    return self.output_layer(x)
    

class DQN_agent():
  def __init__(self, env_name):
    self.env_name = env_name
    self.agent_name = "DQN_Agent_"+self.env_name
    self.env = gym.make(env_name)
    self.action_space = self.env.action_space.n
    self.observation_space = self.env.observation_space.shape[0]
    self.Observations, self.next_Observations, self.Actions, self.Rewards, self.Dones  = [], [], [], [], []

    # Hyper-parameters
    self.gamma = 0.995
    self.learning_rate = 0.5e-3
    self.EPISODES = 10000
    self.epsilon = 1
    self.min_epislon = 0.01
    self.epsilon_decay = 0.0005
    self.memory_size = 2000
    
    self.batch_size = 32
    self.min_batch = 124
    
    # Memory
    self.Memory= deque(maxlen=self.memory_size)

    # Predict and Target Network
    self.Predict_Network = DQN_Model(self.observation_space, self.action_space)
    self.Target_Network = DQN_Model(self.observation_space, self.action_space)

    self.Predict_Network.build(input_shape=(1,self.observation_space))
    self.Target_Network.build(input_shape=(1,self.observation_space))
    
    print(self.Predict_Network.summary())
    print(self.Predict_Network.summary())
    
    # Model Evaluation
    self.scores, self.average, self.episodes = [], [], []

    # Model Path
    self.Save_Path = 'Models'
    if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
    self.path = self.Save_Path + '/{}_{}_LR_{}'.format(self.agent_name, self.env_name, self.learning_rate)


  def act(self, state, decay_step):
    explore_probability = self.min_epislon + (self.epsilon - self.min_epislon) * np.exp(-self.epsilon_decay * decay_step)
    
    if(decay_step % 100 == 0):
      print("explore probability = ", explore_probability)

    if explore_probability > np.random.rand(): # explore
      return random.randrange(self.action_space)
    else: # exploit
      return np.argmax(self.Predict_Network(state))

  def remember(self, state, next_state, action, reward, done):
    mem_entry = np.array([state, next_state, action, reward, done], dtype=object)  
    self.Memory.append(mem_entry)

  def update_target_network(self):

    #target_model_weights = self.Target_Network.get_weights()

    #for layer in self.Predict_Network.layers:
    #  print("=== PREDICT LAYER:", layer.name)
    #  print("weights", layer.get_weights())
    
    #for layer in self.Target_Network.layers:
    #  print("=== TARGET LAYER:", layer.name)
    #  print("weights", layer.get_weights())

    #print("target model weights=", target_model_weights)
    self.Target_Network.set_weights(self.Predict_Network.get_weights())




  def replay(self):
    if(len(self.Memory) < self.min_batch):
      return
   
    # Define network optimizer
    optimizer = tf.keras.optimizers.RMSprop(learning_rate = self.learning_rate, rho=0.95, epsilon=0.01)

    # Define loss function
    network_loss_function = tf.keras.losses.Huber(delta=1.0, reduction="auto", name="huber_loss")

    # Retrieve random batch from past experiences
    batch = random.sample(self.Memory, k=self.batch_size)
    batch_array = np.asarray(batch, dtype=object)
    
    for _, x_train_batch in enumerate(batch_array):
      with tf.GradientTape() as tape:
        # Unravel the batch
        state, next_state, action, reward, done=np.split(x_train_batch, indices_or_sections=5, axis=0)

        state = np.stack(state).astype(None)
        next_state=np.stack(next_state).astype(None)
        
        state = np.squeeze(next_state)
        next_state = np.squeeze(next_state)

        state=np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)
        
        action = action[0]
        reward = reward[0]

        if done:
          q_values_current = self.Predict_Network(state)[0]
          loss = network_loss_function(y_true=[reward], y_pred=q_values_current)
        else:
          # Retrieve Current Q value
          q_values_current = self.Predict_Network(state)
          
          # max_a' Q_target(s', a')
          targets = np.array(self.Target_Network(next_state))
      
          # Calculate Target Value
          targets[0][action] = reward + (self.gamma * np.amax(targets))

          # Calculate the loss for this batch
          loss = network_loss_function(y_true=targets, y_pred=q_values_current)

      grads = tape.gradient(loss, self.Predict_Network.trainable_variables)
      optimizer.apply_gradients(zip(grads, self.Predict_Network.trainable_variables))

  pylab.figure(figsize=(18, 9))
  def PlotModel(self, score, episode):
    self.scores.append(score)
    self.episodes.append(episode)
    self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
    if str(episode)[-2:] == "00":# much faster than episode % 100
        pylab.plot(self.episodes, self.scores, 'b')
        pylab.plot(self.episodes, self.average, 'r')
        pylab.title(self.agent_name + self.env_name, fontsize=18 )
        pylab.ylabel('Score', fontsize=18)
        pylab.xlabel('Episodes', fontsize=18)
        try:
            pylab.savefig(self.path+".png")
        except OSError:
            pass
    return self.average[-1]

  def train(self):
    """Main train loop of the Agent"""
    decay_step = 0
    for e in range(self.EPISODES):
      done = False
      score = 0
      state = self.env.reset()      
      state  = state.reshape([1, state.shape[0]])

      while not done:
        decay_step += 1
        self.env.render()
        action = self.act(state, decay_step)
        next_state, reward, done, _ = self.env.step(action)
        next_state = next_state.reshape([1,next_state.shape[0]])

        if not done:
          reward = reward
        else:
          reward = -100

        self.remember(state, next_state, action, reward, done)
        score += reward
       
        if done:
          self.update_target_network()
          self.PlotModel(score, e)
          break
        
        state = next_state
        self.replay()

if __name__ == "__main__":
    env_name = 'CartPole-v1'
    agent = DQN_agent(env_name)
    
    #agent.update_target_network()
    agent.train()
