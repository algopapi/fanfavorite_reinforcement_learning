from dis import dis
from hashlib import new
import os
import gym
import numpy as np
import tensorflow as tf
import random
import pylab


import keras
from keras.models import Model, load_model
from keras.layers import Flatten, Dense, Input
from keras.losses import mean_squared_error, SparseCategoricalCrossentropy, CategoricalCrossentropy

ACTOR_LOSS_WEIGHT = 1
ENTROPY_LOSS_WEIGHT = 1e-4

def critic_loss(discounted_rewards, predicted_values):
  print("Predicted values", predicted_values)
  print("discounted rewards", discounted_rewards)
  return mean_squared_error(discounted_rewards, predicted_values)

def actor_loss(combined,  policy_logits):
  
  actions, advantages = tf.split(combined, 2, axis=-1)
  sparse_ce = SparseCategoricalCrossentropy(from_logits=True)
  print("policy_loss=", policy_logits)
  print("actions=", actions)
  print("advantages=", advantages)
  actions = tf.cast(actions, tf.int32)
  policy_loss = sparse_ce(y_true=actions, y_pred=policy_logits, sample_weight=advantages)
  probs = tf.nn.softmax(policy_logits)

  cce = CategoricalCrossentropy()
  entropy_loss = cce(probs, probs)

  return (policy_loss * ACTOR_LOSS_WEIGHT ) - (entropy_loss * ENTROPY_LOSS_WEIGHT)

# Actor Critic Network
def Actor_Critic(action_space, observation_space, learning_rate):
  X_Input = Input(shape=(observation_space,))
  X = Dense(512, activation="relu")(X_Input)
  X = Dense(256, activation="relu")(X)
  X = Dense(64,  activation="relu")(X)

  actions = Dense(action_space, activation="softmax")(X)
  value = Dense(1)(X)
  
  Actor = Model(inputs=X_Input, outputs=actions)
  Actor.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate), loss=actor_loss)
  Actor.summary()

  Critic = Model(inputs=X_Input, outputs=value)
  Critic.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate), loss=critic_loss)

  return Actor, Critic

class AC2_agent():
  def __init__(self, env_name):
    self.env_name = env_name
    self.agent_name = "A2C_Agent_"+self.env_name
    self.env = gym.make(env_name)
    self.action_space = self.env.action_space.n
    self.observation_space = self.env.observation_space.shape[0]
    self.Observations, self.next_Observations, self.Actions, self.Rewards, self.Dones  = [], [], [], [], []

    # Hyper-parameters
    self.gamma = 0.995
    self.learning_rate = 0.00025
    self.EPISODES = 10000

    # Actor and critic model
    self.Actor, self.Critic = Actor_Critic(self.action_space, self.observation_space, self.learning_rate)

    # Model Evaluation
    self.scores, self.average, self.episodes = [], [], []

    # Model Path
    self.Save_Path = 'Models'
    if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
    self.path = self.Save_Path + '/{}_{}_LR_{}'.format(self.agent_name, self.env_name, self.learning_rate)

  def act(self, observation):
    policy_logits = self.Actor.predict(observation)[0]
    action = np.random.choice(self.action_space, p=policy_logits)
    return action, policy_logits

  def remember(self, observation, next_observation,  action, reward, done):
    self.Observations.append(observation)
    self.next_Observations.append(next_observation)
    self.Actions.append(action)
    self.Rewards.append(reward)
    self.Dones.append(done)
  
  def forget(self):
    self.Observations, self.next_Observations, self.Actions, self.Rewards, self.Dones = [], [], [], [], []

  def discounted_rewards(self, rewards, dones, target_values):
    discounted_rewards = np.zeros_like(rewards)
    gamma = self.gamma
    sum_r = 0
    for i in reversed(range(rewards.shape[0])):
      sum_r = sum_r * gamma + rewards[i] * (1-dones[i])
      discounted_rewards[i] = sum_r
    
    advantages = discounted_rewards - target_values
    advantages = np.vstack(advantages)
    discounted_rewards = np.vstack(discounted_rewards)

    return discounted_rewards, advantages

  def replay(self):
  
    observations = np.vstack(self.Observations)

    rewards = np.array(self.Rewards)
    actions = np.vstack(self.Actions) 
    dones = np.array(self.Dones)
    target_value = self.Critic.predict(observations)
    target_value = np.squeeze(target_value)
    #print("target_value shape", target_value.shape)
    
    discounted_rewards, advantages = self.discounted_rewards(rewards, dones, target_value)

    combined = np.zeros((len(actions),2,1))
    combined = np.concatenate([actions, advantages], axis=-1)

    # Update the Actor Network
    self.Actor.train_on_batch(x=observations, y=combined)
    
    # Update the Critic Network
    self.Critic.train_on_batch(x=observations, y=discounted_rewards)

    self.forget()


  def train(self):
    for e in range(self.EPISODES):
      done = False
      score = 0
      observation = self.env.reset()      
      observation  = observation.reshape([1, observation.shape[0]])

      while not done:
        self.env.render()
        action, _ = self.act(observation)
        next_observation, reward, done, _ = self.env.step(action)
        next_observation = next_observation.reshape([1,next_observation.shape[0]])

        self.remember(observation, next_observation, action, reward, done)
        score += reward

        if done:
          self.PlotModel(score, e)
          # do something when done
        observation = next_observation
      self.replay()


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



if __name__ == "__main__":
  env_name = 'CartPole-v1'
  A2CAgent = AC2_agent(env_name)
  A2CAgent.train()

