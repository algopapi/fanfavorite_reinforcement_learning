import os
import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import random
import pylab
import cv2


from keras.models import Model, load_model
from keras.layers import Flatten, Dense, Input
from keras.optimizers import Adam, RMSprop


class Model_Gradient_Policy(Model):
  def __init__(self, action_space, observation_space):
    super().__init__()
    self.d1 = Dense(512, activation = 'relu', input_shape = (observation_space, ))
    self.d2 = Dense(256, activation = 'relu')
    self.d3 = Dense(64, activation = 'relu')
    self.out = Dense(action_space, activation = 'softmax')

  def call(self, input_data):
    Input_layer = tf.convert_to_tensor(input_data)
    X = self.d1(input_data)
    X = self.d2(X)
    X = self.d3(X)
    Output = self.out(X)
    return Output
  

class PGAgent():
  #Policy Gradient Main Opimization Algorithm
  def __init__(self, env_name):
    #Environment and PG parameters
    self.env_name = env_name
    self.env = gym.make(env_name)
    self.action_space = self.env.action_space.n
    self.state_space = self.env.observation_space.shape[0]

    self.EPISODES = 3000
    self.lr = 0.000025

    self.ROWS = 80
    self.COLS = 80
    self.REM_STEP = 4

    #instantiate games, plot memory
    self.states, self.actions, self.rewards = [], [], []
    self.episodes, self.scores, self.average = [], [], []

    self.Save_Path = 'Models'
    self.image_memory = np.zeros(self.state_space)

    if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
    self.path = '{}_PG_{}'.format(self.env_name, self.lr)
    self.Model_name = os.path.join(self.Save_Path, self.path)

    self.Actor = Model_Gradient_Policy(action_space = self.action_space, observation_space=self.state_space)
    self.optimizer = RMSprop(learning_rate=self.lr)

    self.max_average = 0

  def remember(self, state, action, reward):
    self.states.append(state)
    action_onehot = np.zeros([self.action_space])
    action_onehot[action] = 1
    self.actions.append(action_onehot)
    self.rewards.append(reward)

  def act(self, state):
    prediction = self.Actor.predict(state)[0]
    action = np.random.choice(self.action_space, p = prediction)
    return action

  def step(self,action):
    next_state, reward, done, info = self.env.step(action)
    return next_state, reward, done, info

  def load(self, Actor_name):
    self.Actor.load_weights(self.Model_name)
  
  def save(self):
    self.Actor.save_weights(self.Model_name, save_format = 'tf')

  def discount_rewards(self, reward):
    gamma = 0.99    # discount rate
    sum_r = 0
    discounted_r = np.zeros_like(reward)
    for i in reversed(range(0,len(reward))):
      sum_r = sum_r * gamma + reward[i]
      discounted_r[i] = sum_r
  
    return discounted_r

  def compute_loss(prob, action, reward):
    dist = tfp.distributions.Categorical(probs = prob)
    log_prob = dist.log_prob(action)
    loss = - log_prob * reward
    return loss


  def replay(self):
    states = np.vstack(self.states)
    actions = np.vstack(self.actions)
    
    #Calculate the discounted rewards
    discounted_r = self.discount_rewards(self.rewards)

    # custom training loop
    # iterate over batches of trainin data
    for state, action, d_reward in zip(states, actions, discounted_r):
        
      print("state = {}, action = {}, reward = {}", state, action, d_reward)

      with tf.GradientTape() as tape:
        # forward pass of the layer
        prob = self.Actor(np.array(state), training=True)
        print("probability = ", prob)
        # Calcualte the policy loss
        loss =  self.compute_loss(prob, action, d_reward)

      # use the gradient tape to automatically retrieve the gradients of the trainable variables with respect to the loss
      grads = tape.gradient(loss, self.Actor.trainable_variables)

      # Run one step of gradient ascent by updating the value of the variables to miminize the loss
      self.optimizer.apply_gradients(zip(grads, self.Actor.trainable_variables))

    self.states, self.actions, self.rewards = [], [], []
  
  def reset(self):
    frame = self.env.reset()
    for i in range(self.REM_STEP):
        state = self.GetImage(frame)
    return state

  def run(self):
    for e in range(self.EPISODES):
      state = self.env.reset()
      state = np.reshape(state, [1, self.state_space])
      done, score, SAVING = False, 0, ''

      while not done:
        
        self.env.render()
        action = self.act(state)

        next_state, reward, done, _ = self.step(action)
        next_state = np.reshape(next_state, [1, self.state_space])

        self.remember(state, action, reward)

        state = next_state
        score += reward

        if done:
          
          average = self.PlotModel(score, e)
          if average >= self.max_average:
            self.max_average = average
            self.save()
            SAVING = "SAVING"
          else:
            SAVING = ""
            print("episode: {}/{}, score: {}, average: {:.2f} {}".format(e, self.EPISODES, score, average, SAVING))
          
          # Update step
          self.replay()
    
    self.env.close()

  pylab.figure(figsize=(18, 9))
  def PlotModel(self, score, episode):
      self.scores.append(score)
      self.episodes.append(episode)
      self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
      if str(episode)[-2:] == "00":# much faster than episode % 100
          pylab.plot(self.episodes, self.scores, 'b')
          pylab.plot(self.episodes, self.average, 'r')
          pylab.ylabel('Score', fontsize=18)
          pylab.xlabel('Steps', fontsize=18)
          try:
              pylab.savefig(self.path+".png")
          except OSError:
              pass
      return self.average[-1]
  
  def test(self):
    self.load("cartpole-ddqn.h5")
    for e in range(self.EPISODES):
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size])
        done = False
        i = 0
        while not done:
          self.env.render()
          action = np.argmax(self.model.predict(state))
          next_state, reward, done, _ = self.env.step(action)
          state = np.reshape(next_state, [1, self.state_size])
          i += 1
          if done:
              print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
              break

if __name__ == "__main__":
    env_name = 'CartPole-v1'
    #env_name = 'PongDeterministic-v4'
    agent = PGAgent(env_name)
    agent.run()
    #agent.test()
    #agent.test('Models/Pong-v0_PG_2.5e-05.h5')