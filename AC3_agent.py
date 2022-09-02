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


def Model_Gradient_Policy(input_shape, action_space, lr):
  X_input = Input(input_shape)
  # X = Flatten(input_shape = input_shape)(X_input)
  X = X_input
  X = Dense(512, input_shape = input_shape, activation = "relu", kernel_initializer = "he_uniform")(X)
  X = Dense(256, input_shape = input_shape, activation = "relu", kernel_initializer = "he_uniform")(X)
  X = Dense(64, input_shape = input_shape, activation = "relu", kernel_initializer = "he_uniform")(X)

  action = Dense(action_space, activation = "softmax", kernel_initializer = "he_uniform")(X)
  value = Dense(1, kernel_initializer='he_uniform')(X)

  Actor = Model(inputs = X_input, outputs = action)
  Actor.compile(loss = 'categorical_crossentropy', optimizer = RMSprop(lr = lr))

  Critic = Model(inputs = X_input, outputs = value) 
  Critic.compile(loss='mse', optimizer = RMSprop(lr = lr))

  return Actor, Critic


class PGAgent():
  #Policy Gradient Main Opimization Algorithm
  def __init__(self, env_name):
    #Environment and PG parameters
    self.env_name = env_name
    self.env = gym.make(env_name)
    self.action_size = self.env.action_space.n
    self.state_size = self.env.observation_space.shape[0]

    self.EPISODES = 3000
    self.lr = 0.000025
    self.ROWS = 80
    self.COLS = 80
    self.REM_STEP = 4

    #instantiate games, plot memory
    self.states, self.actions, self.rewards = [], [], []
    self.episodes, self.scores, self.average = [], [], []

    self.Save_Path = 'Models'
    #self.state_size = (self.REM_STEP, self.ROWS, self.COLS)
    self.image_memory = np.zeros(self.state_size)

    if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
    self.path = '{}_PG_{}'.format(self.env_name, self.lr)
    self.Model_name = os.path.join(self.Save_Path, self.path)

    self.Actor, self.Critic = Model_Gradient_Policy(input_shape = (self.state_size,), action_space = self.action_size, lr = self.lr)
    self.max_average = 0

  def remember(self, state, action, reward):
    self.states.append(state)
    action_onehot = np.zeros([self.action_size])
    action_onehot[action] = 1
    self.actions.append(action_onehot)
    self.rewards.append(reward)

  def act(self, state):
    prediction = self.Actor.predict(state)[0] 
    print("prediction = ", prediction)
    action = np.random.choice(self.action_size, p = prediction)
    print("action = ", action)
    return action

  def step(self,action):
    next_state, reward, done, info = self.env.step(action)
    return next_state, reward, done, info

  def load(self, Actor_name):
    self.Actor = load_model(Actor_name, compile=False)
  
  def save(self):
    self.Actor.save(self.Model_name + '.h5')

  def discount_rewards(self, reward):
    # Compute the gamma-discounted rewards over an episode
    gamma = 0.99    # discount rate
    sum_r = 0
    discounted_r = np.zeros_like(reward)
    for i in reversed(range(0,len(reward))):
      sum_r = sum_r * gamma + reward[i]
      discounted_r[i] = sum_r
  
    return discounted_r

  def replay(self):
    states = np.vstack(self.states)
    
    actions = np.vstack(self.actions)
    
    discounted_r = self.discount_rewards(self.rewards)
    # Get critic network predictions
    critic_predictions = self.Critic.predict(states)[:,0]

    # print("critic prediction shape", critic_predictions.shape)

    # print("discounted_r shape", discounted_r.shape)
    
    #print("critic predictions: ", critic_predictions)
    advantages = discounted_r - critic_predictions

    #we update the actor network with the action feedback
    self.Actor.fit(states, actions, sample_weight = advantages, epochs = 1, verbose = 0)
    
    #We update the critic network with the discounted rewards value
    self.Critic.fit(states, discounted_r, epochs=1, verbose=0)

    #Clear the states, actions, rewards for next episode
    self.states, self.actions, self.rewards = [], [], []
  

  def reset(self):
    frame = self.env.reset()
    for i in range(self.REM_STEP):
        state = self.GetImage(frame)
    return state

  def run(self):
    
    for e in range(self.EPISODES):
      state = self.env.reset()
      state = np.reshape(state, [1, self.state_size])
      done, score, SAVING = False, 0, ''

      while not done:
        
        self.env.render()
        action = self.act(state)

        next_state, reward, done, _ =self.step(action)
        next_state = np.reshape(next_state, [1, self.state_size])

        self.remember(state, action, reward)

        state = next_state
        score += reward

        if done:
          average = self.PlotModel(score, e)
          if e == 2999:
            self.max_average = average
            self.save()
            SAVING = "SAVING"
          else:
            SAVING = ""
            print("episode: {}/{}, score: {}, average: {:.2f} {}".format(e, self.EPISODES, score, average, SAVING))
          break
      
      self.replay()    

        
    
    #self.env.close()

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
          action = np.argmax(self.Actor.predict(state))
          next_state, reward, done, _ = self.env.step(action)
          state = np.reshape(next_state, [1, self.state_size])
          i += 1
          if done:
              print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
              break



if __name__ == "__main__":
    env_name = 'LunarLander-v2'
    #env_name = 'PongDeterministic-v4'
    agent = PGAgent(env_name)
    agent.run()
    #agent.test()
    #agent.test('Models/Pong-v0_PG_2.5e-05.h5')