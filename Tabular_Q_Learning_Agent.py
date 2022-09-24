import gym
import numpy as np
import random

from pyparsing import condition_as_parse_action



class Vanilla_Agent():
  def __init__(self, environment_name):
    self.env_name = environment_name
    self.env = gym.make("Taxi-v3")
    state_size = self.env.observation_space.n
    action_size = self.env.action_space.n

    self.epsilon = 0.1

    self.EPISODES = 100000

    self.learning_rate = 0.9
    self.discount=0.99

    print("State_size", state_size)
    print("Action_space", action_size)

    self.q_table = np.zeros([state_size, action_size])


  def act(self, state):
    if random.uniform(0, 1) <= self.epsilon:
      return self.env.action_space.sample()
    else:
      return np.argmax(self.q_table[state])


  def run(self):
    for e in range(self.EPISODES):
      state = self.env.reset()
      t = 0
      reward, penalties, done = 0,0, False
      score = 0
      while not done:
        #self.env.render()

        action = self.act(state)

        old_q = self.q_table[state, action]

        next_state, reward, done, info = self.env.step(action)

        new_value = reward + ( self.discount * np.max(self.q_table[next_state]))
        td_error = new_value - old_q
        
        self.q_table[state,action] = old_q + self.learning_rate * td_error

        score += reward
        t += 1

        state = next_state

        if reward == -10:
          penalties += 1

        if done:
          if e % 100 == 0: 
            print("Score = {}, Penalties = {}".format(score, penalties))

        


  def test(self):
    state = self.env.reset()
    penalties, score, episodes = 0, 0, 100
    done = False
    for _ in range(episodes):
      score = 0
      while not done:
      
        action = np.argmax(self.q_table[state])
        state, reward, done, info = self.env.step(action)
      

        if reward == -10:
          penalties += 1

        score += reward

        if done:
          print("score = ", score)
 




if __name__ == "__main__":
  agent = Vanilla_Agent("Taxi-v3")
  agent.run()
  #agent.test()

