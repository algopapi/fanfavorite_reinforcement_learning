import tensorflow as tf
import numpy as np

import os
import gym
import time

from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Dense, Layer

from matplotlib import pyplot
from PlotModel import PlotModel

class PPO_Network(Model):
    """ Actor-Critic Style Network"""
    def __init__(self, observation_space, action_space):
        super(self, PPO_Network).__init__()
        self.d1 = Dense(512, input_shape=(observation_space,), activation="relu")
        self.d2 = Dense(256, activation="relu")
        self.d3 = Dense(64, activation="relu")
        
        self.actor = Dense(action_space, activation="softmax")
        self.critic = Dense(1, activation=None)
    
    def call(self, inputs):
        x = self.d1(inputs)
        x = self.d2(x)
        x = self.d3(x)
        return self.actor(x), self.critic(x)


class PPO_Agent():
    def __init(self, env_name):
        self.env_name = env_name
        self.agent_name = "PPO_Agent"
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space.n
        self.state_space = self.env.observation_space.shape[0]

        self.EPISODES = 3000
        self.lr = 0.001

        #instantiate games, plot memory
        self.states, self.actions, self.rewards, self.dones = [], [], [], []
        self.episodes, self.scores, self.average = [], [], []

        self.Save_Path = 'Models'

        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.path = '{}_{}_LR_{}'.format(self.agent_name, self.env_name, self.lr)
        self.Model_name = os.path.join(self.Save_Path, self.path)

        self.PPO_Netowrk = PPO_Network(action_space = self.action_space, observation_space=self.state_space)


    def remember(self, state, action, reward, dones):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(dones)

    def act(self, state):
        actions, value = self.PPO_Netowrk.predict(state)[0]
        action = np.random.choice(self.action_space, p = actions)
        return action, value

    def discounted_rewards(self, rewards, dones):
        discounted_rewards=np.zeros_like(rewards)
        sum = 0
        for i in reversed(range(0, len(rewards))):
            sum = rewards[i] + (self.gamma * sum) * (1 - dones[i])
            discounted_rewards[i] = sum
        return discounted_rewards

    def train(self):
        """ Main Train loop of the PPO algorithm. Performs a gradient ascent step on collected trajectory 
            and clips the update step following to the PPO clipping scheme """
        
        
        

        # Forget past experiences
        self.states, self.actions, self.rewards, self.dones = [],[],[],[]


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
                
                PlotModel(self, score, e)

                if done:
                    average = self.PlotModel(score, e)
                    if average >= self.max_average:
                        self.max_average = average
                        self.save()
                        SAVING = "SAVING"
                    else:
                        SAVING = ""
                        print("episode: {}/{}, score: {}, average: {:.2f} {}".format(e, self.EPISODES, score, average, SAVING))
                
                # Train Loop
                self.train()   
            self.env.close()



if __name__ == "__main__":
    env_name = 'CartPole-v1'
    agent = PPO_Agent(env_name)
    agent.run()