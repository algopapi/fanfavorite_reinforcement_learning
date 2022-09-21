import tensorflow as tf
import numpy as np

import os
import gym
import time

from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam

from matplotlib import pyplot
from PlotModel import PlotModel

class PPO_Network(Model):
    """ Actor-Critic Style Network"""
    def __init__(
        self, 
        observation_space, 
        action_space,
        name="PPO_Network"    
    ):
        super(PPO_Network, self).__init__(name=name)
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
    def __init__(self, env_name):
        self.env_name = env_name
        self.agent_name = "PPO_Agent"
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space.n
        self.state_space = self.env.observation_space.shape[0]

        self.EPISODES = 3000
        self.lr = 0.001
        self.gamma = 0.95
        self.clip_epsilon = 0.3

        #instantiate games, plot memory
        self.states, self.log_probs, self.rewards, self.dones, self.values = [], [], [], [], []
        self.episodes, self.scores, self.average = [], [], []

        self.Save_Path = 'Models'

        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.path = '{}_{}_LR_{}'.format(self.agent_name, self.env_name, self.lr)
        self.Model_name = os.path.join(self.Save_Path, self.path)

        self.PPO_Network = PPO_Network(action_space = self.action_space, observation_space=self.state_space)
        self.huber_loss = Huber()
        self.optimizer = Adam(learning_rate=self.lr)
        self.max_average = 300

    def remember(self, state, log_prob, reward, done, value):
        self.states.append(state)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def act(self, state):
        action_probabilities, value = self.PPO_Network.predict(state)
        action = np.random.choice(self.action_space, p = action_probabilities[0])
        log_prob = np.log(action_probabilities[0][action])
        return action, log_prob, value

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
        states = np.squeeze(np.array(self.states))
        log_probs = np.array(self.log_probs)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values)
        
        discounted_rewards = self.discounted_rewards(rewards, dones)
        advantages = discounted_rewards - np.squeeze(values)

        print("action probs", log_probs)
        print("states", states)
        print("discounted_rewards", discounted_rewards)
        print("advantages", advantages)

        # Train Loop on Critic
        actor_losses = []
        critic_losses = []
        for crit_value, dis_reward, log_prob, advantage in zip(values, discounted_rewards, log_probs, advantages):
            with tf.GradientTape() as tape:
                #Calculate Actor Losses
                actor_losses.append(-log_prob * advantage)

                #Calculate Critic Loss
                critic_losses.append(
                    self.huber_loss(y_pred=tf.expand_dims(crit_value,0), y_true=tf.expand_dims(dis_reward,0))
                )

            # Calculate the total Loss Value
            
        total_loss = sum(actor_losses) + sum(critic_losses)
        print("total loss", total_loss)
        gradients = tape.gradient(total_loss, self.PPO_Network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients,self.PPO_Network.trainable_weights))
      

        # Forget past experiences
        self.states, self.actions, self.rewards, self.dones, self.values = [],[],[],[],[]


    def run(self):

        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_space])
            done, score, SAVING = False, 0, ''

            while not done: 
                self.env.render()
                action, log_prob, value = self.act(state)

                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_space])

                self.remember(state, log_prob, reward, done, value)

                state = next_state
                score += reward
                
                if done:
                    average = PlotModel(self, score, e)
                    self.train()  
                    if average >= self.max_average:
                        self.max_average = average
                        self.save()
                        SAVING = "SAVING"
                    else:
                        SAVING = ""
                        print("episode: {}/{}, score: {}, average: {:.2f} {}".format(e, self.EPISODES, score, average, SAVING))
                
            self.env.close()



if __name__ == "__main__":
    env_name = 'CartPole-v1'
    agent = PPO_Agent(env_name)
    agent.run()