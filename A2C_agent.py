import tensorflow as tf
#import tensorflow_probability as tfp
import numpy as np

import os
import gym
import time

from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.losses import Huber, MeanSquaredError
from tensorflow.keras.optimizers import Adam


from matplotlib import pyplot
from utils.plotmodel import PlotModel


class A2C_Network(Model):
    """ Actor-Critic Style Network"""
    def __init__(
        self, 
        observation_space, 
        action_space,
        name="A2C_Network"    
    ):
        super(A2C_Network, self).__init__(name=name)
        self.d1 = Dense(512, input_shape=(observation_space,), activation="relu", kernel_initializer="he_uniform")
        self.d2 = Dense(256, activation="relu", kernel_initializer="he_uniform")
        self.d3 = Dense(64, activation="relu", kernel_initializer="he_uniform")
        
        self.actor = Dense(action_space, activation="softmax", kernel_initializer="he_uniform")
        self.critic = Dense(1, activation="linear", kernel_initializer="he_uniform")
    
    def call(self, inputs):
        x = self.d1(inputs)
        x = self.d2(x)
        x = self.d3(x)
        return self.actor(x), self.critic(x)


class A2C_Agent():
    def __init__(self, env_name):
        self.env_name = env_name
        self.agent_name = "A2C_Agent"
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space.n
        self.state_space = self.env.observation_space.shape[0]

        self.EPISODES = 3000
        self.lr = 0.000025
        self.gamma = 0.94
        self.clip_epsilon = 0.3
        self.epsilon =  np.finfo(np.float32).eps.item() 

        #instantiate games, plot memory
        self.states, self.next_states, self.action_probs, self.rewards, self.dones = [], [], [], [], []
        self.critic_values = [] 
        self.episodes, self.scores, self.average = [], [], []

        self.Save_Path = 'Models'

        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.path = '{}_{}_LR_{}'.format(self.agent_name, self.env_name, self.lr)
        self.Model_name = os.path.join(self.Save_Path, self.path)

        self.A2C_Network = A2C_Network(action_space = self.action_space, observation_space=self.state_space)
        self.critic_loss = Huber()
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.lr)
        self.max_average = 300

        self.ACTOR_LOSS_WEIGHT = 1
        self.CRITIC_LOSS_WEIGHT = 0.5

    def returns(self):
        reward_array = np.array(self.rewards)
        returns=np.zeros_like(reward_array)
        sum = 0
        for i in reversed(range(0, len(reward_array))):
            sum = reward_array[i] + self.gamma * sum
            returns[i] = sum
        return returns

    def run(self):
        running_reward = 0
        for e in range(self.EPISODES):     
            state = self.env.reset()
            with tf.GradientTape() as tape:
                done, score, SAVING = False, 0, ''
                while not done:       # episode
                    
                    state = tf.convert_to_tensor(state)
                    state = tf.expand_dims(state, 0)

                    action_probabilities, critic_v = self.A2C_Network(state)
                    self.critic_values.append(critic_v[0,0])
                    
                    action = np.random.choice(self.action_space, p=np.squeeze(action_probabilities))
                    self.action_probs.append(tf.math.log(action_probabilities[0, action]))
                    
                    state, reward, done, _ = self.env.step(action)
                    self.rewards.append(reward)
                    self.dones.append(done)

                    score += reward
                    
                    if done:
                        average = PlotModel(self, score, e)
                        if average >= self.max_average:
                            self.max_average = average
                            self.save()
                            SAVING = "SAVING"
                        else:
                            SAVING = ""
                            print("episode: {}/{}, score: {}, average: {:.2f} {}".format(e, self.EPISODES, score, average, SAVING))
                        break
        
                running_reward = 0.05 * score + (1-0.05) * running_reward

                # Calculate returns
                returns = self.returns()
        
                #Normalize
                returns = (returns - np.mean(returns)) / (np.std(returns) + self.epsilon)

                # Calculate Losses
                actor_losses = []
                critic_losses = []
                for action_prob, critic_value, ret in zip(self.action_probs, self.critic_values, returns):

                    # Calculate loss for each action value pair
                    diff = ret - critic_value
                    actor_losses.append(-action_prob * diff)

                    critic_losses.append(
                        self.critic_loss(tf.expand_dims(critic_value,0), tf.expand_dims(ret,0))
                    )
            
                total_loss = sum(actor_losses) + sum(critic_losses)
                #print("total loss", total_loss)
                grads = tape.gradient(total_loss, self.A2C_Network.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.A2C_Network.trainable_variables))

                self.action_probs, self.critic_values, self.rewards, self.dones = [], [], [], []

if __name__ == "__main__":
    env_name = 'CartPole-v1'
    agent = PPO_Agent(env_name)
    agent.run()