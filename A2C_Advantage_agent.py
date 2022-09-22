import tensorflow as tf
import numpy as np

import os
import gym
import datetime

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
        self.d1 = Dense(64, input_shape=(observation_space,), activation="relu", kernel_initializer="he_uniform")
        self.d2 = Dense(64, activation="relu", kernel_initializer="he_uniform")
        self.d3 = Dense(64, activation="relu", kernel_initializer="he_uniform")
        
        self.actor = Dense(action_space, activation="softmax", kernel_initializer="he_uniform")
        self.critic = Dense(1, activation="linear", kernel_initializer="he_uniform")
    
    def call(self, inputs):
        x = self.d1(inputs)
        x = self.d2(x)
        #x = self.d3(x)
        return self.actor(x), self.critic(x)


class A2C_Agent():
    def __init__(self, env_name):
        self.env_name = env_name
        self.agent_name = "A2C_Advantage_Agent"
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space.n
        self.state_space = self.env.observation_space.shape[0]

        self.EPISODES = 3000
        self.lr = 0.001
        self.gamma = 0.97
        self.clip_epsilon = 0.3
        self.epsilon =  np.finfo(np.float32).eps.item() 

        #instantiate games, plot memory
        self.states, self.next_states, self.action_probs, self.rewards, self.dones = [], [], [], [], []
        self.current_critic_values, self.next_critic_values = [], [] 
        self.episodes, self.scores, self.average = [], [], []

        self.Save_Path = './Models'
        self.Plot_Path = './Plots'
      
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        if not os.path.exists(self.Plot_Path): os.makedirs(self.Plot_Path)

        self.path = '{}_{}_LR_{}'.format(self.agent_name, self.env_name, self.lr)
        
        self.Model_name = os.path.join(self.Save_Path, self.path)
        self.Plot_name = os.path.join(self.Plot_Path, self.path)

        self.A2C_Network = A2C_Network(action_space = self.action_space, observation_space=self.state_space)
        self.critic_loss = MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.max_average = 300


    def returns(self):
        reward_array = np.array(self.rewards)
        returns=np.zeros_like(reward_array)
        sum = 0
        for i in reversed(range(0, len(reward_array))):
            sum = reward_array[i] + self.gamma * sum
            returns[i] = sum
        return returns

    def save(self):
        self.A2C_Network.save_weights(self.Model_name)

    def run(self):
        for e in range(self.EPISODES):     
            state = self.env.reset()
            with tf.GradientTape() as tape:
                done, score, SAVING = False, 0, ''
                while not done:      
                    state = tf.convert_to_tensor(state)
                    state = tf.expand_dims(state, 0)

                    action_probabilities, critic_v = self.A2C_Network(state)
                    self.current_critic_values.append(critic_v[0,0])
                    
                    action = np.random.choice(self.action_space, p=np.squeeze(action_probabilities))
                    self.action_probs.append(tf.math.log(action_probabilities[0, action]))
                    
                    next_state, reward, done, _ = self.env.step(action)
                    
                    next_state_t = tf.convert_to_tensor(next_state)
                    next_state_t = tf.expand_dims(next_state_t,0)
                    _, next_critic_v = self.A2C_Network(next_state_t)

                    self.next_critic_values.append(next_critic_v[0,0])
                    self.rewards.append(reward)
                    self.dones.append(done)

                    score += reward
                    state = next_state

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
                
                # Calculate Losses
                actor_losses = []
                critic_losses = []

                actor_loss = tf.keras.metrics.Mean('actor_loss', dtype=tf.float32)
                critic_loss = tf.keras.metrics.Mean('critic_loss', dtype=tf.float32)

                history = zip(self.action_probs, self.current_critic_values, self.next_critic_values, self.rewards, self.dones)
                for action_prob, current_cv, next_cv, reward, done in history:
                    # Calculate Advantage
                    td_target = reward + (1 - done) * self.gamma * next_cv
                    advantage = td_target - current_cv

                    # Calculate loss for each action value pair
                    actor_losses.append(-action_prob * advantage)

                    critic_losses.append(
                        self.critic_loss(tf.expand_dims(td_target,0), tf.expand_dims(current_cv,0))
                    )
            
                actor_sum = sum(actor_losses) * self.ACTOR_LOSS_WEIGHT
                critic_sum = sum(critic_losses) * self.CRITIC_LOSS_WEIGHT
                
                total_loss = actor_sum + critic_sum
                grads = tape.gradient(total_loss, self.A2C_Network.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.A2C_Network.trainable_variables))
                
                # Logging 
                actor_loss(actor_sum)
                critic_loss(critic_sum)

                tf.summary.scalar('actor loss', actor_loss.result(), step=e)
                tf.summary.scalar('critic loss', critic_loss.result(), step=e)

                # Clear Memory
                self.action_probs, self.rewards, self.dones = [], [], []
                self.next_critic_values,  self.current_critic_values = [], [] 

if __name__ == "__main__":

    #session = tf.InteractiveSession()

    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print("logdir", log_dir)
    summary_writer = tf.summary.create_file_writer(log_dir)

    with summary_writer.as_default():
        env_name = 'CartPole-v1'
        agent = A2C_Agent(env_name)
        agent.run()