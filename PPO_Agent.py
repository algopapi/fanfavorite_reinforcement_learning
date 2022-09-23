import tensorflow as tf
#import tensorflow_probability as tfp
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


class PPO_Actor(Model):
    """ Actor Network"""
    def __init__(
        self, 
        observation_space, 
        action_space,
        name="PPO_Actor"    
    ):
        super(PPO_Actor, self).__init__(name=name)
        self.d1 = Dense(512, input_shape=(observation_space,), activation="relu", kernel_initializer="he_uniform")
        self.d2 = Dense(256, activation="relu", kernel_initializer="he_uniform")
        self.d3 = Dense(64, activation="relu", kernel_initializer="he_uniform")
        self.actor = Dense(action_space, activation=None, kernel_initializer="he_uniform")
    
    def call(self, inputs):
        x = self.d1(inputs)
        x = self.d2(x)
        x = self.d3(x)
        return self.actor(x)

class PPO_Critic(Model):
    """ Critic Network"""
    def __init__(
        self, 
        observation_space, 
        name="PPO_Critic"    
    ):
        super(PPO_Critic, self).__init__(name=name)
        self.d1 = Dense(512, input_shape=(observation_space,), activation="relu", kernel_initializer="he_uniform")
        self.d2 = Dense(256, activation="relu", kernel_initializer="he_uniform")
        self.d3 = Dense(64, activation="relu", kernel_initializer="he_uniform")
        self.critic = Dense(1, activation="linear", kernel_initializer="he_uniform")
    
    def call(self, inputs):
        x = self.d1(inputs)
        x = self.d2(x)
        x = self.d3(x)
        return self.critic(x)



class PPO_Agent():
    def __init__(self, env_name):
        self.env_name = env_name
        self.agent_name = "PPO_Agent"
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space.n
        self.state_space = self.env.observation_space.shape[0]

        self.EPISODES = 3000
        self.max_steps_per_episode = 500
        self.lr = 0.0005
        self.gamma = 0.95
        self.clip_epsilon = 0.3
        self.epsilon = np.finfo(np.float32).eps.item() 
        self.clip_epsilon = 0.2

        #instantiate games, plot memory
        self.states, self.actions, self.action_probs, self.rewards, self.dones = [], [], [], [], []
        self.critic_values = []
        self.episodes, self.scores, self.average = [], [], []

        self.Save_Path = './Models'
        self.Plot_Path = './Plots'
      
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        if not os.path.exists(self.Plot_Path): os.makedirs(self.Plot_Path)

        self.path = '{}_{}_LR_{}'.format(self.agent_name, self.env_name, self.lr)
        
        self.Model_name = os.path.join(self.Save_Path, self.path)
        self.Plot_name = os.path.join(self.Plot_Path, self.path)

        self.PPO_Actor = PPO_Actor(action_space = self.action_space, observation_space=self.state_space)
        self.PPO_Critic = PPO_Critic(observation_space = self.state_space)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        self.actor_train_iterations = 10
        self.critic_train_itartions = 5

        self.max_average = 300
        self.n_epochs_per_batch = 5

        self.ACTOR_LOSS_WEIGHT = 1
        self.CRITIC_LOSS_WEIGHT = 1

    def returns(self, reward_array):
        returns=np.zeros_like(reward_array)
        sum = 0
        for i in reversed(range(0, len(reward_array))):
            sum = reward_array[i] + self.gamma * sum
            returns[i] = sum
        return returns

    def logprobabilities(self, logits, a):
        logprobabilities_all = tf.nn.log_softmax(logits)
        onehots = tf.one_hot(a, self.action_space)
        
        logprobability = tf.reduce_sum(
           onehots * logprobabilities_all, axis=1
        )
        return logprobability

    def get_action(self, observation):
        logits = self.PPO_Actor(observation)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        return logits, action


    def run(self):
        for e in range(self.EPISODES):     
            state = self.env.reset()
            done, score, SAVING = False, 0, ''
            while not done:          # episode
                state = tf.convert_to_tensor(state)
                state = tf.expand_dims(state, 0)
                self.states.append(state)
                
                logits, action = self.get_action(state)
                log_probs = self.logprobabilities(logits, action)

                self.action_probs.append(log_probs)
                self.actions.append(action)

                critic_value = self.PPO_Critic(state)
                self.critic_values.append(critic_value[0,0])
                
                state, reward, done, _ = self.env.step(action[0].numpy())
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
    
            
           
            # Calculate returns and Normalize
            reward_array = np.array(self.rewards)
            batch_returns = self.returns(reward_array)
            batch_returns = (batch_returns - np.mean(batch_returns)) / (np.std(batch_returns) + self.epsilon)
            
            # Convert some buffer lists to numpy arrays 
            actions = np.squeeze(np.array(self.actions))
            batch_states = tf.convert_to_tensor(self.states)
            batch_log_probs = np.array(self.action_probs)

            print("batch_log_probs", batch_log_probs)
            print("actions", actions)
            # Calculate Advanatges
            batch_critic_values = np.array(self.critic_values)
            batch_advantages = batch_returns - batch_critic_values

            # Start Policy Training loop
            for _ in range(self.actor_train_iterations):
                with tf.GradientTape() as tape:                    
                    """Get action_probs & critic values from network under current policy and        
                    calculate the ratios of the current batch """

                    logits = self.PPO_Actor(batch_states)
                    logprobs = self.logprobabilities(np.squeeze(logits), actions)
                    logprobs = tf.expand_dims(logprobs, axis=-1)
                    ratios = np.squeeze(tf.exp(logprobs - batch_log_probs))
                    
                    min_advantage = tf.where(
                        batch_advantages > 0, 
                        (1 + self.clip_epsilon) * batch_advantages,
                        (1 - self.clip_epsilon) * batch_advantages
                    )

                    policy_loss = -tf.reduce_mean(
                        tf.minimum(ratios * batch_advantages, min_advantage)
                    )

                # Do gradient ascent step
                grads = tape.gradient(policy_loss, self.PPO_Actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(grads, self.PPO_Actor.trainable_variables))

            # Start Critic Training Loop
            for _ in range(self.critic_train_iterations):
                with tf.GradientTape() as tape:
                
                    critic_values = self.PPO_Critic(batch_states)

                    critic_loss = tf.reduce_mean(
                        (batch_returns - critic_values) ** 2
                    )

                grads = tape.gradient(critic_loss, self.PPO_Critic.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(grads, self.PPO_Critic.trainable_variables))

            # Clear the buffer after each episode
            self.action_probs, self.critic_values, self.rewards, self.states, self.actions, self.dones = [], [], [], [], [], []

if __name__ == "__main__":
    env_name = 'CartPole-v1'

    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print("logdir", log_dir)
    summary_writer = tf.summary.create_file_writer(log_dir)
    with summary_writer.as_default():
        agent = PPO_Agent(env_name)
        agent.run()