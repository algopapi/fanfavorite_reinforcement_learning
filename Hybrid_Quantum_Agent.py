import os
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_quantum as tfq

import gym
import cirq
import sympy
import pylab
import numpy as np
from functools import reduce
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit
tf.get_logger().setLevel('ERROR')


def one_qubit_rotations(qubit, symbols):
    return [
        cirq.rx(symbols[0])(qubit),
        cirq.ry(symbols[1])(qubit),
        cirq.rz(symbols[2])(qubit)
    ]


def entangling_layer(qubits):
    cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    cz_ops += ([cirq.CZ(qubits[0], qubits[-1])]) if len(qubits) != 2 else []
    return cz_ops


def generate_circuit(qubits, n_layers):
    """Perpares data re-uploading circuit on `qubits` with `n_layers` layers """
    print("test")
    n_qubits = len(qubits)
    print("N_qubits", n_qubits)

    # Sympy symbols for variational angels
    params = sympy.symbols(f'theta(0:{3*(n_layers+1)*n_qubits})')
    params = np.asarray(params).reshape((n_layers + 1, n_qubits, 3))

    # Sympy symbols for encoding angles
    inputs = sympy.symbols(f'x(0:{n_layers})' + f'_(0:{n_qubits})')
    inputs = np.asarray(inputs).reshape((n_layers, n_qubits))

    print("Defined parameters")

    # Define Circuit
    circuit = cirq.Circuit()

    # Initialized circuit
    print("initialized circuit")
    for l in range(n_layers):
        # Create Variational Layer
        circuit += cirq.Circuit(one_qubit_rotations(q,
                                params[l, i]) for i, q in enumerate(qubits))
        circuit += entangling_layer(qubits)

        # Create encoding layer
        circuit += cirq.Circuit(cirq.rx(inputs[l, i])(q)
                                for i, q in enumerate(qubits))
    
    # Create final variational layer
    circuit += cirq.Circuit(one_qubit_rotations(q,
                            params[n_layers, i]) for i, q in enumerate(qubits))

    return circuit, list(params.flat), list(inputs.flat)

# Custom Quantum Layer
class ReUploadingPQC(tf.keras.layers.Layer):
    """
    Performs the transoframtion (s_1, ..., s_d) -> (theta_1, ..., theta_N), lmdb[1][1]s_1, ..., lmdb[1][m]s_1, 

    """

    def __init__(self, qubits, n_layers, observables, activation="linear", name="re-uploading_PQC"):
        super(ReUploadingPQC, self).__init__(name=name)
        self.n_layers = n_layers
        self.n_qubits = len(qubits)

        circuit, theta_symbols, input_symbols = generate_circuit(
            qubits, n_layers)
        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)

        self.theta = tf.Variable(
            initial_value=theta_init(
                shape=(1, len(theta_symbols)), dtype="float32"),
            trainable=True, name="thetas"
        )

        lmbd_init = tf.ones(shape=(self.n_qubits * self.n_layers, ))
        self.lmbd = tf.Variable(
            initial_value=lmbd_init, dtype="float32", trainable=True, name="lamdas"
        )

        # Define explicit symbol order
        symbols = [str(symb) for symb in theta_symbols + input_symbols]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])

        self.activation = activation
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(circuit, observables)

    def call(self, inputs):
        # inputs[0] = encoding data for the state.
        batch_dim = tf.gather(tf.shape(inputs[0]), 0)

        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_thetas = tf.tile(self.theta, multiples=[batch_dim, 1])
        tiled_up_inputs = tf.tile(inputs[0], multiples=[1, self.n_layers])
        scaled_inputs = tf.einsum("i,ji->ji", self.lmbd, tiled_up_inputs)
        squashed_inputs = tf.keras.layers.Activation(
            self.activation)(scaled_inputs)

        joined_vars = tf.concat([tiled_up_thetas, squashed_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)

        return self.computation_layer([tiled_up_circuits, joined_vars])


class Alternating(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(Alternating, self).__init__()
        self.w = tf.Variable(
            initial_value=tf.constant([[(-1.)**i for i in range(output_dim)]]),
            dtype="float32",
            trainable=True,
            name="obs-weights"
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w)


def generate_model_policy(qubits, n_layers, n_actions, beta, observables):
    # Print input parameters
    input_tensor = tf.keras.Input(shape=(len(qubits),),)
    re_uploading_pqc = ReUploadingPQC(
        qubits, n_layers, observables)([input_tensor])
    process = tf.keras.Sequential([
        Alternating(n_actions),
        tf.keras.layers.Lambda(lambda x: x * beta),
        tf.keras.layers.Softmax()
    ], name="observable-policy")

    policy = process(re_uploading_pqc)
    model = tf.keras.Model(inputs=[input_tensor], outputs=policy)

    return model


class PGAgent():
    # Policy Gradient Main Opimization Algorithm
    def __init__(self, env_name):
        # Environment and PG parameters
        self.agent_name = "Vanilla_Gradient"
        self.env_name = env_name
        self.env = gym.make(env_name, render_mode="human_mode")
        self.action_space = self.env.action_space.n
        self.state_space = self.env.observation_space.shape[0]
        # print the action space and state space:
        print("Action space: ", self.action_space)
        print("State space: ", self.state_space)

        self.EPISODES = 3000
        self.lr = 0.001

        # instantiate games, plot memory
        self.states, self.actions, self.rewards = [], [], []
        self.episodes, self.scores, self.average = [], [], []

        self.Save_Path = 'Models'
        self.image_memory = np.zeros(self.state_space)

        if not os.path.exists(self.Save_Path):
            os.makedirs(self.Save_Path)
        self.path = '{}_{}_LR_{}'.format(
            self.agent_name, self.env_name, self.lr)
        self.Model_name = os.path.join(self.Save_Path, self.path)

        self.n_qubits = 4
        self.n_layers = 5
        self.n_actions = self.action_space

        self.qubits = cirq.GridQubit.rect(1, self.n_qubits)
        self.ops = [cirq.Z(q) for q in self.qubits]
        self.observables = [reduce((lambda x, y: x * y), self.ops)]

        self.Actor = generate_model_policy(
            self.qubits,
            self.n_layers,
            self.action_space,
            1.0,
            self.observables
        )
        self.optimizer = tf.keras.optimizers.RMSprop(self.lr)
        tf.keras.utils.plot_model(self.Actor, show_shapes=True, dpi=70)
        self.max_average = 300

    def remember(self, state, action, reward):
        self.states.append(state)
        action_onehot = np.zeros([self.action_space])
        action_onehot[action] = 1
        self.actions.append(action_onehot)
        self.rewards.append(reward)

    def act(self, state):
        print('state: ', state)
        prediction = self.Actor.predict(state)[0]
        action = np.random.choice(self.action_space, p = prediction)
        return action

    def step(self, action):
        next_state, reward, done, _ ,info = self.env.step(action)
        return next_state, reward, done, info

    def load(self, Actor_name):
        self.Actor.load_weights(self.Model_name)

    def save(self):
        self.Actor.save_weights(self.Model_name, save_format='tf')

    def discount_rewards(self, reward):
        gamma = 0.99    # discount rate
        sum_r = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0, len(reward))):
            sum_r = sum_r * gamma + reward[i]
            discounted_r[i] = sum_r

        return discounted_r

    def compute_loss(self, prob, action, reward):
        dist = tfp.distributions.Categorical(probs=prob)
        log_prob = dist.log_prob(action)
        loss = - log_prob * reward
        return loss

    def replay(self):
        states = np.vstack(self.states)
        actions = np.vstack(self.actions)

        # Calculate the discounted rewards
        discounted_r = self.discount_rewards(self.rewards)

        # custom training loop
        # iterate over batches of trainin data
        for state, action, d_reward in zip(states, actions, discounted_r):

            with tf.GradientTape() as tape:
                # forward pass of the layer
                prob = self.Actor(np.array(state, ndmin=2), training=True)
                #print("probability = ", prob)
                # Calcualte the policy loss
                loss = self.compute_loss(prob, action, d_reward)

            # use the gradient tape to automatically retrieve the gradients of the trainable variables with respect to the loss
            grads = tape.gradient(loss, self.Actor.trainable_variables)

            # Run one step of gradient ascent by updating the value of the variables to miminize the loss
            self.optimizer.apply_gradients(
                zip(grads, self.Actor.trainable_variables))

        self.states, self.actions, self.rewards = [], [], []

    def run(self):
        print("!! run !!")
        for e in range(self.EPISODES):
            state = self.env.reset()
            print("state", state)
            print("state[0]", state[0])
            state = np.reshape(state[0], [1, self.state_space])
            print("state after reshape", state)
            done, score, SAVING = False, 0, ''

            while not done:

                self.env.render()
                action = self.act(state)

                next_state, reward, done, _ = self.step(action)
                next_state = np.reshape(next_state, [1, self.state_space])

                self.remember(state, action, reward)

                state = next_state
                score += reward
                self.PlotModel(score, e)

                if done:
                    average = self.PlotModel(score, e)
                    if average >= self.max_average:
                        self.max_average = average
                        self.save()
                        SAVING = "SAVING"
                    else:
                        SAVING = ""
                        print("episode: {}/{}, score: {}, average: {:.2f} {}".format(e,
                              self.EPISODES, score, average, SAVING))

                    # Update step
                    self.replay()

        self.env.close()

    pylab.figure(figsize=(18, 9))

    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        if str(episode)[-2:] == "00":  # much faster than episode % 100
            pylab.plot(self.episodes, self.scores, 'b')
            pylab.plot(self.episodes, self.average, 'r')
            pylab.title(self.agent_name + self.env_name, fontsize=18)
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Episodes', fontsize=18)
            try:
                pylab.savefig(self.path+".png")
            except OSError:
                pass
        return self.average[-1]


if __name__ == "__main__":
    env_name = 'CartPole-v1'
    agent = PGAgent(env_name)
    agent.run()
