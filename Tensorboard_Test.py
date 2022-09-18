from re import M
import tensorboard
import tensorflow as tf
import numpy as np

import os
import gym
import time

from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model, save_model, load_model, Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import TensorBoard
from keras.datasets import mnist

from matplotlib import pyplot



NAME = "Test_Model-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
# Load data
(train_X, train_y), (test_X, test_y) = mnist.load_data()

print("Image Shape", train_X[0].shape)


train_X, test_X = train_X / 255.0, test_X / 255.0

model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(512, activation="relu", kernel_initializer="he_uniform"),
    Dense(256, activation="relu", kernel_initializer="he_uniform"),
    Dense(64, activation="relu", kernel_initializer="he_uniform"),
    Dense(10, activation="linear", kernel_initializer="he_uniform")
])

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate = 0.0025),
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(train_X, train_y, epochs=10, callbacks=[tensorboard])