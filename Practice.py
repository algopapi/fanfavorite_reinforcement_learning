import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


class Linear(keras.layers.Layer):
  def __init__(self, units=32, input_dim = 32):
    super(Linear, self).__init__()
    w_init = tf.random_normal_initializer()
    
    self.w = tf.Variable(
      initial_value = w_init(shape=(input_dim, units), dtype = "float32"),
      trainable = True
    )

    self.b = tf.Variable(
      initial_value = w_init(shape=(units,), dtype="float32"), 
      trainable = True
    )

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b

class ComputeSum(keras.layers.Layer):

  def __init__(self, input_dim):
    super(ComputeSum, self).__init__()
    self.total = tf.Variable(initial_value = tf.zeros((input_dim,)), trainable = False)

  def call(self, inputs):
    print("self total, ", self.total.numpy())
    self.total.assign_add(tf.reduce_sum(inputs, axis=0))
    return self.total


x = tf.ones((2,2))
print("x = ", x)
print("x shape = ", x.shape)
linear_layer  = Linear(4, 2)
compute_sum = ComputeSum(2)
y_sum = compute_sum(x)
y = linear_layer(x)
print("y =", y)
print("Y_sum", y_sum.numpy())
print("x = ", x)
y_sum = compute_sum(x)
print("Y_sum", y_sum.numpy())



class Dense(layers.Layer):
  def __init__(self, units):
    super(Dense, self).__init__()
    self.units = units
  
  def build(self, input_shape):
    self.w = self.add_weight(
      name ='w', 
      shape = (input_shape[-1], self.units),
      initializer = "random_normal", 
      trainable=True
    )

    self.b = self.add_weight(
      name = 'b', 
      shape = (self.units, ),
      initializer = 'zeros', 
      trainable = True
    )
  
  def call (self, inputs):
    return tf.matmul(inputs, self.w) + self.b

class MyReLU(layers.Layer):
  def __init__(self):
    super(MyReLU, self).__init__()
    
  def call(self, x):
    return tf.math.maximum(x, 0)

class MyModel(keras.Model):
  def __init__(self, num_classes = 10):
    super(MyModel, self).__init__()
    self.dense1 = Dense(64)
    self.dense2 = Dense(10)
    self.relu = MyReLU()
    
    # self.dense1 = layers.Dense(64)
    # self.dense2 = layers.Dense(num_classes)

  def call(self, input_tensor):
    x = self.relu(self.dense1(input_tensor))
    return self.dense2(x)

model = MyModel()
model.compile(
  loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  optimizer=keras.optimizers.Adam(),
  metrics=['accuracy']
)


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0


