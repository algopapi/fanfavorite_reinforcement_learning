from re import X
import tensorflow as tf
import numpy as np

a = [[1, 2], [3, 4], [5, 6]]
b = [[1, 0], [0, 1],[1, 0]]

a = np.asarray(a)
b = np.asarray(b)


# Use one hot encodings for indices of array b  
c = tf.expand_dims(
    tf.reduce_sum(
        tf.multiply(
            x = a,
            y = b
        ),
        axis=-1
    ), 
    axis=-1
)

c = tf.expand_dims(tf.reduce_sum(tf.multiply(a, b), axis=-1), axis=-1)
print (c)

print("c = ", c)