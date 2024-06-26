import tensorflow as tf

#Tensor Initialization
x = tf.constant(4, shape=(1,1), dtype=tf.float32)
x = tf.constant([[1,2,3],[4,5,6]])
x = tf.ones((3,3))
x = tf.zeros((3,3))
x = tf.random.normal((3,3), mean=0, stddev=1)
print(x)

#Math Operations

#Indexing a Tensor

#Reshape a Tensor