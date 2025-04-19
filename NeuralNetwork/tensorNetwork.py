import tensorflow as tf

scalar = tf.constant(5)

vector = tf.constant([1, 2, 3])

matrix = tf.constant([[1, 2], [3, 4]])

tensor3D = tf.constant([[[1], [2]], [[3], [4]]])

print(scalar)
print(vector)
print(matrix)
print(tensor3D)