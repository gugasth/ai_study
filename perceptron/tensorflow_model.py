import tensorflow as tf

# Initial inputs and weights
X = tf.constant([1.0, 0.0])  # Input features
W = tf.Variable([0.5, 0.5], dtype=tf.float32)  # Weights
bias = tf.Variable(0.1, dtype=tf.float32)  # Bias

# Step activation function
def step_function(x):
    return tf.where(x > 0, 1.0, 0.0)

# Compute perceptron output
output = step_function(tf.tensordot(X, W, axes=1) + bias)

print("Perceptron output (TensorFlow):", output.numpy())
