import tensorflow as tf
import numpy as np

# Define input data and target labels (for OR logic gate)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float32)
y = np.array([0, 1, 1, 1], dtype=np.float32)  # Expected output (target labels)

# Initialize weights and bias
W = tf.Variable(tf.random.normal([2]), dtype=tf.float32)  # Two weights
bias = tf.Variable(tf.random.normal([1]), dtype=tf.float32)  # Single bias

# Hyperparameters
learning_rate = 0.1
epochs = 10

# Step activation function
def step_function(x):
    return tf.where(x > 0, 1.0, 0.0)

# Training loop
for epoch in range(epochs):
    total_error = 0
    for i in range(len(X)):
        with tf.GradientTape() as tape:
            weighted_sum = tf.tensordot(X[i], W, axes=1) + bias  # Compute weighted sum
            y_pred = step_function(weighted_sum)  # Apply step function
            error = y[i] - y_pred  # Compute error
            loss = tf.square(error)  # Squared error loss

        # Compute gradients
        gradients = tape.gradient(loss, [W, bias])

        # Update weights and bias manually
        W.assign_add(learning_rate * gradients[0])
        bias.assign_add(learning_rate * gradients[1])

        total_error += abs(error.numpy())

    print(f"Epoch {epoch+1}: Total Error = {total_error}")

    if total_error == 0:
        break  # Stop if the perceptron converged

# Testing phase
for i in range(len(X)):
    output = step_function(tf.tensordot(X[i], W, axes=1) + bias)
    print(f"Input: {X[i]}, Predicted Output: {output.numpy()}")
