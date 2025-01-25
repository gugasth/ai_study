import numpy as np

# Define input data and target labels (for OR logic gate)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 1, 1, 1])  # Expected output (target labels)

# Initialize weights and bias
W = np.random.rand(2)  # Random small values for weights
bias = np.random.rand(1)[0]
learning_rate = 0.1  # Learning rate (controls update magnitude)
epochs = 10  # Number of iterations

# Step activation function
def step_function(x):
    return 1 if x > 0 else 0

# Training loop
for epoch in range(epochs):
    total_error = 0
    for i in range(len(X)):
        # Compute the perceptron output
        weighted_sum = np.dot(X[i], W) + bias
        y_pred = step_function(weighted_sum)
        
        # Compute the error
        error = y[i] - y_pred

        # Update weights and bias if there's an error
        W += learning_rate * error * X[i]
        bias += learning_rate * error

        total_error += abs(error)

    print(f"Epoch {epoch+1}: Total Error = {total_error}")
    print(f'New Weights: {W}')
    print(f'New bias: {bias}')
    # Stop training if no errors (convergence)
    if total_error == 0:
        break

# Final trained weights and bias
print("Trained Weights:", W)
print("Trained Bias:", bias)

# Test the perceptron
for i in range(len(X)):
    output = step_function(np.dot(X[i], W) + bias)
    print(f"Input: {X[i]}, Predicted Output: {output}")
