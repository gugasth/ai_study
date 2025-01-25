import numpy as np

# Define XOR dataset
X = np.array([[0, 0], 
              [0, 1], 
              [1, 0], 
              [1, 1]])

Y = np.array([[0], [1], [1], [0]])  # Expected output (labels)

# Initialize weights and biases randomly
np.random.seed(42)
W_hidden = np.random.randn(2, 4)  # 2 input neurons, 4 hidden neurons
b_hidden = np.random.randn(1, 4)
W_output = np.random.randn(4, 1)  # 4 hidden neurons, 1 output neuron
b_output = np.random.randn(1, 1)

# Define hyperparameters
learning_rate = 0.1
epochs = 10000

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Training loop
for epoch in range(epochs):
    # Forward pass (input -> hidden)
    Z_hidden = np.dot(X, W_hidden) + b_hidden
    A_hidden = relu(Z_hidden)
    
    # Forward pass (hidden -> output)
    Z_output = np.dot(A_hidden, W_output) + b_output
    A_output = sigmoid(Z_output)

    # Compute loss (Binary Cross Entropy)
    loss = -np.mean(Y * np.log(A_output) + (1 - Y) * np.log(1 - A_output))

    # Backpropagation (gradient calculation)
    dA_output = A_output - Y  # Derivative of loss w.r.t output
    dW_output = np.dot(A_hidden.T, dA_output)
    db_output = np.sum(dA_output, axis=0, keepdims=True)

    dA_hidden = np.dot(dA_output, W_output.T) * relu_derivative(Z_hidden)
    dW_hidden = np.dot(X.T, dA_hidden)
    db_hidden = np.sum(dA_hidden, axis=0, keepdims=True)

    # Update weights and biases
    W_output -= learning_rate * dW_output
    b_output -= learning_rate * db_output
    W_hidden -= learning_rate * dW_hidden
    b_hidden -= learning_rate * db_hidden

    # Print loss occasionally
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Testing the trained model
Z_hidden = np.dot(X, W_hidden) + b_hidden
A_hidden = relu(Z_hidden)
Z_output = np.dot(A_hidden, W_output) + b_output
predictions = sigmoid(Z_output)
predictions = np.round(predictions)

print("\nFinal Predictions:")
print(predictions)
