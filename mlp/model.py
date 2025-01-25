import numpy as np

# Define input (single sample for prediction)
X = np.array([[0, 1]])  # Example input

# Initialize weights and biases manually (for simplicity)
W_hidden = np.array([[0.5, -0.6, 0.1],
                     [0.8, 0.2, -0.3]])  # 2 input features -> 3 hidden neurons
b_hidden = np.array([[0.1, -0.1, 0.05]])  # Bias for hidden layer

W_output = np.array([[0.7],
                     [-0.2],
                     [0.3]])  # 3 hidden neurons -> 1 output neuron
b_output = np.array([[0.05]])  # Bias for output layer

# Define activation functions
def relu(x):
    return np.maximum(0, x) 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Forward pass (input -> hidden layer)
Z_hidden = np.dot(X, W_hidden) + b_hidden  # Linear transformation
print(f'Z_hidden: {Z_hidden}')

A_hidden = relu(Z_hidden)  # Apply ReLU activation
print(f'A_hidden: {A_hidden}')

# Forward pass (hidden -> output layer)
Z_output = np.dot(A_hidden, W_output) + b_output  # Linear transformation
print(f'Z_output: {Z_output}')

A_output = sigmoid(Z_output)  # Apply Sigmoid activation

# Print final output (prediction)
print("Predicted output:", A_output)
