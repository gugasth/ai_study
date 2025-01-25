import numpy as np

# Initial inputs and weights
X = np.array([1, 0])    # Input features, e.g., true for some condition, false for another.
W = np.array([0.5, 0.5])   # Weights associated with the input features
bias = 0.1  # Bias term to shift the decision boundary

# Step activation function (threshold function)
def step_function(x):
    """
    Step function activation.
    Returns 1 if the input x is greater than 0, otherwise returns 0.
    """
    return 1 if x > 0 else 0

# Compute the perceptron output
output = step_function(np.dot(X, W) + bias)  # (1*0.5) + (0*0.5) + 0.1 

print("Perceptron output:", output)
