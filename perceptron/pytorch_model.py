import torch

# Initial inputs and weights
X = torch.tensor([1.0, 0.0])  # Input features
W = torch.tensor([0.5, 0.5], requires_grad=True)  # Weights
bias = torch.tensor(0.1, requires_grad=True)  # Bias

# Step activation function
def step_function(x):
    return torch.where(x > 0, torch.tensor(1.0), torch.tensor(0.0))

# Compute perceptron output
output = step_function(torch.dot(X, W) + bias)

print("Perceptron output (PyTorch):", output.item())
