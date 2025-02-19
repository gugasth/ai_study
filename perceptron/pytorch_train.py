import torch

# Define input data and target labels (for OR logic gate)
X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])
y = torch.tensor([0.0, 1.0, 1.0, 1.0])  # Expected output (target labels)

# Initialize weights and bias
W = torch.randn(2, requires_grad=True)  # Random weights
bias = torch.randn(1, requires_grad=True)  # Random bias

# Hyperparameters
learning_rate = 0.1
epochs = 10

# Step activation function
def step_function(x):
    return torch.where(x > 0, torch.tensor(1.0), torch.tensor(0.0))

# Training loop
for epoch in range(epochs):
    total_error = 0
    for i in range(len(X)):
        weighted_sum = torch.dot(X[i], W) + bias  # Compute weighted sum
        y_pred = step_function(weighted_sum)  # Apply step function
        error = y[i] - y_pred  # Compute error

        # Compute gradients manually (since we're not using an optimizer)
        W.grad = None
        bias.grad = None

        loss = error**2  # Simple squared error
        loss.backward()  # Compute gradients

        # Update weights and bias
        with torch.no_grad():
            W += learning_rate * W.grad
            bias += learning_rate * bias.grad

        total_error += abs(error.item())

    print(f"Epoch {epoch+1}: Total Error = {total_error}")
    
    if total_error == 0:
        break  # Stop if the perceptron converged

# Testing phase
for i in range(len(X)):
    output = step_function(torch.dot(X[i], W) + bias)
    print(f"Input: {X[i].numpy()}, Predicted Output: {output.item()}")
