import matplotlib.pyplot as plt
import numpy as np

# Define a simple loss function
def loss(x):
  return (x - 2)**2

# Define Adam optimizer
def adam_update(param, grad, learning_rate, beta1, beta2, t):
  """
  Adam update algorithm is an optimization method used for training machine
  learning models, particularly neural networks.

  Intuition:
  Adam combines the benefits of two other popular optimization algorithms:
  AdaGrad and RMSProp.

  1. AdaGrad adapts the learning rate to parameters, performing larger updates
     for infrequent parameters and smaller updates for frequent ones. However,
     its continuously accumulating squared gradients can lead to an overly
     aggressive and monotonically decreasing learning rate.

  2. RMSProp modifies AdaGrad by using a moving average of squared gradients to
     adapt the learning rate, which resolves the radical diminishing learning
     rates of AdaGrad.

  Adam takes this a step further by:
  - Calculating an exponentially moving average of the gradients (m) to smooth
    out the gradient descent path, addressing the issue of noisy gradients.
  - Computing an exponentially moving average of the squared gradients (v),
    which scales the learning rate inversely proportional to the square root of
    the second moments of the gradients. This helps in adaptive learning rate
    adjustments.
  - Implementing bias corrections to the first (m_hat) and second (v_hat) moment
    estimates to account for their initialization at the origin, leading to more
    accurate updates at the beginning of the training.

  This results in an optimization algorithm that can handle sparse gradients on
  noisy problems, which is efficient for large datasets and high-dimensional
  parameter spaces.

  Note: The function requires initialization or previous values of m_prev,
  v_prev, and epsilon (a small number to prevent division by zero).
  """

  # Update biased first moment estimate.
  # m is the exponentially moving average of the gradients.
  # beta1 is the decay rate for the first moment.
  m = beta1 * m_prev + (1 - beta1) * grad

  # Update biased second raw moment estimate.
  # v is the exponentially moving average of the squared gradients.
  # beta2 is the decay rate for the second moment.
  v = beta2 * v_prev + (1 - beta2) * grad**2

  # Compute bias-corrected first moment estimate.
  # This corrects the bias in the first moment caused by initialization at origin.
  m_hat = m / (1 - beta1**(t + 1))

  # Compute bias-corrected second raw moment estimate.
  # This corrects the bias in the second moment caused by initialization at origin.
  v_hat = v / (1 - beta2**(t + 1))

  # Update parameters.
  # Parameters are adjusted based on the learning rate, corrected first moment,
  # and the square root of the corrected second moment.
  # epsilon is a small number to avoid division by zero.
  param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

  # Return the updated parameters, as well as the first and second moment estimates.
  return param, m, v


def gradient(x):

  sum = 0
  for i in 


# Initialize parameters and optimizer state
param = np.array([np.random.randn()] * 5)
m_prev = np.zeros_like(param)
v_prev = np.zeros_like(param)
learning_rate = 0.01
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Track parameter values and loss over time
param_history = [param]
loss_history = [loss(param)]

# Training loop
epochs = 500
for t in range(epochs):
  # Calculate gradient
  grad = gradient(param)

  # Update parameter and optimizer state
  param, m_prev, v_prev = adam_update(param, grad, learning_rate, beta1, beta2, t)

  # Track parameter and loss
  param_history.append(param)
  loss_history.append(loss(param))

# Plot results
plt.figure(figsize=(8, 6))

# Plot loss over time
plt.plot(range(len(loss_history)), loss_history, label="Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()

# Plot parameter trajectory
plt.figure(figsize=(8, 6))
plt.plot(range(len(param_history)), param_history, label="Parameter")
plt.axhline(2, color="red", linestyle="--", label="Minimum")
plt.xlabel("Iteration")
plt.ylabel("Parameter Value")
plt.legend()

plt.show()