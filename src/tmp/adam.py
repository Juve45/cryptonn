# import matplotlib.pyplot as plt
# import numpy as np

# # Define a simple loss function
# def loss(x):
#   return (x - 2)**2

# # Define Adam optimizer
# def adam_update(param, grad, learning_rate, beta1, beta2, t):
#   """
#   Adam update algorithm is an optimization method used for training machine
#   learning models, particularly neural networks.

#   Intuition:
#   Adam combines the benefits of two other popular optimization algorithms:
#   AdaGrad and RMSProp.

#   1. AdaGrad adapts the learning rate to parameters, performing larger updates
#      for infrequent parameters and smaller updates for frequent ones. However,
#      its continuously accumulating squared gradients can lead to an overly
#      aggressive and monotonically decreasing learning rate.

#   2. RMSProp modifies AdaGrad by using a moving average of squared gradients to
#      adapt the learning rate, which resolves the radical diminishing learning
#      rates of AdaGrad.

#   Adam takes this a step further by:
#   - Calculating an exponentially moving average of the gradients (m) to smooth
#     out the gradient descent path, addressing the issue of noisy gradients.
#   - Computing an exponentially moving average of the squared gradients (v),
#     which scales the learning rate inversely proportional to the square root of
#     the second moments of the gradients. This helps in adaptive learning rate
#     adjustments.
#   - Implementing bias corrections to the first (m_hat) and second (v_hat) moment
#     estimates to account for their initialization at the origin, leading to more
#     accurate updates at the beginning of the training.

#   This results in an optimization algorithm that can handle sparse gradients on
#   noisy problems, which is efficient for large datasets and high-dimensional
#   parameter spaces.

#   Note: The function requires initialization or previous values of m_prev,
#   v_prev, and epsilon (a small number to prevent division by zero).
#   """

#   # Update biased first moment estimate.
#   # m is the exponentially moving average of the gradients.
#   # beta1 is the decay rate for the first moment.
#   m = beta1 * m_prev + (1 - beta1) * grad

#   # Update biased second raw moment estimate.
#   # v is the exponentially moving average of the squared gradients.
#   # beta2 is the decay rate for the second moment.
#   v = beta2 * v_prev + (1 - beta2) * grad**2

#   # Compute bias-corrected first moment estimate.
#   # This corrects the bias in the first moment caused by initialization at origin.
#   m_hat = m / (1 - beta1**(t + 1))

#   # Compute bias-corrected second raw moment estimate.
#   # This corrects the bias in the second moment caused by initialization at origin.
#   v_hat = v / (1 - beta2**(t + 1))

#   # Update parameters.
#   # Parameters are adjusted based on the learning rate, corrected first moment,
#   # and the square root of the corrected second moment.
#   # epsilon is a small number to avoid division by zero.
#   param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

#   # Return the updated parameters, as well as the first and second moment estimates.
#   return param, m, v


# def gradient(x, w, a):

#   sum = 0
#   for i in 


# # Initialize parameters and optimizer state
# param = np.array([np.random.randn()] * 5)
# m_prev = np.zeros_like(param)
# v_prev = np.zeros_like(param)
# learning_rate = 0.01
# beta1 = 0.9
# beta2 = 0.999
# epsilon = 1e-8

# # Track parameter values and loss over time
# param_history = [param]
# loss_history = [loss(param)]

# # Training loop
# epochs = 500
# for t in range(epochs):
#   # Calculate gradient
#   grad = gradient(param)

#   # Update parameter and optimizer state
#   param, m_prev, v_prev = adam_update(param, grad, learning_rate, beta1, beta2, t)

#   # Track parameter and loss
#   param_history.append(param)
#   loss_history.append(loss(param))

# # Plot results
# plt.figure(figsize=(8, 6))

# # Plot loss over time
# plt.plot(range(len(loss_history)), loss_history, label="Loss")
# plt.xlabel("Iteration")
# plt.ylabel("Loss")
# plt.legend()

# # Plot parameter trajectory
# plt.figure(figsize=(8, 6))
# plt.plot(range(len(param_history)), param_history, label="Parameter")
# plt.axhline(2, color="red", linestyle="--", label="Minimum")
# plt.xlabel("Iteration")
# plt.ylabel("Parameter Value")
# plt.legend()

# plt.show()




# gradient descent optimization with adam for a two-dimensional test function
from math import sqrt
from numpy import asarray
from numpy.random import rand
from numpy.random import seed
import numpy as np


# x = [1, 4, 6, 7, 2, 5, 6]

# w = [
# [0, 0, 1, 1],
# [1, 1, 0, 1],
# [1, 0, 0, 1],
# [0, -1, -1, 1],
# [1, 1, 0, 1],
# [2, 1, 1, 1],
# [0, 1, -1, 1]]

# a = [22, 10, -7, 31]
import attack

x= [61, 44, 48, 56, 78, 96, 112, 117, 123, 125, 108, 104, 120, 118, 111, 103, 112, 110, 101, 116, 115, 109, 110, 113, 126, 137, 142, 134, 135, 137, 130, 128, 18, 0, 10, 32, 58, 88, 95, 93, 94, 86, 77, 73, 76, 81, 77, 79, 86, 78, 74, 94, 97, 90, 96, 89, 86, 96, 98, 101, 102, 94, 90, 94, 23, 8, 31, 56, 79, 98, 99, 89, 83, 84, 85, 80, 78, 97, 93, 99, 96, 92, 85, 91, 100, 101, 108, 96, 95, 95, 98, 111, 100, 90, 90, 80, 26, 23, 60, 71, 79, 82, 80, 71, 75, 90, 96, 81, 83, 111, 98, 92, 84, 84, 94, 88, 99, 105, 105, 99, 92, 97, 97, 100, 97, 94, 91, 68, 36, 37, 72, 88, 86, 85, 86, 81, 78, 94, 96, 75, 77, 90, 86, 75, 70, 80, 87, 87, 100, 111, 101, 107, 100, 92, 87, 86, 73, 62, 52, 47, 52, 58, 79, 91, 97, 93, 92, 84, 77, 89, 88, 72, 77, 81, 79, 70, 66, 69, 62, 64, 75, 74, 62, 86, 107, 104, 87, 75, 60, 44, 16, 17, 74, 81, 91, 92, 97, 92, 83, 83, 91, 102, 87, 86, 77, 54, 47, 48, 62, 66, 42, 40, 42, 56, 104, 85, 101, 115, 95, 76, 66, 69, 66, 50, 88, 83, 97, 104, 96, 90, 99, 98, 95, 97, 98, 77, 46, 31, 26, 41, 121, 146, 87, 46, 43, 59, 89, 72, 94, 112, 102, 83, 75, 72, 98, 79, 107, 91, 98, 94, 88, 89, 93, 93, 87, 104, 100, 49, 32, 33, 32, 50, 129, 135, 109, 69, 50, 50, 47, 45, 75, 103, 103, 97, 95, 89, 92, 102, 126, 120, 122, 107, 95, 93, 93, 97, 103, 104, 65, 40, 39, 46, 56, 64, 90, 86, 95, 86, 68, 64, 56, 47, 70, 100, 100, 100, 100, 99, 93, 100, 128, 124, 124, 115, 105, 107, 109, 107, 116, 94, 48, 62, 55, 65, 100, 96, 101, 87, 94, 96, 104, 140, 106, 69, 74, 97, 95, 98, 104, 102, 94, 101, 120, 115, 115, 103, 98, 101, 90, 79, 107, 90, 70, 96, 89, 125, 125, 110, 124, 107, 133, 120, 159, 174, 152, 87, 72, 104, 103, 100, 102, 98, 95, 100, 119, 112, 111, 111, 116, 95, 61, 65, 84, 66, 89, 158, 130, 156, 176, 154, 170, 170, 173, 151, 163, 154, 127, 93, 86, 96, 99, 100, 101, 99, 98, 99, 135, 125, 113, 110, 100, 65, 49, 57, 54, 67, 117, 182, 141, 161, 218, 195, 193, 191, 180, 179, 175, 171, 164, 143, 126, 103, 103, 97, 99, 106, 98, 105, 119, 116, 115, 112, 103, 53, 63, 75, 61, 91, 159, 154, 108, 169, 203, 227, 227, 219, 219, 199, 195, 210, 205, 209, 194, 124, 97, 101, 101, 103, 96, 99, 121, 108, 108, 108, 102, 45, 48, 59, 76, 152, 197, 104, 117, 165, 171, 235, 251, 249, 233, 218, 210, 197, 218, 242, 228, 133, 98, 101, 97, 96, 94, 94, 121, 104, 105, 109, 96, 45, 64, 78, 128, 205, 186, 102, 146, 165, 193, 229, 238, 232, 224, 218, 181, 136, 193, 240, 214, 140, 124, 125, 120, 114, 95, 86, 117, 97, 106, 112, 99, 71, 80, 111, 184, 240, 151, 130, 183, 201, 210, 219, 213, 188, 166, 161, 172, 154, 163, 217, 205, 134, 116, 113, 113, 119, 116, 102, 121, 83, 86, 96, 97, 90, 65, 159, 227, 242, 141, 142, 217, 218, 171, 162, 148, 123, 121, 129, 159, 195, 196, 222, 215, 130, 92, 89, 89, 90, 105, 126, 113, 90, 96, 99, 97, 101, 148, 227, 247, 215, 142, 129, 188, 172, 95, 104, 96, 119, 132, 103, 128, 184, 228, 227, 200, 122, 81, 81, 73, 69, 75, 119, 111, 89, 94, 93, 94, 108, 198, 233, 200, 176, 135, 113, 153, 150, 94, 110, 113, 137, 113, 98, 116, 152, 186, 173, 178, 128, 75, 84, 92, 95, 84, 116, 111, 85, 94, 90, 88, 123, 123, 145, 170, 161, 141, 124, 156, 159, 116, 104, 113, 116, 108, 108, 105, 106, 117, 111, 115, 115, 96, 101, 109, 111, 114, 158, 109, 90, 99, 97, 99, 118, 97, 104, 144, 143, 158, 168, 165, 137, 115, 106, 108, 113, 117, 114, 110, 111, 111, 109, 100, 101, 114, 114, 111, 106, 129, 179, 111, 97, 98, 106, 120, 114, 111, 112, 130, 132, 148, 181, 174, 181, 118, 104, 109, 101, 114, 118, 116, 107, 80, 66, 63, 92, 126, 120, 100, 97, 142, 170, 115, 90, 97, 108, 113, 116, 120, 126, 140, 154, 131, 174, 201, 178, 169, 150, 113, 90, 105, 119, 116, 81, 49, 55, 74, 111, 122, 107, 94, 117, 163, 132, 135, 96, 94, 97, 98, 102, 116, 126, 129, 146, 128, 117, 155, 127, 129, 137, 116, 99, 93, 96, 91, 90, 100, 111, 112, 112, 104, 94, 105, 159, 182, 110, 161, 134, 106, 88, 85, 97, 106, 113, 114, 109, 98, 91, 116, 129, 107, 96, 95, 116, 115, 98, 103, 118, 121, 116, 119, 110, 98, 97, 136, 183, 156, 90, 176, 139, 124, 114, 94, 89, 102, 107, 104, 100, 85, 94, 107, 108, 102, 92, 91, 103, 117, 112, 103, 99, 110, 112, 115, 115, 106, 112, 170, 175, 97, 52, 182, 150, 145, 136, 120, 106, 106, 117, 114, 105, 93, 100, 111, 108, 97, 91, 91, 98, 110, 114, 110, 86, 92, 107, 113, 117, 103, 90, 157, 146, 34, 22, 172, 153, 156, 152, 144, 134, 123, 112, 101, 95, 102, 108, 105, 107, 100, 98, 99, 110, 114, 112, 118, 101, 84, 92, 108, 107, 71, 30, 109, 133, 35, 38, 146, 128, 143, 150, 155, 160, 149, 135, 120, 104, 104, 106, 102, 102, 100, 101, 105, 106, 105, 105, 103, 93, 86, 79, 88, 73, 36, 13, 100, 152, 69, 59, 150, 136, 146, 151, 166, 184, 185, 176, 161, 145, 136, 134, 134, 130, 122, 128, 131, 123, 112, 113, 117, 105, 93, 95, 102, 93, 72, 83, 144, 188, 123, 98]
attack.show_image(x)


a = [i for i in a]
c = [-1] * len(x)
lb = [0] * len(x)
ub = [256] * len(x)


import scipy.optimize

print(len(x))
w = asarray(w).transpose()#.tolist()
print(w)
# a = asarray(a)
# print(a.shape)

# x = np.linalg.lstsq(w, a)[0]
x = scipy.optimize.linprog(c, A_eq = w, b_eq = a, bounds=[0, 256] )
print(x)

attack.show_image(x['x'])
exit(0)

# a= [97025, 528576, -307106, 92151, -627653, 273075, -398587, 919000, 290708, 320318, 334519, 280166, -526413, 700811, -148981, -284088, 98791, -554414, 193806, -60598]



# objective function
def objective(x):
  ans = 0

  for i in range(len(a)):
    tmp = a[i]
    for j in range(len(x)):
      tmp -= x[j] * w[j][i]
    ans += tmp ** 2
  return ans
  

# derivative of objective function
def derivative(x):
  
  ct = [0] * len(a)
  for i in range(len(a)):
    for j in range(len(x)):
      ct[i] += x[j] * w[j][i]

  ans = [0] * len(x)
  for i in range(len(a)):
    for j in range(len(x)):
      ans[j] += 2 * w[j][i] * (ct[i] - a[i])
  for i in range(len(ans)):
    if x[i] <= 1.0:
      ans[i] = 0.0
    if x[i] > 255.0:
      ans[i] = 0.0
  return asarray(ans)

# gradient descent algorithm with adam
def adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2, eps=1e-8):
  # generate an initial point
  x = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
  score = objective(x)
  # initialize first and second moments
  print(bounds.shape[0])
  m = [0.0 for _ in range(bounds.shape[0])]
  v = [0.0 for _ in range(bounds.shape[0])]
  # run the gradient descent updates
  for t in range(n_iter):
    # calculate gradient g(t)
    g = derivative(x)
    # build a solution one variable at a time
    for i in range(x.shape[0]):
      # m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
      m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
      # v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
      v[i] = beta2 * v[i] + (1.0 - beta2) * g[i]**2
      # mhat(t) = m(t) / (1 - beta1(t))
      mhat = m[i] / (1.0 - beta1**(t+1))
      # vhat(t) = v(t) / (1 - beta2(t))
      vhat = v[i] / (1.0 - beta2**(t+1))
      # x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + eps)
      x[i] = x[i] - alpha * mhat / (sqrt(vhat) + eps)
    # evaluate candidate point
    score = objective(x)
    # report progress
    print('>%d f(%s) = %.5f' % (t, x, score))
  return [x, score]

# seed the pseudo random number generator
# seed(15)

for i in range(4000):
  seed(i)
  # define range for input
  bounds = asarray([[0, 255.0]] * 1024)
  # define the total iterations
  n_iter = 30
  # steps size
  alpha = 0.8
  # factor for average gradient
  beta1 = 0.9
  # factor for average squared gradient
  beta2 = 4.999
  # perform the gradient descent search with adam
  best, score = adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2)
  print('Done!')
  print('f(%s) = %f' % (best, score))






# x= [[61, 44, 48, 56, 78, 96, 112, 117, 123, 125, 108, 104, 120, 118, 111, 103, 112, 110, 101, 116, 115, 109, 110, 113, 126, 137, 142, 134, 135, 137, 130, 128, 18, 0, 10, 32, 58, 88, 95, 93, 94, 86, 77, 73, 76, 81, 77, 79, 86, 78, 74, 94, 97, 90, 96, 89, 86, 96, 98, 101, 102, 94, 90, 94, 23, 8, 31, 56, 79, 98, 99, 89, 83, 84, 85, 80, 78, 97, 93, 99, 96, 92, 85, 91, 100, 101, 108, 96, 95, 95, 98, 111, 100, 90, 90, 80, 26, 23, 60, 71, 79, 82, 80, 71, 75, 90, 96, 81, 83, 111, 98, 92, 84, 84, 94, 88, 99, 105, 105, 99, 92, 97, 97, 100, 97, 94, 91, 68, 36, 37, 72, 88, 86, 85, 86, 81, 78, 94, 96, 75, 77, 90, 86, 75, 70, 80, 87, 87, 100, 111, 101, 107, 100, 92, 87, 86, 73, 62, 52, 47, 52, 58, 79, 91, 97, 93, 92, 84, 77, 89, 88, 72, 77, 81, 79, 70, 66, 69, 62, 64, 75, 74, 62, 86, 107, 104, 87, 75, 60, 44, 16, 17, 74, 81, 91, 92, 97, 92, 83, 83, 91, 102, 87, 86, 77, 54, 47, 48, 62, 66, 42, 40, 42, 56, 104, 85, 101, 115, 95, 76, 66, 69, 66, 50, 88, 83, 97, 104, 96, 90, 99, 98, 95, 97, 98, 77, 46, 31, 26, 41, 121, 146, 87, 46, 43, 59, 89, 72, 94, 112, 102, 83, 75, 72, 98, 79, 107, 91, 98, 94, 88, 89, 93, 93, 87, 104, 100, 49, 32, 33, 32, 50, 129, 135, 109, 69, 50, 50, 47, 45, 75, 103, 103, 97, 95, 89, 92, 102, 126, 120, 122, 107, 95, 93, 93, 97, 103, 104, 65, 40, 39, 46, 56, 64, 90, 86, 95, 86, 68, 64, 56, 47, 70, 100, 100, 100, 100, 99, 93, 100, 128, 124, 124, 115, 105, 107, 109, 107, 116, 94, 48, 62, 55, 65, 100, 96, 101, 87, 94, 96, 104, 140, 106, 69, 74, 97, 95, 98, 104, 102, 94, 101, 120, 115, 115, 103, 98, 101, 90, 79, 107, 90, 70, 96, 89, 125, 125, 110, 124, 107, 133, 120, 159, 174, 152, 87, 72, 104, 103, 100, 102, 98, 95, 100, 119, 112, 111, 111, 116, 95, 61, 65, 84, 66, 89, 158, 130, 156, 176, 154, 170, 170, 173, 151, 163, 154, 127, 93, 86, 96, 99, 100, 101, 99, 98, 99, 135, 125, 113, 110, 100, 65, 49, 57, 54, 67, 117, 182, 141, 161, 218, 195, 193, 191, 180, 179, 175, 171, 164, 143, 126, 103, 103, 97, 99, 106, 98, 105, 119, 116, 115, 112, 103, 53, 63, 75, 61, 91, 159, 154, 108, 169, 203, 227, 227, 219, 219, 199, 195, 210, 205, 209, 194, 124, 97, 101, 101, 103, 96, 99, 121, 108, 108, 108, 102, 45, 48, 59, 76, 152, 197, 104, 117, 165, 171, 235, 251, 249, 233, 218, 210, 197, 218, 242, 228, 133, 98, 101, 97, 96, 94, 94, 121, 104, 105, 109, 96, 45, 64, 78, 128, 205, 186, 102, 146, 165, 193, 229, 238, 232, 224, 218, 181, 136, 193, 240, 214, 140, 124, 125, 120, 114, 95, 86, 117, 97, 106, 112, 99, 71, 80, 111, 184, 240, 151, 130, 183, 201, 210, 219, 213, 188, 166, 161, 172, 154, 163, 217, 205, 134, 116, 113, 113, 119, 116, 102, 121, 83, 86, 96, 97, 90, 65, 159, 227, 242, 141, 142, 217, 218, 171, 162, 148, 123, 121, 129, 159, 195, 196, 222, 215, 130, 92, 89, 89, 90, 105, 126, 113, 90, 96, 99, 97, 101, 148, 227, 247, 215, 142, 129, 188, 172, 95, 104, 96, 119, 132, 103, 128, 184, 228, 227, 200, 122, 81, 81, 73, 69, 75, 119, 111, 89, 94, 93, 94, 108, 198, 233, 200, 176, 135, 113, 153, 150, 94, 110, 113, 137, 113, 98, 116, 152, 186, 173, 178, 128, 75, 84, 92, 95, 84, 116, 111, 85, 94, 90, 88, 123, 123, 145, 170, 161, 141, 124, 156, 159, 116, 104, 113, 116, 108, 108, 105, 106, 117, 111, 115, 115, 96, 101, 109, 111, 114, 158, 109, 90, 99, 97, 99, 118, 97, 104, 144, 143, 158, 168, 165, 137, 115, 106, 108, 113, 117, 114, 110, 111, 111, 109, 100, 101, 114, 114, 111, 106, 129, 179, 111, 97, 98, 106, 120, 114, 111, 112, 130, 132, 148, 181, 174, 181, 118, 104, 109, 101, 114, 118, 116, 107, 80, 66, 63, 92, 126, 120, 100, 97, 142, 170, 115, 90, 97, 108, 113, 116, 120, 126, 140, 154, 131, 174, 201, 178, 169, 150, 113, 90, 105, 119, 116, 81, 49, 55, 74, 111, 122, 107, 94, 117, 163, 132, 135, 96, 94, 97, 98, 102, 116, 126, 129, 146, 128, 117, 155, 127, 129, 137, 116, 99, 93, 96, 91, 90, 100, 111, 112, 112, 104, 94, 105, 159, 182, 110, 161, 134, 106, 88, 85, 97, 106, 113, 114, 109, 98, 91, 116, 129, 107, 96, 95, 116, 115, 98, 103, 118, 121, 116, 119, 110, 98, 97, 136, 183, 156, 90, 176, 139, 124, 114, 94, 89, 102, 107, 104, 100, 85, 94, 107, 108, 102, 92, 91, 103, 117, 112, 103, 99, 110, 112, 115, 115, 106, 112, 170, 175, 97, 52, 182, 150, 145, 136, 120, 106, 106, 117, 114, 105, 93, 100, 111, 108, 97, 91, 91, 98, 110, 114, 110, 86, 92, 107, 113, 117, 103, 90, 157, 146, 34, 22, 172, 153, 156, 152, 144, 134, 123, 112, 101, 95, 102, 108, 105, 107, 100, 98, 99, 110, 114, 112, 118, 101, 84, 92, 108, 107, 71, 30, 109, 133, 35, 38, 146, 128, 143, 150, 155, 160, 149, 135, 120, 104, 104, 106, 102, 102, 100, 101, 105, 106, 105, 105, 103, 93, 86, 79, 88, 73, 36, 13, 100, 152, 69, 59, 150, 136, 146, 151, 166, 184, 185, 176, 161, 145, 136, 134, 134, 130, 122, 128, 131, 123, 112, 113, 117, 105, 93, 95, 102, 93, 72, 83, 144, 188, 123, 98]]
