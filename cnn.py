import numpy as np
import fenc as fe
import cifar_loader as cl
from threading import Thread
import time
from memory_profiler import profile 
import sys
# x =[np.array(a).reshape(1, 30), np.array(b).reshape(1, 30), 
														# np.array(c).reshape(1, 30)]

IMG_COLOR = 3
IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_SZ = IMG_HEIGHT * IMG_WIDTH
LAYER_1 = 2000
# OUTPUT_LAYER = 1
LABEL_CNT = 10

ENCRYPTION = True

def grayscale(pixels):

	 ans = []
	 for i in range(IMG_SZ):
		  ans.append(int(0.299*pixels[i] + 0.587*pixels[i + 1024] + 0.114*pixels[i + 2048]))
	 # print(ans)
	 return np.array(ans)


def add_enc_image(image, lst):

	 enc_image = feip.encrypt(image)
	 lst.append(enc_image)


def load_data(file, limit = None):
	x_init, y = cl.load_data(file)
	enc_x_feip = []

	if limit != None:
		y = y[:limit]
		x_init = x_init[:limit]

	x = [grayscale(i).reshape(1, IMG_SZ) for i in x_init]

	if ENCRYPTION:
		global feip
		feip = fe.FEIP()
		feip.setup(x[1].shape[1])
		threads = []	

		start = time.time()

		k = 0
		for image in x:
			 print("encryption image #", k)
			 k += 1
				  
			 t = Thread(target=add_enc_image, args=[image.tolist()[0], enc_x_feip])
			 threads.append(t)

			 t.start()

		for t in threads:
			 t.join()

		end_time = time.time()

		print("Encryption done in ", end_time - start, "s")
		
	y = np.array(y)
	print("data loaded")
	return x, y, enc_x_feip

# activation function
 
def sigmoid(x):
	  return(1/(1 + np.exp(-x)))
	 
offset = []

def f_forward(X, parameters):
	 # hidden
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]
	
	# compute the activation of the hidden layer
	if ENCRYPTION:
		Z1 = first_layer_prep(X, W1)
	else:
		Z1 = np.dot(W1, X.T) + b1 #why? Here they were working with transpuse
	
	A1 = sigmoid(Z1) # out put of layer 2
	
	# compute the activation of the output layer
	Z2 = np.dot(W2, A1) 
	Z2 += b2
	A2 = sigmoid(Z2)
	
	cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
	
	return A2, cache


def inner_product(v, y):

	 inner_sum = 0.0

	 for i in range(len(y)):
		  inner_sum += v[0][i] * y[i]

	 return inner_sum

def normalize_w(w, k):

	 for i in range(len(w)):
		  w[i] = int(w[i] * k)

	 return w


def secure_inner_product(x, y):

	 y = normalize_w(y, 10 ** 2)

	 skf = feip.key_derive(y)
	 ans = feip.decrypt(x, skf, y)

	 return ans
	 # y = normalize_w(y, 10 **(-6))


def first_layer_prep(x, w1):

	 z = []
	 # threads = []

	 for i in range(w1.shape[1]):
		  # t = threading.Thread(target=, args=[]) 
		  z.append(secure_inner_product(x, w1[ :, i].tolist()) / 100.)

	 return np.array(z).reshape(1, LAYER_1)

# initializing the weights randomly
def generate_wt():
	global IMG_SZ, LAYER_1, LABEL_CNT
	input_size = IMG_SZ
	hidden_size = LAYER_1
	output_size = LABEL_CNT
	np.random.seed(0)
	W1 = np.random.randn(hidden_size, input_size) * 0.01
	b1 = np.zeros((hidden_size, 1))
	W2 = np.random.randn(output_size, hidden_size) * 0.01
	b2 = np.zeros((output_size, 1))
	parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

	return parameters
	
		
# for loss we will be using mean square error(MSE)
def loss(out, y):
    m = y.shape[0]
    loss = -(1/m) * np.sum(y*np.log(out) + (1-y)*np.log(1-out))
    return loss
	# s =(np.square(out-Y))
	# s = np.sum(s)/len(y)
	# pos1 = np.argmax(out)
	# pos2 = np.argmax(Y)
	# outcome = 0
	# if pos1 == pos2:
	# 	outcome = 1
	# return (s, outcome)
	 
# Back propagation of error 
def back_prop(x, y, parameters, cache):
	m = y.shape[0]
	
	# retrieve the intermediate values
	Z1 = cache["Z1"]
	A1 = cache["A1"]
	Z2 = cache["Z2"]
	A2 = cache["A2"]

	# compute the derivative of the loss with respect to A2
	y = y.reshape(A2.shape)
	# dA2 = A2 - y
	dA2 = - (y/A2) + ((1-y)/(1-A2))
	# print(dA2.shape, A2.shape, y.shape)
	
	# compute the derivative of the activation function of the output layer
	dZ2 = dA2 * (A2 * (1-A2))
	
	# compute the derivative of the weights and biases of the output layer
	# print(dZ2.shape, A1.T.shape)
	dW2 = (1/m) * np.dot(dZ2, A1.T)
	db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
	
	# compute the derivative of the activation function of the hidden layer
	dA1 = np.dot(parameters["W2"].T, dZ2)
	dZ1 = dA1 * (A1 * (1-A1))
	
	# compute the derivative of the weights and biases of the hidden layer
	dW1 = (1/m) * np.dot(dZ1, x)
	db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
	gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

	return gradients

	
def update_parameters(parameters, gradients, learning_rate):
	# retrieve the gradients
	dW1 = gradients["dW1"]
	db1 = gradients["db1"]
	dW2 = gradients["dW2"]
	db2 = gradients["db2"]
	
	# retrieve the weights and biases
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]
	
	# update the weights and biases
	W1 = W1 - learning_rate*dW1
	b1 = b1 - learning_rate*db1
	W2 = W2 - learning_rate*dW2
	b2 = b2 - learning_rate*db2
	
	parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
	
	return parameters

def binary_cross_entropy_loss(A2, y):
    m = y.shape[0]
    loss = -(1/m) * np.sum(np.dot(y, np.log(A2)) + np.dot((1-y),np.log(1-A2)))
    return loss

def train(x, Y, parameters, alpha = 0.01, epoch = 10):
	acc =[]
	losss =[]

	for j in range(epoch):
		l =[]
		acc_1 = 0
		loss_1 = 0
		for i in range(len(x)):
			image = x[i]
			if ENCRYPTION:
				image = enc_x_feip[i]
			out, cache = f_forward(image, parameters)
			loss = binary_cross_entropy_loss(out, y)
			# # lo = loss(out, Y[i])
			# acc_1 += lo[1]
			# loss_1 += lo[0]
			gradients = back_prop(x[i], y[i], parameters, cache)
			parameters = update_parameters(parameters, gradients, alpha)
			if i % 100 == 0:
				print(f"iteration {i}: loss = {loss}")

		# print("epochs:", j + 1, "======== acc:", acc_1)  

		# losss.append(loss_1/len(x))
		# acc.append((acc_1/len(x))*100)
	# return(acc, losss, parameters)
	return parameters
 
def test(x, Y, parameters):
	acc =[]
	losss =[]

	l =[]
	for i in range(len(x)):
		image = x[i]
		if ENCRYPTION:
			image = enc_x_feip[i]
		out, cache = f_forward(image, parameters)
		l.append((loss(out, Y[i])))

	print("test  ===X===X=== acc:", (1-(sum(l)/len(x)))*100)   
	acc.append((1-(sum(l)/len(x)))*100)
	losss.append(sum(l)/len(x))
	return(acc, losss, parameters)
  
def predict(x, parameters):

	  Out, cache = f_forward(x, parameters)
	  maxm = 0
	  k = 0
	  for i in range(len(Out[0])):
			 if(maxm<Out[0][i]):
					 maxm = Out[0][i]
					 k = i
	  if(k == 0):
			 print("Image is of letter A.")
	  elif(k == 1):
			 print("Image is of letter B.")
	  else:
			 print("Image is of letter C.")
	  plt.imshow(x.reshape(5, 6))
	  plt.show()   
	 
  
"""The arguments of train function are data set list x, 
correct labels y, weights w1, w2, learning rate = 0.1, 
no of epochs or iteration.The function will return the
matrix of accuracy and loss and also the matrix of 
trained weights w1, w2"""




parameters = generate_wt()

# print(w1, "\n\n", w2)

x, y, enc_x_feip = load_data('cifar10/data_batch_1', 500)
print("aiiici", x[0].shape, y[0].shape)


# acc, losss, parameters = train(x, y, parameters, 0.08, 300)
parameters = train(x, y, parameters, 0.08, 300)

import matplotlib.pyplot as plt1
 
# plotting accuracy
plt1.plot(acc)
plt1.ylabel('Accuracy')
plt1.xlabel("Epochs:")
plt1.show()
 
# plotting Loss
plt1.plot(losss)
plt1.ylabel('Loss')
plt1.xlabel("Epochs:")
plt1.show()

x, y, enc_x_feip = load_data('cifar10/test_batch', 100)
acc, losss, parameters = test(x, y, parameters)



# the trained weights are
print(parameters["W1"], "\n", parameters["W2"])



"""
The predict function will take the following arguments:
1) image matrix
2) w1 trained weights
3) w2 trained weights
"""
# print("x1", x[1], type(x[1]))
# predict(x[1], parameters)
