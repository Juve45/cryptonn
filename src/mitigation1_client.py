import numpy as np
from utils import cifar_loader as cl
from threading import Thread
import time
from memory_profiler import profile 
import sys
from mlsocket import MLSocket
from utils.numpy_socket import NumpySocket
import socket 
# x =[np.array(a).reshape(1, 30), np.array(b).reshape(1, 30), 
														# np.array(c).reshape(1, 30)]

import settings



def create_socket(offset = 0):
	s = NumpySocket()
	s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
	s.connect((settings.CLOUD_IP, settings.CLOUD_PORT + offset)) # Connect to the port and host
	return s


def client_send(s, data):
	# print("send", data)
	s.sendall(data)


def client_recieve(s):
	x = s.recv(1024)
	# print("recv:", x)
	return x
 

def sigmoid(x):
	  return(1/(1 + np.exp(-x)))
	 
offset = []

def encrypt_matrix(X):
	print(X.shape)
	xenc = []

	for i in range(X.shape[0]):
		xenc.append(ipe.encrypt())

def f_forward(X, parameters, s):
	 # hidden
	W1 = parameters["W1"]
	print(X.shape)
	# compute the activation of the hidden layer
	if settings.ENCRYPTION:
		# Z1 = first_layer_prep(X, W1)
		Xenc = encrypt_matrix(X)
		client_send(s, Xenc)	
	else:
		Z1 = np.dot(W1, X.T) # client
		client_send(s, Z1)	
	


# initializing the weights randomly
def generate_wt():

	input_size = settings.IMG_SZ
	hidden_size = settings.LAYER_1
	output_size = settings.LABEL_CNT
	np.random.seed(0)
	W1 = np.random.randn(hidden_size, input_size) * 0.01
	
	parameters = {"W1": W1, "b1": 0, "W2": 0, "b2": 0}

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
def back_prop(x,y, conn):
	m = y.shape[0]
	vect = []


	dZ1 = client_recieve(conn) 

	dW1 = (1/m) * np.dot(dZ1, x)
	db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
	gradients = {"dW1": dW1, "db1": db1}

	return gradients

	
def update_parameters(parameters, gradients, learning_rate):
	# retrieve the gradients

	dW1 = gradients["dW1"]
	
	# retrieve the weights and biases
	W1 = parameters["W1"]
	# update the weights and biases
	W1 = W1 - learning_rate*dW1

	parameters = {"W1": W1}
	
	return parameters

def train(x, Y, parameters, alpha = 0.01, epoch = 10, batch = 15):
	acc =[]
	losss =[]
	s = create_socket()
	for j in range(epoch):
		l =[]
		acc_1 = 0
		loss_1 = 0
		for i in range(0, len(x), batch):

			image = np.concatenate(x[i:i+batch])
			if settings.ENCRYPTION:
				# not working
				image = enc_x_feip[i:i+batch]
			f_forward(image, parameters, s)
			
			gradients = back_prop(np.concatenate(x[i:i+batch]), np.concatenate(y[i:i+batch]), s)
			parameters = update_parameters(parameters, gradients, alpha)
			

		print("epochs:", j + 1, "======== acc:", acc_1)  

		losss.append(loss_1/len(x))
		acc.append((acc_1/len(x))*100)
	s.close()
	return(acc, losss, parameters)
  

def test(x, Y, parameters):
	acc =[]
	losss =[]
	s = create_socket(5)
	acc_1 = 0
	loss_1 = 0
	for i in range(0, len(x)):

		image = np.concatenate(x)
		if settings.ENCRYPTION:
			image = enc_x_feip[i:i+batch]
		f_forward(image, parameters, s)
		

	print("test ======== acc:", acc_1)  

	losss.append(loss_1/len(x))
	acc.append((acc_1/len(x))*100)
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

from utils import MNISTReader
mnist = MNISTReader.MnistDataloader("../MNIST/train-images.idx3-ubyte", "../MNIST/train-labels.idx1-ubyte", "../MNIST/t10k-images.idx3-ubyte", "../MNIST/t10k-labels.idx1-ubyte")


x, y, x_test, y_test = mnist.load_data_nn();
x = x
y = y
x_test = x_test
y_test = y_test
# print("aiiici", x[0].shape, y[0].shape)


acc, losss, parameters = train(x, y, parameters, 0.005, 2, 10)
# parameters = train(x, y, parameters, 0.08, 20)

import matplotlib.pyplot as plt1


 
# plotting accuracy
plt1.plot(acc)
plt1.ylabel('Accuracy')
plt1.xlabel("Epochs:")
plt1.show()
 
# # plotting Loss
# plt1.plot(losss)
# plt1.ylabel('Loss')
# plt1.xlabel("Epochs:")
# plt1.show()


acc, losss, parameters = test(x_test, y_test, parameters)


print(acc, losss)

# plotting accuracy
# plt1.plot(acc)
# plt1.ylabel('Accuracy')
# plt1.xlabel("Epochs:")
# plt1.show()
 
# # plotting Loss
# plt1.plot(losss)
# plt1.ylabel('Loss')
# plt1.xlabel("Epochs:")
# plt1.show()

# the trained weights are
print(parameters["W1"])



"""
The predict function will take the following arguments:
1) image matrix
2) w1 trained weights
3) w2 trained weights
"""
# print("x1", x[1], type(x[1]))
# predict(x[1], parameters)
