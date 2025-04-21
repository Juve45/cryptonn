from math import sqrt
from numpy import asarray
from numpy.random import rand
from numpy.random import seed
import numpy as np
from utils import cifar_loader as cl

from utils import imgview
import random


IMG_COLOR = 3
IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_SZ = IMG_HEIGHT * IMG_WIDTH 
# IMG_SZ = IMG_HEIGHT * IMG_WIDTH * IMG_COLOR


# initializing the weights randomly
def generate_wt(x, y):
     l =[]
     for i in range(x * y):
          l.append(np.random.randn())
     return(np.array(l).reshape(x, y))
      

def augment(w, a, n, t, period): #t is threshold
	# print(len(w[0]))

	positions = []
	for i in range(32):
		for j in range(32):
			if (i + j) % period == 0:
				positions.append((i, j))

	# limit = int(32 * 32 / period)
	# random.shuffle(positions)
	# positions = positions[:limit]


	for (i, j) in positions:
		row = [0] * n
		row[i * 32 + j] = 1
		pos = []
		for x, y in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
			ii = i + x
			jj = j + y
			if 0 <= ii * 32 + jj < n:
				pos.append(ii * 32 + jj) 
		
		for p in pos:
			row[p] = -1. / len(pos)
		# print(row)
		# print(len(row))
		w.append(row)
		a.append(t)

		row = [0] * n
		row[i * 32 + j] = -1
		for p in pos:
			row[p] = 1. / len(pos)
		w.append(row)
		a.append(t)
	# print("a=", a)


				# 		pos.append(ii * 32 + jj) 
				# for p in pos:

def grayscale(pixels):

    ans = []
    for i in range(IMG_SZ):
        ans.append(int(0.299*pixels[i] + 0.587*pixels[i + 1024] + 0.114*pixels[i + 2048]))
    # print(ans)
    return np.array(ans)

import datetime


mse_res = [[], [], [], []]
t_res = [[], [], [], []]

def load_nn_train(count, layer1):

	x_init, y = cl.load_data('../cifar10/data_batch_1')
	x = [grayscale(i).reshape(1, IMG_SZ) for i in x_init[:count]]
	# x = [i.reshape(1, IMG_SZ) for i in x_init[:count]]
	# x = x_init[:count]
	print(x[0].shape)
	print(x[1].shape)
	print(x[2].shape)
	w1 = generate_wt(IMG_SZ, layer1)
	# imgview.show_image(x[0])
	for i in range(len(x)):
		print(i, x[i].shape)
		z1 = x[i].dot(w1)# input of layer 1
		# z1 = z1 + np.random.randint(low = -10, high = 10, size=z1.shape)
		
		np.save("../data/w", w1)
		np.save("../data/x", x[i])
		np.save("../data/a", z1)


		for ineq_period in [2, 4, 9, 16, 1000]:
		# for ineq_period in [9, 10]:
			a = datetime.datetime.now()
			mse = linprog_attack(ineq_period, 
				"../results2/img%i_%i_%i" % (i, layer1, ineq_period))
			b = datetime.datetime.now()
			t = (b-a).microseconds/1000
			mse_res[i].append(mse);
			t_res[i].append(t);
			print("img%i_%i_%i" % (i, layer1, ineq_period), mse, t, "ms")

		# exit(0)
		xx = np.load("../data/x.npy")[0]
		imgview.show_image(xx, "../results2/img%i_original"%(i))
		# imgview.show_image(xx)


def linprog_attack(period, name):
	w = np.load("../data/w.npy")
	a = np.load("../data/a.npy")
	x = np.load("../data/x.npy")[0]
	# imgview.show_image(x, "")

	a = a.tolist()[0]
	c = [0] * len(x)

	# lb = [0] * len(x)
	# ub = [256] * len(x)

	w = w.transpose().tolist()
	ineq = []
	aineq = []
	augment(ineq, aineq, len(x), 3., period)	

	import scipy.optimize

	# x = np.linalg.lstsq(w, a)[0]
	report = scipy.optimize.linprog(c, A_ub = ineq, b_ub = aineq, A_eq = w, b_eq = a, bounds=[0, 255], method = "interior-point")
	# x = scipy.optimize.linprog(c, A_eq = w, b_eq = a, bounds=[0, 255] )
	# print(report)

	mse = ((x/255 - report['x']/255)**2).mean()
	print("MSE ", name, mse)
	imgview.show_image(report['x'], name)
	return mse


print(mse_res)
print(t_res)

load_nn_train(4, 350)
load_nn_train(4, 500)
# load_nn_train(4, 750)
# load_nn_train(4, 1000)



exit(0)

