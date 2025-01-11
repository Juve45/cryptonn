from math import sqrt
from numpy import asarray
from numpy.random import rand
from numpy.random import seed
import numpy as np
from utils import cifar_loader as cl

from utils import imgview



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

	for i in range(32):
		for j in range(32):
			if (i + j) % period == 0 :
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
				print(len(row))
				w.append(row)
				a.append(t)

				row = [0] * n
				row[i * 32 + j] = -1
				for p in pos:
					row[p] = 1. / len(pos)
				w.append(row)
				a.append(t)
	print(a)


				# 		pos.append(ii * 32 + jj) 
				# for p in pos:

def grayscale(pixels):

    ans = []
    for i in range(IMG_SZ):
        ans.append(int(0.299*pixels[i] + 0.587*pixels[i + 1024] + 0.114*pixels[i + 2048]))
    # print(ans)
    return np.array(ans)



def load_nn_train(count, layer1):

	x_init, y = cl.load_data('../cifar10/data_batch_1')
	x = [grayscale(i).reshape(1, IMG_SZ) for i in x_init[:count]]
	# x = [i.reshape(1, IMG_SZ) for i in x_init[:count]]
	# x = x_init[:count]
	print(x[0].shape)
	print(x[1].shape)
	print(x[2].shape)
	w1 = generate_wt(IMG_SZ, layer1)
	imgview.show_image(x[0])
	for i in range(len(x)):
		print(i, x[i].shape)
		z1 = x[i].dot(w1)# input of layer 1
		# z1 = z1 + np.random.randint(low = -10, high = 10, size=z1.shape)
		
		np.save("../data/w", w1)
		np.save("../data/x", x[i])
		np.save("../data/a", z1)

		# linprog_attack(4, "results/img%i_%i_4" % (i, layer1))
		# linprog_attack(9, "results/img%i_%i_9" % (i, layer1))
		# linprog_attack(16, "results/img%i_%i_16" % (i, layer1))

		xx = np.load("../data/x.npy")[0]
		imgview.show_image(xx, "img%i_original"%(i))


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
	x = scipy.optimize.linprog(c, A_ub = ineq, b_ub = aineq, A_eq = w, b_eq = a, bounds=[0, 255], method = "interior-point")
	# x = scipy.optimize.linprog(c, A_eq = w, b_eq = a, bounds=[0, 255] )
	print(x)

	imgview.show_image(x['x'], name)


load_nn_train(4, 1000)



exit(0)

