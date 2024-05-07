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


def augment(w, a, n, t): #t is threshold
	# print(len(w[0]))

	for i in range(32):
		for j in range(32):
			if i % 2 == 0 and j % 2 == 0 :
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


w = np.load("data/w.npy")
a = np.load("data/a.npy")
x = np.load("data/x.npy")[0]
attack.show_image(x)

# print(x)

a = a.tolist()[0]
c = [0] * len(x)
# for i in range(len(x) // 2):
	# c[2 * i] = -1
lb = [0] * len(x)
ub = [256] * len(x)

w = w.transpose().tolist()
ineq = []
aineq = []
augment(ineq, aineq, len(x), 3.)


import scipy.optimize

print(len(x))
# w = asarray(w).transpose()#.tolist()
# print(w)
# a = asarray(a)
# print(a.shape)

# x = np.linalg.lstsq(w, a)[0]
x = scipy.optimize.linprog(c, A_ub = ineq, b_ub = aineq, A_eq = w, b_eq = a, bounds=[0, 255] )
print(len(a))
# x = scipy.optimize.linprog(c, A_eq = w, b_eq = a, bounds=[0, 255] )
print(x)

attack.show_image(x['x'])
exit(0)

