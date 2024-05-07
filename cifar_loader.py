import pickle
import numpy



def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_data(file):

	d = unpickle(file)

	y = []
	for i in d[b'labels']:
		v = [0] * 10
		v[i] = 1

		y.append(v)
	x = d[b'data']
	print(x[0].shape)

	return x, y

# load_data()

