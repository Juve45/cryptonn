#
# This is a sample Notebook to demonstrate how to read "MNIST Dataset"
#
import numpy as np # linear algebra
import struct
from array import array
from os.path  import join

#
# MNIST Data Loader Class
#

IMG_SZ = 28 * 28

def to_np_img(x):
    # print(x[0])
    return [np.array(i).reshape(1, IMG_SZ) for i in x]

def categorical(pos):
    x = np.zeros(10)
    x[pos] = 1
    return x

def to_np_label(x):
    return [categorical(i) for i in x]


class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        print(size)
        for i in range(size):
            images.append([0] * rows * cols)
        print("ok")
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        print("started")
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        print("loaded train set")
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        print("loaded test set")
        return (x_train, y_train),(x_test, y_test)        

    def load_data_nn(self):
        print("started")
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        print("loaded train set")
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        print("loaded test set")
        return to_np_img(x_train), to_np_label(y_train), to_np_img(x_test), to_np_label(y_test)




# mnistloader = MnistDataloader("MNIST/train-images.idx3-ubyte", "MNIST/train-labels.idx1-ubyte", "MNIST/t10k-images.idx3-ubyte", "MNIST/t10k-labels.idx1-ubyte");
# a = mnistloader.load_data();
# print("all data loaded")


# x_train = to_np_img(a[0][0]);
# y_train = to_np_label(a[0][1]);
# x_test = to_np_img(a[1][0]);
# y_test = to_np_label(a[1][1]);
# print(y_test)

