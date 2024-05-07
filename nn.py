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
LAYER_1 = 1000
LABEL_CNT = 10

x_init, y = cl.load_data()

y = y[:70]
print(y)

def grayscale(pixels):

    ans = []
    for i in range(IMG_SZ):
        ans.append(int(0.299*pixels[i] + 0.587*pixels[i + 1024] + 0.114*pixels[i + 2048]))
    # print(ans)
    return np.array(ans)


x = [grayscale(i).reshape(1, IMG_SZ) for i in x_init[:2]]

# print("x=", x[0].tolist())

print(x[0].shape)


enc_x_feip = []
feip = fe.FEIP()
feip.setup(x[1].shape[1])
threads = []

def add_enc_image(image, lst):

    enc_image = feip.encrypt(image)
    lst.append(enc_image)


start = time.time()

k = 0
for image in x:
    print(k)
    k += 1
        
    t = Thread(target=add_enc_image, args=[image.tolist()[0], enc_x_feip])
    threads.append(t)

    t.start()

for t in threads:
    t.join()

end_time = time.time()

print(end_time - start, "s")


# enc_x_febo

# Labels are also converted into NumPy array
y = np.array(y)
 
 
# print(x, "\n\n", y)




# activation function
 
def sigmoid(x):
     return(1/(1 + np.exp(-x)))
    
# Creating the Feed forward neural network
# 1 Input layer(1, 30)
# 1 hidden layer (1, 5)
# 1 output layer(3, 3)


def f_forward(x, w1, w2):
    # hidden

    print("first layer start", file=sys.stderr)
    start = time.time()
    z1 = first_layer_prep(x, w1)
    end = time.time()
    # print(x)
    # printsys.err.
    print("first layer done, time: ", end - start, "s", file=sys.stderr)
    print("w=", w1.tolist())
    print("a=", z1.tolist())
    print("first layer done, time: ", end - start, "s", file=sys.stderr)

    a1 = sigmoid(z1)# out put of layer 2 

    # Output layer
    z2 = a1.dot(w2)# input of out layer
    a2 = sigmoid(z2)# output of out layer


    a1 = None
    z2 = None
    z1 = None
         
    return(a2)


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
def generate_wt(x, y):
     l =[]
     for i in range(x * y):
          l.append(np.random.randn())
     return(np.array(l).reshape(x, y))
      
# for loss we will be using mean square error(MSE)
def loss(out, Y):
     s =(np.square(out-Y))
     s = np.sum(s)/len(y)
     return(s)
    
# Back propagation of error 
def back_prop(x, y, w1, w2, alpha):
      
     # hidden layer
     z1 = x.dot(w1)# input from layer 1 
     a1 = sigmoid(z1)# output of layer 2 

     # Output layer
     z2 = a1.dot(w2)# input of out layer
     a2 = sigmoid(z2)# output of out layer


     # error in output layer
     d2 =(a2-y)
     d1 = np.multiply((w2.dot((d2.transpose()))).transpose(), 
                                              (np.multiply(a1, 1-a1)))
 
     # Gradient for w1 and w2
     w1_adj = x.transpose().dot(d1)
     w2_adj = a1.transpose().dot(d2)
      
     # Updating parameters
     w1 = w1-(alpha*(w1_adj))
     w2 = w2-(alpha*(w2_adj))
      
     a1 = None
     a2 = None
     z2 = None
     z1 = None
     
     return(w1, w2)
 
def train(x, Y, w1, w2, alpha = 0.01, epoch = 10):
     acc =[]
     losss =[]
     for j in range(epoch):
          l =[]
          for i in range(len(x)):
                out = f_forward(enc_x_feip[i], w1, w2)
                l.append((loss(out, Y[i])))
                w1, w2 = back_prop(x[i], y[i], w1, w2, alpha)
          print("epochs:", j + 1, "======== acc:", (1-(sum(l)/len(x)))*100)   
          acc.append((1-(sum(l)/len(x)))*100)
          losss.append(sum(l)/len(x))
     return(acc, losss, w1, w2)
  
def predict(x, w1, w2):

     Out = f_forward(x, w1, w2)
     # print(Out)
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



w1 = generate_wt(IMG_SZ, LAYER_1)
w2 = generate_wt(LAYER_1, LABEL_CNT)
# print(w1, "\n\n", w2)


acc, losss, w1, w2 = train(x, y, w1, w2, 0.1, 10)


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



# the trained weights are
print(w1, "\n", w2)



"""
The predict function will take the following arguments:
1) image matrix
2) w1 trained weights
3) w2 trained weights
"""
# print("x1", x[1], type(x[1]))
# predict(x[1], w1, w2)
