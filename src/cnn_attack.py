import numpy as np
import cifar_loader as cl

# padding for the matrix of images
def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, ), (pad, ), (pad, ), (0, )), 'constant', constant_values=(0, 0))    
    return X_pad


def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    print(W.shape)
    print(A_prev.shape)

    n_H = int((n_H_prev + 2*pad - f)/stride + 1)
    n_W = int((n_W_prev + 2*pad - f)/stride + 1)
    
    # Initialize the output volume Z with zeros. 
    print((m, n_H, n_W, n_C))
    Z = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = zero_pad(A_prev, pad)
    for h in range(n_H):                            # loop over vertical axis of the output volume
        for w in range(n_W):                        # loop over horizontal axis of the output volume
            # Use the corners to define the (3D) slice of a_prev_pad.
            A_slice_prev = A_prev_pad[:, h*stride:h*stride+f, w*stride:w*stride+f, :]
            # print(np.tensordot(A_slice_prev, W, axes=([1,2,3],[0,1,2])).shape, b.shape)
            # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron.
            Z[:, h, w, :] = np.tensordot(A_slice_prev, W, axes=([1,2,3],[0,1,2])) # + b
                            
    assert(Z.shape == (m, n_H, n_W, n_C))
    cache = (A_prev, W, b, hparameters)
    return Z, cache


def initialize(kernel_shape, mode='Fan-in'):
    b_shape = (1,1,1,kernel_shape[-1]) if len(kernel_shape)==4 else (kernel_shape[-1],)
    if mode == 'Gaussian_dist':
        mu, sigma = 0, 0.1
        weight = np.random.normal(mu, sigma,  kernel_shape) 
        bias   = np.ones(b_shape)*0.01
        
    elif mode == 'Fan-in': #original init. in the paper
        Fi = np.prod(kernel_shape)/kernel_shape[-1]
        weight = np.random.uniform(-2.4/Fi, 2.4/Fi, kernel_shape)    
        bias   = np.ones(b_shape)*0.01     
    return weight, bias

class ConvLayer(object):
    def __init__(self, kernel_shape, hparameters, init_mode='Gaussian_dist'):
        """
        kernel_shape: (n_f, n_f, n_C_prev, n_C)
        hparameters = {"stride": s, "pad": p}
        """
        self.hparameters = hparameters
        self.weight, self.bias = initialize(kernel_shape, init_mode)
        self.v_w, self.v_b = np.zeros(kernel_shape), np.zeros((1,1,1,kernel_shape[-1]))
        
    def foward_prop(self, input_map):
        output_map, self.cache = conv_forward(input_map, self.weight, self.bias, self.hparameters)
        return output_map
    
    def SDLM(self, d2Z, mu, lr_global):
        d2A_prev, d2W = conv_SDLM(d2Z, self.cache)
        h = np.sum(d2W)/d2Z.shape[0]
        self.lr = lr_global / (mu + h)
        return d2A_prev


def grayscale(pixels):

    ans = []
    for i in range(32*32):
        ans.append(int(0.299*pixels[i] + 0.587*pixels[i + 1024] + 0.114*pixels[i + 2048]))
    # print(ans)
    return np.array(ans)


def conv2d(input_image, filters, stride=1, padding=0):
    """
    Applies a 2D convolution operation.
    
    Parameters:
    - input_image: numpy array of shape (H, W), the input grayscale image.
    - filters: numpy array of shape (num_filters, filter_height, filter_width), the convolutional filters.
    - stride: int, the stride length for the convolution.
    - padding: int, the amount of zero-padding around the input image.
    
    Returns:
    - output: numpy array of shape (num_filters, output_height, output_width), the result of the convolution.
    """
    # Add padding to the input image
    if padding > 0:
        input_image = np.pad(input_image, ((padding, padding), (padding, padding)), mode='constant')

    # Get dimensions
    H, W = input_image.shape
    num_filters, filter_height, filter_width = filters.shape

    # Calculate output dimensions
    output_height = (H - filter_height) // stride + 1
    output_width = (W - filter_width) // stride + 1

    # Initialize output
    output = np.zeros((num_filters, output_height, output_width))

    A = []
    y = []
    # Perform the convolution
    for f in range(num_filters):
        current_filter = filters[f]
        for i in range(0, H - filter_height + 1, stride):
            for j in range(0, W - filter_width + 1, stride):
                # Extract the current region of interest
                # msk = [0] * (output_width * output_height)
                cf = [0] * (H * W)
                # print(i, j, H, W)
                region = input_image[i:i+filter_height, j:j+filter_width]
                for k in range(i, i + filter_height):
                    for m in range(j, j + filter_width):
                        # for p in range(filter_width):
                            # print(k, m)
                            # msk[k * output_width + m] = 1
                        cf[k * W + m] += np.sum(current_filter[m - j])

                # Perform element-wise multiplication and sum
                output[f, i//stride, j//stride] = np.sum(np.matmul(region, current_filter))
                # print(cf) 
                A.append(np.array(cf))
                y.append(output[f, i//stride, j//stride]);
                
    return output, np.array(A), np.array(y)


x_init, y = cl.load_data('cifar10/data_batch_1')
x = np.array([grayscale(i).reshape(32, 32) for i in x_init[:5]])



# Define input image (32x32)
input_image = x[1]

# Initialize 6 filters of size 5x5 with random values
filters = np.random.randn(6, 5, 5)

# Define parameters
stride = 1
padding = 0  # To maintain the input dimension

# Apply convolution
output, A, y = conv2d(input_image, filters, stride=stride, padding=padding)

print(A.shape)
print(y.shape)

print(f"Output shape: {output.shape}")  # Should be (6, 32, 32)
# print(output)  


print(np.linalg.cond(A))

c = [0] * 1024

# A = A.transpose().tolist()
# print(A.shape)
# print(y.shape)

A = A[:1480]
y = y[:1480]

import scipy.optimize

# x = scipy.linalg.solve(A, y)
# print(x)

import attack
attack.show_image(x[1])

# x = scipy.optimize.linprog(c, A_ub = ineq, b_ub = aineq, A_eq = w, b_eq = a, bounds=[0, 255], method = "interior-point")
x = scipy.optimize.linprog(c, A_eq = A, b_eq = y, bounds=[0, 255] , method = "interior-point", options={'cholesky':True})
# x = scipy.optimize.linprog(c, A_eq = A, b_eq = y, bounds=[0, 255] , method = "interior-point", options={'sym_pos':False})
# x = scipy.optimize.linprog(c, A_eq = A, b_eq = y, bounds=[0, 255] , options={'cholesky':True})
# x = scipy.optimize.linprog(c, A_eq = A, b_eq = y, bounds=[0, 255] )
print(x)


attack.show_image(x['x'])



# hparameters_convlayer = {"stride": 1, "pad": 0}
# C1 = ConvLayer((5, 5, 1, 6), hparameters_convlayer)

# print(C1.foward_prop(x)[0].shape)
