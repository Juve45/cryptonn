import torch
import torch.nn as nn
from utils import cifar_loader as cl
import numpy as np 
import scipy

# Input: batch of 4 CIFAR-10 images
x = torch.randn(1, 3, 32, 32)


x_init, y = cl.load_data('../cifar10/data_batch_1')

x = x_init[:100]



# def visual(x):
#     import numpy as np
#     import matplotlib.pyplot as plt

#     # Assume `img` is your NumPy array of shape (3, 32, 32)
#     # Convert from (C, H, W) to (H, W, C)
#     img = x.reshape(3, 32, 32)  # example image

#     print(img)

#     img_display = np.transpose(img, (1, 2, 0))

#     # Plot
#     plt.imshow(img_display)
#     plt.axis('off')
#     plt.show()


def augment(w, a, t, n = 32*32): #t is threshold

    positions = set()
    for i in range(32):
        positions.add((i, 1))
        positions.add((1, i))
        positions.add((30, i))
        positions.add((i, 30))

    for (i, j) in positions:
        if (i+j) % 2 == 0:
            continue
        pos = []
        for x, y in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
            ii = i + x
            jj = j + y
            if 0 <= ii < 32 and 0 <= jj < 32:
                pos.append(ii * 32 + jj) 
                # pos.append(1024 + ii * 32 + jj) 
                # pos.append(2048 + ii * 32 + jj) 
        
        row = [0] * 3 * n
        for ch in range(3):
            row[n*ch + i * 32 + j] = 1
            for p in pos:
                row[n* ch + p] = -1. / len(pos)
            # print(row)
            # print(len(row))
        w.append(row)
        a.append(t)

        row = [0] * 3 * n
        for ch in range(3):
            row[n * ch + i * 32 + j] = -1
            for p in pos:
                row[n* ch + p] = 1. / len(pos)
        w.append(row)
        a.append(t)

def attack_once(x, type, name):
    x0 = x
    img = x.reshape(1, 3, 32, 32) / 255.0  # Scale to [0.0, 1.0]
    # x = torch.from_numpy(img).unsqueeze(0).float()
    # First convolutional layer
    weight = np.random.randn(16, 3, 3, 3)
    x = img

    eqs = []
    ys=[]
    width = x.shape[2]
    for cc in range(3):
        for ch in range(x.shape[1]):
            for i in range(x.shape[2] - 2):
                for j in range(x.shape[3] - 2):
                    eq = [0] * 3072
                    y = 0
                    for ii in range(3):
                        for jj in range(3):
                            # print(ch*1024 + (i + ii)*width + (j + jj))
                            eq[ch*1024 + (i + ii)*width + (j + jj)] = float(weight[cc][ch][ii][jj])
                            y += float(weight[cc][ch][ii][jj]) * float(x[0][ch][i + ii][j + jj])
                    # print(eq)
                    eqs.append(eq)
                    ys.append(y)

    if type == 'gaussian':
        x = scipy.linalg.lstsq(eqs, ys)[0] #gaussian elimination

    elif type == 'linprog':
        eqs = np.array(eqs[:2700])
        ys = np.array(ys[:2700])
        c = [0] * 3072
        w = []
        a = []
        if True:
            augment(w, a, 0.15)

        x = scipy.optimize.linprog(c, A_ub=w, b_ub=a, A_eq = eqs, b_eq = ys, bounds=[0, 1] , method = "interior-point", options={'cholesky':True})['x']


    # x = scipy.linalg.solve(eqs, ys) #gaussian elimination


    mse = ((x - x0/255)**2).mean()

    if len(name) <= 6:
        from utils import imgview  
        imgview.show_image_color(x, name+type+"_128_allch")
        # imgview.show_image_color(x0/255, name + "original")

    return mse


serr = 0
for i in range(100):
    err = attack_once(x[i], "linprog", "CIFAR"+str(i))
    serr += err
    print(err)

print("mean Err", serr/100)


