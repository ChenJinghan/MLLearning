# -*- coding: utf-8 -*-
# @Time    : 2019/4/20 10:21
# @Author  : Chen Jinghan
# @File    : RegressionDemo.py
import matplotlib.pyplot as plt
import numpy as np

x_data = [338., 333., 328., 207, 226., 25., 179., 60., 208., 606.]
y_data = [640., 633., 619., 393., 428., 27., 193., 66., 226., 1691]

# y_data = b + w * x_data

# Step1: initial b and w
b = -120
w = -4
# learning rate
lr = 0.000001
iteration = 100000
w_his = []
b_his = []

# iterations
# goodness of function F
# get the value of w and b, using Gradient Descent
for i in range(iteration):
    b_grad = 0.0
    w_grad = 0.0

    # Step2.1: partial derivative of function F with respect to variable w and b
    for j in range(len(x_data)):
        w_grad = w_grad + 2 * (y_data[j] - (b + w * x_data[j])) * (-x_data[j])
        b_grad = b_grad + 2 * (y_data[j] - (b + w * x_data[j])) * (-1)

    # Step2.2: update w and b
    w = w - lr * w_grad
    b = b - lr * b_grad

    # store parameters for plotting
    w_his.append(w)
    b_his.append(b)


# bias
x = np.arange(-200,-100,1)
# weight
y = np.arange(-5,5,0.1)
Z = np.zeros((len(x),len(y)))
# grid
X,Y = np.meshgrid(x,y)

for i in range(len(x)):
    for j in range(len(y)):
        b = x[i]
        w = y[j]
        Z[j][i] = 0
        for n in range(len(x_data)):
            Z[j][i] = Z[j][i] + (y_data[n] - b - w * x_data[n]) ** 2
        Z[j][i] = Z[j][i] / len(x_data)

# draw filled contour
# x,y specify the (x,y) coordinates of the surface
# Z：Hight?
# the alpha bleding value(颜色填充范围)
# colormap
plt.contourf(x,y,Z, 50, alpha = 0.5, cmap = plt.get_cmap('jet'))
plt.plot([-188.4],[2.67],'x',ms=12, markeredgewidth=3, color ='black')
plt.plot(b_his, w_his,'o-',ms=3, lw=1.5, color = 'black')
plt.xlim(-200,-100)
plt.ylim(-5,5)
plt.xlabel(r'$b$',fontsize = 16)
plt.ylabel(r'$w$',fontsize = 16)
plt.show()