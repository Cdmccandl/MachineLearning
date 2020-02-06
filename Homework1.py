"""
Conor McCandless
Machine Learning Homework 1
Building a linear regression model
"""
import numpy as np
import matplotlib.pyplot as plt
#Generate data samples
x_data = np.array([35.,38.,31.,20.,22.,25.,17.,60.,8.,60.])
y_data = 2*x_data+50+5*np.random.random()

#Plot landscape function
bb = np.arange(0,100,1) #the bias
ww = np.arange(-5, 5, 0.1) #weight
Z = np.zeros((len(bb),len(ww)))

for i in range(len(bb)):
    for j in range(len(ww)):
        b = bb[i]
        w = ww[j]
        Z[j][i] = 0
        for n in range(len(x_data)):
            Z[j][i] = Z[j][i] + (w*x_data[n] + b - y_data[n])**2 #loss
        Z[j][i] = Z[j][i]/len(x_data)
        
   
#find best w and b      
#ydata = b +w*xdata        
b = 0
w = 0
learnRate = .00001
iteration = 15000

#initial values for plot
b_history = [b]
w_history = [w]
        
#build gradient descent from equation
for i in range(iteration):
    b_grad = 0
    w_grad = 0
    for n in range (len(x_data)):
        b_grad = b_grad + (b + w*x_data[n] - y_data[n])*1.0
        w_grad = w_grad + (b + w*x_data[n] - y_data[n]) * x_data[n]
        #update b and w
        b = b - b_grad * learnRate
        w = w - w_grad * learnRate
        
        b_history.append(b)
        w_history.append(w)
        
plt.plot(b_history,w_history,'o-',ms=3, lw=1.5,color='black')
plt.xlim(0,100)
plt.ylim(-5,5)
plt.contourf(bb,ww,Z,50,alpha=0.5,cmap = plt.get_cmap('jet'))
plt.show()
#plt.plot(b_history, w_history, 'o-', ms=3,lw=1.5,color = 'black')