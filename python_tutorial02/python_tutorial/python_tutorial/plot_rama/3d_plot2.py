import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np

x = range(10)
y = range(10)
z = [None]*10
for i in range(len(x)):
    z[i] = x[i]*y[i] 

fig = plt.figure()
plt.title("An example of 3D plot")
ax = fig.add_subplot(1,1,1,projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.plot(x,y,z)
plt.show()
