# random walk object oriented implementation
# 2 dimensional

#%%

import numpy as np
import random
import math
import matplotlib.pyplot as plt

# Define a class "walk"
class walk:

    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y

    def take_step(self):
        # random angle alpha
        alpha = 2 * math.pi * random.random()
        self.x = math.cos(alpha) + self.x
        self.y = math.sin(alpha) + self.y

# define number of steps:
n = 1000

# set the initial zero vectors
x = np.zeros(n) 
y = np.zeros(n)

walk = walk()

for i in range(1, n):
    walk.take_step()
    x[i] = walk.x
    y[i] = walk.y

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x, y)

# Set title for the plot
ax.set_title(print('Random Walk with ' + str(n) + " steps"))

plt.show()


# %%
