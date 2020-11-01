# Plot a random walk

# random walk with 2 dimensions (x, y)
# it can walk in north, east, south or west direction

import matplotlib.pyplot as plt
import random
import numpy as np

directions = ["N", "E", "S", "W"]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

# define number of steps:
n = 1000

# set the initial zero vectors
x = np.zeros(n) 
y = np.zeros(n)

# filling the vectors with random directions 
for i in range(1, n): 
    val = random.choice(directions) 
    if val == "E": 
        x[i] = x[i - 1] + 1
        y[i] = y[i - 1] 
    elif val == "W": 
        x[i] = x[i - 1] - 1
        y[i] = y[i - 1] 
    elif val == "N": 
        x[i] = x[i - 1] 
        y[i] = y[i - 1] + 1
    else: 
        x[i] = x[i - 1] 
        y[i] = y[i - 1] - 1


ax.plot(x, y)
plt.show()













