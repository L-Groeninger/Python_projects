# Optimization Course
#
# Author: Lukas Gröninger

# Required libraries
import pandas as np
import numpy as np
import matplotlib.pyplot as plt
import math
import random

# Chapter 1.2 vectors
old_loc = np.array([6, -3])
scalar = 2.5
new_vec = np.array([0, 0])
new_loc = np.array([5, -1])

new_loc = old_loc + scalar * new_vec

scalar = (new_loc - old_loc) / new_vec

# Practice Problem 4
new_loc + (1/3) * (new_loc - old_loc)

# Chapter 1.3 Iteration and recursion

# Iteration example

# If a number is even, divide it by 2, 
# if it is odd, multiply by 3 and add 1

n = 325
n_iterations = 0

while n > 1:
    if n % 2 == 0:
        n = n/2
    else:
        n = n*3 + 1
    n_iterations += 1
    
# Fibonacci Series
# 1, 1, 2, 3, 5, 8, 13

# Define starting values (in this case 1, 1)
a = 2
b = 5
c = 0
vector = [a,b]

while len(vector) < 10:
    c = a + b
    vector.append(c)
    a = b
    b = c

# Extended Fibonacci
a = 2
b = 5
c = 4
d = 0
vector = [a,b,c]

while len(vector) < 10:
    d = a * b - c
    vector.append(d)
    a = b
    b = c
    c = d

# Chapter 1.4 iteration and recursion
x = 4

y = x**3 - (15*x) + 12

a = np.array([3,-6])
b = np.array([4,16])

(b[1]-a[1])/(b[0]-a[0])

# Import fsolve of scipy
from scipy.optimize import fsolve

def f(x):
    return -x**3 + 4 * x**2 + 7
    
z = fsolve(f,2)

# Chapter 1.5 iteration and recursion

# Practice Problem 1
# Given function f(x)
# Input x, x+h, x-h

# a
x = 4
h = 0.1

def f(x):
    return x**2 + 3

y1 = f(x)
y2 = f(x+h)
y3 = f(x-h)

# b
x = 2
h = 1

def f(x):
    return 2*x**3 - 4 * x**2 + 17

y1 = f(x)
y2 = f(x+h)
y3 = f(x-h)

# Practice Problem 2
# Find a minimum
x = 0
h = 0.6
minimum = False

def f(x):
    return x**2 - 4 * x

while minimum != True:
    y1 = f(x)
    y2 = f(x+h)
    y3 = f(x-h)

    if y2 == min(y1, y2, y3):
        x += h
    elif y3 == min(y1, y2, y3):
        x -= h
    else:
        minimum = True
        print("A (local) minimum has been found")

# Practice Problem 3
# Find a maximum

x = 0
h = 3
maximum = False

def f(x):
    return x**3 - 10 * x**2 - 400 * x + 400
    
while maximum != True:
    y1 = f(x)
    y2 = f(x+h)
    y3 = f(x-h)

    if y2 == max(y1, y2, y3):
        x += h
    elif y3 == max(y1, y2, y3):
        x -= h
    else:
        maximum = True
        print("A (local) maximum has been found")

    h = h*0.8

# It really depends on how you initialize x and h

# 1.6 Julia Basics
# Practice Problem 
# Write a function that takes a point (a, b)
# and gives information about this point

a = 2
b = 5

def info(a, b):
    slope = b/a
    length = (b**2 + a**2)**0.5
    print("This point has the slope:")
    print(slope)
    print("This point has the length:")
    print(length)

info(a,b)

# Chapter 1.10
# Final Problem
# Define function to find root
# f(x) = m*x + b
# x = (f(x) - b) / m
a = 2
b = 3

def f(x):
    return x**2 - 4 * x

def secant(a,b):
    while abs(a-b) > 0.1:

        m = (f(a) - f(b)) / (b-a)
        x = (f(a) - a) / m
        a = b
        b = x

    print(b)

secant(a, b)


# Chapter 2

# 2.2 Take the problem from chapter 1 and extend it a little

# Define the function to test 
def f(x):
    return x**2 - 3 * x + 5

def find_minimum(f, x = 0, h = 0.1):
    counter = 0
    # Decide the direction
    # Test if the function is decreasing
    if (f(x) < f(x+h)):
    # If not, change direction
        h = -h

    while (f(x+h) < f(x)):
        x = x + h
        counter += 1
        h *= 1.01

    # Print the points to test the code
    print("x-h = ", round(x-h, 2), "f(x-h) = ", round(f(x-h), 2))
    print("x =   ", round(x, 2), "f(x) =   ", round(f(x), 2))
    print("x+h = ", round(x+h, 2),"f(x+h) = ", round(f(x+h), 2))
    print("Number of iterations: ", counter)

find_minimum(f, x = -12, h = 0.01)

# Chapter 2.3
# Find the brute force minimum

# Take an interval of a - b and a step size h
# plug in the values and report the lowest value

a = 0
b = 2
h = 0.001

def find_min_brute(f, a, b, h):
    for i in np.arange(a, b, h):
        low = f(i)
        if (f(i+h) < low):
            lowest_value = low
            x_position = i
    print("Minimum at x = ", x_position, "with value of:", lowest_value)

        
find_min_brute(f, a = -3, b = 3, h = 0.001)


# Chapter 2.4
# Golden Ratio

# Practice Problem

# Start with: Two end points of an interval
# End when certain tolerance is reached – probably the width of the interval is less than ____, or f(var) < ___.
# Divide the interval into three sections by the golden ratio. Choose the section that forms a V (interior point lower than endpoints).
# Using the new endpoints/interval, loop back to step 3.

# Endpoints
a = -2
b = 3

def goldenrule_min(f, a, b, epsilon = 0.01):
    # Golden Number phi
    phi = (-1+(5)**(1/2))/2  

    int = b - a

    while int > epsilon:
        subdiv = phi * int
        lefttest = b - subdiv
        righttest = a + subdiv

        if f(lefttest) < f(righttest):
            b = righttest
        else:
            a = lefttest
        
        int = b - a

# Chapter 2.5 Slope Method

# f is still the same
def f(x):
    return x**2 - 3 * x + 5

# Plot the function
# 100 linearly spaced numbers
x = np.linspace(-np.pi,np.pi,100)
# setting the axes at the centre
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# plot the functions
plt.plot(x,f(x), 'c', label='x**2 - 3 * x + 5')
plt.legend(loc='upper left')
# show the plot
plt.show()

# So we know that between -3 and 3 there is a minimum
# We define an error term

def slope_min(f, a, b, error = 0.01):

    interval = b - a
    
    while interval >= error:
        # Calculation of a slope
        m = (f(b) - f(a)) / interval
        
        if m >= 0:
            b = b - (interval/3)
        else:
            a = a + (interval/3)
        
        interval = b - a
    print(a, b)

slope_min(f, a = -2, b = 3)
# Chapter 2.6
# Now its about finding a maximum

# Define the function to test 
def f(x):
    return -0.05 * x**2

# Plot the function
# 100 linearly spaced numbers
x = np.linspace(-200,200,10000)
# setting the axes at the centre
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# plot the functions
plt.plot(x,f(x), 'c', label='-0.05 * x**2')
plt.legend(loc='upper left')
# show the plot
plt.show()

def find_maximum(f, x = 2, h = 0.1):
    counter = 0
    # Decide the direction
    # Test if the function is decreasing
    if (f(x) > f(x+h)):
    # If yes switch direction
        h = -h

    while (f(x+h) > f(x)):
        x = x + h
        counter += 1
        h *= 1.01

    # Print the points to test the code
    print("x-h = ", round(x-h, 2), "f(x-h) = ", round(f(x-h), 2))
    print("x =   ", round(x, 2), "f(x) =   ", round(f(x), 2))
    print("x+h = ", round(x+h, 2),"f(x+h) = ", round(f(x+h), 2))
    print("Number of iterations: ", counter)

find_maximum(f, x = 2, h = 0.01)

# Golden section maximum program

# Endpoints
a = -4
b = 3

def goldenrule_max(f, a, b, epsilon = 0.01):
    # Golden Number phi 0.618
    phi = (-1+(5)**(1/2))/2  
    int = b - a

    while int > epsilon:
        subdiv = phi * int
        lefttest = b - subdiv
        righttest = a + subdiv

        if f(lefttest) < f(righttest):
            a = lefttest
        else:
            b = righttest
        
        int = b - a
    print(lefttest, righttest)

goldenrule_max(f, a, b)

# Slope Max funtion

def slope_max(f, a, b, error = 0.01):

    interval = b - a
    
    while interval >= error:
        # Calculation of a slope
        m = (f(b) - f(a)) / interval
        
        if m <= 0:
            b = b - (interval/3)
        else:
            a = a + (interval/3)
        
        interval = b - a
    print(a, b)

slope_max(f, a = -2, b = 3)

# Chapter 2.7 Global
# 
# Write Function that checks y values up to a point
# Still the same test function

def f(x):
    return -0.05 * x**2

def checky_values(f):
    
    x = 0
    x_pos = 0
    x_neg = 0
    
    while (f(x) < 1000000 and f(x) > -1000000):
        x += 2
    x_pos = x
    x = 0
    while (f(x) < 1000000 and f(x) > -1000000):
        x += -2
    x_neg = x
    return [x_pos, x_neg]
    
checky_values(f)        

# Find a global minimum
# y = x4 + 35x3 – 1062x2 – 8336x + 47840, 
# given that the global minimum is somewhere between -200 and 200.

# Define Function
def f(x):
    return x**4 + 35 * x**3 - 1062 * x**2 - 8336*x + 47840

# Function to return the interval
def find_minimum_interval(f, x = 0, h = 0.1):
    counter = 0
    # Decide the direction
    # Test if the function is decreasing
    if (f(x) < f(x+h)):
    # If not, change direction
        h = -h

    while (f(x+h) < f(x)):
        x = x + h
        counter += 1
        h *= 1.01
    
    return [x-h, x+h]

# Function to find the minimum
def slope_min(f, a, b, error = 0.001):
    interval = b - a

    while interval >= error:
        # Calculation of a slope
        m = (f(b) - f(a)) / interval
        
        if m >= 0:
            b = b - (interval/3)
        else:
            a = a + (interval/3)
        
        interval = b - a
    return b

# Putting it together
def brute_global_min(f, a, b):
    
    sections = np.linspace(a, b, 50)
    results = []
    for i in sections:
        results.append(f(i))
    
    start_value = sections[np.argmin(results)]
    
    min_interval = find_minimum_interval(f, x = start_value)
    
    return slope_min(f, min_interval[0], min_interval[1])
    
brute_global_min(f, -200, 200)

# Chapter 2.8 Sawtooth method

# y - y1 = m(x - x1) 
# y - y2 = -m(x - x2)

def sawtooth(x1,y1,x2,y2):
    
    m = 450 # Define max slope
    ycross = m*(x1 + x2)/2 - ((y1-y2)/2) - m*x1 + y1
    xcross = (ycross - y1)/m + x1
    
    return [xcross, ycross]

sawtooth(x1 = -5, y1 = 75, x2 = 1, y2 = 183)

# Chapter 2.10
# Going into 3D

# Practice Problems
def f(x1,x2):
    return (x1 +x2)**2 + (math.sin(x1 +2))**2 + (x2)**2 + 10

def return_low_value(f, x1, x2, step = 0.1):
    
    found_center = False
    while found_center == False:
        test_points = np.array([[x1,x2],[x1 + step,x2],
                               [x1 - step,x2], [x1,x2 + step],
                               [x1,x2 - step]])
        results = []
        for i in test_points:
            results.append(f(i[0],i[1]))
        
        x1 = test_points[np.argmin(results)][0]
        x2 = test_points[np.argmin(results)][1]
        
        if np.argmin(results) == 0:
            found_center = True
    
    return test_points[np.argmin(results)]
  
    
return_low_value(f, 3, 5) 
    

# Practice Problem 5 Grid Search
# x1 from -3 to 3, x2 from -2 to 5

def f(x1,x2):
    return 100*(a-b)**2 + (1-b)**2

a = -3
b = 3
c = -2
d = 5

def grid_search(f, a, b, c, d):
    
    x1 = np.linspace(a,b,6)
    x2 = np.linspace(c,d,6)
    # Define startvalue
    smallest = f(x1[0],x2[0])
    
    for i in x1:
        for j in x2:
            test = f(i, j)
            if test < smallest:
                smallest = test
                small_x = [i, j]
    
    return small_x

grid_search(f, a, b, c, d)

# Chapter 2.11
# Implement Hooke-Jeeves algorithm

# Sieht nicht sonderlich schön aus...
# und es ist auch nicht so wie im Kurs
def f(x1,x2):
    return (x1-3)**2 + (x2+1)**2

def hooke_jeeves(f, x1, x2, step = 0.1):
    
    first_x1 = x1
    first_x2 = x2
    best_x1 = False
    best_x2 = False
    
    while best_x1 == False:
        test_x1 = np.array([[x1,x2], [x1+step,x2], [x1-step,x2]])
        
        if np.argmin([f(test_x1[0][0],test_x1[0][1]),
                      f(test_x1[1][0],test_x1[0][1]),
                      f(test_x1[2][0],test_x1[0][1])]) == 1:
            x1 += step
        elif np.argmin([f(test_x1[0][0],test_x1[0][1]),
                      f(test_x1[1][0],test_x1[0][1]),
                      f(test_x1[2][0],test_x1[0][1])]) == 2:
            x1 -= step
        else:
            best_x1 = True
    
    while best_x2 == False:
        test_x2 = np.array([[x1,x2], [x1,x2+step], [x1,x2-step]])
        
        if np.argmin([f(test_x2[0][0],test_x2[0][1]),
                      f(test_x2[0][0],test_x2[1][1]),
                      f(test_x2[0][0],test_x2[2][1])]) == 1:
            x2 += step
        elif np.argmin([f(test_x2[0][0],test_x2[0][1]),
                      f(test_x2[0][0],test_x2[1][1]),
                      f(test_x2[0][0],test_x2[2][1])]) == 2:
            x2 -= step
        else:
            best_x2 = True
    
    Difference = np.subtract([first_x1,first_x2],[x1,x2])
    
    print("Original x1,x2:", first_x1,first_x2,"\n",
          "Improved x1,x2:", x1,x2,"\n",
          "Difference:", Difference)

hooke_jeeves(f, 1, 1)

# Chapter 2.14
# Generate 5 random numbers between 0 and 1
[random.uniform(0,1) for i in range(5)]

# Generate random numbers from normal distribution
[random.gauss(mu = 0, sigma = 1) for i in range(5)]

# Practice Problem 2
# Shipping from A to B
results = []
for i in range(100):
    
    results.append(random.uniform(1, 3) + random.gauss(0.8, 0.35) + 
                   random.uniform(0.5, 2) + random.gauss(4, 0.1) + 
                   0.25 + random.gauss(3, 0.8))
sum(results)/len(results)


# Chapter 3

# Chapter 3.2
# Practice Problem 4c

A = np.array([[2,1,-3,1],
              [1,-2,0,-6],
              [-3,2,-1,3],
              [-1,0,1,-2]])

B = np.array([12,-28,10,-13])

# A*X = B # multiply with inverse of A
# X = A**-1*B
A_inv = np.linalg.inv(A)
X = np.dot(A_inv,B)

A[1] # row vectors
A[0,:] # column vectors

# Chapter 3.3
A = np.array([[3,1,-2],
              [2,-2,5]], dtype = "float")

A[0,0] = 1
A[1,0] = 0

# Divide first row by 3
A[0] = A[0]/3
# Replace second row by sum of rows * 2
A[1] = A[0]+A[1]*2

# Practice Problem
# Write Function that performs gaussian elimination
# and outputs solution for x1 and x2
# Input Matrix: 2*3

A = np.array([[2, 3, 4],
              [3,-5, 5]], dtype = "float")

def gauss_elimination(A):

    A[1] = A[0] * A[1,0] / A[0,0] - A[1] 
    x2 = A[1,2]/A[1,1]
    x1 = (A[0,2] - A[0,1] * x2) / A[0,0]
    # Print solution
    print("Value for x1: ", x1, "\n",
          "Value for x2: ", x2)

gauss_elimination(A)

# Chapter 3.5
# Pivoting the inexperienced worker problem

A = np.array([[15, 10, 1, 0,0, 1200],
              [1, 2, 0, 1, 0, 120],
              [-10, -9, 0, 0, 1, 0]], dtype = "float")


A[1] = A[1] - (1/15)*A[0]
A[2] = A[2] + (10/15)*A[0]

# x1 = 900/15 = 60

A[0] = A[0] - A[0,1]/A[1,1]*A[1]
A[2] = A[2] - A[1]*(A[2,1]/A[1,1])

# x2 = 40/1.33 = 30.075

def fib(n):
    if n == 0 or n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)


fib(5)

cube = 12

for guess in range(abs(cube)+1):
    
    if guess**3 >= abs(cube):
        break
    
if guess**3 != abs(cube):
    print("cube is not a perfect cube")
        
else:
    if cube < 0:
        guess = -guess
    print("Cube root of", str(cube), "is", str(guess))
    





































