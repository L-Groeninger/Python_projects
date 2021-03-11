# Script for Kata (Python) exercises


# import libraries
import pandas as pd
import numpy as np

# Kata: Is a number prime?
#
# Define a function that takes an integer argument and returns logical value 
# true or false depending on if the integer is a prime

def is_prime(num):
    import math

    # There's only one even prime: 2
    if num < 2    : return False
    if num == 2   : return True
    if num %2 == 0: return False

    
    """
    Property:
        Every number n that is not prime has at least one prime divisor p
        such 1 < p < square_root(n)
    """
    root = int(math.sqrt(num))
    
    # We know there's only one even prime, so with that in mind 
    # we're going to iterate only over the odd numbers plus using the above property
    # the performance will be improved

    for i in range(3, root+1, 2):
        if num % i == 0: return False

    return True

# Test the function
is_prime(num = 761)


# Walk exercise
# You want to check if the directions you receive get you to 
# the starting position. You can only move in x or y direction

def isValidWalk(walk):
    if (walk.count('n') == walk.count('s') and 
        walk.count('e') == walk.count('w') and
        len(walk) == 10): # the length has to be 10
            return True
    return False



# Create Phone_number
def create_phone_number(n):

    if len(n) == 10:

        full_string = ''.join([str(x) for x in n])

        return("(" + full_string[0:3] + ") " + full_string[3:6] + "-" + full_string[6:])


create_phone_number(n = [1,2,3,4,5,6,7,8,9,1])


# Valid Braces
  
# Function to check valid braces
def validBraces(string): 

    open_list = ["[","{","("] 
    close_list = ["]","}",")"] 

    stack = [] 
    # stack will only consists of open brackets
    for i in string: 
        if i in open_list: 
            stack.append(i) 
        elif i in close_list: 
            pos = close_list.index(i) 
            # if length of stack == 0 there is no open bracket -> unbalanced
            if ((len(stack) > 0) and
                (open_list[pos] == stack[len(stack)-1])): 
                stack.pop() 
            else: 
                return False
    if len(stack) == 0: 
        # If all open brackets are popped away (removed)
        return True
    else: 
        return False

validBraces("[[(){({})}]]")


# Program Fibonacci series

def fibonacci(n):
    results = [0,1]
    a = 0
    b = 1
    if n <= 0:
        print("Incorrect input")
    elif n == 1:
        return b
    else:
        for i in range(2,n):
            c = a + b
            a = b
            b = c
            results.append(b)
        return results
 
print(fibonacci(120))

# nth fibonacci 
def nth_fib(n):
    return fibonacci(n)[-1]

nth_fib(5)

# def nth_fib(n):
#     if n == 0 or n == 1:
#         return 1
#     else:
#         return nth_fib(n-1) + nth_fib(n-2)

# nth_fib(5)


