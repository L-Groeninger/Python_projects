# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 11:36:57 2020

@author: Administrator
"""
#                                   1.
##############################################################################
# Kata to place someone in a category (7kyu)

def open_or_senior(data):
    
    res = []
    for i in data:
        if i[0] >= 55 and i[1] > 7:
            res.append("Senior")
        else:
            res.append("Open")
            
    return res


#                                   2.
##############################################################################
# Find out which number is different in eveness and return the index (6kyu)

def iq_test(numbers):
    
    numbers = numbers.split(" ")
    numbers = list(map(int, numbers))

    even = []
    odd = []

    for i in numbers:
        if (i % 2) == 0:
            even.append(i)
            count_even = numbers.index(i)
        else:
            odd.append(i)
            count_odd = numbers.index(i)
        
    if len(even) > len(odd):
        res = count_odd + 1
    else:
        res = count_even + 1
    
    return res
    
 
# test the function
iq_test("2 4 7 8 10")

# more elegant solution:

def iq_test(numbers):
    e = [int(i) % 2 == 0 for i in numbers.split()]

    return e.index(True) + 1 if e.count(True) == 1 else e.index(False) + 1


#                                   3.
##############################################################################
# list the sum of all multiples of 3 and 5 of a number

import numpy as np

def solution(number):
    
    numbers_seq = np.arange(1, number)

    booleans = np.logical_or(numbers_seq % 5 == 0, numbers_seq % 3 == 0)

    return numbers_seq[booleans].sum()

# elegant version, but I prefer mine
def solution(number):
    return sum(x for x in range(number) if x % 3 == 0 or x % 5 == 0)


#                                   4.
##############################################################################

# return true or false depending on whether input is a narcistic number
# example of narcistic number:
     # 1^3 + 5^3 + 3^3 = 1 + 125 + 27 = 153

def narcissistic(value):
    
    n_digits = len(str(value))
    count = 0

    for i in range(len(str(value))):

        result = int(str(value)[i])**n_digits
        count += result
        
    return count == value

# short version:
def narcissistic(value):
    return value == sum(int(x) ** len(str(value)) for x in str(value))


#                                   5. 
##############################################################################
# Define a kaskify function which shows only the last four characters 

# return masked string
def maskify(cc):

    keep = cc[-4:]
    add = "#" * len(cc[:-4])
    result = add + keep
    
    return result

# One could compress this code to:
def maskify(cc):
    return "#"*(len(cc)-4) + cc[-4:]    


#                                   6.
##############################################################################


def halving_sum(number) :

    result = 0
    while (number > 0) :
    # In the end the integer division results to 0
        result += number
        # Update the "result" with every iteration
        number = number // 2
        # The last integer division will be 1 // 2 = 0
        # This will break the loop

    return result



halving_sum(25)



























