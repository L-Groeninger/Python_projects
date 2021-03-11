# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 16:53:43 2021

@author: lukas
"""

# Coding sample
iteration = 0
count = 0
while iteration < 5:
    for letter in "hello, world":
        count += 1
    print("Iteration " + str(iteration) + "; count is: " + str(count))
    iteration += 1 
    
    
for iteration in range(5):
    count = 0
    while True:
        for letter in "hello, world":
            count += 1
        print("Iteration " + str(iteration) + "; count is: " + str(count))
        break
    
count = 0
phrase = "hello, world"
for iteration in range(5):
    count += len(phrase)
    print("Iteration " + str(iteration) + "; count is: " + str(count))    
    
    
# Paste your code into this box 
vowels = "aeiou"
s = 'azcbobobegghakl'
count = 0

for i in s:
    if i in vowels:
        count += 1
print("Number of vowels:", count)
    
    
# Write a program that prints the number of times the string 'bob' occurs in s.
# For example, if s = 'azcbobobegghakl', then your program should print
    
s = 'azcbobobegghakl'
count = 0
word = "bob"

for i in range(len(s)-len(word)-1):
    if s[i] + s[i+1] + s[i+2] == word:
        count += 1
    
# Print longest substring in alphabetical order
s = 'azcbobobegghakl'        
res = ''
tmp = ''

for i in range(len(s)):
    tmp += s[i]
    if len(tmp) > len(res):
        res = tmp
    if i > len(s)-2:
        break
    if s[i] > s[i+1]:
        tmp = ''

print("Longest substring in alphabetical order is: {}".format(res))  
    
# Implement bisection search

num = input("Please think of a number between 0 and 100!")   
guess = 50
step = 50
found = False
while found == False:
    print("Is your secret number", guess, "?")
    eval = input("Enter 'h' to indicate the guess is too high. Enter 'l' to indicate  the guess is too low. Enter 'c' to indicate I guessed correctly.")
    while eval not in "lch":
        print("Your input was not correct")
        eval = input("Enter 'h' to indicate the guess is too high. Enter 'l' to indicate  the guess is too low. Enter 'c' to indicate I guessed correctly.")
    step = step/2
    if eval == "h":
        guess = int(guess - step)
    elif eval == "l":
        guess = int(guess + step)
    elif eval == "c":
        print("Game over. Your secret number was:", guess)
        found = True
    
x = 12
def g(x):
    x = x + 1
    def h(y):
        return x + y
    return h(6)
g(x)        
        
def foo (x):
   def bar (z, x = 0):
      return z + x
   return bar(3)
          
foo(5)  


str1 = 'exterminate!' 
str2 = 'number one - the larch'    
    
str1.upper()
str1.isupper() 
str2.capitalize()    
    
str2.swapcase()
    
def power4(x):
    def square(x):
        return x**2
    return square(x)*square(x)
    
    
power4(3)
    
def odd(x):
    
    return x % 2 != 0
    
odd(3) 
    
# Recursive raise to the power
def recursivePower(base, exp):
    '''
    base: int or float.
    exp: int >= 0
 
    returns: int or float, base^exp
    '''
    if exp == 0:
        return 1
    # elif exp == 1:
    #     return base
    else:
        return recursivePower(base, exp - 1) * base
    
recursivePower(2, 3)
    
# Iterative raise to the power
def iterPower(base, exp):
    
    if exp == 0:
        return 1
    else:
        result = base
        for i in range(1, exp):
            result *= base
        return result
        
iterPower(2, 0)

# Greatest common divider problem

def gcd(a,b):
    
    low = min(a, b)
    high = max(a, b)
    found = False
    
    while found != True:
        if high % low == 0 and min(a,b) % low == 0:
            gcd = low
            found = True
            
        else:
            low -= 1
    return gcd
    
    
gcd(182, 42) 


def gcd_recursive(a, b):
    
# if b = 0, then answer is a
# gcd(a, b) is the same as gcd(b, a % b)
    
    if b == 0:
        return a
    else:
        return gcd_recursive(b, a % b)
    
    
gcd_recursive(12, 9) 
    
# Fibonacci  
def fib(n):
    if n == 1 or n == 2:
        return 1
    else:
        return fib(n-1) + fib(n-2)
    
fib(4)  
    
    
# Lecture 5
t = (2, "lukas", 4)   

name = t[1:2]    

l = [1, 2, "string", (2, 3)]    
    
l[3] + 2

warm = ["red", "orange", "purple"]    
hot = warm
hot.append("pink")    


class Rabbit(object):
    tag = 1
    def __init__(self, age, name):
        self.age = age,
        self.name = name
        self.rid = Rabbit.tag
        Rabbit.tag += 1
        
    def get_name(self):
        return self.name
    def get_age(self):
        return self.age
    def set_name(self, name = ""):
        self.name = name
    def set_age(self, age):
        self.age = age
    
    
    
rabbit_1 = Rabbit(2, "Micky")   
rabbit_2 = Rabbit(1, "Brownie")    

rabbit_1.rid    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    