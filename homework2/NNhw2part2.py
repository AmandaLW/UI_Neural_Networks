#!/usr/bin/env python
# coding: utf-8

# In[9]:


'''
Amanda Ward
Neural Networks HW2 Part 2

The purpose of this code is to show that an XOR NN can be trained in linearly separable
parts. In this case, by using an OR, NAND and NAND

code was adapted from:
https://towardsdatascience.com/perceptrons-logical-functions-and-the-xor-problem-37ca5025790a

'''
import numpy as np

def unit_step(v):

    if v >= 0:
        return 1
    else:
        return 0

def perceptron(x, w, b):
    """ Function implemented by a perceptron with 
        weight vector w and bias b """
    v = np.dot(w, x) + b
    y = unit_step(v)
    return y

def AND_percep(x):
    w = np.array([1, 1])
    b = -1.5
    return perceptron(x, w, b)

# Test
example1 = np.array([1, 1])
example2 = np.array([1, 0])
example3 = np.array([0, 1])
example4 = np.array([0, 0])

print("AND({}, {}) = {}".format(1, 1, AND_percep(example1)))
print("AND({}, {}) = {}".format(1, 0, AND_percep(example2)))
print("AND({}, {}) = {}".format(0, 1, AND_percep(example3)))
print("AND({}, {}) = {}".format(0, 0, AND_percep(example4)))

def OR_percep(x):
    w = np.array([1, 1])
    b = -0.5
    return perceptron(x, w, b)

# Test
example1 = np.array([1, 1])
example2 = np.array([1, 0])
example3 = np.array([0, 1])
example4 = np.array([0, 0])

print("OR({}, {}) = {}".format(1, 1, OR_percep(example1)))
print("OR({}, {}) = {}".format(1, 0, OR_percep(example2)))
print("OR({}, {}) = {}".format(0, 1, OR_percep(example3)))
print("OR({}, {}) = {}".format(0, 0, OR_percep(example4)))

#I added the NAND
# I got the weights and bias here: 
#https://stackoverflow.com/questions/45450366/using-simple-weights-1-1-and-bias-2-for-nand-perceptron
def NAND_percep(x):
    w = np.array([-1, -1])
    b = 1.5
    return perceptron(x, w, b)

# Test
example1 = np.array([1, 1])
example2 = np.array([1, 0])
example3 = np.array([0, 1])
example4 = np.array([0, 0])

print("NAND({}, {}) = {}".format(1, 1, NAND_percep(example1)))
print("NAND({}, {}) = {}".format(1, 0, NAND_percep(example2)))
print("NAND({}, {}) = {}".format(0, 1, NAND_percep(example3)))
print("NAND({}, {}) = {}".format(0, 0, NAND_percep(example4)))

    
# I modified the XOR_net to include the OR NAND and AND architecture 
# set forth in the homeowrk assignment
def XOR_net(x):
    gate_1 = OR_percep(x)
    gate_2 = NAND_percep(x)
    gate_3 = AND_percep(x)
    new_x = np.array([gate_1, gate_2])
    output = AND_percep(new_x)
    return output

print("XOR({}, {}) = {}".format(1, 1, XOR_net(example1)))
print("XOR({}, {}) = {}".format(1, 0, XOR_net(example2)))
print("XOR({}, {}) = {}".format(0, 1, XOR_net(example3)))
print("XOR({}, {}) = {}".format(0, 0, XOR_net(example4)))


# In[ ]:




