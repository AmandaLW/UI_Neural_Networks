#!/usr/bin/env python
# coding: utf-8

# In[8]:


'''
Amanda Ward 
University of Idaho - CS 578 Neural Network
Spring 2020

The purpose of this code is to explore applying the perceptron delta rule
to the XOR gate.

XOR is not linearly separable. I realize this code will not actually produce correct weights.
Our asignment asked us to try to train the weights all at once and then to try to train an XOR that
is a compilation of linearly separable functions. This code is for the former. 

I have two classes, one for training the weights and one for implementing hte weights.
I have listed the expected output for OR AND and XOR. I tested my code with two linearly separable
functions to test that my training and testing classes work. Then I applied it to the XOR.

The assignment asks me to adjust the learning rate and to test having the weights initialized to 
zero and then again randomly. Learning rate and initialization will have no impact on training the
XOR in this scenario. For the sake of being thorough on my report I ran the code for 100,000 epochs.

I used the following website as inspiration for my implementation
https://pythonmachinelearning.pro/perceptrons-the-first-neural-networks/
'''


import numpy as np

# The 
class Perceptron(object):
    """Implements a perceptron network"""
    def __init__(self, input_size, lr, epochs):
        self.W = np.zeros(input_size+1) # This will initialize the weights to zero plus one for the bias
        # add one for bias
        self.epochs = epochs
        self.lr = lr
        #self.epochCount = 0
    
    def activation_fn(self, x):
        #return (x >= 0).astype(np.float32)
        return 1 if x >= 0 else 0
 
    def predict(self, x):
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a
 
    def fit(self, X, d):
        for epoch in range(self.epochs):
    
            print(str(epoch + 1),"epoch has started...")
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, 1)
                y = self.predict(x)
                if (d[i] == y):
                    pass
                else:
                    e = d[i] - y
                    self.W = self.W + self.lr * e * x
'''
This class takes the weights that are trained previously and runs it through a single epoch to see if 
the actual output matches the expected output. 
'''
class PerceptronTest(object):
    """Implements a perceptron network"""
    def __init__(self, d,savedWeights, epochs = 1):
        self.W = savedWeights # This will initialize the weights what we trained them to
        self.epochs = epochs
            
    def activation_fn(self, x):
        #return (x >= 0).astype(np.float32)
        return 1 if x >= 0 else 0
 
    def predict(self, x):
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a
 
    def fit(self, X, d):
        for epoch in range(self.epochs):
    
            print(str(epoch + 1),"Testing has started...")
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, 1)
                y = self.predict(x)
                if (d[i] == y):
                    print("There was a match between actual and desired output")
                    pass
                else:
                   print("Trained weights did not deliver desired output")
    

                
if __name__ == '__main__':
    
# X is the input for all my functions
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    
    # d is my expected output for the AND function
    d = np.array([0, 0, 0, 1]) 
    # e is my expected output for the OR function
    e = np.array([0, 1, 1, 1])
    # f is my expected output for the NAND function
    f = np.array([1,1,1,0])
    #g is my expected output for the XOR function
    g = np.array([1,0,0,1])
 
    perceptron = Perceptron(input_size=2, lr = 0.50, epochs = 100000)
    perceptron.fit(X, g)
    savedWeights = perceptron.W
    print(savedWeights)
    test = PerceptronTest(g, savedWeights)
    test.fit(X,g)
  


# In[ ]:





# In[ ]:




