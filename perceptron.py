import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Perceptron1(object):
    def __init__(self, lr=0.01, epochs=2000):
        self.lr = lr
        self.bias = 1
        self.epochs = epochs
        self.weights = None
        self.errors_ = []


    def fit(self, X, y):
        """Train perceptron. 
        X and y need to be the same length"""
        assert len(X) == len(y), "X and y need to be the same length"
        
        # Initialise weights
        weights = np.zeros(X.shape[1])
        self.weights = np.insert(weights, 0, self.bias, axis=0)

        for _ in range(self.epochs):
            errors = 0
            for xi, y_target in zip(X, y):
                z = self.__linear(xi)  # weighted sum
                y_hat = self.__activation(z)  # activation function
                delta = self.lr * (y_target - y_hat)  # loss
        
                # Update weights - back propagation
                self.weights[1:] += delta * xi
                self.weights[0] += delta
                
                errors += int(delta != 0.0)

            self.errors_.append(errors)          
            if not errors:
                break
    
    
    def __linear(self, X):
        """weighted sum"""
        return np.dot(X, self.weights[1:]) + self.weights[0] 

    
    def __activation(self, X):
        return np.where(X>=0, 1, 0)
                        
    
    def predict(self, X):
        assert type(self.weights) != 'NoneType', "You must run the fit method before making predictions."
        y_hat = np.zeros(X.shape[0],)
        for i, xi in enumerate(X):
            y_hat[i] = self.__activation(self.__linear(xi))
        return y_hat


    def score(sef, predictions, labels):
        return accuracy_score(labels, predictions)
    
    
    def plot(self, predictions, labels):
        assert type(self.weights) != 'NoneType', "You must run the fit method before being able to plot results."
        plt.figure(figsize=(10,8))
        plt.grid(True)

        for input, target in zip(predictions, labels):
            plt.plot(input[0],input[1],'ro' if (target == 1.0) else 'go')

        for i in np.linspace(np.amin(predictions[:,:1]),np.amax(predictions[:,:1])):
            slope = -(self.weights[0]/self.weights[2])/(self.weights[0]/self.weights[1])  
            intercept = -self.weights[0]/self.weights[2]

            # y = mx+b, equation of a line. mx = slope, n = intercept
            y = (slope*i) + intercept
            plt.plot(i, y, color='black', marker='x', linestyle='dashed')
