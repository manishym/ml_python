#!/usr/bin/env python

# Description: Solves ex1 of machine learning course in python
# Purpose: This is part of machine learning course.
# Author:  Manish M Yathnalli
# Date:    Sun-28-April-2013
# Copyright 2013 <"Manish M Yathnalli", manish.ym@gmail.com>
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from demo1 import cost, gradient_descent


def add_ones_to(X):
    j = np.asmatrix(np.ones(len(X))).getT()
    return np.append(j, X, 1)


def plotData(X, y):
    """This function will plot data as explained in ex1 plotData
    (none) -> none
    Since it just plots data, we cannot do doctests


    Author: Manish M Yathnalli
    Date:   Sun-28-April-2013
    """
    plt.plot(X, y, 'rx')
    plt.xlabel("X")
    plt.ylabel("y")
    plt.show()


def main():
    """This function will setup variables and will do the doctest
    Author: Manish M Yathnalli
    Date:   Sun-28-April-2013
    """
    import doctest

    d = scipy.io.loadmat("ex1data1.mat")
    X = np.asmatrix(d['ex1data1'])[:, 0]
    y = np.asmatrix(d['ex1data1'])[:, 1]
    theta = np.matrix([[0], [0]])
    print "Plotting data..."
    iterations = 1500
    alpha = 0.01

    plotData(X, y)
    X = add_ones_to(X)
    print cost(X, y, theta)
    theta, hist = gradient_descent(X, y, alpha, iter=iterations)  # No need for theta in my gradient descent.
    print "Theta found by gradient_descent:", theta
    htheta = theta.getT() * X.getT()
    plt.plot(X[:, 1], y, 'rx', X[:, 1], htheta.getT())
    plt.xlabel("Training Data")
    plt.ylabel("Hypothesis")
    plt.show()
    #plotData(y, htheta)
    doctest.testmod()


if __name__ == '__main__':
    main()