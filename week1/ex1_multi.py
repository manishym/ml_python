#!/usr/bin/env python

# Description: This is equivalent to ex1_multi.m of exercises.
# Purpose: I am trying to port all machine learning exercises to python.
# Author:  Manish M Yathnalli
# Date:    Tue-14-May-2013
# Copyright 2013 <"Manish M Yathnalli", manish.ym@gmail.com>

import numpy as np
import scipy.io
from ex1 import add_ones_to
from demo1 import gradient_descent
import matplotlib.pyplot as plt
from scipy.linalg import pinv

def feature_normalize(mat):
    mu = sum(mat)/len(mat)
    sigma = np.std(mat, 0)
    return (mat - mu) / sigma, mu, sigma

def normal_equation(x, y):
    """Normal equation method to fit a hypothesis into a dataset.
    (mat, mat) -> mat
    
    Author: Manish M Yathnalli
    Date:   Wed-15-May-2013
    """
    return pinv(x.getT() * x) * (x.getT() * y)

def run_ex1_multi():
    """This is exact translation of ex1_multi.m from matlab to python
    Author: Manish M Yathnalli
    Date:   Wed-15-May-2013
    """
    print "Loading data ...."
    data = scipy.io.loadmat('ex1data2.mat')
    X = np.asmatrix(data['data'])[:, 0:2]
    y = np.asmatrix(data['data'])[:, 2]
    m = len(y)
    X, mu, sigma = feature_normalize(X)
    X = add_ones_to(X)
    print "Running gradiend descent"
    theta, history = gradient_descent(X, y, 0.01, 400)
    print "Theta computed by gradient_descent:", theta
    theta = normal_equation(X, y)
    print "Theta computed by normal equation:", theta


def main():
    run_ex1_multi()


if __name__ == '__main__':
    main()