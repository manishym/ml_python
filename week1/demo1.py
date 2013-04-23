#!/usr/bin/env python


# Demo of some features of numpy and python
import numpy as np  # Need to import numpy in order
                    # to use matrix related data and functions


def quiz(x, y, theta):
    """This function solves one of the quiz question in the quiz: Linear
    regression with one variable.
    () -> int
     >>> quiz(np.matrix([[1,1,1,1], [3,2,4,0]]), np.matrix([[4,1,3,1]]), np.matrix([[0],[1]]))
     0.5

    Author: Manish M Yathnalli
    Date:   Tue-23-April-2013
    """
    # This is solution for the quiz. Theta will be a column vector of form [theta0; theta1]
    # htheta(x) is theta' * x, which in numpy language theta.getT() * x
    htheta = theta.getT() * x
    # so, difference is theta' * x - y. To exponentiate each element of the matrix, we need to 
    diff = htheta - y
    # convert the matrix into array.
    squared = np.asarray(diff) ** 2
    # then we need to sum the squared
    squared.sum()
    # then multiply it by 1/2m to get answer.
    jtheta = 1.0/(2 * y.size) * squared.sum()
    # return that
    return jtheta
    # Same can be acomplished in one line using 
    # return 1.0/(2*y.size) * (np.asarray(theta.getT() * x - y)**2).sum()


if __name__ == '__main__':
    import doctest
    doctest.testmod()
