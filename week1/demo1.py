#!/usr/bin/env python


# Demo of some features of numpy and python
import numpy as np  # Need to import numpy in order
                    # to use matrix related data and functions
import matplotlib.pyplot as plt

def cost(x, y, theta):
    """This function solves one of the quiz question in the quiz: Linear
    regression with one variable.
    (mat, mat, mat) -> int
     >>> cost(np.matrix([[1,1,1,1], [3,2,4,0]]), np.matrix([[4,1,3,1]]), np.matrix([[0],[1]]))
     0.5
     >>> cost(np.matrix([[1,1,1,1], [1,2,3,4]]), np.matrix([[1,2,3,4]]), np.matrix([[0],[1]]))
     0.0
     >>> cost(np.matrix([[1,1,1,1], [1,2,3,4]]), np.matrix([[1,2,3,4]]), np.matrix([[0],[2]]))
     3.75
     >>> cost(np.matrix([[1,1,1,1], [1,2,3,4]]), np.matrix([[1,2,3,4]]), np.matrix([[0],[3]]))
     15.0
     >>> cost(np.matrix([[1,1,1,1], [1,2,3,4]]), np.matrix([[1,2,3,4]]), np.matrix([[0],[0.5]]))
     0.9375

     This example is from video lecture 2.3 Cost function.
     >>> cost(np.matrix([[1,1,1], [1,2,3]]), np.matrix([[1,2,3]]), np.matrix([[0],[1]]))
     0.0
     >>> cost(np.matrix([[1,1,1], [1,2,3]]), np.matrix([[1,2,3]]), np.matrix([[0],[0.5]]))
     0.58333333333333326
     >>> cost(np.matrix([[1,1,1], [1,2,3]]), np.matrix([[1,2,3]]), np.matrix([[0],[0]]))
     2.333333333333333

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


def plot_cost_theta():      
    """Lets see if we can plot the cost vs theta for different values of theta
    as shown in lecture 2.3

    () -> None
    Shows a plot
    >>> plot_cost_theta()
    None

    Author: Manish M Yathnalli
    Date:   Press [ctrl-alt-d] to insert date
    """
    # using plot: http://matplotlib.org/users/pyplot_tutorial.html
    # matplotlib is the standard plotting library in python. Check tutorial.
    x = np.matrix([[1, 1, 1], [1, 2, 3]])
    y = np.matrix([[1, 2, 3]])
    thetas = np.arange(0, 2.1, 0.1)
    average_cost = []
    # The below statement is called list comprehensions. It is same as calling
    # a for loop like below.
    # for t in thetas:
    #     average_cost.append(cost(x, y, np.matrix([[0],[t]])))
    average_cost = [cost(x, y, np.matrix([[0], [t]])) for t in thetas]
    plt.plot(thetas, average_cost)
    plt.xlabel("Theta")
    plt.ylabel("Cost")
    plt.show()


if __name__ == '__main__':
    import doctest
    doctest.testmod()
