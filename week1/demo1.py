#!/usr/bin/env python


# Demo of some features of numpy and python
import numpy as np  # Need to import numpy in order
                    # to use matrix related data and functions
import matplotlib.pyplot as plt


def cost(x, y, theta):
    """This function solves one of the quiz question in the quiz: Linear
    regression with one variable.
    (mat, mat, mat) -> float
     >>> cost(np.matrix([[1,1,1,1], [3,2,4,0]]).getT(), np.matrix([[4,1,3,1]]).getT(), np.matrix([[0],[1]]))
     0.5
     >>> cost(np.matrix([[1,1,1,1], [1,2,3,4]]).getT(), np.matrix([[1,2,3,4]]).getT(), np.matrix([[0],[1]]))
     0.0
     >>> cost(np.matrix([[1,1,1,1], [1,2,3,4]]).getT(), np.matrix([[1,2,3,4]]).getT(), np.matrix([[0],[2]]))
     3.75
     >>> cost(np.matrix([[1,1,1,1], [1,2,3,4]]).getT(), np.matrix([[1,2,3,4]]).getT(), np.matrix([[0],[3]]))
     15.0
     >>> cost(np.matrix([[1,1,1,1], [1,2,3,4]]).getT(), np.matrix([[1,2,3,4]]).getT(), np.matrix([[0],[0.5]]))
     0.9375

     This example is from video lecture 2.3 Cost function.
     >>> cost(np.matrix([[1,1,1], [1,2,3]]).getT(), np.matrix([[1,2,3]]).getT(), np.matrix([[0],[1]]))
     0.0
     >>> cost(np.matrix([[1,1,1], [1,2,3]]).getT(), np.matrix([[1,2,3]]).getT(), np.matrix([[0],[0.5]]))
     0.5833333333333334
     >>> cost(np.matrix([[1,1,1], [1,2,3]]).getT(), np.matrix([[1,2,3]]).getT(), np.matrix([[0],[0]]))
     2.3333333333333335

    Author: Manish M Yathnalli
    Date:   Tue-23-April-2013
    """
    # This is solution for the quiz. Theta will be a column vector of form [theta0; theta1]
    # htheta(x) is theta' * x, which in numpy language theta.getT() * x
    htheta = x * theta
    # so, difference is theta' * x - y. To exponentiate each element of the matrix, we need to 
    # diff = htheta - y
    # # convert the matrix into array.
    # squared = np.asarray(diff) ** 2
    # # then we need to sum the squared
    # squared.sum()
    # # then multiply it by 1/2m to get answer.
    # jtheta = 1.0/(2 * y.size) * squared.sum()
    # return that
    temp = htheta - y
    jtheta = temp.getT() * temp / (2.0 * y.size)
    return float(jtheta)
    # Same can be acomplished in one line using 
    # return 1.0/(2*y.size) * (np.asarray(theta.getT() * x - y)**2).sum()




def gradient_descent(x, y, alpha, iter=500):
    """This is my attempt at getting gradient descent algorithm working with python.
    Gradient descent is described in lectures 2.6 and 2.7 of the course.
    (mat, mat, float) -> mat
    Writing doctest for gradient_descent() will be difficult, since values returned will
    be floats and the match will be approximate. So I will write a unittest module for gradient descent.
    >>> x = np.matrix([[1, 1], [1, 2], [1, 3]])
    >>> y = np.matrix([[1], [2], [3]])
    >>> theta = gradient_descent(x, y, 0.01)[0]
    >>> theta.size == 2
    True


    Author: Manish M Yathnalli
    Date:   Tue-23-April-2013
    """
    # Gradient descent is defined by improved_theta = alpha/m x(j) sum(t-to-n htheta(x) - y)
    #  I was calcluating the sum first then trying to multiply it with x.
    #  I have to compute the sum after multiplying with x. 
    #  One iteration of linear regression uses all samples.
    #  check dairy for more information.
    row, col = x.shape
    theta = np.asmatrix(np.zeros(col)).getT()
    temp = []
    for i in range(iter):
        theta = improve(x, y, alpha, theta)
        temp.append(theta)
    return theta, temp


def improve(x, y, alpha, theta):
    """ improves the theta using gradient descent formula
    Gradient descent is defined by improved_theta = alpha/m x(j) sum(t-to-n htheta(x) - y)
    (mat, mat, float, mat) -> mat
    Doctest not possible.
    Author: Manish M Yathnalli
    Date:   Tue-23-April-2013
    """

    row, col = x.shape
    # print "theta shape", theta.shape
    # print "x shape", x.shape
    # return theta
    m = float(y.size)
    assert(theta.shape == (col, 1))

    htheta = (theta.getT() * x.getT()).getT()

    theta = theta - alpha * (((htheta - y).getT() * x).getT() / m)
    return theta



def plot_cost_theta():      
    """Lets see if we can plot the cost vs theta for different values of theta
    as shown in lecture 2.3

    () -> None
    Shows a plot
    >>> plot_cost_theta()
    

    Author: Manish M Yathnalli
    Date:   Sun-28-April-2013
    """
    # using plot: http://matplotlib.org/users/pyplot_tutorial.html
    # matplotlib is the standard plotting library in python. Check tutorial.
    x = np.matrix([[1, 1], [1, 2], [1, 3]])
    y = np.matrix([[1], [2], [3]])
    thetas = np.arange(0, 2.1, 0.1)
    average_cost = []
    # The below statement is called list comprehensions. It is same as calling
    # a for loop like below.
    # for t in thetas:
    #     average_cost.append(cost(x, y, np.matrix([[0],[t]])))
    average_cost = [cost(x, y, np.matrix([[0], [t]])) for t in thetas]
    new_theta, theta_change = gradient_descent(x, y, 0.001, 500)
    # theta_change gives a list of thetas and how it changed
    plt.figure(1)
    plt.subplot(211)
    plt.plot(thetas, average_cost, 'r--')  # 
    plt.xlabel("Theta")
    plt.ylabel("Cost")
    plt.subplot(212)
    print [t[0] for t in theta_change]
    # Plots cost vs iteration in blue, theta[0] vs iteration in red and theta[1] vs iteration in yellow.
    plt.plot(range(500), [cost(x, y, t) for t in theta_change], 'b-', range(500), [float(t[0]) for t in theta_change], 'r-', range(500), [float(t[1]) for t in theta_change], 'y-')
    plt.xlabel("iteration")
    plt.ylabel("Cost")
    plt.show()


if __name__ == '__main__':
    import doctest
    doctest.testmod()


# Notes:
#  I implemented everything in octave first and then ported it to numpy. Since I am new to both octave and python,
#  I found it easy to implement in octave, since octave's first language is matrix, whereas in python, it is a 
#  added language.
