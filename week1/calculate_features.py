#!/usr/bin/python 


def calculate_features(list, feature):
    """caclulates the said feature
    This is with reference to 
    (list, index)
    >>> calculate_features([7921, 5184, 8836,4761], 0)
    0.30576687116564416
    >>> calculate_features([89,72,94,69], 2)
    0.52

    
    Author: Manish M Yathnalli
    Date:   Fri-26-April-2013
    """
    mean = sum(list) / len(list)
    diff = max(list) - min(list)
    return ((list[feature] - mean) * 1.0) / diff


def main():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
      main()  