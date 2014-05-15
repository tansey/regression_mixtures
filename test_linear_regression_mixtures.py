import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
from linear_regression_mixtures import *


if __name__ == '__main__':
    print 'Testing mixture of 3 linear regression models with known number of components'

    NUM_COMPONENTS = 3
    NUM_SUBSETS = 300
    POINTS_PER_SUBSET = 10
    TRUE_COEFFICIENTS = np.array([[2, 0.2], [1, 0.4], [0.8, -0.1]])
    TRUE_VARIANCES = np.array([0.1, 0.2, 0.1])**2
    TRUE_COMPONENT_WEIGHTS = np.array([0.4, 0.2, 0.4])
    TRUE_ASSIGNMENTS = np.random.choice(NUM_COMPONENTS, p=TRUE_COMPONENT_WEIGHTS, size=NUM_SUBSETS)
    KEYS = np.arange(NUM_SUBSETS)

    DATA = {}
    for key,assignment in zip(KEYS, TRUE_ASSIGNMENTS):
        # Generate some random points
        x = np.random.uniform(size=(POINTS_PER_SUBSET, TRUE_COEFFICIENTS.shape[1]))
        x[:,0] = 1

        # Get the parameters of this subset's component
        coefficients = TRUE_COEFFICIENTS[assignment]
        variance = TRUE_VARIANCES[assignment]

        # Generate a noisy version of the response variables
        y = x.dot(coefficients) + np.random.normal(0, np.sqrt(variance), size=POINTS_PER_SUBSET)

        # Add the results to the dataset
        DATA[key] = (x, y)

    results = fit_with_restarts(DATA, KEYS, 3, 20, 8, stochastic=False, verbose=True, num_workers=4)

    # Plot the data
    COMPONENT_COLORS = ['red', 'blue', 'green', 'yellow', 'orange', 'brown', 'gray']
    for key, assignment in zip(KEYS, TRUE_ASSIGNMENTS):
        # Get the data
        x, y = DATA[key]

        # Draw the data points
        plt.scatter(x[:,1], y, color=COMPONENT_COLORS[assignment])

    # Plot the true lines
    for i,coefficients in enumerate(TRUE_COEFFICIENTS):
        x = np.linspace(0, 1, 6)
        features = np.ones((6, 2))
        features[:,1] = x
        y = features.dot(coefficients)
        

        plt.plot(x, y, color='gray', linestyle='--')

    # Plot the resulting lines
    for i,coefficients in enumerate(results.best.coefficients):
        x = np.linspace(0, 1, 6)
        features = np.ones((6, 2))
        features[:,1] = x
        y = features.dot(coefficients)

        plt.plot(x, y, color='black')

    plt.xlim(0,1)
    plt.savefig('figures/test_cem_results.png')
    plt.clf()

    print 'Testing cross-validation on mixture of 3 linear regression components with unknown number of components'
    NUM_FOLDS = 10
    NUM_WORKERS = 3
    results = cross_validate(DATA, KEYS, 5, 1, 10, 20, stochastic=False, verbose=True, num_workers=3)

    print 'Best result: {0} components'.format(results.best)
    results.plot('figures/test_cross_validation_results.png')









