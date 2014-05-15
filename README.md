Mixtures of Linear Regressions for Subsets
==========================================

This is an implementation of a classification expectation-maximization (CEM) algorithm for fitting mixtures of linear regressions.

Note that this case is slightly different than the classic mixture model. Rather than a set of points, we have a set of sets of points. That is, each subset is their own set of (x, y) points and a subset can only belong to a single component. The algorithm implemented is a variant of one used for simple mixtures of linear regressions for points [1].

Since EM is a local optimization procedure, it's recommended that you do multiple random restarts to get a good approximation of the true optimum.

Fitting a finite mixture model
------------------------------
To fit a finite mixture model with a known number of mixture components:

```python
from linear_regression_mixtures import *

# Choose some parameter values for the CEM algorithm
num_components = 3 # the number of mixture components
max_iterations = 20 # the maximum iterations per CEM run
num_restarts = 5 # the number of random restarts to try
stoachastic = False # whether to use stochastic EM instead of deterministic CEM
verbose = True # print output as the algorithm progresses
num_workers = 4 # the number of worker processes to use

# Load your data from file
data = load_data('mydata.csv')

# Create a list of the keys
keys = list(keys)

# Run the CEM algorithm
results = fit_with_restarts(data, keys, num_components, max_iterations, num_restarts, stochastic=stochastic, verbose=verbose, num_workers=num_workers)

# Save the results to file
save_results(results, 'myresults.csv')
```

Or just use the command-line interface:

```
python linear_regression_mixtures mydata.csv --outfile myresults.csv --max_iterations 20 --num_restarts 5 --verbose --num_workers 4 --num_components 3
```

![Finite Linear Regression Mixture Results](https://github.com/tansey/regression_mixtures/raw/master/figures/test_cem_results.png)

The parallelization will be implemented at the level of random restarts.

Finding the number of mixture components
----------------------------------------
If you do not know the number of mixture components generating your data, you can use cross-validation to find it:

```
python linear_regression_mixtures mydata.csv --outfile myresults.csv --max_iterations 20 --num_restarts 5 --verbose --num_workers 4 --min_components 1 --max_components 10 --num_folds 5 --plot_cross_validation cv_results.png
```

![Cross-Validation Results](https://github.com/tansey/regression_mixtures/raw/master/figures/test_cross_validation_results.png)

The cross-validation will parallelize every component count choice during cross-validation. After cross-validation is finished, the count with the highest data log-likelihood will be selected and the complete dataset will be fit using that component count. During this stage, the random restarts will be parallelized just like in the known-count case.

To-do
-----
The current approach to measuring out-of-sample performance for cross-validation is to hold out a collection of subsets, then fix the model parameters and perform ML inference on the held-out subsets. The total data log-likelihood of the held-out subsets is then measured based on which component they were assigned. The problem with this method is that there is not much penalty for having too many components. A better approach may be to hold out individual points from the subsets and then see how well they are predicted after fitting.


References
----------
[1] Faria, Susana, and Gilda Soromenho. "Fitting mixtures of linear regressions." Journal of Statistical Computation and Simulation 80.2 (2010): 201-225.