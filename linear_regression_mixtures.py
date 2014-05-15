'''
Tools for fitting and evaluating finite mixtures of linear regressions.

Note that this case is slightly different than the classic mixture model.
Rather than a set of points, we have a set of sets of points. That is, each
subset is their own set of (x, y) points and a subset can only belong to a single
component.

The algorithm implemented is a classification expectation-maximization (CEM)
variant. Since EM is a local optimization procedure, it's recommended that you
do multiple random restarts to get a good approximation of the true optimum.

Author: Wesley Tansey
Date: 5/15/2014
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import csv
import argparse
from scipy.stats import norm
from multiprocessing import Pool

class MixtureModel(object):
    '''Container to store the results of a single iteration of the CEM algorithm.'''
    def __init__(self, assignments, component_weights, coefficients, variances):
        self.assignments = assignments
        self.component_weights = component_weights
        self.coefficients = coefficients
        self.variances = variances

class MixtureResults(object):
    '''Container to store the results of a CEM iteration.'''
    def __init__(self, num_components):
        self.num_components = num_components
        self.iterations = []
        self.log_likelihoods = []
        self.best = None

    def add_iteration(self, assignments, component_weights, coefficients, variances, data_log_likelihood):
        self.iterations.append(MixtureModel(assignments, component_weights, coefficients, variances))
        self.log_likelihoods.append(data_log_likelihood)

    def finish(self):
        '''Tell the container we're done running CEM.'''
        self.log_likelihoods = np.array(self.log_likelihoods)
        self.best = self.iterations[np.argmax(self.log_likelihoods)]

class CrossValidationResults(object):
    '''Container to store the results of a cross-validation run.'''
    def __init__(self, num_folds, min_components, max_components):
        self.min_components = min_components
        self.max_components = max_components
        self.components = np.arange(min_components, max_components+1)
        self.num_folds = num_folds
        self.results = [[] for _ in self.components]
        self.test_likelihoods = [[] for _ in self.components]
        self.means = None
        self.stderrs = None
        self.best = None

    def add(self, result, test_likelihood):
        idx = result.num_components - self.min_components
        self.results[idx].append(result)
        self.test_likelihoods[idx].append(test_likelihood)

    def create_folds(self, data):
        '''Create num_folds training and testing sets'''
        subsets = []
        keys = list(data.keys())
        samples_per_fold = len(data) / self.num_folds
        for i in xrange(self.num_folds):
            # Get the number of samples for this fold
            num_samples = samples_per_fold + 1 if i < len(data) % self.num_folds else samples_per_fold

            # Sample without replacement from the available keys
            selected = np.random.choice(keys, size=samples_per_fold, replace=False)
            
            # Remove the keys from available set
            for s in selected:
                keys.remove(s)

            # Add the selected subsets to the fold
            subsets.append({s: data[s] for s in selected})

        # Create training and testing sets from the folds
        folds = []

        # Note that this is terrible big-O, but folds are generally small so who cares
        for i,testing in enumerate(subsets):
            training = {}
            for j,fold in enumerate(subsets):
                if i == j:
                    continue
                for key, value in fold.iteritems():
                    training[key] = value
            folds.append((training, testing))

        return folds

    def finish(self):
        '''Tell the container we're done performing cross-validation.'''
        # Calculate the array of raw test error scores
        self.test_likelihoods = np.array(self.test_likelihoods)

        # Calculate the mean test errors for each number of components
        self.means = self.test_likelihoods.mean(axis=1)

        # Calculate the standard error for each number of components
        self.stderrs = self.test_likelihoods.std(axis=1) / np.sqrt(self.num_folds)

        # Get the best number of components
        self.best = self.components[np.argmax(self.means)]

        # Sanity check
        assert(len(self.means) == len(self.components))

    def plot(self, filename):
        '''Plot the results curve for the out-of-sample data log-likelihood.'''
        plt.plot(self.components, self.means, color='blue')
        plt.fill_between(self.components, self.means - self.stderrs, self.means + self.stderrs, facecolor='blue', alpha=0.2)
        plt.title('Avg. Out-of-Sample Performance For Regression Mixtures\n([{0}, {1}] components, {2} folds)'.format(self.min_components, self.max_components, self.num_folds))
        plt.xlim([self.min_components, self.max_components])
        plt.xlabel('Number of mixture components')
        plt.ylabel('Avg. data log-likelihood')
        plt.savefig(filename)
        plt.clf()

def weighted_linear_regression(x, y, weights):
    '''
    Perform linear regression and return the residuals. Note this assumes
    the weights are a vector rather than the standard diagonal matrix-- this is
    for computational efficiency.
    '''
    return np.linalg.pinv((weights[:,np.newaxis] * x).T.dot(x)).dot((weights[:,np.newaxis] * x).T.dot(y))
    
def weighted_regression_variance(x, y, weights, coefficients):
    '''Calculate the variance of a regression model where each observation is weighted.'''
    # TODO: Vectorize
    result = 0.
    for i in xrange(len(y)):
        result += weights[i] * (y[i] - x[i].T.dot(coefficients)) ** 2
    return result / weights.sum()

def calculate_assignments(assignment_weights, stochastic):
    '''
    Assign each set of points to a component.
    If stochastic is true, randomly sample proportional to the assignment_weights.
    Otherwise, assign the component with the maximum weight.

    This is the C-step in the CEM algorithm.
    '''
    if stochastic:
        return np.array([np.random.choice(len(row),p=row) for row in assignment_weights])
    return np.argmax(assignment_weights, axis=1)

def calculate_assignment_weights(data, keys, component_weights, coefficients, variances):
    '''
    Determine a probability for each component to generate each set of points.

    This is the E-step in the CEM algorithm.
    '''
    num_components = len(component_weights)

    # Initialize the new assignment weights
    assignment_weights = np.ones((len(data), num_components), dtype=float)

    # Calculate the new weights for every set of points
    for i,key in enumerate(keys):
        # Get the set of points
        x, y = data[key]

        # Calculate the likelihood of the points one at a time
        # to prevent underflow issues
        for xi, yi in zip(x, y):
            # Get the mean of each component
            mu = np.array([xi.dot(b) for b in coefficients])

            # Get the standard deviation of each component
            sigma = np.array([np.sqrt(v) for v in variances])

            # Calculate the likelihood of this data point coming from each component
            temp_weights = norm.pdf(yi, loc=mu, scale=sigma)

            # Update the likelihood of each component generating this set
            assignment_weights[i] *= temp_weights / temp_weights.sum()
            assignment_weights[i] /= assignment_weights[i].sum()

        # Multiply in the component weightings
        assignment_weights[i] *= component_weights
        assignment_weights[i] /= assignment_weights[i].sum()

    return assignment_weights


def maximum_likelihood_parameters(data, keys, num_components, num_features, assignments, assignment_weights):
    '''
    Calculate the parameter values that maximize the likelihood of the data.

    This is the M-step of the CEM algorithm.
    '''
    # Calculate the weight of each component in the mixture
    component_weights = np.array([(assignments == i).sum() for i in xrange(num_components)]) / float(len(assignments))
    
    # Calculate the regression coefficients and variance for each component
    coefficients = np.zeros((num_components, num_features))
    variances = np.zeros(num_components)
    for i in xrange(num_components):
        # Get the points that are members of this component
        points = np.where(assignments == i)[0]

        # Get the weights for each set
        subset_weights = assignment_weights[points][:,i]

        # If no points were assigned to this cluster, soft-assign it random points
        # TODO: Is there a better way to proceed here? Some sort of split-merge type thing?
        if len(points) == 0:
            points = np.random.choice(len(assignments), size=np.random.randint(1, len(assignments)), replace=False)
            subset_weights = np.ones(len(points)) / float(len(points))

        # Get the data associated with this component
        component_x = []
        component_y = []
        weights = []
        for key, subset_weight in zip(keys[points], subset_weights):
            # Get the data for this subset
            x, y = data[key]

            # Add the points to the overall values to regress on
            component_x.extend(x)
            component_y.extend(y)

            # Each point in a set gets equal weight
            weights.extend([subset_weight / float(len(y))] * len(y))

        # Convert the results to numpy arrays
        component_x = np.array(component_x)
        component_y = np.array(component_y)
        weights = np.array(weights)

        # Get the weighted least squares coefficients
        coefficients[i] = weighted_linear_regression(component_x, component_y, weights)

        # Get the variance of the component given the coefficients
        variances[i] = weighted_regression_variance(component_x, component_y, weights, coefficients[i])

    return (component_weights, coefficients, variances)


def data_log_likelihood(data, keys, assignments, component_weights, coefficients, variances):
    '''
    Calculate the log-likelihood of the data being generated by the mixture model
    with the given parameters.
    '''
    log_likelihood = 0
    for i,key in enumerate(keys):
        x, y = data[key]

        assigned = assignments[i]

        mu = x.dot(coefficients[assigned])
        sigma = np.sqrt(variances[assigned])

        log_likelihood += np.log(norm.pdf(y, loc=mu, scale=sigma)).sum()
        log_likelihood += np.log(component_weights[assigned])
    
    return log_likelihood


def fit_mixture(data, keys, num_components, max_iterations, stochastic=False, verbose=False, threshold=0.00001):
    '''
    Run the classification expecatation-maximization (CEM) algorithm to fit a maximum likelihood model.
    Note that the result is a local optimum, not necessarily a global one.
    '''
    num_features = data.values()[0][0].shape[1]

    # Initialize the results
    results = MixtureResults(num_components)

    prev_log_likelihood = 1
    cur_log_likelihood = 0
    cur_iteration = 0

    if verbose:
        print '\t\t\t\tRandomly initializing assignment weights'

    # Random initialization
    assignment_weights = np.random.uniform(size=(len(data), num_components))
    assignment_weights /= assignment_weights.sum(axis=1)[:, np.newaxis]
    
    if verbose:
        print '\t\t\t\tSampling assignments'

    # Initialize using the normal steps now
    assignments = calculate_assignments(assignment_weights, True)

    if verbose:
        print '\t\t\t\tCalculating parameters'

    component_weights, coefficients, variances = maximum_likelihood_parameters(data, keys, num_components, num_features, assignments, assignment_weights)

    while np.abs(prev_log_likelihood - cur_log_likelihood) > threshold and cur_iteration < max_iterations:
        if verbose:
            print '\t\t\tStarting iteration #{0}'.format(cur_iteration)

        # Calculate the expectation weights
        assignment_weights = calculate_assignment_weights(data, keys, component_weights, coefficients, variances)

        # Assign a value to each of the points
        assignments = calculate_assignments(assignment_weights, stochastic=stochastic)

        # Maximize the likelihood of the parameters
        component_weights, coefficients, variances = maximum_likelihood_parameters(data, keys, num_components, num_features, assignments, assignment_weights)

        # Calculate the total data log-likelihood
        prev_log_likelihood = cur_log_likelihood
        cur_log_likelihood = data_log_likelihood(data, keys, assignments, component_weights, coefficients, variances)

        # Add the iteration to the results
        results.add_iteration(assignments, component_weights, coefficients, variances, cur_log_likelihood)

        if verbose:
            print '\t\t\tLog-Likelihood: {0}'.format(cur_log_likelihood)

        cur_iteration += 1

    # Tell the results that we're done fitting
    results.finish()

    return results

def out_of_sample_log_likelihood(data, keys, component_weights, coefficients, variances):
    '''
    Given a set of subsets that we haven't seen before, assign them each to and
    figure out the total data log-likelihood.
    '''
    # Calculate the expectation weights
    assignment_weights = calculate_assignment_weights(data, keys, component_weights, coefficients, variances)

    # Assign a value to each of the points
    assignments = calculate_assignments(assignment_weights, stochastic=False)

    # Get the log-likelihood for the out-of-sample data
    return data_log_likelihood(data, keys, assignments, component_weights, coefficients, variances)

def fit_worker(worker_params):
    data, keys, num_components, max_iterations, stochastic, verbose = worker_params
    return fit_mixture(data, keys, num_components, max_iterations, stochastic=stochastic, verbose=verbose)

def fit_with_restarts(data, keys, num_components, max_iterations, num_restarts, stochastic=False, verbose=False, num_workers=1):
    '''
    Run the CEM algorithm for num_restarts times and return the best result.
    '''
    max_result = None
    max_likelihood = None

    # Fit the mixture with every restart done in parallel
    if num_workers > 1:
        pool = Pool(num_workers)
        worker_params = [(data, keys, num_components, max_iterations, stochastic, verbose) for _ in xrange(num_restarts)]
        results = pool.map(fit_worker, worker_params)
    else:
        results = [fit_mixture(data, keys, num_components, max_iterations, stochastic=stochastic, verbose=verbose) for _ in xrange(num_restarts)]

    for trial in xrange(num_restarts):
        result = results[trial]

        if max_likelihood is None or result.log_likelihoods.max() > max_likelihood:
            max_result = result
            max_likelihood = result.log_likelihoods.max()

    return max_result

def fit_with_restarts_worker(worker_params):
    data, keys, num_components, max_iterations, num_restarts, stochastic, verbose = worker_params
    return fit_with_restarts(data, keys, num_components, max_iterations, num_restarts, stochastic=stochastic, verbose=verbose)

def cross_validate(data, keys, num_folds, min_components, max_components, max_iterations, stochastic=False, verbose=False, initialization_trials=5, num_workers=1):
    '''
    Perform n-fold cross-validation for all values in the [min, max] range of
    mixture component counts.
    '''
    results = CrossValidationResults(num_folds, min_components, max_components)

    # Create folds for training and testing
    folds = results.create_folds(data)

    if num_workers > 1:
        pool = Pool(num_workers)

    # Test every fold across all possible mixture numbers
    for fold, (training_data, testing_data) in enumerate(folds):
        if verbose:
            print '\tFold #{0} Training: {1} Testing: {2}'.format(fold, len(training_data), len(testing_data))

        training_keys = np.array(training_data.keys())
        testing_keys = np.array(testing_data.keys())
        component_results = []

        # Fit every number of components on this fold
        if num_workers == 1:
            # If we're running single threaded, just build the list iteratively
            for num_components in results.components:
                if verbose:
                    print '\t\t{0} components'.format(num_components)

                component_results.append(fit_with_restarts(training_data, training_keys, num_components, max_iterations, initialization_trials, stochastic=stochastic, verbose=verbose))
        else:
            # Fit each component in parallel
            if verbose:
                print '\t\tFitting with {0} processes'.format(num_workers)
            worker_params = [(training_data, training_keys, num_components, max_iterations, initialization_trials, stochastic, verbose) for num_components in results.components]
            component_results = pool.map(fit_with_restarts_worker, worker_params)

        if verbose:
            print '\t\tTesting on out-of-sample data'

        # Test every result on out-of-sample data
        for max_result in component_results:
            # Get the model from the best iteration
            model = max_result.best

            # Get the out-of-sample error
            oos_log_likelihood = out_of_sample_log_likelihood(testing_data, testing_keys, model.component_weights, model.coefficients, model.variances)

            # Add the result
            results.add(max_result, oos_log_likelihood)

    # Tell the results that we're done testing
    results.finish()

    if num_workers > 1:
        pool.terminate()

    return results

def load_data(filename):
    '''
    Load the data from file and return a map from subset ID to subset data.
    '''
    data = {}
    with open(filename, 'rb') as f:
        reader = csv.reader(f)

        # Skip the header
        header = reader.next()

        # Load the data for each subset
        for line in reader:
            subset_id = int(float(line[0]))
            x = np.array([float(a) for a in line[1:-1]])
            y = float(line[-1])

            if subset_id not in data:
                data[subset_id] = ([], [])

            data[subset_id][0].append(x)
            data[subset_id][1].append(y)

    # Convert all the lists to numpy arrays
    for subset_id in data.iterkeys():
        x, y = data[subset_id]
        data[subset_id] = (np.array(x), np.array(y))

    return data

def save_results(results, keys, filename):
    '''
    Save the results of the CEM algorithm to file with the format:
    row 0: num_components
    rows 1-n: weight,coefficient0,coefficient1,...,coefficientP,variance
    rows > n: subset_id,assignment
    '''
    with open(filename, 'wb') as f:
        writer = csv.writer(f)

        # Write the header info about how many components are in the mixture
        writer.writerow([results.num_components])

        # Get the best model
        best = results.best
    
        # Create the component rows
        print 'component_weights: {0} coefficients: {1} variances: {2}'.format(best.component_weights.shape, best.coefficients.shape, best.variances.shape)
        rows = np.zeros((results.num_components, best.coefficients.shape[1] + 2))
        rows[:,0] = best.component_weights
        rows[:,1:1+best.coefficients.shape[1]] = best.coefficients
        rows[:,1+best.coefficients.shape[1]] = best.variances

        # Write the components to file
        writer.writerows(rows)

        # Write the subset assignments to file
        for key, assignment in zip(keys, best.assignments):
            writer.writerow([key, assignment])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fits a finite mixture of linear regression models over sets of sets of data points.')

    parser.add_argument('datafile', help='The file containing the data.')
    parser.add_argument('--num_components', type=int, help='The number of linear regression mixture components. If not specified, cross-validation will be used to determine the number of components.')
    parser.add_argument('--outfile', default='results.csv', help='The file to output the resulting model parameters and point assignments.')
    parser.add_argument('--stochastic', dest='stochastic', action='store_true', help='Use stochastic sampling for point classification. Deterministic EM assigns points to their MAP component at every iteration; stochastic EM samples proportional to likelihood. After max_iterations is over, the MAP assignments will be calculated based on the parameters of the last iteration.')
    parser.add_argument('--max_iterations', type=int, default=1000, help='The maximum number of EM iterations to perform.')
    parser.add_argument('--min_components', type=int, default=1, help='The minimum number of mixture components to use during cross-validation.')
    parser.add_argument('--max_components', type=int, default=20, help='The maximum number of mixture components to use during cross-validation.')
    parser.add_argument('--num_folds', type=int, default=10, help='The number of folds to use when performing n-fold cross-validation.')
    parser.add_argument('--num_restarts', type=int, default=5, help='The number of random restarts to try.')
    parser.add_argument('--plot_cross_validation', help='Plot the results of n-fold cross-validation and save the results to file.')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Print detailed progress information to the console.')
    parser.add_argument('--num_workers', type=int, default=1, help='The number of parallel processes to use when performing cross-validation.')

    parser.set_defaults(stochastic=False, verbose=False)

    # Get the arguments from the command line
    args = parser.parse_args()

    # Load the data from file
    print 'Loading data from {0}'.format(args.datafile)
    data = load_data(args.datafile)

    keys = list(data.keys())

    # Deterministic EM assigns points to their MAP component at every iteration. Stochastic samples proportional to likelihood.
    use_stochastic_em = args.stochastic
    verbose = args.verbose

    # Get the number of regression model components in our mixture
    num_components = args.num_components

    # If we don't know the number of components, do cross-validation to find them.
    if num_components is None:
        print 'Performing {0}-fold cross-validation to find the best component count in the range [{1},{2}]'.format(args.num_folds, args.min_components, args.max_components)
        cv_results = cross_validate(data, keys, args.num_folds, args.min_components, args.max_components, args.max_iterations, stochastic=use_stochastic_em, initialization_trials=args.num_restarts, verbose=verbose, num_workers=args.num_workers)

        if args.plot_cross_validation is not None:
            print 'Plotting cross-validation results to {0}'.format(args.plot_cross_validation)
            cv_results.plot(args.plot_cross_validation)

        num_components = cv_results.best

    keys = np.array(keys)

    # Now we know the number of components, so just fit the finite mixture
    print 'Fitting a mixture of {0} components with {1} restarts parallelized over {2} processes'.format(num_components, args.num_restarts, args.num_workers)
    results = fit_with_restarts(data, keys, num_components, args.max_iterations, args.num_restarts, stochastic=use_stochastic_em, verbose=verbose, num_workers=args.num_workers)
    
    # Save the resulting mixture model to file, along with all the point assignments
    print 'Saving results to {0}'.format(args.outfile)
    save_results(results, keys, args.outfile)














