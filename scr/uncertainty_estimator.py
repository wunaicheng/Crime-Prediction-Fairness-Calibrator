import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from sklearn.metrics import pairwise_kernels


def empirical_percentile_estimator(X, precentile, prob):
    """
    We then define empirical percentile $\hat{r}$ as the fraction of data points in the test set
    $\datat = \{(\bfy_i, \bfx_i)\}_{i=1}^{N_t}$ with true quantities less than or equal to the predictive quantile.
    \begin{equation}\label{eqn:rhat}
    \hat{r} = \frac{1}{N_\mathrm{test}}\sum_{i=1}^{N_\mathrm{test}} \bbI \left[\bfy_i\leq \bfy_q\left(r;\bfx_i,
               \bfeta, \calD\right) \right],
    \end{equation}
    where $\bbI[\cdot]$ is the indicator function. Under this construction, a perfect posterior would produce predictive
    percentiles which exactly match empirical percentiles, $\hat{r}(r)=r$ for $r\in[0,1]$.

    Parameters
    ----------
        samples: numpy-array
            data points

        x: numpy-array
            theoretical percentiles should be between one and zero

        prop: scipy.stats.probability_function,
            the probability function of samples

    return
    ------
        numpy-array: empirical percentile
    """

    size = float(len(X))

    cdf = prob.cdf(X)

    y = np.array([sum(cdf < precentile[i])/size for i in range(len(precentile))])

    return y


def MPCE2_null_estimator(K_xx, e, rng):
    """
    Compute the MPCE2^2_u for one bootstrap realization.

    Parameters
    ----------
        e: numpy-array
            one-dimensional percentile/probability error vector.

        K_xx: numpy-array
            evaluated kernel function.

        rng: type(np.random.RandomState())
             a numpy random function

    return
    ------
        float: an unbiased estimate of MMD^2_u
    """
    idx = rng.permutation(len(e))
    return MPCE2_estimator(K_xx, e[idx])


def MPCE2_estimator(K_xx, e):
    """
    The estimator MPCE2 = \sum (e Kxx e^T) / n / (n-1)

    Parameters
    ----------
        e: numpy array
            one-dimensional percentile/probability error vector.

        K_xx: numpy array
            evaluated kernel function.

    return
    ------
        float: estimated MPCE2
    """

    size = len(e)

    K = (e.flatten() * K_xx.T).T * e.flatten()

    """
    s = 0
    s1 = 0
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            else:
                s += K[i, j] / (size * (size - 1.0))
                s1 += e[i] * K_xx[i, j] * e[j] / (size * (size - 1.0))
    print(s, s1, (K.sum() - K.diagonal().sum()) / (size * (size - 1.0)))
    exit() """

    return ( K.sum() - K.diagonal().sum() ) / (size * (size-1.0))


def compute_null_distribution(K_xx, e, iterations=1000, n_jobs=1,
                              verbose=True, random_state=None):
    """
    Compute the null-distribution of test statistics via a bootstrap procedure.

    Parameters
    ----------
        e: numpy-array
            one-dimensional percentile/probability error vector.

        K_xx: numpy array
            evaluated kernel function.

        iterations: int
            controls the number of bootstrap realizations

        verbose: bool
            controls the verbosity of the model's output.

        random_state: type(np.random.RandomState()) or None
            defines the initial random state.

    return
    ------
    numpy-array: a boostrap samples of the test null distribution
    """

    if type(random_state) == type(np.random.RandomState()):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    if verbose:
        iterations_list = tqdm(range(iterations))
    else:
        iterations_list = range(iterations)

    # compute the null distribution
    # for 1 cpu run the normal code, for more cpu use the Parallel library. This maximize the speed.
    if n_jobs == 1:
        test_null = [MPCE2_null_estimator(K_xx, e, rng) for _ in iterations_list]

    else:
        test_null = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(MPCE2_null_estimator)(K_xx, e, rng)
                                                              for _ in iterations_list)
    return np.array(test_null)


def MPCE2_test_estimator(X, Cerr, kernel_function='rbf',
                         iterations=1000, verbose=True,
                         random_state=None, n_jobs=1,
                         **kwargs):
    """
    This function estimate MPCE^2_u employing a kernel trick. MPCE^2_u tests if a proposed posterior credible interval
      is calibrated employing a randomly drawn calibration test. The null hypothesis is that the posteriors are
      properly calibrated This function perform a bootstrap algorithm to estimate the null distribution,
      and corresponding p-value.

    Parameters
    ----------
        X: numpy-array
            data, of size NxD [N is the number of data points, D is the features dimension]

        Cerr: numpy-array
            credible error vector, of size Nx1 [N is the number of data points]

        kernel_function: string
            defines the kernel function. For the list of implemented kernel please consult with
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.kernel_metrics.html#sklearn.metrics.pairwise.kernel_metrics

        iterations: int
            controls the number of bootstrap realizations

        verbose: bool
            controls the verbosity of the model's output.

        random_state: type(np.random.RandomState()) or None
            defines the initial random state.

        n_jobs: int
            number of jobs to run in parallel.

        **kwargs:
            extra parameters, these are passed to `pairwise_kernels()` as kernel parameters o
            as the number of k. E.g., if `kernel_two_sample_test(..., kernel_function='rbf', gamma=0.1)`

    return
    ------
    tuple of size 3 (float, numpy-array, float)
        - first element is the test value,
        - second element is samples from the null distribution via a bootstraps algorithm,
        - third element is the estimated p-value.
    """

    check_attributes(X, Cerr, iterations=1000, n_jobs=1)

    # p-value's resolution
    resolution = 1.0/iterations

    # pre-compute the kernel function
    K_xx = pairwise_kernels(X, X, metric=kernel_function, **kwargs)

    # estimate the test value
    test_value = MPCE2_estimator(K_xx, Cerr)

    if verbose:
        print("test value = %s"%test_value)
        print("Start computing the null distribution.")

    # compute the null distribution via a bootstrap algorithm
    test_null = compute_null_distribution(K_xx, Cerr, iterations=iterations, verbose=verbose,
                                          n_jobs=n_jobs, random_state=random_state)

    # compute the local p-value, if less then the resolution set it to the resolution
    p_value_global = max(resolution, resolution*(0 > test_null).sum())

    if verbose:
        if p_value_global == resolution:
            print("p-value [global] < %s \t (resolution : %s)" % (p_value_global, resolution))
        else:
            print("p-value [global] ~= %s \t (resolution : %s)" % (p_value_global, resolution))

    # compute the global p-value, if less then the resolution set it to the resolution
    p_value_local = max(resolution, resolution*(test_null > test_value).sum())

    if verbose:
        if p_value_local == resolution:
            print("p-value [local] < %s \t (resolution : %s)" % (p_value_local, resolution))
        else:
            print("p-value [local] ~= %s \t (resolution : %s)" % (p_value_local, resolution))

    return test_value, test_null, p_value_global, p_value_local


def error_witness_function(X, e, grid, kernel_function='rbf', **kwargs):

    check_attributes(X, e)

    # def witness_function(e, K_xx):
    #    return np.sum(e * K_xx.T, axis=1) / len(e)

    # pre-compute the kernel function
    K_xx = pairwise_kernels(X, grid, metric=kernel_function, **kwargs)

    ewf = np.sum(e.flatten() * K_xx.T, axis=1) / np.sum(K_xx.T, axis=1)  # len(e) #witness_function(e, K_xx) / np.sum(K_xx.T, axis=1) #

    return ewf


def construct_credible_error_vector(Y, Yr_up, Yr_down, alpha):
    """
    For a one dimensional output prediction Y it construct the credible error vector. It takes the lower and upper
      percentiles and assuming credible level alpha is fixed.

    Parameters
    ----------
        Y: numpy-array
            data, of size Nx1 [N is the number of data points]

        Yr_up: numpy-array
            upper percentile vector, of size Nx1 [N is the number of data points]

        Yr_up: numpy-array
            upper percentile vector, of size Nx1 [N is the number of data points]

        alpha: float
            the theoretical credible level alpha

    return
    ------
    numpy-array: credible error vector
    """

    if Y.flatten().shape[0] != Yr_up.flatten().shape[0]:
        raise ValueError("Incompatible dimension for Y and Yr_up matrices. Y and Yr_up should have the same feature dimension,"
                         ": Y.shape[0] == %i while Yr.shape[0] == %i." % (Y.shape[0], Yr_up.shape[0]))

    if Y.flatten().shape[0] != Yr_down.flatten().shape[0]:
        raise ValueError("Incompatible dimension for Y and Yr matrices. Y and Yr should have the same feature dimension,"
                         ": Y.shape[0] == %i while Yr_down.shape[0] == %i." % (Y.shape[0], Yr_down.shape[0]))

    if alpha < 0 or alpha > 1:
        raise ValueError("Incompatible value for alpha. alpha should be a real value between 0 and 1: alpha == " + alpha)

    # indicator of Y less than posterior percentile r
    Yind = 1.0 * ((Y < Yr_up) * (Y > Yr_down))

    # percentile/probability error vector
    e = (Yind - alpha)

    return e


def check_attributes(X, e, iterations=1000, n_jobs=1):
    """
        Check whether the input attributes are in proper format. If not exit with an Error message.
    """
    if X.shape[0] != e.shape[0]:
        raise ValueError("Incompatible dimension for X and e matrices. X and e should have the same feature dimension,"
                         ": X.shape[0] == %i while e.shape[0] == %i." % (X.shape[0], e.shape[0]))

    if not (isinstance(iterations, int) and iterations > 1):
        raise ValueError('iterations has incorrect type or less than 2.')

    if not (isinstance(n_jobs, int) and n_jobs > 0):
        raise ValueError('n_jobs is incorrect type or <1. n_jobs:%s' % n_jobs)