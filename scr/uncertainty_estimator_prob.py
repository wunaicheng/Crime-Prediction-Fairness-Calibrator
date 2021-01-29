import numpy as np
from joblib import Parallel, delayed
from sklearn.gaussian_process.kernels import pairwise_kernels
from tqdm import tqdm

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

    Return
    ------
        numpy-array: empirical percentile
    """

    size = float(len(X))

    cdf = prob.cdf(X)

    y = np.array([sum(cdf < precentile[i])/size for i in range(len(precentile))])

    return y

def ELCE2_null_estimator(p_err, K, rng):
    """
    Compute the ELCE^2_u for one bootstrap realization.

    Parameters
    ----------
        p_err: numpy-array
            one-dimensional probability error vector.

        K: numpy-array
            evaluated kernel function.

        rng: type(np.random.RandomState())
             a numpy random function

    return
    ------
        float: an unbiased estimate of ELCE^2_u
    """

    idx = rng.permutation(len(p_err))

    return ELCE2_estimator(K, p_err[idx])


def ELCE2_estimator(K_xx, e):
    """
    The estimator ELCE^2 = \sum (e Kxx e^T) / n / (n-1)

    Parameters
    ----------
        e: numpy array
            one-dimensional percentile/probability error vector.

        K_xx: numpy array
            evaluated kernel function.

    return
    ------
        float: estimated ELCE^2
    """

    size = len(e)

    K = (e.flatten() * K_xx.T).T * e.flatten()

    return (K.sum() - K.diagonal().sum()) / (size * (size-1.0))


def compute_null_distribution(p_err, K, iterations=1000, n_jobs=1,
                              verbose=False, random_state=None):
    """
    Compute the null-distribution of test statistics via a bootstrap procedure.

    Parameters
    ----------
        p_err: numpy-array
            one-dimensional probability error vector.

        K: numpy array
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
        test_null = [ELCE2_null_estimator(p_err, K, rng) for _ in iterations_list]

    else:
        test_null = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(ELCE2_null_estimator)(p_err, K, rng)
                                                              for _ in iterations_list)
    return np.array(test_null)


def ELCE2_test_estimator(X, Y, p, kernel_function='rbf',
                         prob_kernel_wdith=0.1,
                         iterations=None, verbose=False,
                         random_state=None, n_jobs=1,
                         **kwargs):
    """
    This function estimate ELCE^2_u employing a kernel trick. ELCE^2_u tests if a proposed posterior credible interval
      is calibrated employing a randomly drawn calibration test. The null hypothesis is that the posteriors are
      properly calibrated This function perform a bootstrap algorithm to estimate the null distribution,
      and corresponding p-value.

    Parameters
    ----------
        X: numpy-array
            data, of size NxD [N is the number of data points, D is the features dimension]

        Y: numpy-array
            credible error vector, of size Nx1 [N is the number of data points]

        p: numpy-array
            credible error vector, of size Nx1 [N is the number of data points]

        kernel_function: string
            defines the kernel function. For the list of implemented kernel please consult with
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.kernel_metrics.html#sklearn.metrics.pairwise.kernel_metrics

        prob_kernel_wdith: float
            Width of the probably kernel function.

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

    Parameters
    ----------
    tuple of size 3 (float, numpy-array, float)
        - first element is the test value,
        - second element is samples from the null distribution via a bootstraps algorithm,
        - third element is the estimated p-value.
    """

    check_attributes(X, Y, iterations=1000, n_jobs=1)

    # pre-compute the kernel function
    K_xx = pairwise_kernels(X, X, metric=kernel_function, **kwargs)
    K_pp = pairwise_kernels(p[:, np.newaxis], p[:, np.newaxis], metric='rbf', gamma=1.0/prob_kernel_wdith**2)
    K = K_pp * K_xx

    # error vector
    p_err = Y - p

    # estimate the test value
    test_value = ELCE2_estimator(K, p_err)

    if verbose:
        print("test value = %s"%test_value)
        print("Computing the null distribution.")

    if iterations is None: return test_value

    # p-value's resolution
    resolution = 1.0/iterations

    # compute the null distribution via a bootstrap algorithm
    test_null = compute_null_distribution(p_err, K, iterations=iterations, verbose=verbose,
                                          n_jobs=n_jobs, random_state=random_state)

    # compute the p-value, if less then the resolution set it to the resolution
    p_value = max(resolution, resolution*(test_null > test_value).sum())

    if verbose:
        if p_value == resolution:
            print("p-value < %s \t (resolution : %s)" % (p_value, resolution))
        else:
            print("p-value ~= %s \t (resolution : %s)" % (p_value, resolution))

    return test_value, test_null, p_value


def error_witness_function(X, Y, p, X_grid, p_grid, prob_kernel_wdith=0.1, kernel_function='rbf', **kwargs):

    check_attributes(X, Y)

    # def witness_function(e, K_xx):
    #    return np.sum(e * K_xx.T, axis=1) / len(e)

    # pre-compute the kernel function
    # K_xx = pairwise_kernels(X, grid, metric=kernel_function, **kwargs)

    # pre-compute the kernel function
    K_xx = pairwise_kernels(X, X_grid, metric=kernel_function, **kwargs)
    K_pp = pairwise_kernels(p[:, np.newaxis], p_grid[:, np.newaxis], metric='rbf', gamma=1.0 / prob_kernel_wdith ** 2)
    K = K_pp * K_xx

    # error vector
    p_err = Y - p

    ewf = np.sum(p_err.flatten() * K.T, axis=1) / np.sum(K.T, axis=1) # / len(p_err)

    return ewf


def local_bias_estimator(X, Y, p, X_grid, model='KRR', kernel_function='rbf', **kwargs):

    check_attributes(X, Y)

    if model == 'KRR':
        from sklearn.kernel_ridge import KernelRidge
        model = KernelRidge(kernel=kernel_function, **kwargs)
        # kr = KernelRidge(alpha=alpha, kernel='rbf', **kwargs)
    elif model == 'SVR':
        from sklearn.svm import SVR
        model = SVR(kernel=kernel_function, **kwargs)
    elif model == 'EWF':
        K = pairwise_kernels(X, X_grid, metric=kernel_function, **kwargs)
        p_err = Y - p
        bias = np.sum(p_err.flatten() * K.T, axis=1) / np.sum(K.T, axis=1)
        return bias
    else:
        raise ValueError("Model %s is not defined." % model)

    bias_calibration = Y - p

    model.fit(X, bias_calibration)
    bias = model.predict(X_grid)

    return bias


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


def calibrate(Xtrain, prob_train, Ytrain, Xtest = None, Xtrun_simulationest=None, prob_test=None, method='platt', **kwargs):
    """
        A calibration method that takes the predicted probabilties and positive cases and recalibrate the probabilities.

        Parameters
        ----------
        y_true : array, shape (n_samples_train,)
            True targets for the training set.

        y_prob_train : array, shape (n_samples_train,)
            Probabilities of the positive class to train a calibration model.

        y_prob_test : array, shape (n_samples_test,)
            Probabilities of the positive class to be calibrated (test set). If None it re-calibrate the training set.

        method: string, 'platt', 'isotonic', 'temperature_scaling', 'beta', 'HB', 'BBG', 'ENIR'
            The method to use for calibration. Can be ‘sigmoid’ which corresponds to Platt’s method
            (i.e. a logistic regression model) or ‘isotonic’ which is a non-parametric approach.
            It is not advised to use isotonic calibration with too few calibration samples (<<1000) since it tends to overfit.

        Returns
        -------
        p_calibrated : array, shape (n_bins,)
            The calibrated error for test set.


        References
        ----------
        Küppers et al., "Multivariate Confidence Calibration for Object Detection." CVPR Workshops, 2020.

        Leeuw, Hornik, Mair, Isotone, "Optimization in R : Pool-Adjacent-Violators Algorithm (PAVA) and Active
        Set Methods." Journal of Statistical Software, 2009.

        Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht, "Obtaining well calibrated probabilities
        using bayesian binning." Twenty-Ninth AAAI Conference on Artificial Intelligence, 2015.

        Kull, Meelis, Telmo Silva Filho, and Peter Flach: "Beta calibration: a well-founded and easily implemented
        improvement on logistic calibration for binary classifiers." Artificial Intelligence and Statistics,
        PMLR 54:623-631, 2017.

        Zadrozny, Bianca and Elkan, Charles: "Obtaining calibrated probability estimates from decision
        trees and naive bayesian classifiers." In ICML, pp. 609–616, 2001.

        Zadrozny, Bianca and Elkan, Charles: "Transforming classifier scores into accurate
        multiclass probability estimates." In KDD, pp. 694–699, 2002.

        Ryan J Tibshirani, Holger Hoefling, and Robert Tibshirani: "Nearly-isotonic regression."
        Technometrics, 53(1):54–61, 2011.

        Naeini, Mahdi Pakdaman, and Gregory F. Cooper: "Binary classifier calibration using an ensemble of near
        isotonic regression models." 2016 IEEE 16th International Conference on Data Mining (ICDM). IEEE, 2016.

        Chuan Guo, Geoff Pleiss, Yu Sun and Kilian Q. Weinberger: "On Calibration of Modern Neural Networks."
        Proceedings of the 34th International Conference on Machine Learning, 2017.

        Pereyra, G., Tucker, G., Chorowski, J., Kaiser, L. and Hinton, G.: “Regularizing neural networks by
        penalizing confident output distributions.” CoRR, 2017.

        Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P.,
        Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M. and Duchesnay, E.:
        "Scikit-learn: Machine Learning in Python." In Journal of Machine Learning Research, volume 12 pp 2825-2830,
        2011.

        Platt, John: "Probabilistic outputs for support vector machines and comparisons to regularized likelihood
        methods." Advances in large margin classifiers, 10(3): 61–74, 1999.

        Neumann, Lukas, Andrew Zisserman, and Andrea Vedaldi: "Relaxed Softmax: Efficient Confidence Auto-Calibration
        for Safe Pedestrian Detection." Conference on Neural Information Processing Systems (NIPS) Workshop MLITS, 2018.

        Nilotpal Chakravarti, Isotonic Median Regression: A Linear Programming Approach, Mathematics of Operations
        Research Vol. 14, No. 2 (May, 1989), pp. 303-308.
    """

    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    from netcal.scaling import TemperatureScaling, BetaCalibration
    from netcal.binning import HistogramBinning, BBQ, ENIR

    if prob_test is None:
        probs = prob_train[:, np.newaxis]
    else:
        probs = prob_test[:, np.newaxis]

    if Xtest is None:
        Xtest = Xtrain
    else:
        Xtest = Xtest

    if method == 'platt':
        model = LogisticRegression()
        model.fit(prob_train[:, np.newaxis], Ytrain)  # LR needs X to be 2-dimensional
        p_calibrated = model.predict_proba(probs)[:, 1]

    elif method == 'isotonic':
        model = IsotonicRegression(out_of_bounds='clip')
        model.fit(prob_train, Ytrain)  # LR needs X to be 2-dimensional
        p_calibrated = model.transform(probs.flatten())

    elif method == 'temperature_scaling':
        model = TemperatureScaling()
        model.fit(prob_train, Ytrain)
        p_calibrated = model.transform(probs)

    elif method == 'beta':
        model = BetaCalibration()
        model.fit(prob_train, Ytrain)
        p_calibrated = model.transform(probs)

    elif method == 'HB':
        model = HistogramBinning()
        model.fit(prob_train, Ytrain)
        p_calibrated = model.transform(probs)

    elif method == 'BBQ':
        model = BBQ()
        model.fit(prob_train, Ytrain)
        p_calibrated = model.transform(probs)

    elif method == 'ENIR':
        model = ENIR()
        model.fit(prob_train, Ytrain)
        p_calibrated = model.transform(probs)

    elif method == 'KRR':
        model = KRR_calibration()
        model.fit(Xtrain, prob_train, Ytrain, **kwargs)
        p_calibrated = model.predict(Xtest, probs, mode='prob')

    elif method == 'EWF':
        model = EWF_calibration()
        model.fit(Xtrain, prob_train, Ytrain, **kwargs)
        p_calibrated = model.predict(Xtest, probs, mode='prob')

    else:
        raise ValueError("Method %s is not defined."%method)

    p_calibrated[np.isnan(p_calibrated)] = 0

    # normalize the large numbers and small numbers to one and zero
    p_calibrated[p_calibrated > 1.0] = 1.0
    p_calibrated[p_calibrated < 0.0] = 0.0

    return p_calibrated


def _counts_per_bin(prob, n_bins):
    '''
    Taken from https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/calibration.py#L513
    '''

    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(prob, bins) - 1
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    return bin_total[nonzero]


def calibration_error(y_true, y_prob, n_bins=10, method='ECE'):
    """
    Compute calibration error given true targets and predicted probabilities.
     Calibration curves may also be referred to as reliability diagrams.

    Parameters
    ----------
    y_true : array, shape (n_samples,)
        True targets.

    y_prob : array, shape (n_samples,)
        Probabilities of the positive class.

    method : string, default='ECE', {'ECE', 'MCE', 'BS'}
        Which method to be used to compute calibration error.

    n_bins : int
        Number of bins. Note that a bigger number requires more data.

    Returns
    -------
    score : float
        calibration error score

    References
    ----------
    Alexandru Niculescu-Mizil and Rich Caruana (2005) Predicting Good
    Probabilities With Supervised Learning, in Proceedings of the 22nd
    International Conference on Machine Learning (ICML).
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    # compute fraction of positive cases per y_prob in a bin. See scikit-learn documentation for details
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, normalize=False,
                                                                    n_bins=n_bins, strategy='uniform')

    if method == 'ECE':
        hist_count = _counts_per_bin(y_prob, n_bins)
        return np.sum(hist_count * np.abs(fraction_of_positives - mean_predicted_value)) / np.sum(hist_count)
    elif method == 'MCE':
        return np.max(np.abs(fraction_of_positives - mean_predicted_value))
    elif method == 'BS':
        return brier_score_loss(y_true, y_prob, pos_label=1)
    else:
        raise ValueError("Method %s is not defined."%method)


class KRR_calibration:

    def __init__(self):
        self.model = 'KRR'

    def fit(self, X, p, Y, kernel_function='rbf', **kwargs):

        from sklearn.kernel_ridge import KernelRidge

        check_attributes(X, Y)

        self.model = KernelRidge(kernel=kernel_function, **kwargs)

        observed_bias = Y - p

        self.model.fit(X, observed_bias)

        return self.model

    def predict(self, X, p=None, mode='prob'):

        if mode == 'bias':
            return self.model.predict(X)
        elif mode == 'prob':
            return self.model.predict(X) + p.flatten()
        else:
            raise ValueError("Mode %s is not defined." % mode)


class EWF_calibration:

    def __init__(self):
        self.model = 'KRR'

    def fit(self, X, p, Y, kernel_function='rbf', **kwargs):

        check_attributes(X, Y)

        self.X = X
        self.bias = Y - p

        self.kernel_function = kernel_function
        self.kwargs = kwargs

    def predict(self, Xtest, ptest=None, mode='prob'):

        K = pairwise_kernels(self.X, Xtest, metric=self.kernel_function, **self.kwargs)
        bias = np.sum(self.bias.flatten() * K.T, axis=1) / np.sum(K.T, axis=1)

        if mode == 'bias':
            return bias
        elif mode == 'prob':
            return bias + ptest.flatten()
        else:
            raise ValueError("Mode %s is not defined." % mode)


