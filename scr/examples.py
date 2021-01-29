from .uncertainty_estimator import empirical_percentile_estimator
from .uncertainty_estimator import MPCE2_test_estimator, error_witness_function, construct_credible_error_vector
import matplotlib.pylab as plt
import numpy as np
import scipy.stats

import matplotlib as mpl
mpl.rcParams['xtick.direction'], mpl.rcParams['ytick.direction'] = 'in', 'in'
mpl.rcParams['xtick.major.size'], mpl.rcParams['xtick.minor.size'] = 14, 8
mpl.rcParams['xtick.major.width'], mpl.rcParams['xtick.minor.width'] = 1.0, 0.8
mpl.rcParams['xtick.major.pad'], mpl.rcParams['xtick.minor.pad'] = 10, 10
mpl.rcParams['ytick.major.size'], mpl.rcParams['ytick.minor.size'] = 14, 8
mpl.rcParams['ytick.major.width'], mpl.rcParams['ytick.minor.width'] = 1.0, 0.8
mpl.rcParams['xtick.labelsize'], mpl.rcParams['ytick.labelsize'] = 15, 15
default_cmap = plt.cm.coolwarm

props = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)

def p_value_textstr(p_value_global, p_value_local):

    textstr = ''

    if p_value_global <= 0.001: textstr += "p-value [global] < 0.001 \n"
    else: textstr += "p-value [global] = %0.3f \n" % p_value_global

    if p_value_local <= 0.001: textstr += "p-value [local] < 0.001"
    else: textstr += "p-value [local] = %0.3f" % p_value_local

    return textstr


def interpretation_percentile_estimator():

    np.random.seed(1)
    import scipy.stats
    scipy.stats.norm(0, 1).cdf(0)

    base_probability = scipy.stats.norm(0.0, 1.0)
    x = np.linspace(0.0, 1.0, 201)
    percentile_theory = x.copy() #base_probability.cdf(x)

    # draw samples
    # evaluate empirical percentiles per x
    plt.figure(figsize=(10, 5))

    ax = plt.subplot(1, 2, 1)

    plt.plot(percentile_theory, percentile_theory, ':k', lw=1.5, label='Unbias')

    samples = np.random.normal(0.0, 2.0, 4001)
    percentile_empirical = empirical_percentile_estimator(samples, x, base_probability)
    plt.plot(percentile_theory, percentile_empirical, '-', lw=3.0, label='Underestimated')

    samples = np.random.normal(0.0, 0.5, 4001)
    percentile_empirical = empirical_percentile_estimator(samples, x, base_probability)
    plt.plot(percentile_theory, percentile_empirical, '--', lw=3.0, label='Overestimated')

    # sig_err = np.random.uniform(0.5, 1.5, 1001)
    # samples = np.random.normal(0.0, sig_err, 1001)
    # percentile_empirical = empirical_percentile_estimator(samples, x, base_probability)
    # plt.plot(percentile_theory, percentile_empirical, ':', lw=4.0, label='Random')

    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1'])
    plt.ylabel(r'Empirical percentile $\hat{r}$', size=22)
    plt.xlabel(r'Predictive percentile $r$', size=22)
    plt.legend(loc=2, prop={'size':14})

    plt.grid()

    ax = plt.subplot(1, 2, 2)

    plt.plot(percentile_theory, percentile_theory, ':k',  lw=1.5, label='Unbias')

    samples = np.random.normal(0.5, 1.0, 4001)
    percentile_empirical = empirical_percentile_estimator(samples, x, base_probability)
    plt.plot(percentile_theory, percentile_empirical, '-', lw=3.0, label='Biased low')

    samples = np.random.normal(-0.5, 1.0, 4001)
    percentile_empirical = empirical_percentile_estimator(samples, x, base_probability)
    plt.plot(percentile_theory, percentile_empirical, '--', lw=3.0, label='Biased high')

    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1'])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 6*[''])
    plt.xlabel(r'Predictive percentile $r$', size=22)
    plt.legend(loc=2, prop={'size': 14})

    plt.grid()

    plt.subplots_adjust(hspace=0.04, wspace=0.08)

    plt.savefig('./plots/estimator_interpretation.png', bbox_inches='tight')


def biased_mean(size=500, scale=0.25):

    x = np.random.normal(0.0, 1.0, size)
    y = np.random.normal(x, 1.0)
    yErr = np.random.uniform(0.5, 1.0, size)

    yPostCal = y + np.random.normal(0.0, yErr)
    yPostUnCal = y + np.random.normal(x * scale, yErr)

    return x, y, yErr, yPostCal, yPostUnCal


def biased_variance(size=500, scale=0.2):

    x = np.random.normal(0.0, 1.0, size)
    y = np.random.normal(x, 1.0)
    yErr = np.random.uniform(0.8, 1.0, size)
    yErr_ext = x * scale

    yPostCal = y + np.random.normal(0.0, yErr)
    yPostUnCal = y + np.random.normal(0.0, yErr + yErr_ext)

    return x, y, yErr, yPostCal, yPostUnCal


def naive_percentile_estimator_sims():

    np.random.seed(1)
    import scipy.stats
    scipy.stats.norm(0, 1).cdf(0)

    base_probability = scipy.stats.norm(0.0, 1.0)
    x = np.linspace(0.0, 1.0, 201)
    percentile_theory = x.copy()

    _, y, yErr, yPostCal, yPostUnCal = biased_mean(size=500)
    sample_cal = (yPostCal - y) / yErr
    sample_uncal = (yPostUnCal - y) / yErr

    plt.figure(figsize=(10, 5))

    ax = plt.subplot(1, 2, 1)

    plt.plot(percentile_theory, percentile_theory, ':k', lw=2.5, label='reference', zorder=10)

    percentile_empirical = empirical_percentile_estimator(sample_cal, x, base_probability)
    plt.plot(percentile_theory, percentile_empirical, '-', lw=6.0, label='calibrated')

    percentile_empirical = empirical_percentile_estimator(sample_uncal, x, base_probability)
    plt.plot(percentile_theory, percentile_empirical, '--', lw=5.0, label='uncalibrated')

    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1'])
    plt.ylabel(r'Empirical percentile $\hat{r}$', size=21)
    plt.xlabel(r'Predictive percentile $r$', size=21)

    plt.title('Model with mis-calibrated mean', size=16)
    plt.legend(loc=2, prop={'size':16})
    plt.grid()

    _, y, yErr, yPostCal, yPostUnCal = biased_variance(size=500)
    sample_cal = (yPostCal - y) / yErr
    sample_uncal = (yPostUnCal - y) / yErr

    ax = plt.subplot(1, 2, 2)

    plt.plot(percentile_theory, percentile_theory, ':k', lw=2.5, label='reference', zorder=10)

    percentile_empirical = empirical_percentile_estimator(sample_cal, x, base_probability)
    plt.plot(percentile_theory, percentile_empirical, '-', lw=6.0, label='calibrated')

    percentile_empirical = empirical_percentile_estimator(sample_uncal, x, base_probability)
    plt.plot(percentile_theory, percentile_empirical, '--', lw=5.0, label='uncalibrated')

    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1'])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 6*[''])

    plt.title('Model with mis-calibrated variance', size=16)
    plt.xlabel(r'Predictive percentile $r$', size=21)
    plt.legend(loc=2, prop={'size':16})
    plt.grid()

    plt.subplots_adjust(hspace=0.05, wspace=0.08)

    # plt.show()
    plt.savefig('./plots/naive_estimator_biased_sims.png', bbox_inches='tight')


def experiment_1(test_statitics=True, witness_func=True):

    size = 500

    x, y, yErr, yPostCal, yPostUnCal = biased_mean(size=size)

    percentile = 0.5
    ePostCal = construct_credible_error_vector(yPostCal, y, y-200, percentile)[:, np.newaxis]
    ePostUnCal = construct_credible_error_vector(yPostUnCal, y, y-200, percentile)[:, np.newaxis]

    x = x[:, np.newaxis]
    c = 5000.0

    if test_statitics:

        plt.figure(figsize=(9, 7))

        ax = plt.subplot(2, 1, 1)
        test_estimate, null, p_value_global, p_value_local = MPCE2_test_estimator(x, ePostCal, iterations=2000,
                                                                                  kernel_function='rbf', gamma=1.0)
        prob, bins, patches = plt.hist(c * null, range=[-5, 10], bins=40, density=True, color='green',
                                       label='Null distribution')
        plt.plot(c * test_estimate, prob.max() / 25, 'wv', markersize=14, markeredgecolor='k', markeredgewidth=2)

        textstr = p_value_textstr(p_value_global, p_value_local)
        ax.text(0.55, 0.95, textstr, transform=ax.transAxes, fontsize=16, verticalalignment='top', bbox=props)

        plt.title('Calibrated model (top) vs. uncalibrated model (bottom)', size=18)
        plt.ylabel('PDF', size=20)

        ax = plt.subplot(2, 1, 2)
        test_estimate, null, p_value_global, p_value_local = MPCE2_test_estimator(x, ePostUnCal, iterations=2000,
                                                            kernel_function='rbf', gamma= 1.0)
        prob, bins, patches = plt.hist(c * null, bins=40, density=True, color='orange', label='Null distribution')
        plt.plot(c * test_estimate, prob.max() / 25, 'wv', markersize=14, markeredgecolor='k', markeredgewidth=2)

        textstr = p_value_textstr(p_value_global, p_value_local)
        ax.text(0.55, 0.95, textstr, transform=ax.transAxes, fontsize=16, verticalalignment='top', bbox=props)

        plt.ylabel('PDF', size=20)
        plt.xlabel(r'MPCE$^2_{u}$ test statistics', size=20)

        plt.savefig('./plots/example_calibrated_vs_uncalibrated.pdf', bbox_inched='tight')
        plt.show()

    if witness_func:

        grid = np.linspace(-1.5, 1.5, 101)[:, np.newaxis]

        plt.figure(figsize=(8, 7))

        # WITNESS FUNCTION
        wef_cal = error_witness_function(x, ePostCal, grid, kernel_function='rbf', gamma=1.0/1.0)
        wef_uncal = error_witness_function(x, ePostUnCal, grid, kernel_function='rbf', gamma=1.0/1.0)

        plt.plot(grid.flatten(), 0.0 * grid.flatten(), '--', lw=2.0, color='k', label='reference')
        # plt.plot(grid, 150*wef_cal, ':', color='green', lw=4.0, label='calibrated')
        plt.plot(grid, 150*wef_uncal, '-', color='orange', lw=4.0, label='uncalibrated model')

        plt.legend(loc=3, prop={'size':18})
        plt.xlim([-1.0 , 1.0])
        plt.ylim([-8, 8])
        plt.ylabel(r'Error Witness Function $(\alpha=0.5)$', size=22)
        plt.xlabel(r'$x$', size=27)
        plt.yticks([-8, -4, 0, 4, 8])
        plt.xticks([-1, -0.5, 0, 0.5, 1], ['-1', '-0.5', '0', '0.5', '1'])

        plt.grid()
        plt.savefig('./plots/experiment_1_witness.pdf', bbox_inched='tight')

        # plt.show()


def experiment_2(test_statitics=True, witness_func=True):


    size = 500
    x = np.random.normal(0.0, 1.0, size)
    y = np.random.normal(x, 1.0)
    yErr = np.random.uniform(0.6, 1.0, size)
    yErr_ext = np.sqrt(yErr**2 + 0.3 * x)

    yPostCal = y + np.random.normal(0.0, yErr)
    yPostUnCal = y + np.random.normal(0.0, yErr_ext)

    percentile = 0.5
    ePostCal = construct_credible_error_vector(yPostCal, y, y-200, percentile)[:, np.newaxis]
    ePostUnCal = construct_credible_error_vector(yPostUnCal, y, y-200, percentile)[:, np.newaxis]

    percentile = 0.683
    ePostCalErr = construct_credible_error_vector(yPostCal, y+yErr, y-yErr, percentile)[:, np.newaxis]
    ePostUnCalErr = construct_credible_error_vector(yPostUnCal, y+yErr, y-yErr, percentile)[:, np.newaxis]

    x = x[:, np.newaxis]
    c = 5000.0

    # plt.plot(x, yPostCal, '.')
    # plt.plot(x, yPostUnCal, '.')
    # plt.show()
    # exit()

    if test_statitics:

        plt.figure(figsize=(9, 7))

        ax = plt.subplot(2, 1, 1)
        test_estimate, null, p_value_global, p_value_local = MPCE2_test_estimator(x, ePostCal, iterations=10000,
                                                                                   kernel_function='rbf', gamma=1.0)
        prob, bins, patches = plt.hist(c * null, range=[-5, 10], bins=40, density=True, color='green',
                                       label='Null distribution')
        plt.plot(c * test_estimate, prob.max() / 25, 'wv', markersize=14, markeredgecolor='k', markeredgewidth=2)

        textstr = p_value_textstr(p_value_global, p_value_local)
        ax.text(0.55, 0.95, textstr, transform=ax.transAxes, fontsize=16, verticalalignment='top', bbox=props)

        plt.title('Bias - Calibrated model (top) vs. uncalibrated model (bottom)', size=16)
        plt.ylabel('PDF', size=20)

        ax = plt.subplot(2, 1, 2)
        test_estimate, null, p_value_global, p_value_local = MPCE2_test_estimator(x, ePostUnCal, iterations=10000,
                                                            kernel_function='rbf', gamma= 1.0)
        prob, bins, patches = plt.hist(c * null, bins=40, density=True, color='orange', label='Null distribution')
        plt.plot(c * test_estimate, prob.max() / 25, 'wv', markersize=14, markeredgecolor='k', markeredgewidth=2)

        textstr = p_value_textstr(p_value_global, p_value_local)
        ax.text(0.55, 0.95, textstr, transform=ax.transAxes, fontsize=16, verticalalignment='top', bbox=props)

        plt.ylabel('PDF', size=20)
        plt.xlabel(r'MPCE$^2_{u}$ test statistics', size=20)

        plt.savefig('./plots/example_calibrated_vs_uncalibrated_variance_bias.pdf', bbox_inched='tight')

        plt.figure(figsize=(9, 7))

        ax = plt.subplot(2, 1, 1)
        test_estimate, null, p_value_global, p_value_local = MPCE2_test_estimator(x, ePostCalErr, iterations=10000,
                                                            kernel_function='rbf', gamma=1.0)
        prob, bins, patches = plt.hist(c * null, range=[-5, 10], bins=40, density=True, color='green',
                                       label='Null distribution')

        plt.plot(c * test_estimate, prob.max() / 25, 'wv', markersize=14, markeredgecolor='k', markeredgewidth=2)

        textstr = p_value_textstr(p_value_global, p_value_local)
        ax.text(0.55, 0.95, textstr, transform=ax.transAxes, fontsize=16, verticalalignment='top', bbox=props)

        plt.title('Variance - Calibrated model (top) vs. uncalibrated model (bottom)', size=16)
        plt.ylabel('PDF', size=20)

        ax = plt.subplot(2, 1, 2)
        test_estimate, null, p_value_global, p_value_local = MPCE2_test_estimator(x, ePostUnCalErr, iterations=10000,
                                                            kernel_function='rbf', gamma=1.0)
        prob, bins, patches = plt.hist(c * null, bins=40, density=True, color='orange', label='Null distribution')
        plt.plot(c * test_estimate, prob.max() / 25, 'wv', markersize=14, markeredgecolor='k', markeredgewidth=2)

        textstr = p_value_textstr(p_value_global, p_value_local)
        ax.text(0.55, 0.95, textstr, transform=ax.transAxes, fontsize=16, verticalalignment='top', bbox=props)

        plt.ylabel('PDF', size=20)
        plt.xlabel(r'MPCE$^2_{u}$ test statistics', size=20)

        plt.savefig('./plots/example_calibrated_vs_uncalibrated_variance_variance.pdf', bbox_inched='tight')

    if witness_func:

        grid = np.linspace(-1.5, 1.5, 101)[:, np.newaxis]

        plt.figure(figsize=(8, 7))

        # WITNESS FUNCTION
        wef_cal = error_witness_function(x, ePostCalErr, grid, kernel_function='rbf', gamma=1.0/1.0)
        wef_uncal = error_witness_function(x, ePostUnCalErr, grid, kernel_function='rbf', gamma=1.0/1.0)

        plt.plot(grid.flatten(), 0.0 * grid.flatten(), '--', lw=2.0, color='k', label='reference')
        # plt.plot(grid, 150*wef_cal, ':', color='green', lw=4.0, label='calibrated')
        plt.plot(grid, 150*wef_uncal, '-', color='orange', lw=4.0, label='uncalibrated model')

        plt.legend(loc=1, prop={'size':18})
        plt.xlim([-1, 1])
        plt.ylim([-8, 8])
        plt.ylabel(r'Error Witness Function $(\alpha=0.68)$', size=22)
        plt.xlabel(r'$x$', size=27)
        plt.yticks([-8, -4, 0, 4, 8])
        plt.xticks([-1, -0.5, 0, 0.5, 1], ['-1', '-0.5', '0', '0.5', '1'])

        plt.grid()
        plt.savefig('./plots/experiment_2_witness.pdf', bbox_inched='tight')


def experiment_3():

    from .uncertainty_estimator import MPCE2_estimator
    from sklearn.metrics import pairwise_kernels

    plt.figure(figsize=(9, 5))
    ax = plt.subplot(1, 1, 1)

    size = 2000
    percentile = 0.5
    scale = 0.25

    c = 5000.0

    x, y, yErr, yPostCal, yPostUnCal = biased_mean(size=size, scale=scale)
    ePostUnCalErr = construct_credible_error_vector(yPostUnCal, y+yErr, y-yErr, percentile)[:, np.newaxis]

    grid = np.linspace(-1.5, 1.5, 101)[:, np.newaxis]

    # WITNESS FUNCTION
    wef_uncal = error_witness_function(x, ePostCal, grid, kernel_function='rbf', gamma=1.0 / 1.0)
    wef_cal = error_witness_function(x, ePostUncal, grid, kernel_function='rbf', gamma=1.0 / 1.0)

    plt.plot(grid, grid * 0, '--k', lw=2.0, label='reference')

    plt.plot(grid, wef_cal, '-', color='green', lw=4.0, label='unbiased Data')
    plt.plot(grid, wef_uncal, '-', color='orange', lw=4.0, label='biased Data')
    # plt.plot(grid, grid*0.25, ':', color='red', lw=1.0, label='unbiased Data')

    base_probability = scipy.stats.norm(0.0, 1.0)
    cdf = base_probability.cdf(grid)

    plt.legend(loc=1)
    plt.grid()
    plt.show()


def experiment_4():

    from .uncertainty_estimator import MPCE2_estimator
    from sklearn.metrics import pairwise_kernels

    plt.figure(figsize=(9, 5))
    ax = plt.subplot(1, 1, 1)

    size = 2000
    percentile = 0.68

    c = 5000.0
    """
    config = np.zeros(7)

    for i in range(2000):

        if i % 100 == 0: print(i)

        try:
                scale = 0.01
                x, y, yErr, yPostCal, yPostUnCal = biased_variance(size=size, scale=scale)
                ePostUnCalErr = construct_credible_error_vector(yPostUnCal, y+yErr, y-yErr, percentile)[:, np.newaxis]
                x = x[:, np.newaxis]
                K_xx = pairwise_kernels(x, x, metric='rbf', gamma=1.0)
                test1 = MPCE2_estimator(K_xx, ePostUnCalErr)

                scale = 0.05
                x, y, yErr, yPostCal, yPostUnCal = biased_variance(size=size, scale=scale)
                ePostUnCalErr = construct_credible_error_vector(yPostUnCal, y+yErr, y-yErr, percentile)[:, np.newaxis]
                x = x[:, np.newaxis]
                K_xx = pairwise_kernels(x, x, metric='rbf', gamma=1.0)
                test2 = MPCE2_estimator(K_xx, ePostUnCalErr)

                scale = 0.10
                x, y, yErr, yPostCal, yPostUnCal = biased_variance(size=size, scale=scale)
                ePostUnCalErr = construct_credible_error_vector(yPostUnCal, y+yErr, y-yErr, percentile)[:, np.newaxis]
                x = x[:, np.newaxis]
                K_xx = pairwise_kernels(x, x, metric='rbf', gamma=1.0)
                test3 = MPCE2_estimator(K_xx, ePostUnCalErr)

                if test1 < test2 and test2 < test3:
                    config[0] += 1
                elif test2 < test1 and test1 < test3:
                    config[1] += 1
                elif test1 < test3 and test3 < test2:
                    config[2] += 1
                elif test2 < test3 and test3 < test1:
                    config[3] += 1
                elif test3 < test1 and test1 < test2:
                    config[4] += 1
                elif test3 < test2 and test2 < test1:
                    config[5] += 1
                else:
                    config[6] += 1
        except:
            pass

    print(config / sum(config))
    exit()
    """

    for scale in [0.000000001, 0.05, 0.1, 0.2]:

        test = []
        for i in range(2000):

            if i%500 == 0: print(i)

            try:
                x, y, yErr, yPostCal, yPostUnCal = biased_variance(size=size, scale=scale)
                ePostUnCalErr = construct_credible_error_vector(yPostUnCal, y+yErr, y-yErr, percentile)[:, np.newaxis]
                x = x[:, np.newaxis]
                K_xx = pairwise_kernels(x, x, metric='rbf', gamma=1.0)
                test += [MPCE2_estimator(K_xx, ePostUnCalErr)]
            except: pass

        if scale > 0.001:
            prob, bins, patches = plt.hist(c * np.array(test), range=[-5, 20], histtype='step', lw=4.0,
                                           bins=125, density=True, label='scale = %0.2f'%scale)
        else:
            prob, bins, patches = plt.hist(c * np.array(test), range=[-5, 20], color='grey',
                                           bins=125, density=True, alpha=0.5, label='scale = 0 (null)')

    plt.legend(loc=1, prop={'size':18})
    plt.xlim(-3, 20)
    plt.ylabel('PDF', size=20)
    plt.xlabel(r'MPCE$^2_{u}$ test statistics', size=20)
    plt.legend(loc=1)
    plt.savefig('./plots/example_calibration_distance.pdf', bbox_inches='tight')
    plt.show()


def kernel_example():

    np.random.seed(123853)
    import scipy.stats

    naive_percentile_estimator_sims()

    # experiment_1(test_statitics=True, witness_func=True)
    # experiment_2(test_statitics=True, witness_func=True)
    # experiment_3()
    # experiment_4()
    # experiment_5()
    # real_data_predictive_percentile()

    exit()

    """
    base_probability = scipy.stats.norm(0.0, 1.0)

    xlin = np.linspace(0.0, 1.0, 201)
    percentile_theory = xlin.copy()

    plt.plot(percentile_theory, percentile_theory, ':k', lw=1.5, label='Unbias')

    percentile_empirical = empirical_percentile_estimator((yPost-y)/yErr, xlin, base_probability)
    plt.plot(percentile_theory, percentile_empirical, '-', lw=3.0, label='All')

    mask = x > 0.0
    percentile_empirical = empirical_percentile_estimator((yPost-y)[mask]/yErr[mask], xlin, base_probability)
    plt.plot(percentile_theory, percentile_empirical, '-', lw=3.0, label='x > 0')

    mask = x <= 0.0
    percentile_empirical = empirical_percentile_estimator((yPost-y)[mask]/yErr[mask], xlin, base_probability)
    plt.plot(percentile_theory, percentile_empirical, '-', lw=3.0, label='x < 0')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.grid()
    plt.xlabel(r'Predictive percentile $r$', size=22)
    plt.ylabel(r'Empirical percentile $\hat{r}$', size=22)
    plt.legend(loc=2, prop={'size': 14})
    plt.show()
    """


def experiment_5():

    from .uncertainty_estimator import MPCE2_estimator
    from sklearn.metrics import pairwise_kernels
    import pandas as pd

    df = pd.read_csv('./data/regression/cluster.csv')[::1].reset_index()
    mask = (df.target < 15.0) * (df.target > 14.0)
    df = df[mask].reset_index()

    print(len(df))

    c = 1000.0

    x = np.array(df.target)[:, np.newaxis]
    y = np.array(df.target)
    yMed = np.array(df.pred_means)
    yStd = np.array(df.pred_vars)

    eMed = construct_credible_error_vector(y, yMed, yMed-1000000.0, 0.5)[:, np.newaxis]
    eStd = construct_credible_error_vector(y, yMed+yStd, yMed-yStd, 0.68)[:, np.newaxis]

    """
    plt.figure(figsize=(9, 7))
    
    ax = plt.subplot(2, 1, 1)
    test_estimate, null, p_value_global, p_value_local = MPCE2_test_estimator(x, eMed, iterations=10000, verbose=True,
                                                                              kernel_function='rbf', gamma=10.0)
    prob, bins, patches = plt.hist(c * null, bins=40, density=True, color='green', label='Null distribution')
    plt.plot(c * test_estimate, prob.max() / 25, 'wv', markersize=14, markeredgecolor='k', markeredgewidth=2)

    textstr = p_value_textstr(p_value_global, p_value_local)
    ax.text(0.55, 0.95, textstr, transform=ax.transAxes, fontsize=16, verticalalignment='top', bbox=props)

    plt.title('Cluster sample - median [top], std [bottom]', size=16)
    plt.ylabel('PDF', size=20)

    ax = plt.subplot(2, 1, 2)
    test_estimate, null, p_value_global, p_value_local = MPCE2_test_estimator(x, eStd, iterations=10000, verbose=True,
                                                                             kernel_function='rbf', gamma=10.0)
    prob, bins, patches = plt.hist(c * null, bins=40, density=True, color='green', label='Null distribution')
    plt.plot(c * test_estimate, prob.max() / 25, 'wv', markersize=14, markeredgecolor='k', markeredgewidth=2)

    textstr = p_value_textstr(p_value_global, p_value_local)
    ax.text(0.55, 0.95, textstr, transform=ax.transAxes, fontsize=16, verticalalignment='top', bbox=props)

    plt.ylabel('PDF', size=20)
    plt.xlabel(r'MPCE$^2_{u}$ test statistics', size=20)

    plt.savefig('./plots/Clusters_calibration_bias.pdf', bbox_inched='tight')
    """

    grid = np.linspace(14.0, 15.0, 201)[:, np.newaxis]

    plt.figure(figsize=(9, 7))

    # WITNESS FUNCTION
    wef = error_witness_function(x, eMed, grid, kernel_function='rbf', gamma=100.0)

    plt.plot(grid.flatten(), 0.0 * grid.flatten(), '--', lw=2.0, color='k', label='reference')
    plt.plot(grid, wef, '-', color='green', lw=4.0, label='calibrated')

    # plt.legend(loc=1, prop={'size': 18})
    plt.xlim([14, 15])
    plt.ylim([-0.05, 0.02])
    plt.ylabel(r'Error Witness Function $(\alpha=0.5)$', size=22)
    plt.xlabel(r'$\log(M)$', size=27)
    # plt.yticks([-8, -4, 0, 4, 8])
    # plt.xticks([-1, -0.5, 0, 0.5, 1], ['-1', '-0.5', '0', '0.5', '1'])

    plt.grid()
    plt.savefig('./plots/Cluster_witness.pdf', bbox_inched='tight')


def real_data_predictive_percentile():

    import pandas as pd

    df = pd.read_csv('./data/regression/cluster.csv')[::100].reset_index()
    dfh = pd.read_csv('./data/regression/housing.csv').reset_index()

    mask = (df.target < 15.0) * (df.target > 14.0)
    y = np.array(df.target[mask])
    yMed = np.array(df.pred_means[mask])
    yStd = np.array(df.pred_vars[mask])

    mask = (dfh.target < 40) * (dfh.target > 20)
    yh = np.array(dfh.target[mask])
    yhMed = np.array(dfh.pred_means[mask])
    yhStd = np.array(dfh.pred_vars[mask])

    # plt.plot(yh, yhMed, '.')
    # plt.show()

    base_probability = scipy.stats.norm(0.0, 1.0)
    percentile_theory = np.linspace(0.0, 1.0, 51)

    percentile_empirical = empirical_percentile_estimator((yMed-y)/yStd, percentile_theory, base_probability)
    percentile_empirical_h = empirical_percentile_estimator((yhMed-yh)/yhStd, percentile_theory, base_probability)

    plt.plot(percentile_theory, percentile_theory, ':k', lw=1.5, label='reference')
    plt.plot(percentile_theory, percentile_empirical, '-', lw=3.0, label='Cluster Sample')
    plt.plot(percentile_theory, percentile_empirical_h, '-', lw=3.0, label='Housing Sample')

    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2', '0.4', '0.6', '0.8', '1'])
    plt.xlabel(r'Predictive percentile $r$', size=22)
    plt.ylabel(r'Empirical percentile $\hat{r}$', size=22)
    plt.legend(loc=2, prop={'size': 14})

    plt.grid()
    plt.show()


def experiment_6():

    from .uncertainty_estimator import MPCE2_estimator
    from sklearn.metrics import pairwise_kernels
    import pandas as pd

    df = pd.read_csv('./data/moments_perc_P_lob_ltr_z_LSS_SDSS.csv', sep=r",\s*")
    data = np.load('./data/ltr_lob_lss_mock.npy')
    print(df[['z', 'lam_tr', 'mean', 'std', 'MAD', 'skew', 'ex_kurt', 'median', 'sigma']])
    print(data.T[:2])

