from .uncertainty_estimator import empirical_percentile_estimator
from .uncertainty_estimator import MPCE2_test_estimator, error_witness_function, construct_credible_error_vector
import matplotlib.pylab as plt
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB



import matplotlib as mpl

mpl.rcParams['xtick.direction'], mpl.rcParams['ytick.direction'] = 'in', 'in'
mpl.rcParams['xtick.major.size'], mpl.rcParams['xtick.minor.size'] = 14, 8
mpl.rcParams['xtick.major.width'], mpl.rcParams['xtick.minor.width'] = 1.0, 0.8
mpl.rcParams['xtick.major.pad'], mpl.rcParams['xtick.minor.pad'] = 10, 10
mpl.rcParams['ytick.major.size'], mpl.rcParams['ytick.minor.size'] = 14, 8
mpl.rcParams['ytick.major.width'], mpl.rcParams['ytick.minor.width'] = 1.0, 0.8
mpl.rcParams['xtick.labelsize'], mpl.rcParams['ytick.labelsize'] = 15, 15
default_cmap = plt.cm.coolwarm
import pandas as pd
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)
def _check_directory(my_folder):
    """
    check if `my_folder` directory exists otherwise make it.
    """
    if not os.path.exists(my_folder):
        os.makedirs(my_folder)


def _calssifier():

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB

    # classifier
    return RandomForestClassifier()





def run_simulation(n_sample=15000, sklearn=True, rand=None, grid=False):

    from sklearn import datasets

    if sklearn == True:

        X, y = datasets.make_classification(n_samples=n_sample, n_features=40, n_informative=15,
                                   n_redundant=2, n_repeated=0, n_classes=2, n_clusters_per_class=4,
                                   class_sep=0.2, flip_y=0, weights=[0.4, 0.6], random_state=rand)
        return X, y

    else:

        if rand is not None: np.random.seed(rand)

        X = np.random.normal(0.0, 1.0, [2, n_sample])
        prob_cal = 1.0 / (1.0 + np.exp(X[0]+X[1]))
        loc_mis_prob = 1.0 / (1.0 + np.exp(X[0]))
        glo_mis_prob = 1.0 / (1.0 + np.exp(-0.4+1.3*X[0]))

        y = np.random.binomial(1, prob_cal, n_sample)

        if grid:

            x1 = np.linspace(-2, 2, 5)
            x2 = np.linspace(-2, 2, 1001)

            Xgrid = []
            for j in range(len(x2)):
                for i in range(len(x1)):
                    Xgrid += [[x1[i], x2[j]]]
            Xgrid = np.array(Xgrid).T

            prob_cal_grid = 1.0 / (1.0 + np.exp(Xgrid[0]+Xgrid[1]))
            loc_mis_prob_grid = 1.0 / (1.0 + np.exp(Xgrid[0]))
            glo_mis_prob_grid = 1.0 / (1.0 + np.exp(-0.4+1.3*Xgrid[0]))

            return X.T, y, glo_mis_prob, loc_mis_prob, prob_cal, Xgrid.T, glo_mis_prob_grid, loc_mis_prob_grid, prob_cal_grid

        else:
            return X.T, y, glo_mis_prob, loc_mis_prob, prob_cal,


def example_2():

    from sklearn import datasets
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.calibration import calibration_curve
    from .uncertainty_estimator_prob import MLCE2_test_estimator, calibrate
    from sklearn.metrics import pairwise_distances
    from netcal.metrics import ECE

    X, y = run_simulation(n_sample=15000)

    # datasets.make_classification(n_samples=50000, n_features=20,
    #                                    n_informative=2, n_redundant=2, random_state=15636)

    train_samples = 5000  # Samples used for training the models
    cv_samples = 5000  # Samples used for training the models

    X_train = X[:train_samples]
    X_cv = X[train_samples:train_samples+cv_samples]
    X_test = X[train_samples+cv_samples:]

    y_train = y[:train_samples]
    y_cv = y[train_samples:train_samples+cv_samples]
    y_test = y[train_samples+cv_samples:]

    # Create classifiers
    lr = LogisticRegression()
    gnb = GaussianNB()
    svc = LinearSVC(C=1.0)
    rfc = RandomForestClassifier()

    # Error Metrics
    n_bins = 20
    ece = ECE(n_bins, detection=False)

    # kernel hyperparameter
    gamma = 1.0 / np.median(pairwise_distances(X_test, metric='euclidean')) ** 2

    # #############################################################################
    # Plot calibration plots

    plt.figure(figsize=(7, 9))
    ax1 = plt.subplot2grid((7, 1), (0, 0), rowspan=4)
    ax2 = plt.subplot2grid((7, 1), (4, 0), rowspan=2)
    ax3 = plt.subplot2grid((7, 1), (6, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    clf = rfc
    clf.fit(X_train, y_train)
    prob_test = clf.predict_proba(X_test)[:, 1]
    prob_cv = clf.predict_proba(X_cv)[:, 1]

    for method in ['No calibration', 'platt', 'isotonic']: #, 'temperature_scaling', 'beta', 'HB', 'BBQ', 'ENIR']:

        if method == 'No calibration':
            prob_cal = prob_test.copy()
        else:
            prob_cal = calibrate(y_cv, prob_cv, prob_test=prob_test, method=method)

        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_cal, n_bins=20)
        MLCE2 = MLCE2_test_estimator(X_test, y_test, prob_cal, prob_kernel_wdith=0.05, kernel_function='rbf', gamma=gamma)

        ece_score = ece.measure(prob_cal, y_test, return_map = False, return_num_samples = False)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % (method,))
        # ax2.hist(prob_cal, range=(0, 1), bins=20, histtype="step", lw=2)
        ax2.plot(ece_score, 1, 'v', markersize=14, markeredgewidth=2)
        ax3.plot(MLCE2 * 1000, 1, 'v', markersize=14, markeredgewidth=2)

    ax1.set_ylabel("Fraction of positives", size=18)
    ax1.set_xlabel("Mean predicted value", size=18)
    ax1.set_ylim([0.0, 1.])
    ax1.set_xlim([0.0, 1.])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots (reliability curve)')

    ax2.set_xlabel("ECE", size=18)
    ax2.set_yticks([])

    ax3.set_xlabel(r"ELCE$^2_{u}$", size=18)
    ax3.set_yticks([])
    ax3.set_xlim([0.0, 30.])

    plt.tight_layout()
    plt.show()


def example_3():

    from sklearn.calibration import calibration_curve
    from .uncertainty_estimator_prob import ELCE2_test_estimator, calibrate, calibration_error
    from sklearn.metrics import pairwise_distances

    X, y, mis_prob, prob, prob_exp = run_simulation(n_sample=10000, sklearn=False, rand=5362279)

    # plt.plot(X.T[0], prob_exp, '.')
    # plt.plot(X.T[0], prob, '.')
    # plt.plot(X.T[0], mis_prob, '.')
    # plt.show()
    # exit()

    cv_samples = 5000  # Samples used for training the models

    X_cv = X[:cv_samples]
    X_test = X[cv_samples:]

    y_cv = y[:cv_samples]
    y_test = y[cv_samples:]

    mis_prob_cv = mis_prob[:cv_samples]
    mis_prob_test = mis_prob[cv_samples:]

    prob_cv = prob[:cv_samples]
    prob_test = prob[cv_samples:]

    prob_exp_cv = prob_exp[:cv_samples]
    prob_exp_test = prob_exp[cv_samples:]

    # kernel hyperparameter
    gamma = 1.0 / np.median(pairwise_distances(X_test, metric='euclidean')) ** 2

    # error calibration setup
    n_bins = 20

    # #############################################################################
    # Plot calibration plots

    methods = ['No calibration', 'platt', 'isotonic', 'temperature_scaling', 'BBQ'] # 'temperature_scaling', 'beta', 'HB', 'BBQ', 'ENIR'

    l_methods  = len(methods)

    plt.figure(figsize=(l_methods * 5, 11))

    for i, method in enumerate(methods):

        ax1 = plt.subplot2grid((9, l_methods * 1), (0, i), rowspan=4)
        ax2 = plt.subplot2grid((9, l_methods * 1), (4, i))
        ax3 = plt.subplot2grid((9, l_methods * 1), (5, i))
        ax4 = plt.subplot2grid((9, l_methods * 1), (6, i))
        ax5 = plt.subplot2grid((9, l_methods * 1), (7, i))
        ax6 = plt.subplot2grid((9, l_methods * 1), (8, i))

        ax1.plot([0, 1], [0, 1], "k:", label="reference")

        plot_metrics(X_test, prob_exp_test, y_test, prob_exp_cv, y_cv, method, ax1, ax2, ax3, ax4, ax5, ax6,
                     legend="Calibrated", n_bins=20,
                     prob_kernel_wdith=0.05, gamma=gamma, marker='^', markersize=16, markeredgewidth=2)

        plot_metrics(X_test, prob_test, y_test, prob_cv, y_cv, method, ax1, ax2, ax3, ax4, ax5, ax6,
                     legend='Locally miscalibrated', n_bins=20,
                     prob_kernel_wdith=0.05, gamma=gamma, marker='v', markersize=16, markeredgewidth=2)

        plot_metrics(X_test, mis_prob_test, y_test, mis_prob_cv, y_cv, method, ax1, ax2, ax3, ax4, ax5, ax6,
                     legend='Globaly miscalibrated', n_bins=20,
                     prob_kernel_wdith=0.05, gamma=gamma, marker='*', markersize=16, markeredgewidth=2)

        """
        if method == 'No calibration': prob_cal = prob_exp_test.copy()
        else: prob_cal = calibrate(y_cv, prob_exp_cv, prob_test=prob_exp_test, method=method)

        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_cal, n_bins=20)
        ece_score = calibration_error(y_test, prob_cal, n_bins=n_bins, method='ECE')
        mce_score = calibration_error(y_test, prob_cal, n_bins=n_bins, method='MCE')
        ELCE2 = ELCE2_test_estimator(X_test, y_test, prob_cal, prob_kernel_wdith=0.05, kernel_function='rbf', gamma=gamma)

        ax1.plot(mean_predicted_value, fraction_of_positives, "^-", lw=2.0, label="Calibrated")
        ax2.plot(ece_score, 1, '^', markersize=14, markeredgewidth=2)
        ax3.plot(mce_score, 1, '^', markersize=14, markeredgewidth=2)
        ax4.plot(ELCE2 * 40, 1, '^', markersize=14, markeredgewidth=2)

        if method == 'No calibration': prob_cal = prob_test.copy()
        else: prob_cal = calibrate(y_cv, prob_cv, prob_test=prob_test, method=method)

        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_cal, n_bins=20)
        ece_score = calibration_error(y_test, prob_cal, n_bins=n_bins, method='ECE')
        mce_score = calibration_error(y_test, prob_cal, n_bins=n_bins, method='MCE')
        ELCE2 = ELCE2_test_estimator(X_test, y_test, prob_cal, prob_kernel_wdith=0.05, kernel_function='rbf', gamma=gamma)

        ax1.plot(mean_predicted_value, fraction_of_positives, "v--", lw=2.0, label="Locally miscalibrated")
        ax2.plot(ece_score, 1, 'v', markersize=14, markeredgewidth=2)
        ax3.plot(mce_score, 1, 'v', markersize=14, markeredgewidth=2)
        ax4.plot(ELCE2 * 40, 1, 'v', markersize=14, markeredgewidth=2)

        if method == 'No calibration': prob_cal = mis_prob_test.copy()
        else: prob_cal = calibrate(y_cv, mis_prob_cv, prob_test=mis_prob_test, method=method)

        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_cal, n_bins=20)
        ece_score = calibration_error(y_test, prob_cal, n_bins=n_bins, method='ECE')
        mce_score = calibration_error(y_test, prob_cal, n_bins=n_bins, method='MCE')
        ELCE2 = ELCE2_test_estimator(X_test, y_test, prob_cal, prob_kernel_wdith=0.05, kernel_function='rbf', gamma=gamma)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-.", lw=2.0, label="Globaly + locally miscalibrated")
        ax2.plot(ece_score, 1, '*', markersize=9, markeredgewidth=2)
        ax3.plot(mce_score, 1, '*', markersize=9, markeredgewidth=2)
        ax4.plot(ELCE2 * 40, 1, '*', markersize=9, markeredgewidth=2)
        """

        ax1.set_xlabel("Mean predicted value", size=18)
        ax1.set_ylim([0.0, 1.])
        ax1.set_xlim([0.0, 1.])
        ax1.set_title(method, size=20)
        ax1.grid()

        ax2.set_yticks([])
        ax2.set_xlim([0.0, 0.25])
        ax3.set_yticks([])
        ax3.set_xlim([0, 0.102])
        ax4.set_yticks([])
        ax4.set_xlim([0, 0.4])
        ax5.set_yticks([])
        ax5.set_xlim([-0.02, 0.25])
        ax6.set_yticks([])
        ax6.set_xlim([-0.01, 0.5])

        if i == 0:
            ax1.set_ylabel("Fraction of positives", size=18)
            ax1.legend(loc="upper left", prop={'size':13})
            ax2.set_ylabel("BS", size=18)
            ax3.set_ylabel("ECE", size=18)
            ax4.set_ylabel("MCE", size=18)
            ax5.set_ylabel(r"ELCE$^2_{u}$", size=18)
            ax6.set_ylabel(r"p-val", size=18)

    plt.tight_layout()
    plt.savefig('./plots/simulation_3.png', bbox_inches='tight')


def example_3_table():

    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.calibration import calibration_curve
    from .uncertainty_estimator_prob import ELCE2_test_estimator, calibrate, calibration_error
    from sklearn.metrics import pairwise_distances
    import pandas as pd

    n_sims = 1000
    cv_samples = 5000  # Samples used for training the models
    tot_samples = 10000

    # error calibration setup
    n_bins = 20

    # kernel hyperparameter
    gamma = 0.35

    for i_func in range(3):

        methods = {'no_calibration': [], 'platt': [], 'isotonic': [],
                   'BBQ': [], 'temperature_scaling': [], 'error_type':[]} # 'beta': [], 'HB': [], 'BBQ': [], 'ENIR': [], }

        for isim in range(n_sims):

            if isim%20 == 0: print('Simulation # (f_%i) : %i'%(i_func, isim))

            X, y, prob_f2, prob_f1, prob_f = run_simulation(n_sample=tot_samples, sklearn=False)
            f = [prob_f, prob_f1, prob_f2]

            X_cv = X[:cv_samples]
            X_test = X[cv_samples:]

            y_cv = y[:cv_samples]
            y_test = y[cv_samples:]

            prob_cv = f[i_func][:cv_samples]
            prob_test = f[i_func][cv_samples:]

            for i, method in enumerate(list(methods.keys())[:-1]):

                if method == 'no_calibration': prob_cal = prob_test.copy()
                else: prob_cal = calibrate(y_cv, prob_cv, prob_test=prob_test, method=method)

                mask = prob_cal <= 1.0
                mask *= prob_cal >= 0.0

                methods[method] += [calibration_error(y_test[mask], prob_cal[mask], n_bins=n_bins, method='BS')]
                methods[method] += [calibration_error(y_test[mask], prob_cal[mask], n_bins=n_bins, method='ECE')]
                methods[method] += [calibration_error(y_test[mask], prob_cal[mask], n_bins=n_bins, method='MCE')]
                methods[method] += [ELCE2_test_estimator(X_test, y_test, prob_cal, verbose=False,
                                                         prob_kernel_wdith=0.05, kernel_function='rbf', gamma=gamma)*40]

            methods['error_type'] += ['BS']
            methods['error_type'] += ['ECE']
            methods['error_type'] += ['MCE']
            methods['error_type'] += ['MLCE']

        pd.DataFrame(methods).to_csv('./data/f_%i_sims_3_table_data.csv'%i_func, index=False)


def real_data_example():

    from sklearn.metrics import pairwise_distances

    np.random.seed(12324)

    # load data and train wdbc
    X_test_sg, y_test_sg, X_cv_sg, y_cv_sg, prob_test_sg, prob_cv_sg = load_sg()
    X_test_sim, y_test_sim, X_cv_sim, y_cv_sim, prob_test_sim, prob_cv_sim = load_sims()
    X_test_wdbc, y_test_wdbc, X_cv_wdbc, y_cv_wdbc, prob_test_wdbc, prob_cv_wdbc = load_wdbc()
    X_test_SAheart, y_test_SAheart, X_cv_SAheart, y_cv_SAheart, prob_test_SAheart, prob_cv_SAheart = load_SAheart()
    X_test_DR, y_test_DR, X_cv_DR, y_cv_DR, prob_test_DR, prob_cv_DR = load_DR()
    X_test_HA, y_test_HA, X_cv_HA, y_cv_HA, prob_test_HA, prob_cv_HA = load_heart_attack()
    X_test_Pima, y_test_Pima, X_cv_Pima, y_cv_Pima, prob_test_Pima, prob_cv_Pima = load_Pima()

    # kernel hyperparameter
    gamma_sg = 1.0 / np.median(pairwise_distances(X_test_sg, metric='euclidean')) ** 2
    gamma_sim = 1.0 / np.median(pairwise_distances(X_test_sim, metric='euclidean')) ** 2
    gamma_wdbc = 1.0 / np.median(pairwise_distances(X_test_wdbc, metric='euclidean')) ** 2
    gamma_SAheart = 1.0 / np.median(pairwise_distances(X_test_SAheart, metric='euclidean')) ** 2
    gamma_DR = 1.0 / np.median(pairwise_distances(X_test_DR, metric='euclidean')) ** 2
    gamma_HA = 1.0 / np.median(pairwise_distances(X_test_HA, metric='euclidean')) ** 2
    gamma_Pima = 1.0 / np.median(pairwise_distances(X_test_Pima, metric='euclidean')) ** 2

    # error calibration setup
    n_bins = 8

    # #############################################################################
    # Plot calibration plots

    methods = ['No calibration', 'platt', 'isotonic', 'temperature_scaling', 'BBQ']

    l_methods = len(methods)

    plt.figure(figsize=(l_methods * 5, 12))

    for i, method in enumerate(methods):

        ax1 = plt.subplot2grid((8, l_methods * 1), (0, i), rowspan=4)
        ax2 = plt.subplot2grid((8, l_methods * 1), (4, i))
        ax3 = plt.subplot2grid((8, l_methods * 1), (5, i))
        ax4 = plt.subplot2grid((8, l_methods * 1), (6, i))
        ax5 = plt.subplot2grid((8, l_methods * 1), (7, i))

        ax1.plot([0, 1], [0, 1], "k:", label="reference")

        plot_metrics(X_test_SAheart, prob_test_SAheart, y_test_SAheart, prob_cv_SAheart, y_cv_SAheart, method, ax1, ax2, ax3, ax4, ax5,
                     legend='Heart Disease', n_bins=n_bins,
                     prob_kernel_wdith=0.2, gamma=gamma_SAheart, marker='^', markersize=14, markeredgewidth=2)
        plot_metrics(X_test_DR, prob_test_DR, y_test_DR, prob_cv_DR, y_cv_DR, method, ax1, ax2, ax3, ax4, ax5,
                     legend='Diabetic Retinopathy', n_bins=n_bins,
                     prob_kernel_wdith=0.2, gamma=gamma_DR, marker='v', markersize=14, markeredgewidth=2)
        plot_metrics(X_test_HA, prob_test_HA, y_test_HA, prob_cv_HA, y_cv_HA, method, ax1, ax2, ax3, ax4, ax5,
                     legend='Heart Attack', n_bins=n_bins,
                     prob_kernel_wdith=0.2, gamma=gamma_HA, marker='*', markersize=14, markeredgewidth=2)
        plot_metrics(X_test_Pima, prob_test_Pima, y_test_Pima, prob_cv_Pima, y_cv_Pima, method, ax1, ax2, ax3, ax4, ax5,
                     legend='Pima Diabetes', n_bins=n_bins,
                     prob_kernel_wdith=0.2, gamma=gamma_Pima, marker='o', markersize=10, markeredgewidth=2)
        plot_metrics(X_test_wdbc, prob_test_wdbc, y_test_wdbc, prob_cv_wdbc, y_cv_wdbc, method, ax1, ax2, ax3, ax4, ax5,
                     legend='Breast Cancer', n_bins=n_bins,
                     prob_kernel_wdith=0.2, gamma=gamma_wdbc, marker='>', markersize=14, markeredgewidth=2)

        ax1.set_xlabel("Mean predicted value", size=18)
        ax1.set_ylim([0.0, 1.])
        ax1.set_xlim([0.0, 1.])
        ax1.set_title(method, size=20)
        ax1.grid()

        ax2.set_yticks([])
        ax2.set_xlim([0.0, 0.3])
        ax3.set_yticks([])
        ax3.set_xlim([-0.01, 0.2])
        ax4.set_yticks([])
        ax4.set_xlim([-0.01, 0.6])
        ax5.set_yticks([])
        ax5.set_xlim([-0.02, 0.15])

        if i == 0:
            ax1.set_ylabel("Fraction of positives", size=18)
            ax1.legend(loc="upper left", prop={'size':10})
            ax2.set_ylabel("BS", size=18)
            ax3.set_ylabel("ECE", size=18)
            ax4.set_ylabel("MCE", size=18)
            ax5.set_ylabel(r"ELCE$^2_{u}$", size=18)

    plt.tight_layout()
    plt.savefig('./plots/RF_.png', bbox_inches='tight')
    plt.show()


def real_data_example2():

    from sklearn.metrics import pairwise_distances

    np.random.seed(12524)

    # load data and train wdbc
    X_test_sg, y_test_sg, X_cv_sg, y_cv_sg, prob_test_sg, prob_cv_sg = load_sg()
    X_test_sim, y_test_sim, X_cv_sim, y_cv_sim, prob_test_sim, prob_cv_sim = load_sims()
    X_test_wdbc, y_test_wdbc, X_cv_wdbc, y_cv_wdbc, prob_test_wdbc, prob_cv_wdbc = load_wdbc()
    X_test_SAheart, y_test_SAheart, X_cv_SAheart, y_cv_SAheart, prob_test_SAheart, prob_cv_SAheart = load_SAheart()
    X_test_DR, y_test_DR, X_cv_DR, y_cv_DR, prob_test_DR, prob_cv_DR = load_DR()
    X_test_HA, y_test_HA, X_cv_HA, y_cv_HA, prob_test_HA, prob_cv_HA = load_heart_attack()
    X_test_Pima, y_test_Pima, X_cv_Pima, y_cv_Pima, prob_test_Pima, prob_cv_Pima = load_Pima()

    # kernel hyperparameter
    gamma_sg = 1.0 / np.median(pairwise_distances(X_test_sg, metric='euclidean')) ** 2
    gamma_sim = 1.0 / np.median(pairwise_distances(X_test_sim, metric='euclidean')) ** 2
    gamma_wdbc = 1.0 / np.median(pairwise_distances(X_test_wdbc, metric='euclidean')) ** 2
    gamma_SAheart = 1.0 / np.median(pairwise_distances(X_test_SAheart, metric='euclidean')) ** 2
    gamma_DR = 1.0 / np.median(pairwise_distances(X_test_DR, metric='euclidean')) ** 2
    gamma_HA = 1.0 / np.median(pairwise_distances(X_test_HA, metric='euclidean')) ** 2
    gamma_Pima = 1.0 / np.median(pairwise_distances(X_test_Pima, metric='euclidean')) ** 2

    # error calibration setup
    n_bins = 8

    # #############################################################################
    # Plot calibration plots

    methods = ['No calibration', 'platt', 'isotonic', 'temperature_scaling', 'BBQ']

    l_methods = len(methods)

    plt.figure(figsize=(l_methods * 5, 12))

    for i, method in enumerate(methods):

        ax1 = plt.subplot2grid((9, l_methods * 1), (0, i), rowspan=4)
        ax2 = plt.subplot2grid((9, l_methods * 1), (4, i))
        ax3 = plt.subplot2grid((9, l_methods * 1), (5, i))
        ax4 = plt.subplot2grid((9, l_methods * 1), (6, i))
        ax5 = plt.subplot2grid((9, l_methods * 1), (7, i))
        ax6 = plt.subplot2grid((9, l_methods * 1), (8, i))

        ax1.plot([0, 1], [0, 1], "k:", label="reference")

        plot_metrics(X_test_sg, prob_test_sg, y_test_sg, prob_cv_sg, y_cv_sg, method, ax1, ax2,ax3, ax4, ax5, ax6,
                     legend='Star-Galaxy', n_bins=n_bins, prob_kernel_wdith=0.2, gamma=gamma_sg, marker='^', markersize=20, markeredgewidth=2)
        # plot_metrics(X_test_SAheart, prob_test_SAheart, y_test_SAheart, prob_cv_SAheart, y_cv_SAheart, method, ax1, ax2, ax3, ax4, ax5, ax6,
        #             legend='Heart Disease', n_bins=n_bins, prob_kernel_wdith=0.2, gamma=gamma_SAheart, marker='^', markersize=14, markeredgewidth=2)
        # plot_metrics(X_test_DR, prob_test_DR, y_test_DR, prob_cv_DR, y_cv_DR, method, ax1, ax2, ax3, ax4, ax5,
        #             legend='Diabetic Retinopathy', n_bins=n_bins, prob_kernel_wdith=0.2, gamma=gamma_DR, marker='v', markersize=14, markeredgewidth=2)
        plot_metrics(X_test_HA, prob_test_HA, y_test_HA, prob_cv_HA, y_cv_HA, method, ax1, ax2, ax3, ax4, ax5, ax6, legend='Heart Attack', n_bins=n_bins,
                     prob_kernel_wdith=0.2, gamma=gamma_HA, marker='o', markersize=15, markeredgewidth=2)
        # plot_metrics(X_test_Pima, prob_test_Pima, y_test_Pima, prob_cv_Pima, y_cv_Pima, method, ax1, ax2, ax3, ax4, ax5, ax6,
        #               legend='Pima Diabetes', n_bins=n_bins, prob_kernel_wdith=0.2, gamma=gamma_Pima, marker='s', markersize=15, markeredgewidth=2)
        plot_metrics(X_test_wdbc, prob_test_wdbc, y_test_wdbc, prob_cv_wdbc, y_cv_wdbc, method, ax1, ax2, ax3, ax4, ax5, ax6,
                     legend='Breast Cancer', n_bins=n_bins, prob_kernel_wdith=0.2, gamma=gamma_wdbc, marker='*', markersize=20, markeredgewidth=2)

        ax1.set_xlabel("Mean predicted value", size=18)
        ax1.set_ylim([0.0, 1.])
        ax1.set_xlim([0.0, 1.])
        ax1.set_title(method, size=20)
        ax1.grid()

        ax2.set_yticks([])
        ax2.set_xlim([0.0, 0.3])
        ax3.set_yticks([])
        ax3.set_xlim([-0.01, 0.2])
        ax4.set_yticks([])
        ax4.set_xlim([-0.01, 0.6])
        ax5.set_yticks([])
        ax5.set_xlim([-0.02, 0.25])
        ax6.set_yticks([])
        ax6.set_xlim([-0.01, 0.5])

        if i == 0:
            ax1.set_ylabel("Fraction of positives", size=18)
            ax1.legend(loc="upper left", prop={'size':17})
            ax2.set_ylabel("BS", size=18)
            ax3.set_ylabel("ECE", size=18)
            ax4.set_ylabel("MCE", size=18)
            ax5.set_ylabel(r"ELCE$^2_{u}$", size=18)
            ax6.set_ylabel(r"p-val", size=18)

    plt.tight_layout()
    plt.savefig('./plots/RF_.png', bbox_inches='tight')
    # plt.show()


def real_data_example_table():

    def print_test_score(score, i):
        print(r' & $%0.3f \pm %0.3f$'%(np.mean(score[i]), np.std(score[i])), end = '')

    from sklearn.metrics import pairwise_distances

    np.random.seed(12324)

    # error calibration setup
    n_bins = 8

    n_realizations = 100
    _ = None

    # #############################################################################
    # Plot calibration plots
    methods = ['No calibration', 'platt', 'isotonic', 'temperature_scaling', 'BBQ']
    l_methods = len(methods)

    # load_data = [load_SAheart(), load_DR(), load_heart_attack(), load_Pima(), load_wdbc()]
    # data_label = ['Heart Disease', 'Diabetic Retinopathy', 'Heart Attack', 'Pima Diabetes', 'Breast Cancer']
    data_label = ['Star-Galaxy', 'Heart Attack', 'Breast Cancer']

    # load data and train wdbc
    for ilabel in data_label:

        BS_test_scores = np.zeros([l_methods, n_realizations])
        ece_test_scores = np.zeros([l_methods, n_realizations])
        mce_test_scores = np.zeros([l_methods, n_realizations])
        elce_test_scores = np.zeros([l_methods, n_realizations])

        for irealization in range(n_realizations):

            if ilabel == 'Star-Galaxy': X_test, y_test, X_cv, y_cv, prob_test, prob_cv = load_sg()
            if ilabel == 'Heart Disease': X_test, y_test, X_cv, y_cv, prob_test, prob_cv = load_SAheart()
            if ilabel == 'Diabetic Retinopathy': X_test, y_test, X_cv, y_cv, prob_test, prob_cv = load_DR()
            if ilabel == 'Heart Attack': X_test, y_test, X_cv, y_cv, prob_test, prob_cv = load_heart_attack()
            if ilabel == 'Pima Diabetes': X_test, y_test, X_cv, y_cv, prob_test, prob_cv = load_Pima()
            if ilabel == 'Breast Cancer': X_test, y_test, X_cv, y_cv, prob_test, prob_cv = load_wdbc()

            # kernel hyperparameter
            gamma = 1.0 / np.median(pairwise_distances(X_test, metric='euclidean')) ** 2

            for i, method in enumerate(methods):

                BS_test_scores[i, irealization], ece_test_scores[i, irealization],\
                mce_test_scores[i, irealization], elce_test_scores[i, irealization] =\
                    plot_metrics(X_test, prob_test, y_test, prob_cv, y_cv, method, _, _, _, _, _, _,
                                 n_bins=n_bins, prob_kernel_wdith=0.2, gamma=gamma, data=True)

        print(ilabel, end='')
        print(r' & BS ', end='')
        for i, method in enumerate(methods): print_test_score(BS_test_scores, i)
        print(r' \\ \cline{2-7} ')
        print(r' & ECE ', end='')
        for i, method in enumerate(methods): print_test_score(ece_test_scores, i)
        print(r' \\ \cline{2-7} ')
        print(r' & MCE ', end='')
        for i, method in enumerate(methods): print_test_score(mce_test_scores, i)
        print(r' \\ \cline{2-7} ')
        print(r' & ELCE ', end='')
        for i, method in enumerate(methods): print_test_score(elce_test_scores, i)
        print(r' \\ \hline ')

        print()


def plot_metrics(X_test, prob_test, y_test, X_cv, prob_cv, y_cv, method, ax1, ax2, ax3, ax4, ax5, ax6, legend=None, n_bins=20,
                 prob_kernel_wdith=0.05, gamma=0.5, marker='^', markersize=14, markeredgewidth=2, data=False, **kwargs):

    import time
    from sklearn.calibration import calibration_curve
    from .uncertainty_estimator_prob import ELCE2_test_estimator, calibrate, calibration_error

    if method == 'No calibration': prob_cal = prob_test.copy()
    elif method == 'EWF': prob_cal = calibrate(X_cv, prob_cv, y_cv, Xtest=X_test, prob_test=prob_test, method=method, gamma=gamma)
    else: prob_cal = calibrate(X_cv, prob_cv, y_cv, Xtest=X_test, prob_test=prob_test, method=method, **kwargs)

    print(y_test.shape, prob_cal.shape)

    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_cal, n_bins=n_bins)
    BS_score = calibration_error(y_test, prob_cal, n_bins=n_bins, method='BS')
    ece_score = calibration_error(y_test, prob_cal, n_bins=n_bins, method='ECE')
    mce_score = calibration_error(y_test, prob_cal, n_bins=n_bins, method='MCE')
    if data:
        ELCE2 = ELCE2_test_estimator(X_test, y_test, prob_cal, prob_kernel_wdith=prob_kernel_wdith,
                                                kernel_function='rbf', gamma=gamma, iterations=None, verbose=False)
    else:
        ELCE2, _, pvalue = ELCE2_test_estimator(X_test[::], y_test[::], prob_cal[::], prob_kernel_wdith=prob_kernel_wdith,
                                                kernel_function='rbf', gamma=gamma, iterations=1000, verbose=True)

    if data: return BS_score, ece_score, mce_score, ELCE2*100

    if pvalue > 0.49: pvalue = 0.49
    if ELCE2 < 0.0: ELCE2 = -0.00005

    ax1.plot(mean_predicted_value, fraction_of_positives, "%s-"%marker, lw=2.0, label=legend)
    ax2.plot(BS_score, 1, marker, markersize=markersize, markeredgewidth=markeredgewidth)
    ax3.plot(ece_score, 1, marker, markersize=markersize, markeredgewidth=markeredgewidth)
    ax4.plot(mce_score, 1, marker, markersize=markersize, markeredgewidth=markeredgewidth)
    ax5.plot(ELCE2*100, 1, marker, markersize=markersize, markeredgewidth=markeredgewidth)
    ax6.plot(pvalue, 1, marker, markersize=markersize, markeredgewidth=markeredgewidth)

    print(legend + ' ' + method + ' BS : %0.3f '%BS_score + ' ECE : %0.3f '%ece_score + ' MCE : %0.3f '%mce_score + ' ELCE2 : %0.3f'%(ELCE2*40))
    time.sleep(1)


def load_sg():
    """

    """
    import pandas as pd
    # 'redshift', 'u', 'g', 'r', 'i', 'z', 'class'
    df = pd.read_csv('./data/Skyserver_SQL2.csv')[['redshift', 'u', 'g', 'r', 'i', 'z', 'class']]
    df = df[df['class'] != 'QSO'].dropna()[1::2]

    df['class'] = df['class'].map({'GALAXY':1, 'STAR':0})

    #print(len(df))
    #exit()

    # classifier
    clf = _calssifier()

    train, validate, test = np.split(df.sample(frac=1), [int(.33 * len(df)), int(.66 * len(df))])

    X_train = np.array(train[train.columns[1:-1]])
    X_cv = np.array(validate[validate.columns[1:-1]])
    X_test = np.array(test[test.columns[1:-1]])

    X_train_redshift = np.array(train[train.columns[0]])[:, np.newaxis]
    X_cv_redshift = np.array(validate[validate.columns[0]])[:, np.newaxis]
    X_test_redshift = np.array(test[test.columns[0]])[:, np.newaxis]

    y_train = np.array(train[train.columns[-1]])
    y_cv = np.array(validate[validate.columns[-1]])
    y_test = np.array(test[test.columns[-1]])

    # train the model
    clf.fit(X_train, y_train)
    prob_test = clf.predict_proba(X_test)[:, 1]
    prob_cv = clf.predict_proba(X_cv)[:, 1]

    return X_test_redshift, y_test, X_cv_redshift, y_cv, prob_test, prob_cv


def load_wdbc():

    import pandas as pd
    df = pd.read_csv('./data/Irvine_database/wdbc.data', header=None).drop(columns=[0])

    df[1] = df[1].map({'M':1, 'B':0})

    # classifier
    clf = _calssifier()

    train, validate, test = np.split(df.sample(frac=1), [int(.33 * len(df)), int(.66 * len(df))])

    X_train = np.array(train[train.columns[1:]])
    X_cv = np.array(validate[validate.columns[1:]])
    X_test = np.array(test[test.columns[1:]])

    y_train = np.array(train[train.columns[0]])
    y_cv = np.array(validate[validate.columns[0]])
    y_test = np.array(test[test.columns[0]])

    # train the model
    clf.fit(X_train, y_train)
    prob_test = clf.predict_proba(X_test)[:, 1]
    prob_cv = clf.predict_proba(X_cv)[:, 1]

    return X_test, y_test, X_cv, y_cv, prob_test, prob_cv


def load_SAheart():
    """
    A retrospective sample of males in a heart-disease high-risk region
    of the Western Cape, South Africa. There are roughly two controls per
    case of CHD. Many of the CHD positive men have undergone blood
    pressure reduction treatment and other programs to reduce their risk
    factors after their CHD event. In some cases the measurements were
    made after these treatments. These data are taken from a larger
    dataset, described in  Rousseauw et al, 1983, South African Medical
    Journal. 

    sbp		systolic blood pressure
    tobacco		cumulative tobacco (kg)
    ldl		low densiity lipoprotein cholesterol
    adiposity
    famhist		family history of heart disease (Present, Absent)
    typea		type-A behavior
    obesity
    alcohol		current alcohol consumption
    age		age at onset
    chd		response, coronary heart disease
    """

    import pandas as pd

    df = pd.read_csv('http://www-stat.stanford.edu/~tibs/ElemStatLearn/datasets/SAheart.data').drop(columns=['row.names'])
    df['famhist'] = df['famhist'].map({'Present':1, 'Absent':0})

    # classifier
    clf = _calssifier()

    train, validate, test = np.split(df.sample(frac=1), [int(.33 * len(df)), int(.66 * len(df))])

    X_train = np.array(train[train.columns[:9]])
    X_cv = np.array(validate[validate.columns[:9]])
    X_test = np.array(test[test.columns[:9]])

    y_train = np.array(train[train.columns[9]])
    y_cv = np.array(validate[validate.columns[9]])
    y_test = np.array(test[test.columns[9]])

    # train the model
    clf.fit(X_train, y_train)
    prob_test = clf.predict_proba(X_test)[:, 1]
    prob_cv = clf.predict_proba(X_cv)[:, 1]

    return X_test, y_test, X_cv, y_cv, prob_test, prob_cv


def load_DR():
    """
    title = {Feedback on a publicly distributed database: the Messidor database},
	volume = {33},
	copyright = {Copyright (c) 2014 Image Analysis \& Stereology},
	issn = {1854-5165},
	shorttitle = {Feedback on a publicly distributed database},
	url = {http://www.ias-iss.org/ojs/IAS/article/view/1155},
	doi = {10.5566/ias.1155},
	abstract = {The Messidor database, which contains hundreds of eye fundus images, has been publicly distributed since 2008. It was created by the Messidor project in order to evaluate automatic lesion segmentation and diabetic retinopathy grading methods. Designing, producing and maintaining such a database entails significant costs. By publicly sharing it, one hopes to bring a valuable resource to the public research community. However, the real interest and benefit of the research community is not easy to quantify. We analyse here the feedback on the Messidor database, after more than 6 years of diffusion. This analysis should apply to other similar research databases.},
	language = {en},
	number = {3},
	journal = {Image Analysis \& Stereology},
	author = {Decencière, Etienne and Zhang, Xiwei and Cazuguel, Guy and Lay, Bruno and Cochener, Béatrice and Trone, Caroline and Gain, Philippe and Ordonez, Richard and Massin, Pascale and Erginay, Ali and Charton, Béatrice and Klein, Jean-Claude},
	month = aug,
	year = {2014},
	keywords = {Diabetic Retinopathy, image database, Image Processing, Messidor},
	pages = {231--234},
    """

    import pandas as pd

    df = pd.read_csv('./data/Irvine_database/messidor_features.csv', header=None).drop(columns=[0, 1, 18])

    # classifier
    clf = _calssifier()

    train, validate, test = np.split(df.sample(frac=1), [int(.33 * len(df)), int(.66 * len(df))])

    X_train = np.array(train[train.columns[:16]])
    X_cv = np.array(validate[validate.columns[:16]])
    X_test = np.array(test[test.columns[:16]])

    y_train = np.array(train[train.columns[16]])
    y_cv = np.array(validate[validate.columns[16]])
    y_test = np.array(test[test.columns[16]])

    # train the model
    clf.fit(X_train, y_train)
    prob_test = clf.predict_proba(X_test)[:, 1]
    prob_cv = clf.predict_proba(X_cv)[:, 1]

    return X_test, y_test, X_cv, y_cv, prob_test, prob_cv


def load_heart_attack():
    """
    source: "https://www.kaggle.com/nareshbhat/health-care-data-set-on-heart-attack-possibility/data?select=heart.csv"
    """
    import pandas as pd

    df = pd.read_csv('./data/Irvine_database/heart_attack.csv')

    # classifier
    clf = _calssifier()

    train, validate, test = np.split(df.sample(frac=1), [int(.33 * len(df)), int(.66 * len(df))])

    X_train = np.array(train[train.columns[1:13]])
    X_cv = np.array(validate[validate.columns[1:13]])
    X_test = np.array(test[test.columns[1:13]])

    X_train_age = np.array(train[train.columns[0]])[:, np.newaxis]
    X_cv_age = np.array(validate[validate.columns[0]])[:, np.newaxis]
    X_test_age = np.array(test[test.columns[0]])[:, np.newaxis]

    y_train = np.array(train[train.columns[13]])
    y_cv = np.array(validate[validate.columns[13]])
    y_test = np.array(test[test.columns[13]])

    # train the model
    clf.fit(X_train, y_train)
    prob_test = clf.predict_proba(X_test)[:, 1]
    prob_cv = clf.predict_proba(X_cv)[:, 1]

    return X_test_age, y_test, X_cv_age, y_cv, prob_test, prob_cv


def load_Pima():
    """
    Context
    This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The
    objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain
    diagnostic measurements included in the dataset. Several constraints were placed on the selection of these
    instances from a larger database. In particular, all patients here are females at least 21 years old
    of Pima Indian heritage.

    Content
    The datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables
    includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.

    Acknowledgements
    Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). Using the ADAP learning
    algorithm to forecast the onset of diabetes mellitus. In Proceedings of the Symposium on Computer Applications and
    Medical Care (pp. 261--265). IEEE Computer Society Press.
    """

    import pandas as pd

    df = pd.read_csv('./data/Irvine_database/Pima_diabetes.csv')

    clf = _calssifier()

    train, validate, test = np.split(df.sample(frac=1), [int(.33 * len(df)), int(.66 * len(df))])

    X_train = np.array(train[train.columns[:7]])
    X_cv = np.array(validate[validate.columns[:7]])
    X_test = np.array(test[test.columns[:7]])

    X_train_age = np.array(train[train.columns[7]])[:, np.newaxis]
    X_cv_age = np.array(validate[validate.columns[7]])[:, np.newaxis]
    X_test_age = np.array(test[test.columns[7]])[:, np.newaxis]

    y_train = np.array(train[train.columns[8]])
    y_cv = np.array(validate[validate.columns[8]])
    y_test = np.array(test[test.columns[8]])

    # train the model
    clf.fit(X_train, y_train)
    prob_test = clf.predict_proba(X_test)[:, 1]
    prob_cv = clf.predict_proba(X_cv)[:, 1]

    return X_test_age, y_test, X_cv_age, y_cv, prob_test, prob_cv


def load_sims_global(n_sample=500):
    X, y, mis_prob, prob, prob_exp = run_simulation(n_sample=n_sample, sklearn=False, rand=5362279)

    cv_samples = int(n_sample/2)  # Samples used for training the models

    X_cv = X[:cv_samples]
    X_test = X[cv_samples:]

    y_cv = y[:cv_samples]
    y_test = y[cv_samples:]

    mis_prob_cv = mis_prob[:cv_samples]
    mis_prob_test = mis_prob[cv_samples:]

    prob_cv = prob[:cv_samples]
    prob_test = prob[cv_samples:]

    prob_exp_cv = prob_exp[:cv_samples]
    prob_exp_test = prob_exp[cv_samples:]

    return X_test, y_test, X_cv, y_cv, mis_prob_test, mis_prob_cv


def load_sims_local(n_sample=500):
    X, y, mis_prob, prob, prob_exp = run_simulation(n_sample=n_sample, sklearn=False, rand=5362279)

    cv_samples = int(n_sample / 2)

    X_cv = X[:cv_samples]
    X_test = X[cv_samples:]

    y_cv = y[:cv_samples]
    y_test = y[cv_samples:]

    mis_prob_cv = mis_prob[:cv_samples]
    mis_prob_test = mis_prob[cv_samples:]

    prob_cv = prob[:cv_samples]
    prob_test = prob[cv_samples:]

    prob_exp_cv = prob_exp[:cv_samples]
    prob_exp_test = prob_exp[cv_samples:]

    return X_test, y_test, X_cv, y_cv, prob_test, prob_cv


def example_3_wf():

    from sklearn.calibration import calibration_curve
    from .uncertainty_estimator_prob import ELCE2_test_estimator, calibrate, calibration_error, error_witness_function, local_bias_estimator
    from sklearn.metrics import pairwise_distances

    X, y, glo_mis_prob, loc_mis_prob, prob_cal, Xgrid, glo_mis_prob_grid, loc_mis_prob_grid, prob_cal_grid = run_simulation(n_sample=20000, sklearn=False, rand=165856, grid=True)

    cv_samples = 10000  # Samples used for training the models

    X_cv = X[:cv_samples]
    X_test = X[cv_samples:]

    y_cv = y[:cv_samples]
    y_test = y[cv_samples:]

    glo_mis_prob_cv = glo_mis_prob[:cv_samples]
    glo_mis_prob_test = glo_mis_prob[cv_samples:]

    loc_mis_prob_cv = loc_mis_prob[:cv_samples]
    loc_mis_prob_test = loc_mis_prob[cv_samples:]

    prob_cal_cv = prob_cal[:cv_samples]
    prob_cal_test = prob_cal[cv_samples:]

    # kernel hyper-parameter
    gamma = 1.0 / np.median(pairwise_distances(X_test, metric='euclidean')) ** 2 * 6.0

    # error calibration setup
    n_bins = 20

    # #############################################################################
    # Plot calibration plots

    methods = ['No calibration', 'platt', 'isotonic', 'temperature_scaling', 'BBQ']
    # l_methods = len(methods)
    # plt.figure(figsize=(l_methods * 5, 19))

    kmax = 5

    def plot_ewf(k, prob_cv, prob_test, prob_test_grid, method, ax):

        ax.plot([-2, 2], [0, 0], "k-", label="reference")

        if method == 'No calibration':
            prob = prob_test.copy()
            prob_grid = prob_test_grid.copy()
        else:
            prob = calibrate(y_cv, prob_cv, prob_test= prob_test, method=method)
            prob_grid = calibrate(y_cv, prob_cv, prob_test= prob_test_grid, method=method)

        ewf = error_witness_function(X_test, y_test, prob, Xgrid, prob_grid, prob_kernel_wdith=0.1, kernel_function='rbf', gamma=gamma)
        # bias1 = local_bias_estimator(X_test, y_test, prob, Xgrid, prob_grid, model='SVR', prob_kernel_wdith=0.1, kernel_function='rbf', C=0.1,  gamma=gamma)
        # bias2 = local_bias_estimator(X_test, y_test, prob, Xgrid, prob_grid, model='SVR',prob_kernel_wdith=0.1, kernel_function='rbf', C=1.0, gamma=gamma)
        # bias3 = local_bias_estimator(X_test, y_test, prob, Xgrid, prob_grid, model='SVR',prob_kernel_wdith=0.1, kernel_function='rbf', C=10.0, gamma=gamma)

        bias1 = local_bias_estimator(X_test, y_test, prob, Xgrid,  model='KRR', kernel_function='rbf', alpha=0.1,  gamma=gamma/40.0)
        bias2 = local_bias_estimator(X_test, y_test, prob, Xgrid, model='KRR', kernel_function='rbf', alpha=1.0, gamma=gamma/40.0)
        bias3 = local_bias_estimator(X_test, y_test, prob, Xgrid,  model='KRR', kernel_function='rbf', alpha=10.0, gamma=gamma/40.0)

        ax.plot(Xgrid[k::kmax].T[1], prob_cal_grid[k::kmax] - prob_grid[k::kmax], 'k:', lw=5.0, label=r'$P_{\rm true} - \widehat{f}$ (true bias)')
        ax.plot(Xgrid[k::kmax].T[1], ewf[k::kmax], lw=4.0, label='Globally+locally mis-calibrated')
        ax.plot(Xgrid[k::kmax].T[1], bias1[k::kmax], lw=4.0, label=r'KRR ($\alpha=0.1$)')
        ax.plot(Xgrid[k::kmax].T[1], bias2[k::kmax], lw=4.0, label=r'KRR ($\alpha=1$)')
        ax.plot(Xgrid[k::kmax].T[1], bias3[k::kmax], lw=4.0, label=r'KRR ($\alpha=10$)')

        return ax

    plt.figure(figsize=(kmax * 5, 5))

    for k in range(kmax):
        ax = plt.subplot2grid((1, kmax), (0, k))
        ax = plot_ewf(k, glo_mis_prob_cv, glo_mis_prob_test, glo_mis_prob_grid, 'No calibration', ax)

        ax.set_ylim([-0.5, 0.5])
        ax.set_xlim([-2.0, 2.0])

        ax.set_title(r'$x_1 = %0.0f$' % Xgrid[k, 0], size=25, color='indianred')

        if k == 0:
            ax.set_ylabel("EWF", size=18)
            plt.yticks([-0.5, -0.25, 0.0, 0.25])
        else:
            plt.yticks([-0.5, -0.25, 0, 0.25], [])

        plt.xticks([-1, 0, 1, 2])
        ax.set_xlabel(r"$x_2$", size=27)

        if k == 1: ax.legend(loc="upper right", prop={'size': 15})

        ax.grid()

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    #plt.savefig('./plots/simulation_3_ewf_Estimator.png', bbox_inches='tight')
    plt.show()


    """
    for k in range(kmax):
        for i, method in enumerate(methods):

            ax = plt.subplot2grid((7, l_methods * 1), (k, i))
            ax = plot_ewf(k, glo_mis_prob_cv, glo_mis_prob_test, glo_mis_prob_grid, method, ax)

            ax.set_ylim([-0.5, 0.5])
            ax.set_xlim([-2.0, 2.0])
            if k == 0:
                ax.set_title(method, size=22)

            if i == 0:
                ax.set_ylabel("EWF", size=18)
                plt.yticks([-0.5, -0.25, 0.0, 0.25])
            else:
                plt.yticks([-0.5, -0.25, 0, 0.25], [])

            if k == kmax-1:
                plt.xticks([-1, 0, 1, 2])
                ax.set_xlabel(r"$x_2$", size=25)
            else:
                plt.xticks([-1, 0, 1, 2], [])

            if k == 0 and i == 0:
                ax.legend(loc="upper right", prop={'size': 13})

            if i == len(methods)-1:
                ax.text(1.0, 0.5, r'$x_1 = %0.0f$'%Xgrid[k, 0],
                        horizontalalignment='left', verticalalignment='center',
                        rotation=-90, fontsize=25, color='indianred', transform=ax.transAxes)

            ax.grid()

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.savefig('./plots/simulation_3_ewf.png', bbox_inches='tight')
    # plt.show() """
import random

def load_wdbc1():
    import pandas as pd
    #df = pd.read_csv('data.csv', header=None).drop(columns=[0])
    clf = LogisticRegression()
    df = pd.read_csv('diabetic_data.csv', header=None).fillna(0)
    random.seed(10)
    print (df.shape)
    df = df.sample(n=6000, axis=0)
    df = df.astype(float)

    df = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    train, validate, test = np.split(df.sample(frac=1), [int(.33 * len(df)), int(.66 * len(df))])
    X_train = np.array(train[train.columns[0:15]])
    X_cv = np.array(validate[validate.columns[0:15]])
    X_test = np.array(test[test.columns[0:15]])
    y_train = np.array(train[train.columns[15]])
    y_cv = np.array(validate[validate.columns[15]])
    y_test = np.array(test[test.columns[15]])
    clf.fit(X_train, y_train)
    prob_test = clf.predict_proba(X_test)
    prob_cv = clf.predict_proba(X_cv)
    return X_test, y_test, X_cv, y_cv, prob_test[:, 1], prob_cv[:, 1]

def load_wdbc2():
    import pandas as pd
    #df = pd.read_csv('data.csv', header=None).drop(columns=[0])
    clf = RandomForestClassifier()
    df = pd.read_csv('diabetic_data.csv', header=None).fillna(0)
    random.seed(10)
    df = df.sample(n=6000, axis=0)
    df = df.astype(float)
    df = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    train, validate, test = np.split(df.sample(frac=1), [int(.33 * len(df)), int(.66 * len(df))])
    X_train = np.array(train[train.columns[0:15]])
    X_cv = np.array(validate[validate.columns[0:15]])
    X_test = np.array(test[test.columns[0:15]])
    y_train = np.array(train[train.columns[15]])
    y_cv = np.array(validate[validate.columns[15]])
    y_test = np.array(test[test.columns[15]])
    clf.fit(X_train, y_train)
    prob_test = clf.predict_proba(X_test)
    prob_cv = clf.predict_proba(X_cv)
    return X_test, y_test, X_cv, y_cv, prob_test[:, 1], prob_cv[:, 1]

def load_wdbc3():
    import pandas as pd
    #df = pd.read_csv('data.csv', header=None).drop(columns=[0])
    clf = svm.SVC(probability=True)
    df = pd.read_csv('diabetic_data.csv', header=None).fillna(0)
    random.seed(10)
    df = df.sample(n=6000, axis=0)
    df = df.astype(float)
    df = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

    train, validate, test = np.split(df.sample(frac=1), [int(.33 * len(df)), int(.66 * len(df))])
    X_train = np.array(train[train.columns[0:15]])
    X_cv = np.array(validate[validate.columns[0:15]])
    X_test = np.array(test[test.columns[0:15]])
    y_train = np.array(train[train.columns[15]])
    y_cv = np.array(validate[validate.columns[15]])
    y_test = np.array(test[test.columns[15]])
    clf.fit(X_train, y_train)
    prob_test = clf.predict_proba(X_test)
    prob_cv = clf.predict_proba(X_cv)

    return X_test, y_test, X_cv, y_cv, prob_test[:, 1], prob_cv[:, 1]

def load_wdbc4():
    import pandas as pd
    random.seed(10)
    #df = pd.read_csv('data.csv', header=None).drop(columns=[0])
    clf = MultinomialNB()
    df = pd.read_csv('diabetic_data.csv', header=None).fillna(0)

    df = df.sample(n=6000, axis=0)
    df = df.astype(float)

    df = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    train, validate, test = np.split(df.sample(frac=1), [int(.33 * len(df)), int(.66 * len(df))])
    X_train = np.array(train[train.columns[0:15]])
    X_cv = np.array(validate[validate.columns[0:15]])
    X_test = np.array(test[test.columns[0:15]])
    y_train = np.array(train[train.columns[15]])
    y_cv = np.array(validate[validate.columns[15]])
    y_test = np.array(test[test.columns[15]])
    clf.fit(X_train, y_train)
    prob_test = clf.predict_proba(X_test)
    prob_cv = clf.predict_proba(X_cv)
    return X_test, y_test, X_cv, y_cv, prob_test[:, 1], prob_cv[:, 1]

def assess_calibration_model1():
    from sklearn.calibration import calibration_curve
    from .uncertainty_estimator_prob import ELCE2_test_estimator, calibrate, calibration_error, \
            error_witness_function, local_bias_estimator, EWF_calibration, KRR_calibration
    from sklearn.metrics import pairwise_distances
    np.random.seed(12524)
    X_test_wdbc, y_test_wdbc, X_cv_wdbc, y_cv_wdbc, prob_test_wdbc, prob_cv_wdbc = load_wdbc1()
    X_test_wdbc2, y_test_wdbc2, X_cv_wdbc2, y_cv_wdbc2, prob_test_wdbc2, prob_cv_wdbc2 = load_wdbc2()
    X_test_wdbc3, y_test_wdbc3, X_cv_wdbc3, y_cv_wdbc3, prob_test_wdbc3, prob_cv_wdbc3 = load_wdbc3()
    X_test_wdbc4, y_test_wdbc4, X_cv_wdbc4, y_cv_wdbc4, prob_test_wdbc4, prob_cv_wdbc4 = load_wdbc4()


    gamma_wdbc = 1.0 / np.median(pairwise_distances(X_test_wdbc, metric='euclidean')) ** 2


    n_bins = 8

    methods = ['No calibration', 'EWF', 'KRR', 'temperature_scaling']

    l_methods = len(methods)

    plt.figure(figsize=(l_methods * 5, 12))

    for i, method in enumerate(methods):

        ax1 = plt.subplot2grid((9, l_methods * 1), (0, i), rowspan=4)
        ax2 = plt.subplot2grid((9, l_methods * 1), (4, i))
        ax3 = plt.subplot2grid((9, l_methods * 1), (5, i))
        ax4 = plt.subplot2grid((9, l_methods * 1), (6, i))
        ax5 = plt.subplot2grid((9, l_methods * 1), (7, i))
        ax6 = plt.subplot2grid((9, l_methods * 1), (8, i))

        ax1.plot([0, 1], [0, 1], "k:", label="reference")


        plot_metrics(X_test_wdbc, prob_test_wdbc, y_test_wdbc, X_cv_wdbc, prob_cv_wdbc, y_cv_wdbc, method, ax1, ax2, ax3, ax4, ax5, ax6,
                         legend='Logistic Regression', n_bins=n_bins, prob_kernel_wdith=0.2, gamma=gamma_wdbc/10.0, marker='o', markersize=20, markeredgewidth=2, alpha=0.5)
        plot_metrics(X_test_wdbc2, prob_test_wdbc2, y_test_wdbc2, X_cv_wdbc2, prob_cv_wdbc2, y_cv_wdbc2, method, ax1, ax2,
                     ax3, ax4, ax5, ax6,
                     legend='Random Forest', n_bins=n_bins, prob_kernel_wdith=0.2, gamma=gamma_wdbc / 10.0,
                     marker='*', markersize=20, markeredgewidth=2, alpha=0.5)
        plot_metrics(X_test_wdbc3, prob_test_wdbc3, y_test_wdbc3, X_cv_wdbc3, prob_cv_wdbc3, y_cv_wdbc3, method, ax1, ax2,
                     ax3, ax4, ax5, ax6,
                     legend='SVM', n_bins=n_bins, prob_kernel_wdith=0.2, gamma=gamma_wdbc / 10.0,
                     marker='s', markersize=20, markeredgewidth=2, alpha=0.5)
        plot_metrics(X_test_wdbc4, prob_test_wdbc4, y_test_wdbc4, X_cv_wdbc4, prob_cv_wdbc4, y_cv_wdbc4, method, ax1, ax2,
                     ax3, ax4, ax5, ax6,
                     legend='Naive Bayes', n_bins=n_bins, prob_kernel_wdith=0.2, gamma=gamma_wdbc / 10.0,
                     marker='P', markersize=20, markeredgewidth=2, alpha=0.5)

        ax1.set_xlabel("Mean predicted value", size=18)
        ax1.set_ylim([0.0, 1.])
        ax1.set_xlim([0.0, 1.])
        ax1.set_title(method, size=20)
        ax1.grid()

        ax2.set_yticks([])
        ax2.set_xlim([0.0, 0.3])
        ax3.set_yticks([])
        ax3.set_xlim([-0.01, 0.2])
        ax4.set_yticks([])
        ax4.set_xlim([-0.01, 0.6])
        ax5.set_yticks([])
        ax5.set_xlim([-0.02, 0.50])
        ax6.set_yticks([])
        ax6.set_xlim([-0.01, 0.5])

        if i == 0:
            ax1.set_ylabel("Fraction of positives", size=18)
            ax1.legend(loc="upper left", prop={'size': 17})
            ax2.set_ylabel("BS", size=18)
            ax3.set_ylabel("ECE", size=18)
            ax4.set_ylabel("MCE", size=18)
            ax5.set_ylabel(r"ELCE$^2_{u}$", size=18)
            ax6.set_ylabel(r"p-val", size=18)

    plt.tight_layout()
    plt.show()
def assess_calibration_model():
    from sklearn.calibration import calibration_curve
    from .uncertainty_estimator_prob import ELCE2_test_estimator, calibrate, calibration_error, \
        error_witness_function, local_bias_estimator, EWF_calibration, KRR_calibration
    from sklearn.metrics import pairwise_distances
    """
    X, y, glo_mis_prob, loc_mis_prob, prob_cal, Xgrid, glo_mis_prob_grid, loc_mis_prob_grid, prob_cal_grid = \
        run_simulation(n_sample=20000, sklearn=False, rand=165856, grid=True)

    cv_samples = 10000  # Samples used for training the models

    X_cv = X[:cv_samples]; X_test = X[cv_samples:]

    y_cv = y[:cv_samples]; y_test = y[cv_samples:]

    glo_mis_prob_cv = glo_mis_prob[:cv_samples]; glo_mis_prob_test = glo_mis_prob[cv_samples:]
    loc_mis_prob_cv = loc_mis_prob[:cv_samples]; loc_mis_prob_test = loc_mis_prob[cv_samples:]

    # kernel hyper-parameter
    gamma = 1.0 / np.median(pairwise_distances(X_test, metric='euclidean')) ** 2 * 6.0

    # error calibration setup
    n_bins = 20; kmax = 5

    ewf_model = EWF_calibration()
    krr_model = KRR_calibration()

    # #############################################################################
    # Plot calibration plots
    def plot_ewf(k, prob_cv, prob_test, prob_test_grid, method, ax):

        ax.plot([-2, 2], [0, 0], "k-", label="reference")

        if method == 'No calibration':
            prob = prob_test.copy()
            prob_grid = prob_test_grid.copy()
        else:
            prob = calibrate(y_cv, prob_cv, prob_test=prob_test, method=method)
            prob_grid = calibrate(y_cv, prob_cv, prob_test=prob_test_grid, method=method)

        ewf_model.fit(X_test, prob, y_test,  kernel_function='rbf', gamma=gamma)
        ewf = ewf_model.predict(Xgrid, mode='bias')

        krr_model.fit(X_test, prob, y_test, kernel_function='rbf', gamma=gamma/40.0, alpha=1.)
        krr_1 = krr_model.predict(Xgrid, mode='bias')

        krr_model.fit(X_test, prob, y_test, kernel_function='rbf', gamma=gamma / 40.0, alpha=10.)
        krr_10 = krr_model.predict(Xgrid, mode='bias')

        ax.plot(Xgrid[k::kmax].T[1], prob_cal_grid[k::kmax] - prob_grid[k::kmax], 'k:', lw=5.0, label=r'$f_{\rm true} - \widehat{f}$ (true bias)')
        ax.plot(Xgrid[k::kmax].T[1], ewf[k::kmax], lw=4.0, label='Naive Estimator')
        ax.plot(Xgrid[k::kmax].T[1], krr_1[k::kmax], lw=4.0, label=r'KRR ($\lambda=1$)')
        ax.plot(Xgrid[k::kmax].T[1], krr_10[k::kmax], lw=4.0, label=r'KRR ($\lambda=10$)')

        return ax
    
    plt.figure(figsize=(kmax * 5, 10))

    for k in range(kmax):
        ax = plt.subplot2grid((2, kmax), (0, k))
        ax = plot_ewf(k, loc_mis_prob_cv, loc_mis_prob_test, loc_mis_prob_grid, 'No calibration', ax)

        ax.set_ylim([-0.5, 0.5])
        ax.set_xlim([-2.0, 2.0])

        ax.set_title(r'$x_1 = %0.0f$' % Xgrid[k, 0], size=25, color='indianred')

        if k == 0:
            ax.set_ylabel("bias $ = f - \widehat{f}_1$", size=18)
            plt.yticks([-0.5, -0.25, 0.0, 0.25])
        else: plt.yticks([-0.5, -0.25, 0, 0.25], [])

        plt.xticks([-1, 0, 1, 2], [' ', ' ', ' ', ' '])

        if k == 1: ax.legend(loc="upper right", prop={'size': 15})

        ax.grid()

        ax = plt.subplot2grid((2, kmax), (1, k))
        ax = plot_ewf(k, glo_mis_prob_cv, glo_mis_prob_test, glo_mis_prob_grid, 'No calibration', ax)

        ax.set_ylim([-0.5, 0.5])
        ax.set_xlim([-2.0, 2.0])

        if k == 0:
            ax.set_ylabel("bias $= f - \widehat{f}_2$", size=18)
            plt.yticks([-0.5, -0.25, 0.0, 0.25])
        else: plt.yticks([-0.5, -0.25, 0, 0.25], [])

        plt.xticks([-1, 0, 1, 2])
        ax.set_xlabel(r"$x_2$", size=27)

        if k == 1: ax.legend(loc="upper right", prop={'size': 15})

        ax.grid()

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.savefig('./plots/simulation_3_bias_Estimators.png', bbox_inches='tight')
    # plt.show()
    """

    from sklearn.metrics import pairwise_distances

    np.random.seed(12524)

    # load data and train wdbc
    #X_test_sg, y_test_sg, X_cv_sg, y_cv_sg, prob_test_sg, prob_cv_sg = load_sg()
    X_test_sim_glo, y_test_sim_glo, X_cv_sim_glo, y_cv_sim_glo, prob_test_sim_glo, prob_cv_sim_glo = load_sims_global(n_sample=2000)
    X_test_sim_loc, y_test_sim_loc, X_cv_sim_loc, y_cv_sim_loc, prob_test_sim_loc, prob_cv_sim_loc = load_sims_local(n_sample=2000)
    #X_test_wdbc, y_test_wdbc, X_cv_wdbc, y_cv_wdbc, prob_test_wdbc, prob_cv_wdbc = load_wdbc()

    # kernel hyperparameter
    #gamma_sg = 1.0 / np.median(pairwise_distances(X_test_sg, metric='euclidean')) ** 2
    gamma_sim_glo = 1.0 / np.median(pairwise_distances(X_test_sim_glo, metric='euclidean')) ** 2 / 4.0
    gamma_sim_loc = 1.0 / np.median(pairwise_distances(X_test_sim_loc, metric='euclidean')) ** 2 / 4.0
    #gamma_wdbc = 1.0 / np.median(pairwise_distances(X_test_wdbc, metric='euclidean')) ** 2

    # error calibration setup
    n_bins = 8

    # #############################################################################
    # Plot calibration plots

    methods = ['No calibration', 'EWF', 'KRR', 'temperature_scaling']

    l_methods = len(methods)

    plt.figure(figsize=(l_methods * 5, 12))

    for i, method in enumerate(methods):

        ax1 = plt.subplot2grid((9, l_methods * 1), (0, i), rowspan=4)
        ax2 = plt.subplot2grid((9, l_methods * 1), (4, i))
        ax3 = plt.subplot2grid((9, l_methods * 1), (5, i))
        ax4 = plt.subplot2grid((9, l_methods * 1), (6, i))
        ax5 = plt.subplot2grid((9, l_methods * 1), (7, i))
        ax6 = plt.subplot2grid((9, l_methods * 1), (8, i))

        ax1.plot([0, 1], [0, 1], "k:", label="reference")

        plot_metrics(X_test_sim_glo, prob_test_sim_glo, y_test_sim_glo, X_cv_sim_glo, prob_cv_sim_glo, y_cv_sim_glo, method, ax1, ax2, ax3, ax4, ax5, ax6,
                     legend='Globally miscalibrated', n_bins=n_bins, prob_kernel_wdith=0.2, gamma=gamma_sim_glo, marker='*', markersize=15, markeredgewidth=2, alpha=1.0)
        plot_metrics(X_test_sim_loc, prob_test_sim_loc, y_test_sim_loc, X_cv_sim_loc, prob_cv_sim_loc, y_cv_sim_loc, method, ax1, ax2, ax3, ax4, ax5, ax6,
                     legend='Locally miscalibrated', n_bins=n_bins, prob_kernel_wdith=0.2, gamma=gamma_sim_loc, marker='v', markersize=15, markeredgewidth=2, alpha=1.0)
        # plot_metrics(X_test_sg, prob_test_sg, y_test_sg, X_cv_sg, prob_cv_sg, y_cv_sg, method, ax1, ax2, ax3, ax4, ax5, ax6,
        #            legend='Star-Galaxy', n_bins=n_bins, prob_kernel_wdith=0.2, gamma=gamma_sg/10.0, marker='^', markersize=20, markeredgewidth=2, alpha=0.5)
        # plot_metrics(X_test_wdbc, prob_test_wdbc, y_test_wdbc, X_cv_wdbc, prob_cv_wdbc, y_cv_wdbc, method, ax1, ax2, ax3, ax4, ax5, ax6,
        #             legend='Breast Cancer', n_bins=n_bins, prob_kernel_wdith=0.2, gamma=gamma_wdbc/10.0, marker='*', markersize=20, markeredgewidth=2, alpha=0.5)

        ax1.set_xlabel("Mean predicted value", size=18)
        ax1.set_ylim([0.0, 1.])
        ax1.set_xlim([0.0, 1.])
        ax1.set_title(method, size=20)
        ax1.grid()

        ax2.set_yticks([])
        ax2.set_xlim([0.0, 0.3])
        ax3.set_yticks([])
        ax3.set_xlim([-0.01, 0.2])
        ax4.set_yticks([])
        ax4.set_xlim([-0.01, 0.6])
        ax5.set_yticks([])
        ax5.set_xlim([-0.02, 0.50])
        ax6.set_yticks([])
        ax6.set_xlim([-0.01, 0.5])

        if i == 0:
            ax1.set_ylabel("Fraction of positives", size=18)
            ax1.legend(loc="upper left", prop={'size': 17})
            ax2.set_ylabel("BS", size=18)
            ax3.set_ylabel("ECE", size=18)
            ax4.set_ylabel("MCE", size=18)
            ax5.set_ylabel(r"ELCE$^2_{u}$", size=18)
            ax6.set_ylabel(r"p-val", size=18)

    plt.tight_layout()
    plt.show()

def example_1():

    from sklearn import datasets
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import pairwise_distances
    from .uncertainty_estimator_prob import  calibrate, ELCE2_test_estimator
    from sklearn.preprocessing import normalize


    '''
    data = np.genfromtxt ('diabetic_data.csv', delimiter=',')
    print(data.shape)
    np.random.shuffle(data)
    X = data[:,0:40]
    X = np.nan_to_num(X)
    y = data[:,40]
    y = np.nan_to_num(y)
    X = normalize(X,axis = 0)



    train_samples = 50000  # Samples used for training the models
    cv_samples = 0  # Samples used for training the models

    X_train = X[:train_samples]
    X_cv = X_train
    X_test = X[train_samples+cv_samples:train_samples+cv_samples+10000]
    y_train = y[:train_samples]
    y_cv = y_train
    y_test = y[train_samples+cv_samples:train_samples+cv_samples+10000]

'''
    data = np.genfromtxt ('data.csv', delimiter=',')
    print(data.shape)
    np.random.shuffle(data)
    X = data[:,1:7]
    X = np.nan_to_num(X)
    y = data[:,7]
    y = np.nan_to_num(y)
    X = normalize(X,axis = 0)



    train_samples = 3000  # Samples used for training the models
    cv_samples = 0  # Samples used for training the models

    X_train = X[:train_samples]
    X_cv = X_train
    X_test = X[train_samples+cv_samples:train_samples+cv_samples+1500]
    y_train = y[:train_samples]
    y_cv = y_train
    y_test = y[train_samples+cv_samples:train_samples+cv_samples+1500]
    # kernel hyperparameter
    gamma = 1.0 / np.median(pairwise_distances(X_test, metric='euclidean')) ** 2

    # Create classifiers
    lr = LogisticRegression()
    gnb = GaussianNB()
    svc = LinearSVC(C=1.0)
    rfc = RandomForestClassifier()

    # #############################################################################
    # Plot calibration plots

    plt.figure(figsize=(16, 9))
    ax1 = plt.subplot2grid((7, 2), (0, 0), rowspan=4)
    ax2 = plt.subplot2grid((7, 2), (4, 0), rowspan=2)
    ax3 = plt.subplot2grid((7, 2), (6, 0))

    ax4 = plt.subplot2grid((7, 2), (0, 1), rowspan=4)
    ax5 = plt.subplot2grid((7, 2), (4, 1), rowspan=2)
    ax6 = plt.subplot2grid((7, 2), (6, 1))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax4.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for clf, name in [(lr, 'Logistic'),
                      (gnb, 'Naive Bayes'),
                      (svc, 'Support Vector Classification'),
                      (rfc, 'Random Forest')]:
        clf.fit(X_train, y_train)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
            prob_cv = clf.predict_proba(X_cv)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
            prob_cv = clf.decision_function(X_cv)
            prob_cv = \
                (prob_cv - prob_cv.min()) / (prob_cv.max() - prob_cv.min())

        prob_cal = calibrate(X_cv, prob_cv,y_cv, prob_test=prob_pos, method='isotonic')
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=20)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % (name,))

        ax2.hist(prob_pos, range=(0, 1), bins=20, label=name, histtype="step", lw=2)

        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_cal, n_bins=20)
        ax4.plot(mean_predicted_value, fraction_of_positives, "o-", label="%s (calibrated)" % (name,))

        ax5.hist(prob_cal, range=(0, 1), bins=20, label=name, histtype="step", lw=2)

        MLCE2 = ELCE2_test_estimator(X_test, y_test, prob_pos, prob_kernel_wdith=0.1, kernel_function='rbf', gamma=gamma)
        ax3.plot(MLCE2 * 100, 1, 'v', markersize=14, markeredgewidth=2)

        MLCE2 = ELCE2_test_estimator(X_test, y_test, prob_cal, prob_kernel_wdith=0.1, kernel_function='rbf', gamma=gamma)
        ax6.plot(MLCE2 * 100, 1, 'v', markersize=14, markeredgewidth=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([0.0, 1.])
    ax1.set_xlim([0.0, 1.])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots (reliability curve) -- uncalibrated models')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("PDF")
    ax2.legend(loc="upper center", ncol=2)
    ax2.set_yticks([])

    ax3.set_xlabel(r"ELCE$^2_{u}$", size=18)
    ax3.set_yticks([])
    # ax3.set_xlim([0.0, 50.])


    ax4.set_ylabel("Fraction of positives")
    ax4.set_ylim([0.0, 1.])
    ax4.set_xlim([0.0, 1.])
    ax4.legend(loc="lower right")
    ax4.set_title('Calibration plots (reliability curve) -- calibrated models')

    ax5.set_xlabel("Mean predicted value")
    ax5.set_ylabel("PDF")
    ax5.legend(loc="upper center", ncol=2)
    ax5.set_yticks([])

    ax6.set_xlabel(r"ELCE$^2_{u}$", size=20)
    ax6.set_yticks([])
    # ax6.set_xlim([0.0, 50.])
    plt.title('Diabete Prediction')
    plt.tight_layout()
    plt.show()

