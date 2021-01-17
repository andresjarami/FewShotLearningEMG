import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import Experiments.Experiment1.VisualizationFunctions as VF1
from scipy import stats
import ast


# %% Upload results of the three databases
def uploadResults(place, samples, people, windowSize, featureSet):
    resultsTest = pd.read_csv(place + "_FeatureSet_" + str(featureSet) + "_startPerson_" + str(1) + "_endPerson_" + str(
        people) + '_windowSize_' + windowSize + ".csv")
    if len(resultsTest) != samples * people:
        print('error' + ' 1')
        print(len(resultsTest))
    return resultsTest.drop(columns='Unnamed: 0')


# %% Accuracy of the Cote Allard interface and our classifiers for the three databases
def AnalysisCote(placeOur260, placeOur295, placeCote, featureSet):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
    shotsSet = np.array([1, 2, 3, 4])
    shots = len(shotsSet)

    idx = 0
    # the experiment with the CNN was perform 20 times to get more accurated results
    CoteAllardExperiment_Repetitions = 20

    for base in ['Nina5', 'Cote', 'EPN']:

        if base == 'Nina5':
            samples = 4
            people = 10
            title = 'NinaPro5'
        elif base == 'Cote':
            samples = 4
            people = 17
            title = 'Côté-Allard'
        elif base == 'EPN':
            samples = 25
            people = 30
            title = base
        DataFrame260 = uploadResults(placeOur260 + base, samples, people, windowSize='260', featureSet=featureSet)
        DataFrame295 = uploadResults(placeOur295 + base, samples, people, windowSize='295', featureSet=featureSet)

        vectOurQDA260 = np.zeros(shots)
        vectOurLDA260 = np.zeros(shots)
        vectOurQDA295 = np.zeros(shots)
        vectOurLDA295 = np.zeros(shots)
        vectCote = np.zeros(shots)
        for s in range(1, shots + 1):
            place = placeCote + "Cote_CWT_" + base + "/"
            cote = pd.read_csv(place + "Pytorch_results_" + str(s) + "_cycles.csv", header=None)
            coteResults = []

            if base == 'Nina5':
                for i in range(people):
                    coteResults.append(np.array(ast.literal_eval(cote.loc[0][i])).T[0].mean())
                coteResults = np.array(coteResults)
            elif base == 'EPN':
                for i in range(CoteAllardExperiment_Repetitions):
                    coteResults.append(ast.literal_eval(cote.loc[0][i]))
                coteResults = np.mean(np.array(coteResults), axis=0)
            elif base == 'Cote':
                for i in range(CoteAllardExperiment_Repetitions):
                    coteResults.append(ast.literal_eval(cote.loc[0][i]))
                    coteResults.append(ast.literal_eval(cote.loc[1][i]))
                coteResults = np.mean(np.array(coteResults), axis=0)

            vectOurLDA260[s - 1] = np.mean(DataFrame260['AccLDAProp'].loc[(DataFrame260['# shots'] == s)].values) * 100
            vectOurQDA260[s - 1] = np.mean(DataFrame260['AccQDAProp'].loc[(DataFrame260['# shots'] == s)].values) * 100
            vectOurLDA295[s - 1] = np.mean(DataFrame295['AccLDAProp'].loc[(DataFrame295['# shots'] == s)].values) * 100
            vectOurQDA295[s - 1] = np.mean(DataFrame295['AccQDAProp'].loc[(DataFrame295['# shots'] == s)].values) * 100
            vectCote[s - 1] = np.mean(coteResults)

        ax[idx].grid(color='gainsboro', linewidth=1)
        ax[idx].set_axisbelow(True)

        ax[idx].plot(shotsSet, vectCote, marker='x', label='Côté-Allard approach (window 260ms)', color='tab:green')
        ax[idx].plot(shotsSet, vectOurLDA260, marker='o', label='Our LDA classifier (window 260ms)', color='tab:orange')
        ax[idx].plot(shotsSet, vectOurQDA260, marker='^', label='Our QDA classifier (window 260ms)', color='tab:orange')
        ax[idx].plot(shotsSet, vectOurLDA295, marker='o', label='Our LDA classifier (window 295ms)', color='tab:blue')
        ax[idx].plot(shotsSet, vectOurQDA295, marker='^', label='Our QDA classifier (window295ms)', color='tab:blue')
        ax[idx].yaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
        ax[idx].xaxis.set_ticks(np.arange(1, shots + .2, 1))
        ax[idx].set_xlabel('repetitions')
        ax[idx].set_ylabel('accuracy [%]')
        ax[idx].set_title(title)
        idx += 1

    ax[2].legend(loc='lower center', bbox_to_anchor=(2, -1.5), ncol=3)
    fig.tight_layout(pad=0.1)
    plt.savefig("coteAcc.png", bbox_inches='tight', dpi=600)
    plt.show()

    return


# %% Friedman rank test for the Cote Allard interface and our classifiers

def friedman_test(*args):
    """
        From: https://github.com/citiususc/stac/blob/master/stac/nonparametric_tests.py
        Performs a Friedman ranking test.
        Tests the hypothesis that in a set of k dependent samples groups (where k >= 2) at least two of the groups represent populations with different median values.

        Parameters
        ----------
        sample1, sample2, ... : array_like
            The sample measurements for each group.

        Returns
        -------
        F-value : float
            The computed F-value of the test.
        p-value : float
            The associated p-value from the F-distribution.
        rankings : array_like
            The ranking for each group.
        pivots : array_like
            The pivotal quantities for each group.

        References
        ----------
        M. Friedman, The use of ranks to avoid the assumption of normality implicit in the analysis of variance, Journal of the American Statistical Association 32 (1937) 674–701.
        D.J. Sheskin, Handbook of parametric and nonparametric statistical procedures. crc Press, 2003, Test 25: The Friedman Two-Way Analysis of Variance by Ranks
    """
    k = len(args)
    if k < 2: raise ValueError('Less than 2 levels')
    n = len(args[0])
    if len(set([len(v) for v in args])) != 1: raise ValueError('Unequal number of samples')

    rankings = []
    for i in range(n):
        row = [col[i] for col in args]
        row_sort = sorted(row, reverse=True)
        rankings.append([row_sort.index(v) + 1 + (row_sort.count(v) - 1) / 2. for v in row])

    rankings_avg = [np.mean([case[j] for case in rankings]) for j in range(k)]
    rankings_cmp = [r / np.sqrt(k * (k + 1) / (6. * n)) for r in rankings_avg]

    chi2 = ((12 * n) / float((k * (k + 1)))) * (
            (np.sum(r ** 2 for r in rankings_avg)) - ((k * (k + 1) ** 2) / float(4)))
    iman_davenport = ((n - 1) * chi2) / float((n * (k - 1) - chi2))

    p_value = 1 - stats.f.cdf(iman_davenport, k - 1, (k - 1) * (n - 1))

    return iman_davenport, p_value, rankings_avg, rankings_cmp

def holm_test(ranks, control=None):
    """
        From: https://github.com/citiususc/stac/blob/master/stac/nonparametric_tests.py
        Performs a Holm post-hoc test using the pivot quantities obtained by a ranking test.
        Tests the hypothesis that the ranking of the control method is different to each of the other methods.

        Parameters
        ----------
        pivots : dictionary_like
            A dictionary with format 'groupname':'pivotal quantity'
        control : string optional
            The name of the control method (one vs all), default None (all vs all)

        Returns
        ----------
        Comparions : array-like
            Strings identifier of each comparison with format 'group_i vs group_j'
        Z-values : array-like
            The computed Z-value statistic for each comparison.
        p-values : array-like
            The associated p-value from the Z-distribution wich depends on the index of the comparison
        Adjusted p-values : array-like
            The associated adjusted p-values wich can be compared with a significance level

        References
        ----------
        O.J. S. Holm, A simple sequentially rejective multiple test procedure, Scandinavian Journal of Statistics 6 (1979) 65–70.
    """
    k = len(ranks)
    values = list(ranks.values())
    keys = list(ranks.keys())
    if not control:
        control_i = values.index(min(values))
    else:
        control_i = keys.index(control)

    comparisons = [keys[control_i] + " vs " + keys[i] for i in range(k) if i != control_i]
    z_values = [abs(values[control_i] - values[i]) for i in range(k) if i != control_i]
    p_values = [2 * (1 - stats.norm.cdf(abs(z))) for z in z_values]
    # Sort values by p_value so that p_0 < p_1
    p_values, z_values, comparisons = map(list, zip(*sorted(zip(p_values, z_values, comparisons), key=lambda t: t[0])))
    adj_p_values = [min(max((k - (j + 1)) * p_values[j] for j in range(i + 1)), 1) for i in range(k - 1)]

    return comparisons, z_values, p_values, adj_p_values, keys[control_i]

def dataFrame_Friedman_Holm(dataFrame):
    data = np.asarray(dataFrame)
    num_datasets, num_methods = data.shape
    print("Number of classifiers: ", num_methods,
          "\nNumber of evaluations (10(people NinaPro5) x 4(shots) + 17(people Cote) x 4(shots) 30(people EPN) x 7(shots)): ",
          num_datasets, '\n')

    alpha = 0.05  # Set this to the desired alpha/signifance level

    stat, p = stats.friedmanchisquare(*data)

    reject = p <= alpha
    print("Should we reject H0 (i.e. is there a difference in the means) at the", (1 - alpha) * 100,
          "% confidence level?", reject, '\n')

    if not reject:
        print(
            "Exiting early. The rankings are only relevant if there was a difference in the means i.e. if we rejected h0 above")
    else:
        statistic, p_value, ranking, rank_cmp = friedman_test(*np.transpose(data))
        ranks = {key: ranking[i] for i, key in enumerate(list(dataFrame.columns))}
        ranksComp = {key: rank_cmp[i] for i, key in enumerate(list(dataFrame.columns))}

        comparisons, z_values, p_values, adj_p_values, best = holm_test(ranksComp)

        adj_p_values = np.asarray(adj_p_values)

        for method, rank in ranks.items():
            print(method + ":", "%.1f" % rank)
        print('\n The best classifier is: ', best)
        holm_scores = pd.DataFrame({"p": adj_p_values, "sig": adj_p_values < alpha}, index=comparisons)
        print(holm_scores)

def AnalysisFriedman(placeOur260, placeOur295, placeCote, featureSet):
    shots = 4
    # the experiment with the CNN was perform 20 times to get more accurated results
    CoteAllardExperiment_Repetitions = 20
    vectOurQDA260 = []
    vectOurLDA260 = []
    vectOurQDA295 = []
    vectOurLDA295 = []
    vectCote = []
    for base in ['Nina5', 'Cote', 'EPN']:

        if base == 'Nina5':
            samples = 4
            people = 10
        elif base == 'Cote':
            samples = 4
            people = 17
        elif base == 'EPN':
            samples = 25
            people = 30
        DataFrame260 = uploadResults(placeOur260 + base, samples, people, windowSize='260', featureSet=featureSet)
        DataFrame295 = uploadResults(placeOur295 + base, samples, people, windowSize='295', featureSet=featureSet)

        for s in range(1, shots + 1):
            place = placeCote + "Cote_CWT_" + base + "/"
            cote = pd.read_csv(place + "Pytorch_results_" + str(s) + "_cycles.csv", header=None)
            coteResults = []

            if base == 'Nina5':
                for i in range(people):
                    coteResults.append(np.array(ast.literal_eval(cote.loc[0][i])).T[0].mean())
                coteResults = np.array(coteResults)
            elif base == 'EPN':
                for i in range(CoteAllardExperiment_Repetitions):
                    coteResults.append(ast.literal_eval(cote.loc[0][i]))
                coteResults = np.mean(np.array(coteResults), axis=0)
            elif base == 'Cote':
                for i in range(CoteAllardExperiment_Repetitions):
                    coteResults.append(ast.literal_eval(cote.loc[0][i]))
                    coteResults.append(ast.literal_eval(cote.loc[1][i]))
                coteResults = np.mean(np.array(coteResults), axis=0)

            vectOurLDA260 = np.hstack(
                (vectOurLDA260, DataFrame260['AccLDAProp'].loc[(DataFrame260['# shots'] == s)].values * 100))
            vectOurQDA260 = np.hstack(
                (vectOurQDA260, DataFrame260['AccQDAProp'].loc[(DataFrame260['# shots'] == s)].values * 100))
            vectOurLDA295 = np.hstack(
                (vectOurLDA295, DataFrame295['AccLDAProp'].loc[(DataFrame295['# shots'] == s)].values * 100))
            vectOurQDA295 = np.hstack(
                (vectOurQDA295, DataFrame295['AccQDAProp'].loc[(DataFrame295['# shots'] == s)].values * 100))
            vectCote = np.hstack((vectCote, coteResults))
    #Analysis with a window size of 260ms
    print('\n\nANALYSIS OF WINDOW SIZE 260ms')
    dataFrame_Friedman_Holm(pd.DataFrame(
        data={'vectOurLDA260': vectOurLDA260, 'vectOurQDA260': vectOurQDA260,'vectCote': vectCote}))
    # Analysis with a window size of 295ms
    print('\n\nANALYSIS OF WINDOW SIZE 295ms')
    dataFrame_Friedman_Holm(pd.DataFrame(
        data={'vectOurLDA295': vectOurLDA295, 'vectOurQDA295': vectOurQDA295, 'vectCote': vectCote}))
    # Analysis with all window sizes
    print('\n\nANALYSIS OF ALL WINDOW SIZES')
    dataFrame_Friedman_Holm(dataFrame = pd.DataFrame(
        data={'vectOurLDA260': vectOurLDA260, 'vectOurQDA260': vectOurQDA260, 'vectOurLDA295': vectOurLDA295,
              'vectOurQDA295': vectOurQDA295, 'vectCote': vectCote}))
