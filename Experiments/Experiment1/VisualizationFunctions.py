import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy import stats


# %% Upload results of the three databases
def uploadResults(place, samples, people, windowSize):
    resultsTest = pd.read_csv(place + "_FeatureSet_1_startPerson_" + str(1) + "_endPerson_" + str(
        people) + '_windowSize_' + windowSize + ".csv")
    if len(resultsTest) != samples * people:
        print('error' + ' 1')
        print(len(resultsTest))
    for j in range(2, 4):
        auxFrame = pd.read_csv(
            place + "_FeatureSet_" + str(j) + "_startPerson_" + str(1) + "_endPerson_" + str(
                people) + '_windowSize_' + windowSize + ".csv")
        resultsTest = pd.concat([resultsTest, auxFrame], ignore_index=True)
        if len(auxFrame) != samples * people:
            print('error' + ' ' + str(j))
            print(len(auxFrame))
    return resultsTest.drop(columns='Unnamed: 0')


def uploadResultsDatabases(folder, database, windowSize):
    if database == 'Nina5':
        samples = 4
        people = 10
        shots = 4
    elif database == 'Cote':
        samples = 4
        people = 17
        shots = 4
    elif database == 'EPN':
        samples = 25
        people = 30
        shots = 25
    elif database == 'Capgmyo_dba':
        samples = 9
        people = 18
        shots = 9
    elif database == 'Capgmyo_dbc':
        samples = 9
        people = 10
        shots = 9
    elif database == 'Nina1':
        samples = 9
        people = 27
        shots = 9
    elif database == 'Nina3':
        samples = 5
        people = 9
        shots = 5

    place = folder + database

    return analysisResults(uploadResults(place, samples, people, windowSize), shots)


def analysisResults(resultDatabase, shots):
    results = pd.DataFrame(columns=['Feature Set', '# shots'])
    timeOurTechnique = pd.DataFrame(columns=[])

    idx = 0
    for j in range(1, 4):
        for i in range(1, shots + 1):
            # subset = str(tuple(range(1, i + 1)))
            results.at[idx, 'Feature Set'] = j
            results.at[idx, '# shots'] = i

            # the accuracy for all LDA and QDA approaches
            IndLDA = resultDatabase['AccLDAInd'].loc[
                (resultDatabase['# shots'] == i) & (resultDatabase['Feature Set'] == j)]
            IndQDA = resultDatabase['AccQDAInd'].loc[
                (resultDatabase['# shots'] == i) & (resultDatabase['Feature Set'] == j)]
            MultiLDA = resultDatabase['AccLDAMulti'].loc[
                (resultDatabase['# shots'] == i) & (resultDatabase['Feature Set'] == j)]
            MultiQDA = resultDatabase['AccQDAMulti'].loc[
                (resultDatabase['# shots'] == i) & (resultDatabase['Feature Set'] == j)]
            LiuLDA = resultDatabase['AccLDALiu'].loc[
                (resultDatabase['# shots'] == i) & (resultDatabase['Feature Set'] == j)]
            LiuQDA = resultDatabase['AccQDALiu'].loc[
                (resultDatabase['# shots'] == i) & (resultDatabase['Feature Set'] == j)]
            VidLDA = resultDatabase['AccLDAVidovic'].loc[
                (resultDatabase['# shots'] == i) & (resultDatabase['Feature Set'] == j)]
            VidQDA = resultDatabase['AccQDAVidovic'].loc[
                (resultDatabase['# shots'] == i) & (resultDatabase['Feature Set'] == j)]
            OurLDA = resultDatabase['AccLDAProp'].loc[
                (resultDatabase['# shots'] == i) & (resultDatabase['Feature Set'] == j)]
            OurQDA = resultDatabase['AccQDAProp'].loc[
                (resultDatabase['# shots'] == i) & (resultDatabase['Feature Set'] == j)]
            results.at[idx, 'IndLDA'] = IndLDA.mean(axis=0)
            results.at[idx, 'IndLDAstd'] = IndLDA.std(axis=0)
            results.at[idx, 'IndQDA'] = IndQDA.mean(axis=0)
            results.at[idx, 'IndQDAstd'] = IndQDA.std(axis=0)
            results.at[idx, 'MultiLDA'] = MultiLDA.mean(axis=0)
            results.at[idx, 'MultiLDAstd'] = MultiLDA.std(axis=0)
            results.at[idx, 'MultiQDA'] = MultiQDA.mean(axis=0)
            results.at[idx, 'MultiQDAstd'] = MultiQDA.std(axis=0)
            results.at[idx, 'LiuLDA'] = LiuLDA.mean(axis=0)
            results.at[idx, 'LiuLDAstd'] = LiuLDA.std(axis=0)
            results.at[idx, 'LiuQDA'] = LiuQDA.mean(axis=0)
            results.at[idx, 'LiuQDAstd'] = LiuQDA.std(axis=0)
            results.at[idx, 'VidLDA'] = VidLDA.mean(axis=0)
            results.at[idx, 'VidLDAstd'] = VidLDA.std(axis=0)
            results.at[idx, 'VidQDA'] = VidQDA.mean(axis=0)
            results.at[idx, 'VidQDAstd'] = VidQDA.std(axis=0)
            results.at[idx, 'OurLDA'] = OurLDA.mean(axis=0)
            results.at[idx, 'OurLDAstd'] = OurLDA.std(axis=0)
            results.at[idx, 'OurQDA'] = OurQDA.mean(axis=0)
            results.at[idx, 'OurQDAstd'] = OurQDA.std(axis=0)

            # the weights λ and w for our LDA and QDA adaptive classifiers
            wLDA = resultDatabase['wTargetMeanLDA'].loc[
                (resultDatabase['# shots'] == i) & (resultDatabase['Feature Set'] == j)]
            lLDA = resultDatabase['wTargetCovLDA'].loc[
                (resultDatabase['# shots'] == i) & (resultDatabase['Feature Set'] == j)]
            wQDA = resultDatabase['wTargetMeanQDA'].loc[
                (resultDatabase['# shots'] == i) & (resultDatabase['Feature Set'] == j)]
            lQDA = resultDatabase['wTargetCovQDA'].loc[
                (resultDatabase['# shots'] == i) & (resultDatabase['Feature Set'] == j)]
            results.at[idx, 'wLDA'] = wLDA.mean(axis=0)
            results.at[idx, 'lLDA'] = lLDA.mean(axis=0)
            results.at[idx, 'wQDA'] = wQDA.mean(axis=0)
            results.at[idx, 'lQDA'] = lQDA.mean(axis=0)

            idx += 1

        ### Times of the adaptive classifier
        # the time of the adaptation plus the training of our adaptive classifier both LDA and QDA
        ourLDAtrainTime = resultDatabase['tPropLDA'].loc[(resultDatabase['Feature Set'] == j)]
        ourQDAtrainTime = resultDatabase['tPropQDA'].loc[(resultDatabase['Feature Set'] == j)]
        timeOurTechnique.at[j, 'meanTrainLDA'] = round(ourLDAtrainTime.mean(axis=0) * 1000, 2)
        timeOurTechnique.at[j, 'stdTrainLDA'] = round(ourLDAtrainTime.std(axis=0) * 1000, 2)
        timeOurTechnique.at[j, 'varTrainLDA'] = round(ourLDAtrainTime.var(axis=0) * 1000, 2)
        timeOurTechnique.at[j, 'meanTrainQDA'] = round(ourQDAtrainTime.mean(axis=0) * 1000, 2)
        timeOurTechnique.at[j, 'stdTrainQDA'] = round(ourQDAtrainTime.std(axis=0) * 1000, 2)
        timeOurTechnique.at[j, 'varTrainQDA'] = round(ourQDAtrainTime.var(axis=0) * 1000, 2)

        # the time of the classification and the preprocessing (min-maxd normalization) of our adaptive classifier both LDA and QDA
        ourLDAclassifyTime = resultDatabase['tCLPropL'].loc[(resultDatabase['Feature Set'] == j)]
        ourQDAclassifyTime = resultDatabase['tCLPropQ'].loc[(resultDatabase['Feature Set'] == j)]
        ourNormTime = resultDatabase['tPre'].loc[(resultDatabase['Feature Set'] == j)]
        # means, standar deviation, and variance
        timeOurTechnique.at[j, 'meanClLDA'] = round(ourLDAclassifyTime.mean(axis=0) * 1000, 2)
        timeOurTechnique.at[j, 'stdClLDA'] = round(ourLDAclassifyTime.std(axis=0) * 1000, 2)
        timeOurTechnique.at[j, 'varClLDA'] = round(ourLDAclassifyTime.var(axis=0) * 1000, 2)
        timeOurTechnique.at[j, 'meanClQDA'] = round(ourQDAclassifyTime.mean(axis=0) * 1000, 2)
        timeOurTechnique.at[j, 'stdClQDA'] = round(ourQDAclassifyTime.std(axis=0) * 1000, 2)
        timeOurTechnique.at[j, 'varClQDA'] = round(ourQDAclassifyTime.var(axis=0) * 1000, 2)
        timeOurTechnique.at[j, 'meanNorm'] = round(ourNormTime.mean(axis=0) * 1000000, 2)
        timeOurTechnique.at[j, 'stdNorm'] = round(ourNormTime.std(axis=0) * 1000000, 2)
        timeOurTechnique.at[j, 'varNorm'] = round(ourNormTime.var(axis=0) * 1000000, 2)

    return results, timeOurTechnique


# %%Graph of the accuracy of the all DA classifiers for the three databasese and three feature sets
def graphACC(resultsNina5, resultsCote, resultsEPN, resultsCap_A, resultsCap_C, resultsNina1):
    numDatabases = 6
    fig, ax = plt.subplots(nrows=numDatabases, ncols=6, sharey='row', figsize=(15, 22))
    shotsSet = np.arange(1, 5)
    shots = len(shotsSet)
    for Data in range(numDatabases):
        for DA in ['LDA', 'QDA']:

            for FeatureSet in range(3):
                idx = FeatureSet
                if Data == 0:
                    results = resultsNina1
                    ax[Data, idx].yaxis.set_ticks(np.arange(20, 86, 10))
                elif Data == 1:
                    results = resultsNina5
                    ax[Data, idx].yaxis.set_ticks(np.arange(30, 100, 7))
                elif Data == 2:
                    results = resultsCote
                    ax[Data, idx].yaxis.set_ticks(np.arange(70, 100, 7))
                elif Data == 3:
                    results = resultsEPN
                    ax[Data, idx].yaxis.set_ticks(np.arange(55, 100, 6))
                elif Data == 4:
                    results = resultsCap_A
                    ax[Data, idx].yaxis.set_ticks(np.arange(20, 100, 11))
                elif Data == 5:
                    results = resultsCap_C
                    ax[Data, idx].yaxis.set_ticks(np.arange(30, 100, 11))

                if DA == 'QDA':
                    idx += 3
                ax[Data, idx].grid(color='gainsboro', linewidth=1)
                ax[Data, idx].set_axisbelow(True)

                if Data == 5 and idx == 5:
                    Model = 'Ind' + DA
                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1]) * 100
                    ax[Data, idx].plot(shotsSet, Y[:shots], marker='x', label='Individual', color='tab:orange')
                    Model = 'Multi' + DA
                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1]) * 100
                    ax[Data, idx].plot(shotsSet, Y[:shots], marker='s', label='Multi-user', color='tab:purple')
                    Model = 'Liu' + DA
                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1]) * 100
                    ax[Data, idx].plot(shotsSet, Y[:shots], marker='o', label='Liu', color='tab:green')
                    Model = 'Vid' + DA
                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1]) * 100
                    ax[Data, idx].plot(shotsSet, Y[:shots], marker='^', label='Vidovic', color='tab:red')
                    Model = 'Our' + DA
                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1]) * 100
                    ax[Data, idx].plot(shotsSet, Y[:shots], marker='v', label='Our classifier', color='tab:blue')
                    ax[Data, idx].yaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
                else:
                    Model = 'Ind' + DA
                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1]) * 100
                    ax[Data, idx].plot(shotsSet, Y[:shots], marker='x', color='tab:orange')
                    Model = 'Multi' + DA
                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1]) * 100
                    ax[Data, idx].plot(shotsSet, Y[:shots], marker='s', color='tab:purple')
                    Model = 'Liu' + DA
                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1]) * 100
                    ax[Data, idx].plot(shotsSet, Y[:shots], marker='o', color='tab:green')
                    Model = 'Vid' + DA
                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1]) * 100
                    ax[Data, idx].plot(shotsSet, Y[:shots], marker='^', color='tab:red')
                    Model = 'Our' + DA
                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1]) * 100
                    ax[Data, idx].plot(shotsSet, Y[:shots], marker='v', color='tab:blue')
                    ax[Data, idx].yaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))

                if shots == 25:
                    ax[Data, idx].xaxis.set_ticks([1, 5, 10, 15, 20, 25])
                elif shots == 9:
                    ax[Data, idx].xaxis.set_ticks([1, 3, 5, 7, 9])
                else:
                    ax[Data, idx].xaxis.set_ticks(np.arange(1, shots + .2, 1))

    ax[numDatabases - 1, 0].set_xlabel('repetitions')
    ax[numDatabases - 1, 1].set_xlabel('repetitions')
    ax[numDatabases - 1, 2].set_xlabel('repetitions')
    ax[numDatabases - 1, 3].set_xlabel('repetitions')
    ax[numDatabases - 1, 4].set_xlabel('repetitions')
    ax[numDatabases - 1, 5].set_xlabel('repetitions')

    ax[0, 0].set_title('LDA\n Feature Set 1')
    ax[0, 1].set_title('LDA\n Feature Set 2')
    ax[0, 2].set_title('LDA\n Feature Set 3')

    ax[0, 3].set_title('QDA\n Feature Set 1')
    ax[0, 4].set_title('QDA\n Feature Set 2')
    ax[0, 5].set_title('QDA\n Feature Set 3')
    ax[0, 0].set_ylabel('NinaPro1\naccuracy [%]')
    ax[1, 0].set_ylabel('NinaPro5\naccuracy [%]')
    ax[2, 0].set_ylabel('Côté-Allard_db\naccuracy [%]')
    ax[3, 0].set_ylabel('EMG-EPN-120 \naccuracy [%]')
    ax[4, 0].set_ylabel('CapgMyo_dba\naccuracy [%]')
    ax[5, 0].set_ylabel('CapgMyo_dbc\naccuracy [%]')

    # fig.legend(loc='lower center', bbox_to_anchor=(0.5,-0.1), ncol=5)
    fig.tight_layout(pad=0.1)
    plt.savefig("PaperFigures/acc.png", bbox_inches='tight', dpi=300)
    plt.show()


# %% Friedman rank test for all DA approaches

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
    aux = [r ** 2 for r in rankings_avg]
    chi2 = ((12 * n) / float((k * (k + 1)))) * (
            (np.sum(aux)) - ((k * (k + 1) ** 2) / float(4)))
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

def AnalysisFriedman(folder, windowSize):
    base = 'Nina5'
    samples = 4
    people = 10
    place = folder + base
    DataFrameN5 = uploadResults(place, samples, people, windowSize)
    base = 'Cote'
    samples = 4
    people = 17
    place = folder + base
    DataFrameCo = uploadResults(place, samples, people, windowSize)
    base = 'EPN'
    samples = 25
    people = 30
    place = folder + base
    DataFrameE = uploadResults(place, samples, people, windowSize)
    base = 'Capgmyo_dba'
    samples = 9
    people = 18
    place = folder + base
    DataFrameCA = uploadResults(place, samples, people, windowSize)
    base = 'Capgmyo_dbc'
    samples = 9
    people = 10
    place = folder + base
    DataFrameCC = uploadResults(place, samples, people, windowSize)
    base = 'Nina1'
    samples = 9
    people = 27
    place = folder + base
    DataFrameN1 = uploadResults(place, samples, people, windowSize)
    base = 'Nina3'
    samples = 5
    people = 9
    place = folder + base
    DataFrameN3 = uploadResults(place, samples, people, windowSize)

    TotalDataframe = pd.concat(
        [DataFrameN5, DataFrameCo, DataFrameE, DataFrameCA, DataFrameCC, DataFrameN1, DataFrameN3])
    # TotalDataframe = pd.concat([DataFrameN1])

    # for shot in range(1,4):
    shot = 4
    for f in range(1, 4):
        for DA in ['LDA', 'QDA']:
            print('\n\nTYPE OF DA CLASSIFIER: ' + DA + ' FEATURE SET: ' + str(f))
            dataFrame = pd.DataFrame()
            dataFrame['Individual ' + DA + ' ' + str(f)] = TotalDataframe['Acc' + DA + 'Ind'].loc[
                (TotalDataframe['Feature Set'] == f) & (TotalDataframe['# shots'] <= shot)].values
            dataFrame['Multi-User ' + DA + ' ' + str(f)] = TotalDataframe['Acc' + DA + 'Multi'].loc[
                (TotalDataframe['Feature Set'] == f) & (TotalDataframe['# shots'] <= shot)].values
            dataFrame['Liu ' + DA + ' ' + str(f)] = TotalDataframe['Acc' + DA + 'Liu'].loc[
                (TotalDataframe['Feature Set'] == f) & (TotalDataframe['# shots'] <= shot)].values
            dataFrame['Vidovic ' + DA + ' ' + str(f)] = TotalDataframe['Acc' + DA + 'Vidovic'].loc[
                (TotalDataframe['Feature Set'] == f) & (TotalDataframe['# shots'] <= shot)].values
            dataFrame['Our ' + DA + ' ' + str(f)] = TotalDataframe['Acc' + DA + 'Prop'].loc[
                (TotalDataframe['Feature Set'] == f) & (TotalDataframe['# shots'] <= shot)].values

            data = np.asarray(dataFrame)
            num_datasets, num_methods = data.shape
            print("Number of classifiers: ", num_methods,
                  "\nNumber of evaluations (18(Nina1)+9(Nina3)+10(Nina5)+17(Cote)+30(EPN)+18(Capg_A)+10(Capg_C)) x 4(shots): ",
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

    return


# %% Analysis using a large database (EPN)
def largeDatabase(results):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
    shotsTotal = 25
    shotsSet = np.array([0, 4, 9, 14, 19, 24])
    xAxis = shotsSet + 1

    idx = 2
    title = 'Weights'
    wL = np.array(results['wLDA'])
    lL = np.array(results['lLDA'])
    wQ = np.array(results['wQDA'])
    lQ = np.array(results['lQDA'])
    ax[idx].grid(color='gainsboro', linewidth=1)
    ax[idx].set_axisbelow(True)
    ax[idx].plot(xAxis, np.mean(wL.reshape((3, shotsTotal)), axis=0)[shotsSet], label='$\hat{\omega}_c$ LDA',
                 marker='*', color='black')
    ax[idx].plot(xAxis, np.mean(lL.reshape((3, shotsTotal)), axis=0)[shotsSet], label='$\hat{\lambda}_c$ LDA',
                 marker='<', color='black')
    ax[idx].plot(xAxis, np.mean(wQ.reshape((3, shotsTotal)), axis=0)[shotsSet], label='$\hat{\omega}_c$ QDA',
                 marker='*', color='tab:cyan')
    ax[idx].plot(xAxis, np.mean(lQ.reshape((3, shotsTotal)), axis=0)[shotsSet], label='$\hat{\lambda}_c$ QDA',
                 marker='<', color='tab:cyan')
    ax[idx].set_title(title)
    ax[idx].xaxis.set_ticks(shotsSet)
    ax[idx].set_xlabel('repetitions')
    ax[idx].set_ylabel('weight value')
    ax[idx].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    # ax[idx].legend(loc='lower center', bbox_to_anchor=(1.2, -0.8), ncol=2)

    idx = 0
    bestFeatureSet = [3, 1]
    DA = ['LDA', 'QDA']
    for i in range(len(DA)):
        shotsSet_featureSet = (bestFeatureSet[i] - 1) * shotsTotal + shotsSet
        title = DA[i]
        ind = np.array(results['Ind' + DA[i]]) * 100
        liu = np.array(results['Liu' + DA[i]]) * 100
        vid = np.array(results['Vid' + DA[i]]) * 100
        our = np.array(results['Our' + DA[i]]) * 100
        ax[idx].grid(color='gainsboro', linewidth=1)
        ax[idx].set_axisbelow(True)
        ax[idx].plot(xAxis, ind[shotsSet_featureSet], label='Individual', marker='x',
                     color='tab:orange')
        ax[idx].plot(xAxis, liu[shotsSet_featureSet], label='Liu', marker='o',
                     color='tab:green')
        ax[idx].plot(xAxis, vid[shotsSet_featureSet], label='Vid', marker='^',
                     color='tab:red')
        ax[idx].plot(xAxis, our[shotsSet_featureSet], label='Our classifier', marker='v',
                     color='tab:blue')
        ax[idx].set_title(title)
        ax[idx].xaxis.set_ticks(shotsSet)
        ax[idx].set_xlabel('repetitions')
        ax[idx].set_ylabel('accuracy [%]')
        ax[idx].yaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
        idx += 1
    # ax[0].legend(loc='lower center', bbox_to_anchor=(1.2, -1.2), ncol=2)

    fig.tight_layout(pad=1)
    plt.savefig("PaperFigures/largeDatabase.png", bbox_inches='tight', dpi=600)
    plt.show()


# %% Time Analysis

def analysisTime(extractionTime, timeOurTechnique):
    # from seconds to miliseconds
    extractionTime = extractionTime * 1000

    for featureSet in range(3):
        for DA in ['LDA', 'QDA']:
            print('\nOur ' + DA + ' Technique for feature set ' + str(featureSet + 1))
            print('Feature set: ' + str(featureSet + 1))
            print('Training Time [s]: ',
                  round(timeOurTechnique.loc[featureSet + 1, 'meanTrain' + DA] / (60 * 1000), 1), '±',
                  round(timeOurTechnique.loc[featureSet + 1, 'stdTrain' + DA] / (60 * 1000), 1))

            print('Extraction time [ms]: ', round(extractionTime.loc[featureSet, :].mean(), 2), '±',
                  round(extractionTime.loc[featureSet, :].std(), 2))
            print('Classification time [ms]: ', timeOurTechnique.loc[featureSet + 1, 'meanCl' + DA], '±',
                  timeOurTechnique.loc[featureSet + 1, 'stdCl' + DA])
            print('Prprocessing time (min-max normalization) [µs]: ', timeOurTechnique.loc[featureSet + 1, 'meanNorm'],
                  '±', timeOurTechnique.loc[featureSet + 1, 'stdNorm'])
            print('Analysis time (the sum of the extraction, classification, and preprocessing times) [ms]: ',
                  round(extractionTime.loc[featureSet, :].mean() + timeOurTechnique.loc[featureSet + 1, 'meanCl' + DA] +
                        timeOurTechnique.loc[featureSet + 1, 'meanNorm'] / 1000, 2), '±',
                  round(np.sqrt((extractionTime.loc[featureSet, :].var() + timeOurTechnique.loc[
                      featureSet + 1, 'varCl' + DA] + timeOurTechnique.loc[featureSet + 1, 'varNorm'] / 1000)), 2))


def analysisTimeTotal(extractionTimeN1, timeOurTechniqueN1, extractionTimeN3, timeOurTechniqueN3, extractionTimeN5,
                      timeOurTechniqueN5, extractionTimeC, timeOurTechniqueC, extractionTimeE, timeOurTechniqueE,
                      extractionTimeCap_a, timeOurTechniqueCap_a, extractionTimeCap_c, timeOurTechniqueCap_c):
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex='col', figsize=(9, 6))
    classifiers = 7 * 2
    vectTrainingTime = np.zeros(classifiers)
    vectTrainingTimeSTD = np.zeros(classifiers)
    vectExtractionTime = np.zeros(classifiers)
    vectExtractionTimeSTD = np.zeros(classifiers)
    vectClassificationTime = np.zeros(classifiers)
    vectClassificationTimeSTD = np.zeros(classifiers)
    vectPreprocessingTime = np.zeros(classifiers)
    vectPreprocessingTimeSTD = np.zeros(classifiers)
    vectAnalysisTime = np.zeros(classifiers)
    vectAnalysisTimeSTD = np.zeros(classifiers)
    idx = 0
    for BD in range(7):
        if BD == 0:
            extractionTime = extractionTimeN1
            timeOurTechnique = timeOurTechniqueN1
            bestFeatureSet = [3, 3]
        elif BD == 1:
            extractionTime = extractionTimeN3
            timeOurTechnique = timeOurTechniqueN3
            bestFeatureSet = [3, 3]
        elif BD == 2:
            extractionTime = extractionTimeN5
            timeOurTechnique = timeOurTechniqueN5
            bestFeatureSet = [3, 1]
        elif BD == 3:
            extractionTime = extractionTimeC
            timeOurTechnique = timeOurTechniqueC
            bestFeatureSet = [1, 1]
        elif BD == 4:
            extractionTime = extractionTimeE
            timeOurTechnique = timeOurTechniqueE
            bestFeatureSet = [3, 1]
        elif BD == 5:
            extractionTime = extractionTimeCap_a
            timeOurTechnique = timeOurTechniqueCap_a
            bestFeatureSet = [1, 1]
        elif BD == 6:
            extractionTime = extractionTimeCap_c
            timeOurTechnique = timeOurTechniqueCap_c
            bestFeatureSet = [1, 1]
        # from seconds to miliseconds
        extractionTime = extractionTime * 1000
        DA = ['LDA', 'QDA']
        for i in range(len(DA)):
            TrainingTime = 0
            TrainingTimeSTD = 0
            ExtractionTime = 0
            ExtractionTimeSTD = 0
            ClassificationTime = 0
            ClassificationTimeSTD = 0
            PreprocessingTime = 0
            PreprocessingTimeSTD = 0
            AnalysisTime = 0
            AnalysisTimeVAR = 0
            TrainingTime += timeOurTechnique.loc[bestFeatureSet[i], 'meanTrain' + DA[i]] / (60 * 1000)
            TrainingTimeSTD += timeOurTechnique.loc[bestFeatureSet[i], 'stdTrain' + DA[i]] / (60 * 1000)
            ExtractionTime += extractionTime.loc[bestFeatureSet[i] - 1, :].mean()
            ExtractionTimeSTD += extractionTime.loc[bestFeatureSet[i] - 1, :].std()
            ClassificationTime += timeOurTechnique.loc[bestFeatureSet[i], 'meanCl' + DA[i]]
            ClassificationTimeSTD += timeOurTechnique.loc[bestFeatureSet[i], 'stdCl' + DA[i]]
            PreprocessingTime += timeOurTechnique.loc[bestFeatureSet[i], 'meanNorm'] / 1000
            PreprocessingTimeSTD += timeOurTechnique.loc[bestFeatureSet[i], 'stdNorm'] / 1000
            AnalysisTime += extractionTime.loc[bestFeatureSet[i] - 1, :].mean() + timeOurTechnique.loc[
                bestFeatureSet[i], 'meanCl' + DA[i]] + timeOurTechnique.loc[bestFeatureSet[i], 'meanNorm'] / 1000
            AnalysisTimeVAR += np.sqrt((extractionTime.loc[bestFeatureSet[i] - 1, :].var() + timeOurTechnique.loc[
                bestFeatureSet[i], 'varCl' + DA[i]] + timeOurTechnique.loc[bestFeatureSet[i], 'varNorm'] / 1000))
            vectTrainingTime[idx] = TrainingTime / 3
            vectTrainingTimeSTD[idx] = TrainingTimeSTD / 3
            vectExtractionTime[idx] = ExtractionTime / 3
            vectExtractionTimeSTD[idx] = ExtractionTimeSTD / 3
            vectClassificationTime[idx] = ClassificationTime / 3
            vectClassificationTimeSTD[idx] = ClassificationTimeSTD / 3
            vectPreprocessingTime[idx] = PreprocessingTime / 3
            vectPreprocessingTimeSTD[idx] = PreprocessingTimeSTD / 3
            vectAnalysisTime[idx] = AnalysisTime / 3
            vectAnalysisTimeSTD[idx] = AnalysisTimeVAR / 3
            idx += 1

    print('Training Time [min]', vectTrainingTime)
    print('Training Time [min] STD', vectTrainingTimeSTD, '\n')
    print('Feature extraction Time [ms]', vectExtractionTime)
    print('Feature extraction Time [ms] STD', vectExtractionTimeSTD)
    print('Pre-processing Time [ms]', vectPreprocessingTime)
    print('Pre-processing Time [ms] STD', vectPreprocessingTimeSTD)
    print('Classification Time [ms]', vectClassificationTime)
    print('Classification Time [ms] STD', vectClassificationTimeSTD)
    print('Data-Analysis Time [ms]', vectAnalysisTime)
    print('Data-Analysis Time [ms] STD', vectAnalysisTimeSTD, '\n')

    xAxis = np.arange(classifiers)  # the x locations for the groups
    width = 0.5  # the width of the bars: can also be len(x) sequence
    ax[0].grid(color='gainsboro', linewidth=1)
    ax[0].set_axisbelow(True)

    vectPreprocessingTime *= 300
    ax[0].bar(xAxis, vectExtractionTime, width, label='Feature-extraction Time')
    ax[0].bar(xAxis, vectPreprocessingTime, width, bottom=vectExtractionTime, label='Pre-processing Time')
    ax[0].bar(xAxis, vectClassificationTime, width, bottom=vectExtractionTime + vectPreprocessingTime,
              yerr=vectAnalysisTimeSTD, label='Classification Time')

    ax[0].set_ylabel('time (ms)')
    ax[0].yaxis.set_ticks(np.arange(0, 3.1, 1))
    ax[0].set_title('Data-Analysis Time')
    ax[0].set_xticks(xAxis)

    ax[1].grid(color='gainsboro', linewidth=1)
    ax[1].set_axisbelow(True)

    ax[1].bar(xAxis, vectTrainingTime, width, yerr=vectTrainingTimeSTD, label='Training Time', color='tab:red')
    ax[1].set_xlabel('Our classifiers over the seven databases')
    ax[1].set_ylabel('time (min)')
    ax[1].yaxis.set_ticks(np.arange(0, 3.1, 1))
    ax[1].set_title('Training Time')
    ax[1].set_xticks(xAxis)
    ax[1].set_xticklabels(
        ['LDA_Nina1', 'QDA_Nina1', 'LDA_Nina3', 'QDA_Nina3', 'LDA_Nina5', 'QDA_Nina5', 'LDA_Cote', 'QDA_Cote',
         'LDA_EPN', 'QDA_EPN', 'LDA_CapgMyoA', 'QDA_CapgMyoA', 'LDA_CapgMyoC', 'QDA_CapgMyoC'], rotation=20)
    # ax[1].legend(loc='lower center', bbox_to_anchor=(2, -1), ncol=5)

    fig.tight_layout(pad=1)
    plt.savefig("PaperFigures/times.png", bbox_inches='tight', dpi=600)

    plt.show()


# %% Graph NinaPro3 dataset that has EMG data from amputees

def graphACCNina3(results):
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey='row', figsize=(7, 3))
    shotsSet = np.arange(1, 6)
    shots = len(shotsSet)

    bestFeatureSet = [3, 3]
    DA = ['LDA', 'QDA']
    for i in range(len(DA)):
        FeatureSet = bestFeatureSet[i]
        ax[i].yaxis.set_ticks(np.arange(20, 86, 10))
        ax[i].grid(color='gainsboro', linewidth=1)
        ax[i].set_axisbelow(True)

        Model = 'Ind' + DA[i]
        Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet]) * 100
        ax[i].plot(shotsSet, Y[:shots], marker='x', label='Individual', color='tab:orange')
        Model = 'Multi' + DA[i]
        Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet]) * 100
        ax[i].plot(shotsSet, Y[:shots], marker='s', label='Multi-user', color='tab:purple')
        Model = 'Liu' + DA[i]
        Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet]) * 100
        ax[i].plot(shotsSet, Y[:shots], marker='o', label='Liu', color='tab:green')
        Model = 'Vid' + DA[i]
        Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet]) * 100
        ax[i].plot(shotsSet, Y[:shots], marker='^', label='Vidovic', color='tab:red')
        Model = 'Our' + DA[i]
        Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet]) * 100
        ax[i].plot(shotsSet, Y[:shots], marker='v', label='Our classifier', color='tab:blue')
        ax[i].yaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))

        ax[i].set_xlabel('repetitions')

    ax[0].set_title('LDA')
    ax[1].set_title('QDA')

    ax[0].set_ylabel('NinaPro3\naccuracy [%]')

    # ax[2, 5].legend(loc='lower center', bbox_to_anchor=(2, -0.7), ncol=5)

    fig.tight_layout(pad=0.1)
    plt.savefig("PaperFigures/accNina3.png", bbox_inches='tight', dpi=600)
    plt.show()


# %%
def uploadResultsCapgmyo(place1, place2, samples, people, times, windowSize, dataset):
    for featureSet in range(1, 4):
        place = place1 + dataset
        try:
            resultsTest = pd.read_csv(
                place + "_FeatureSet_" + str(featureSet) + "_startPerson_" + str(1) + "_endPerson_" + str(
                    1) + '_windowSize_' + windowSize + ".csv")
            if len(resultsTest) != samples * times:
                print('1 error ' + str(featureSet) + ' ' + str(1))
                print(len(resultsTest), '/' + str(samples * times))

            for per in range(2, people + 1):
                try:
                    auxFrame = pd.read_csv(
                        place + "_FeatureSet_" + str(featureSet) + "_startPerson_" + str(per) + "_endPerson_" + str(
                            per) + '_windowSize_' + windowSize + ".csv")
                    resultsTest = pd.concat([resultsTest, auxFrame], ignore_index=True)
                    if len(auxFrame) != samples * times:
                        print('1 error ' + str(featureSet) + ' ' + str(per))
                        print(len(auxFrame), '/' + str(samples * times))
                except:
                    print('1 Not Found ' + str(featureSet) + ' ' + str(per))

            # place = place2 + dataset
            # for per in range(1, people + 1):
            #     try:
            #         auxFrame = pd.read_csv(
            #             place + "_FeatureSet_" + str(featureSet) + "_startPerson_" + str(per) + "_endPerson_" + str(
            #                 per) + '_windowSize_' + windowSize + ".csv")
            #         resultsTest = pd.concat([resultsTest, auxFrame], ignore_index=True)
            #         if len(auxFrame) != samples * times:
            #             print('2 error ' + str(featureSet) + ' ' + str(per))
            #             print(len(auxFrame), '/' + str(samples * times))
            #     except:
            #         print('2 Not Found ' + str(featureSet) + ' ' + str(per))

            resultsTest = resultsTest.drop(columns='Unnamed: 0')
            if dataset == 'Capgmyo_dba':
                resultsTest['# shots'] = resultsTest['# shots'] / 16
                resultsTest = resultsTest.iloc[::2]
            elif dataset == 'Capgmyo_dbc' or dataset == 'Nina1':
                resultsTest['# shots'] = resultsTest['# shots'] / 24
                resultsTest = resultsTest.iloc[::2]
            elif dataset == 'Nina3':
                resultsTest['# shots'] = resultsTest['# shots'] / 36
                resultsTest = resultsTest.iloc[::2]
            resultsTest.to_csv(
                '../Experiment1/results/' + dataset + "_FeatureSet_" + str(featureSet) + "_startPerson_" + str(
                    1) + "_endPerson_" + str(
                    people) + '_windowSize_' + windowSize + ".csv")

        except:
            print(dataset, featureSet, 'error databse')


# windowSize = '295'
# place1 = '../Experiment1/results2/'
# place2 = '../Experiment1/results2/2_'
#
# dataset = 'Capgmyo_dba'
# print(dataset)
# samples = 9
# people = 18
# times = 2
# uploadResultsCapgmyo(place1, place2, samples, people, times, windowSize, dataset)
#
# dataset = 'Capgmyo_dbc'
# print(dataset)
# samples = 9
# people = 10
# times = 2
# uploadResultsCapgmyo(place1, place2, samples, people, times, windowSize, dataset)
#
# dataset = 'Nina1'
# print(dataset)
# samples = 9
# people = 27
# times = 2
# uploadResultsCapgmyo(place1, place2, samples, people, times, windowSize, dataset)
#
# dataset = 'Nina3'
# print(dataset)
# samples = 5
# people = 9
# times = 2
# uploadResultsCapgmyo(place1, place2, samples, people, times, windowSize, dataset)
#
# dataset = 'Cote'
# print(dataset)
# samples = 4
# people = 17
# times = 1
# uploadResultsCapgmyo(place1, place2, samples, people, times, windowSize, dataset)
#
# dataset = 'EPN'
# print(dataset)
# samples = 25
# people = 30
# times = 1
# uploadResultsCapgmyo(place1, place2, samples, people, times, windowSize, dataset)
#
# dataset = 'Nina5'
# print(dataset)
# samples = 4
# people = 10
# times = 1
# uploadResultsCapgmyo(place1, place2, samples, people, times, windowSize, dataset)

#
#
# windowSize = '100'
#
# dataset = 'Capgmyo_dba'
# print(dataset)
# samples = 9
# people = 18
# times = 2
# uploadResultsCapgmyo(place1, place2, samples, people, times, windowSize, dataset)
#
# dataset = 'Capgmyo_dbc'
# print(dataset)
# samples = 9
# people = 10
# times = 2
# uploadResultsCapgmyo(place1, place2, samples, people, times, windowSize, dataset)
#
# windowSize = '280'
# dataset = 'Nina1'
# print(dataset)
# samples = 9
# people = 27
# times = 2
# uploadResultsCapgmyo(place1, place2, samples, people, times, windowSize, dataset)
#

# place = '../Experiment1/results/'
# windowSize = '295'
#
# database = 'Nina5'
# resultsNina5, timeNina5 = uploadResultsDatabases(place, database, windowSize)
#
# database = 'Cote'
# resultsCote, timeCote = uploadResultsDatabases(place, database, windowSize)
#
# database = 'EPN'
# resultsEPN, timeEPN = uploadResultsDatabases(place, database, windowSize)
#
# database = 'Capgmyo_dba'
# resultsCapgmyoA, timeCapgmyoA = uploadResultsDatabases(place, database, windowSize)
#
# database = 'Capgmyo_dbc'
# resultsCapgmyoC, timeCapgmyoC = uploadResultsDatabases(place, database, windowSize)
#
# database = 'Nina1'
# resultsNina1, timeNina1 = uploadResultsDatabases(place, database, windowSize)
#
# database = 'Nina3'
# resultsNina3, timeNina3 = uploadResultsDatabases(place, database, windowSize)

# graphACC(resultsNina5, resultsCote, resultsEPN, resultsCapgmyoA, resultsCapgmyoC, resultsNina1)
#
# AnalysisFriedman(place, windowSize)

# largeDatabase(resultsEPN)
#
# graphACCNina3(resultsNina3)

