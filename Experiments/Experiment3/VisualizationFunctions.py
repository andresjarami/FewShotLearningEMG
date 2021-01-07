import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import Experiments.Experiment1.VisualizationFunctions as VF1
from scipy import stats
import ast

def AnalysisCote(placeOur,placeCote):
    bases = ['NinaPro5', 'Cote', 'EPN']
    confidence = 0.05
    results = pd.DataFrame()
    featureSet = 1
    idxD = 0
    CoteAllardExperiment_Repetitions=20
    for base in bases:
        if base == 'NinaPro5':
            samples = 4
            people = 10
            shots = 5
        elif base == 'Cote':
            samples = 4
            people = 17
            shots = 5

        elif base == 'EPN':
            samples = 25
            people = 30
            shots = 5
        idx = 0

        for s in range(1, shots):

            place = placeCote+"Cote_CWT_" + base + "/"
            cote = pd.read_csv(place + "Pytorch_results_" + str(s) + "_cycles.csv", header=None)
            coteResults = []

            if base == 'NinaPro5':
                for i in range(people):
                    coteResults.append(np.array(ast.literal_eval(cote.loc[0][i])).T[0].mean())
                coteResults = np.array(coteResults)
                place = placeOur + 'Nina5'
            elif base == 'EPN':
                for i in range(CoteAllardExperiment_Repetitions):
                    coteResults.append(ast.literal_eval(cote.loc[0][i]))
                coteResults = np.mean(np.array(coteResults), axis=0)
                place = placeOur + base
            elif base == 'Cote':
                for i in range(CoteAllardExperiment_Repetitions):
                    coteResults.append(ast.literal_eval(cote.loc[0][i]))
                    coteResults.append(ast.literal_eval(cote.loc[1][i]))
                coteResults = np.mean(np.array(coteResults), axis=0)
                place = placeOur+ base
            DataFrame = VF1.uploadResults2(place, samples, people)

            OurResultsQDA = DataFrame['AccQDAProp'].loc[
                        (DataFrame['Feature Set'] == featureSet) & (DataFrame['# shots'] == s)].values * 100
            OurResultsLDA = DataFrame['AccLDAProp'].loc[
                                (DataFrame['Feature Set'] == featureSet) & (DataFrame['# shots'] == s)].values * 100


            OurQDAMedian = np.mean(OurResultsQDA)
            CoteMedian = np.mean(coteResults)

            WilcoxonMethod = 'wilcox'
            alternativeMethod = 'greater'

            results.at['Our Interface\'s accuracy [%] over database: ' + base, idx] = np.round(OurQDAMedian, 2)
            results.at['Cote Interface\'s accuracy [%] over database: ' + base, idx] = np.round(CoteMedian, 2)
            results.at['Difference [%] over database: ' + base, idx] = round(OurQDAMedian - CoteMedian, 2)
            results.at['p-value over database: ' + base, idx] = 1
            if OurQDAMedian > CoteMedian:
                p = stats.wilcoxon(OurResultsQDA, coteResults, alternative=alternativeMethod, zero_method=WilcoxonMethod)[1]
                if p < confidence:
                    results.at['p-value over database: ' + base, idx] = p
            elif CoteMedian > OurQDAMedian:
                p = stats.wilcoxon(coteResults, OurResultsQDA, alternative=alternativeMethod, zero_method=WilcoxonMethod)[1]
                if p < confidence:
                    results.at['p-value over database: ' + base, idx] = p

            idx += 1
        idxD += 2
    return results


def graphACC(resultsNina5T, resultsCoteT, resultsEPNT,coteFolder):
    FeatureSetM = 3
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(13, 6))
    #     shot=np.arange(1,5)
    shots=4

    databases=['NinaPro5','EPN','Cote']
    people=[10,30,20]
    for Data in range(FeatureSetM):
        base=databases[Data]
        place = coteFolder + "Cote_CWT_" + base + "/"
        coteY=np.zeros(shots)
        for s in range(1,shots+1):
            cote = pd.read_csv(place + "Pytorch_results_" + str(s) + "_cycles.csv", header=None)
            coteResults = []
            if base == 'NinaPro5':
                for i in range(people[Data]):
                    coteResults.append(np.array(ast.literal_eval(cote.loc[0][i])).T[0].mean())

                coteResults = np.array(coteResults)
            elif base == 'EPN':
                for i in range(people[Data]):
                    coteResults.append(ast.literal_eval(cote.loc[0][i]))

                coteResults = np.mean(np.array(coteResults), axis=0)
            elif base == 'Cote':
                for i in range(people[Data]):
                    coteResults.append(ast.literal_eval(cote.loc[0][i]))
                    coteResults.append(ast.literal_eval(cote.loc[1][i]))
                coteResults = np.mean(np.array(coteResults), axis=0)
            coteY[s-1]=np.mean(coteResults)
        for classifier in range(2):
            for FeatureSet in range(FeatureSetM):

                if Data == 0:
                    shot = np.arange(1, 5)
                    results = resultsNina5T

                    ax[Data, FeatureSet].yaxis.set_ticks(np.arange(30, 100, 6))
                elif Data == 1:
                    shot = np.arange(1, 5)
                    results = resultsCoteT

                    ax[Data, FeatureSet].yaxis.set_ticks(np.arange(74, 100, 5))
                elif Data == 2:
                    results = resultsEPNT
                    shot = np.arange(1, 5)

                    ax[Data, FeatureSet].yaxis.set_ticks(np.arange(50, 100, 6))

                if classifier == 0:
                    Model = 'LDA_Ind'

                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1]) * 100
                    ax[Data, FeatureSet].scatter(shot, Y[:len(shot)], marker='x', label='Individual',
                                                 color='tab:orange')

                    Model = 'LDA_Multi'

                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1]) * 100
                    ax[Data, FeatureSet].scatter(shot, Y[:len(shot)], marker='v', label='Multi-user',
                                                 color='tab:purple')

                    Model = 'LiuL'

                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1]) * 100
                    ax[Data, FeatureSet].scatter(shot, Y[:len(shot)], marker='o', label='Liu',
                                                 color='tab:green')

                    Model = 'VidL'

                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1]) * 100
                    ax[Data, FeatureSet].scatter(shot, Y[:len(shot)], marker='^', label='Vidovic',
                                                 color='tab:red')

                    Model = 'PropQ_L'
                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1]) * 100
                    ax[Data, FeatureSet].plot(shot, Y[:len(shot)], label='Our technique', color='tab:blue')

                    ax[Data, FeatureSet].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))

                    if len(shot) == 25:
                        ax[Data, FeatureSet].xaxis.set_ticks([1, 5, 10, 15, 20, 25])
                    else:

                        ax[Data, FeatureSet].xaxis.set_ticks(np.arange(1, len(shot) + .2, 1))
                    ax[Data, FeatureSet].grid()






                elif classifier == 1:

                    Model = 'QDA_Ind'

                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1]) * 100
                    ax[Data, FeatureSet + 3].scatter(shot, Y[:len(shot)], marker='x', label='Individual',
                                                     color='tab:orange')

                    Model = 'QDA_Multi'

                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1]) * 100
                    ax[Data, FeatureSet + 3].scatter(shot, Y[:len(shot)], marker='v', label='Multi-user',
                                                     color='tab:purple')

                    Model = 'LiuQ'

                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1]) * 100
                    ax[Data, FeatureSet + 3].scatter(shot, Y[:len(shot)], marker='o', label='Liu',
                                                     color='tab:green')

                    Model = 'VidQ'

                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1]) * 100
                    ax[Data, FeatureSet + 3].scatter(shot, Y[:len(shot)], marker='^', label='Vidovic',
                                                     color='tab:red')

                    Model = 'PropQ'

                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1]) * 100

                    ax[Data, FeatureSet + 3].plot(shot, Y[:len(shot)], label='Our technique', color='tab:blue')

                    ax[Data, FeatureSet + 3].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))

                    if len(shot) == 25:
                        ax[Data, FeatureSet + 3].xaxis.set_ticks([1, 5, 10, 15, 20, 25])
                    else:
                        ax[Data, FeatureSet + 3].xaxis.set_ticks(np.arange(1, len(shot) + .2, 1))
                    ax[Data, FeatureSet + 3].grid()



    ax[2, 0].set_xlabel('repetitions')
    ax[2, 1].set_xlabel('repetitions')
    ax[2, 2].set_xlabel('repetitions')
    ax[2, 3].set_xlabel('repetitions')
    ax[2, 4].set_xlabel('repetitions')
    ax[2, 5].set_xlabel('repetitions')

    ax[0, 0].set_title('LDA\n Feature Set 1')
    ax[0, 1].set_title('LDA\n Feature Set 2')
    ax[0, 2].set_title('LDA\n Feature Set 3')

    ax[0, 3].set_title('QDA\n Feature Set 1')
    ax[0, 4].set_title('QDA\n Feature Set 2')
    ax[0, 5].set_title('QDA\n Feature Set 3')
    ax[0, 0].set_ylabel('NinaPro5\naccuracy')
    ax[1, 0].set_ylabel('Cote Allard\naccuracy')
    ax[2, 0].set_ylabel('EPN \naccuracy')

    ax[2, 5].legend(loc='lower center', bbox_to_anchor=(2, -0.7), ncol=5)
    # ax2.legend(loc='lower center', bbox_to_anchor=(2, -1.2), ncol=5)

    fig.tight_layout(pad=0.1)
    plt.savefig("fig1.png", bbox_inches='tight', dpi=600)
    plt.show()

