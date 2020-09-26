import numpy as np
import pandas as pd
import Experiments.Experiment1.VisualizationFunctions as VF1
from scipy import stats
import ast

def AnalysisCote(placeOur,placeCote):
    bases = ['NinaPro5', 'Cote', 'EPN']
    confidence = 0.05
    results = pd.DataFrame()
    featureSet = 1
    idxD = 0
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
            elif base == 'EPN':
                for i in range(20):
                    coteResults.append(ast.literal_eval(cote.loc[0][i]))
                coteResults = np.mean(np.array(coteResults), axis=0)
            elif base == 'Cote':
                for i in range(20):
                    coteResults.append(ast.literal_eval(cote.loc[0][i]))
                    coteResults.append(ast.literal_eval(cote.loc[1][i]))
                coteResults = np.mean(np.array(coteResults), axis=0)
            place = placeOur+ base
            DataFrame = VF1.uploadResults(place, samples, people)

            OurResultsQDA = DataFrame['AccQDAProp'].loc[
                        (DataFrame['Feature Set'] == featureSet) & (DataFrame['# shots'] == s)].values * 100


            OurQDAMedian = np.median(OurResultsQDA)
            CoteMedian = np.median(coteResults)

            WilcoxonMethod = 'wilcox'
            alternativeMethod = 'greater'

            results.at['Our Interface\'s accuracy [%] over database: ' + base, idx] = round(OurQDAMedian, 2)
            results.at['Cote Interface\'s accuracy [%] over database: ' + base, idx] = round(CoteMedian, 2)
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
