import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_spd_matrix
import Experiments.DatabasesEvaluation.functions as F
import sys
import random
from sklearn.manifold import TSNE
import time


def DataGenerator(classes=2, Features=2, seed=None, samples=100, people=10, Graph=True, covRandom=True, meanRandom=True,
                  nameFile=None, mix=False):
    classCovFactor = 1
    vU = 1
    np.random.seed(seed)
    random.seed(seed)

    covSet = make_spd_matrix(Features)
    meanSet = np.zeros([classes, Features])
    for cl in range(classes):
        auxMeanSet = []
        for fea in range(Features):
            auxMeanSet = np.hstack([auxMeanSet, random.uniform(-vU, vU)])
        meanSet[cl, :] = auxMeanSet

    DataFrame = pd.DataFrame(columns=['data', 'mean', 'cov', 'person', 'class'])
    idx = 0
    if Graph:
        colors_list = list(['blue', 'red', 'green', 'orange', 'gray', 'brown', 'yellow'])
        fig1, ax = plt.subplots(1, 1, figsize=(8, 8))

    for person in range(people):
        if mix and person < (people / 2):
            classCovFactor = 10
        else:
            classCovFactor = 1
        auxPoint = np.random.uniform(low=-classCovFactor, high=classCovFactor, size=(1, Features))
        # auxPoint = np.random.multivariate_normal(np.zeros(Features), make_spd_matrix(Features) * classCovFactor,1)
        for cl in range(classes):

            if meanRandom:
                auxMean = []
                for fea in range(Features):
                    auxMean = np.hstack([auxMean, random.uniform(-vU, vU)])
            else:
                auxMean = meanSet[cl, :]

            DataFrame.at[idx, 'mean'] = auxPoint[0] + auxMean
            if covRandom:
                DataFrame.at[idx, 'cov'] = make_spd_matrix(Features)
            else:
                DataFrame.at[idx, 'cov'] = covSet
            DataFrame.at[idx, 'data'] = np.random.multivariate_normal(DataFrame.loc[idx, 'mean'],
                                                                      DataFrame.loc[idx, 'cov'],
                                                                      samples).T
            DataFrame.at[idx, 'class'] = cl
            DataFrame.at[idx, 'person'] = person
            if Graph and Features == 2:
                x1 = DataFrame.loc[idx, 'data'][0, :]
                x2 = DataFrame.loc[idx, 'data'][1, :]
            elif Graph and Features >= 3:

                xTSNE = TSNE(n_components=2).fit_transform(DataFrame.loc[idx, 'data'].T)
                x1 = xTSNE[:, 0] + auxPoint[0, 0] * 5
                x2 = xTSNE[:, 1] + auxPoint[0, 1] * 5

            if Graph:
                if mix and person < (people / 2):
                    color = colors_list[cl + 2]
                    label = 'clase ' + str(cl) + ' Second Distribution'
                else:
                    color = colors_list[cl]
                    label = 'clase ' + str(cl) + ' First Distribution'
                if person == people - 1 or (mix and (person == people / 2 - 1)):
                    ax.scatter(x1, x2, s=0.5, color=color)
                    F.confidence_ellipse(x1, x2, ax, edgecolor=color, label=label)
                else:
                    ax.scatter(x1, x2, s=0.5, color=color)
                    F.confidence_ellipse(x1, x2, ax, edgecolor=color)

            idx += 1

    if Graph and Features == 2:
        ax.set_title(str(classes) + ' Clases - ' + str(people) + ' People - ' + str(Features) + 'Features')
        plt.grid()
        plt.legend(loc='best', prop={'size': 7})
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
    elif Graph and Features >= 3:
        ax.set_title('T-SNE: ' + str(classes) + ' Clases - ' + str(people) + ' People - ' + str(Features) + 'Features')
        plt.grid()
        plt.legend(loc='best', prop={'size': 7})
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.show()

    # time.sleep(10)
    # plt.close("all")
    if nameFile is not None:
        DataFrame.to_csv(nameFile + '.csv', index=False)
        DataFrame.to_pickle(nameFile + '.pkl')
    return DataFrame


def DataGenerator_TwoCL_TwoFeat(seed=None, samples=100, people=5, Graph=True, covRandom=True, nameFile=None,
                                mix=False):
    classes = 2
    Features = 2
    meansDistance = 2
    np.random.seed(seed)
    random.seed(seed)

    if covRandom:
        covSet1 = make_spd_matrix(Features)
        covSet2 = make_spd_matrix(Features)
        covSet = np.vstack((covSet1, covSet2))
    else:
        covSet = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    meanSet = np.zeros([classes, Features])
    meanSet[0, :] = np.array([0, 0])
    meanSet[1, :] = np.array([meansDistance, 0])

    DataFrame = pd.DataFrame(columns=['data', 'mean', 'cov', 'person', 'class'])
    idx = 0
    if Graph:
        colors_list = list(['blue', 'red', 'green', 'orange', 'gray', 'brown', 'yellow'])
        fig1, ax = plt.subplots(1, 1, figsize=(8, 8))

    for person in range(people):
        if mix and person < (people / 2):
            classCovFactor = 10
            auxPoint = np.random.uniform(low=-classCovFactor, high=classCovFactor, size=(1, Features))[0]
        else:
            auxPoint = np.zeros(2)

        for cl in range(classes):

            DataFrame.at[idx, 'data'] = np.random.multivariate_normal(meanSet[cl, :] + auxPoint,
                                                                      covSet[(cl + 1) * Features -
                                                                             Features:(cl + 1) * Features,
                                                                      0:Features], samples).T
            DataFrame.at[idx, 'mean'] = np.mean(DataFrame.loc[idx, 'data'], axis=1)
            DataFrame.at[idx, 'cov'] = np.cov(DataFrame.loc[idx, 'data'])

            DataFrame.at[idx, 'class'] = cl
            DataFrame.at[idx, 'person'] = person
            if Graph:
                x1 = DataFrame.loc[idx, 'data'][0, :]
                x2 = DataFrame.loc[idx, 'data'][1, :]

            if Graph:
                if mix and person < (people / 2):
                    color = colors_list[cl + 2]
                    label = 'clase ' + str(cl) + ' People with different distributions'
                else:
                    color = colors_list[cl]
                    label = 'clase ' + str(cl) + ' People with the same distribution'
                if person == people - 1 or (mix and (person == people / 2 - 1)):
                    ax.scatter(x1, x2, s=0.5, color=color)
                    F.confidence_ellipse(x1, x2, ax, edgecolor=color, label=label)
                else:
                    ax.scatter(x1, x2, s=0.5, color=color)
                    F.confidence_ellipse(x1, x2, ax, edgecolor=color)

            idx += 1

    if Graph:
        ax.set_title(str(classes) + ' Clases - ' + str(people) + ' People - ' + str(Features) + 'Features')
        plt.grid()
        plt.legend(loc='best', prop={'size': 7})
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

    if nameFile is not None:
        DataFrame.to_pickle(nameFile + '.pkl')
    return DataFrame


# classes = int(sys.argv[1])
# Features = int(sys.argv[2])
# seed = int(sys.argv[3])
# samples = int(sys.argv[4])
# people = int(sys.argv[5])
# shots = int(sys.argv[6])
# Graph = bool(int(sys.argv[7]))
# covRandom = bool(int(sys.argv[8]))
# meanRandom = bool(int(sys.argv[9]))
# times = int(sys.argv[10])


classes = 2
Features = 2
seed = 1
samples = 100
people = 10
shots = 50
Graph = True
covRandom = False
meanRandom = False
times = 100
per = 78
t = 20

nameFile = None
mix = True

DataGenerator1(classes, Features, seed, samples, people, Graph, covRandom, meanRandom, nameFile, mix)
