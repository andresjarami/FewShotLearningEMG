# %% Libraries
import Experiments.Experiment1.DA_BasedAdaptiveModels as adaptive
import Experiments.Experiment1.DA_Classifiers as DA_Classifiers
import Experiments.Experiment2.VisualizationFunctions as VF2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


# %% Generator of Synthetic data
def DataGenerator_TwoCL_TwoFeat(seed=None, samples=100, people=5, peopleSame=0, Graph=True):
    global colors_list
    peopleDiff = people - (peopleSame + 1)

    classes = 2
    Features = 2
    meansDistance = 2
    np.random.seed(seed)
    random.seed(seed)

    meanSet = np.zeros([classes, Features])
    meanSet[0, :] = np.array([0, 0])
    meanSet[1, :] = np.array([meansDistance, 0])

    DataFrame = pd.DataFrame(columns=['data', 'mean', 'cov', 'person', 'class'])
    idx = 0
    if Graph:
        colors_list = list(['blue', 'red', 'deepskyblue', 'lightcoral', 'orange', 'green'])
        fig1, ax = plt.subplots(1, 1, figsize=(9, 9))

    for person in range(people):

        covSet = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        if person == people - 1:
            auxPoint = np.zeros(2)
            covSet = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])

        else:
            if person < peopleDiff:
                classCovFactor = 10
                auxPoint = np.random.uniform(low=5, high=classCovFactor * 2 + 5, size=(1, Features))[0]

            else:
                auxPoint = np.zeros(2)

        for cl in range(classes):

            DataFrame.at[idx, 'data'] = np.random.multivariate_normal(
                meanSet[cl, :] + auxPoint, covSet[(cl + 1) * Features - Features:(cl + 1) * Features, 0:Features],
                samples).T
            DataFrame.at[idx, 'mean'] = np.mean(DataFrame.loc[idx, 'data'], axis=1)
            DataFrame.at[idx, 'cov'] = np.cov(DataFrame.loc[idx, 'data'])

            DataFrame.at[idx, 'class'] = cl
            DataFrame.at[idx, 'person'] = person
            if Graph:
                x1 = DataFrame.loc[idx, 'data'][0, :]
                x2 = DataFrame.loc[idx, 'data'][1, :]

            if Graph:
                sizeM = 7

                if person == people - 1:
                    color = colors_list[cl]
                    label = 'clase ' + str(cl) + ' Target person'
                    if cl == 0:
                        markerCL = 'o'
                    else:
                        markerCL = '^'
                    ax.scatter(x1, x2, s=sizeM, color=color, marker=markerCL)
                    VF2.confidence_ellipse(x1, x2, ax, edgecolor=color, label=label)
                else:
                    if person < peopleDiff:
                        color = colors_list[cl + 4]
                        label = 'clase ' + str(cl) + ' PK: ' + str(
                            people - 1 - peopleSame) + ' people different to the target user'
                        lineStyle = '-.'
                        if cl == 0:
                            markerCL = 's'
                        else:
                            markerCL = 'p'
                    else:
                        color = colors_list[cl + 2]

                        label = 'clase ' + str(cl) + ' PK: ' + str(peopleSame) + ' person similar to the target user'
                        lineStyle = ':'
                        if cl == 0:
                            markerCL = '*'
                        else:
                            markerCL = 'x'

                if person != people - 1:
                    if person == people - 2 or person == peopleDiff - 1:
                        ax.scatter(x1, x2, s=sizeM, color=color, marker=markerCL)
                        VF2.confidence_ellipse(x1, x2, ax, edgecolor=color, label=label, linestyle=lineStyle)
                    else:
                        ax.scatter(x1, x2, s=sizeM, color=color, marker=markerCL)
                        VF2.confidence_ellipse(x1, x2, ax, edgecolor=color, linestyle=lineStyle)

            idx += 1

    if Graph:
        ax.set_title(
            'A Traget User and Two Source Users \n (' + str(classes) + ' Clases - ' + str(Features) + ' Features)')

        plt.grid()
        plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0., prop={'size': 8})
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        fig1.tight_layout(pad=0.1)
        # plt.savefig("distr.png", bbox_inches='tight', dpi=600)
        plt.show()
    return DataFrame


# %% Analysis of Synthetic data
def ResultsSyntheticData(DataFrame, nameFile, numberShots=30, peoplePK=0, samples=500, Features=2,
                         classes=2, times=10, Graph=False, printValues=False):
    iSample = 3
    people = 1

    resultsData = pd.DataFrame()

    idx = 0

    per = peoplePK

    currentPerson = DataFrame[DataFrame['person'] == per].reset_index(drop=True)
    lenTest = int(samples / 2)
    preTrainedDataMatrix = DataFrame[DataFrame['person'] != per].reset_index(drop=True)
    preTrainedDataMatrix.at[:, 'class'] = preTrainedDataMatrix.loc[:, 'class'] + 1

    currentPersonTrain = pd.DataFrame(columns=['data', 'class'])
    currentValues = pd.DataFrame(columns=['mean', 'cov', 'class'])
    pkValues = pd.DataFrame(columns=['mean', 'cov', 'class'])

    x_test = currentPerson.loc[0, 'data'][0:Features, 0:lenTest]
    y_test = np.ones(lenTest)

    x_PK = np.empty((2, 0))
    y_PK = []
    for cla in range(1, 3):
        for person in range(peoplePK):
            auxPre = preTrainedDataMatrix[
                (preTrainedDataMatrix['person'] == person) & (preTrainedDataMatrix['class'] == cla)][
                'data'].reset_index(drop=True)
            x_PK = np.hstack((x_PK, auxPre[0]))
            y_PK = np.hstack((y_PK, cla * np.ones(samples * people)))

    currentPersonTrain.at[0, 'data'] = currentPerson.loc[0, 'data'][0:Features, lenTest:samples]
    currentPersonTrain.at[0, 'class'] = 1
    for cl in range(1, classes):
        currentPersonTrain.at[cl, 'data'] = currentPerson.loc[cl, 'data'][0:Features, lenTest:samples]
        currentPersonTrain.at[cl, 'class'] = cl + 1
        x_test = np.hstack((x_test, currentPerson.loc[cl, 'data'][0:Features, 0:lenTest]))
        y_test = np.hstack((y_test, np.ones(lenTest) * cl + 1))
    x_test = x_test.T

    # for t in range(times): (for spliting the task)
    t = times

    for i in range(numberShots):
        x_train = currentPersonTrain.loc[0, 'data'][:, t:i + t + iSample]
        y_train = np.ones(i + iSample)
        currentValues.at[0, 'mean'] = np.mean(x_train, axis=1)
        currentValues.at[0, 'cov'] = np.cov(x_train, rowvar=True)
        currentValues.at[0, 'class'] = 1

        for cl in range(1, classes):
            x_train = np.hstack((x_train, currentPersonTrain.loc[cl, 'data'][:, t:i + t + iSample]))
            y_train = np.hstack((y_train, np.ones(i + iSample) * cl + 1))
            currentValues.at[cl, 'mean'] = np.mean(currentPersonTrain.loc[cl, 'data'][:, t:i + t + iSample], axis=1)
            currentValues.at[cl, 'cov'] = np.cov(currentPersonTrain.loc[cl, 'data'][:, t:i + t + iSample], rowvar=True)
            currentValues.at[cl, 'class'] = cl + 1

        x_Multi = np.hstack((x_PK, x_train))
        y_Multi = np.hstack((y_PK, y_train))
        for cl in range(classes):
            pkValues.at[cl, 'mean'] = np.mean(x_Multi[:, y_Multi == cl + 1], axis=1)
            pkValues.at[cl, 'cov'] = np.cov(x_Multi[:, y_Multi == cl + 1], rowvar=True)
            pkValues.at[cl, 'class'] = cl + 1

        resultsData.at[idx, 'person'] = per
        resultsData.at[idx, 'times'] = t
        resultsData.at[idx, 'shots'] = i + iSample

        step = 1
        k = 1 - (np.log(i + 1) / np.log(samples + 1))

        propModelLDA, _, resultsData.at[idx, 'wTargetMeanLDA'], _, resultsData.at[idx, 'wTargetCovLDA'], resultsData.at[
            idx, 'tPropLDA'] = adaptive.OurModel(currentValues, preTrainedDataMatrix, classes, Features, x_train.T,
                                                 y_train, step, 'LDA', k)
        propModelQDA, _, resultsData.at[idx, 'wTargetMeanQDA'], _, resultsData.at[idx, 'wTargetCovQDA'], resultsData.at[
            idx, 'tPropQDA'] = adaptive.OurModel(currentValues, preTrainedDataMatrix, classes, Features, x_train.T,
                                                 y_train, step, 'QDA', k)

        liuModel = adaptive.LiuModel(currentValues, preTrainedDataMatrix, classes, Features)
        vidovicModelL, vidovicModelQ = adaptive.VidovicModel(currentValues, preTrainedDataMatrix, classes, Features)

        resultsData.at[idx, 'AccLDAInd'], _ = DA_Classifiers.accuracyModelLDA(x_test, y_test, currentValues, classes)
        resultsData.at[idx, 'AccQDAInd'], _ = DA_Classifiers.accuracyModelQDA(x_test, y_test, currentValues, classes)
        resultsData.at[idx, 'AccLDAMulti'], _ = DA_Classifiers.accuracyModelLDA(x_test, y_test, pkValues, classes)
        resultsData.at[idx, 'AccQDAMulti'], _ = DA_Classifiers.accuracyModelQDA(x_test, y_test, pkValues, classes)
        resultsData.at[idx, 'AccLDALiu'], _ = DA_Classifiers.accuracyModelLDA(x_test, y_test, liuModel, classes)
        resultsData.at[idx, 'AccQDALiu'], _ = DA_Classifiers.accuracyModelQDA(x_test, y_test, liuModel, classes)
        resultsData.at[idx, 'AccLDAVidovic'], _ = DA_Classifiers.accuracyModelLDA(x_test, y_test, vidovicModelL,
                                                                                  classes)
        resultsData.at[idx, 'AccQDAVidovic'], _ = DA_Classifiers.accuracyModelQDA(x_test, y_test, vidovicModelQ,
                                                                                  classes)
        resultsData.at[idx, 'AccLDAProp'], _ = DA_Classifiers.accuracyModelLDA(x_test, y_test, propModelLDA, classes)
        resultsData.at[idx, 'AccQDAProp'], _ = DA_Classifiers.accuracyModelQDA(x_test, y_test, propModelQDA, classes)

        if nameFile is not None:
            resultsData.to_csv(nameFile)
        idx += 1
        if printValues:
            print(per + 1, t + 1, i + 1)

    if Graph:
        VF2.graphSyntheticData(resultsData, numberShots, iSample)
