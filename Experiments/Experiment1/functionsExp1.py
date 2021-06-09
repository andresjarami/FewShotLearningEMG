# %% Libraries
import DA_BasedAdaptiveModels as adaptive
import DA_Classifiers as DA_Classifiers
import numpy as np
import pandas as pd
import time
import math
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# %% Upload Databases

def uploadDatabases(Database, featureSet, windowSize):
    # Setting general variables
    path = '../../'

    if Database == 'EPN':
        classes = 5
        peoplePriorK = 30
        peopleTest = 30
        numberShots = 50
        CH = 8
    elif Database == 'Nina5':
        classes = 18
        peoplePriorK = 10
        peopleTest = 10
        numberShots = 6
        CH = 8
    elif Database == 'Nina3':
        classes = 18
        peoplePriorK = 9
        peopleTest = 9
        numberShots = 6
        CH = 12
    elif Database == 'Nina1':
        classes = 12
        peoplePriorK = 27
        peopleTest = 27
        numberShots = 10
        CH = 10
    elif Database == 'Cote':
        classes = 7
        peoplePriorK = 19
        peopleTest = 17
        numberShots = 4
        CH = 8
    elif Database == 'Capgmyo_dba':
        classes = 8
        peoplePriorK = 18
        peopleTest = 18
        numberShots = 10
        CH = 128
    elif Database == 'Capgmyo_dbc':
        classes = 12
        peoplePriorK = 10
        peopleTest = 10
        numberShots = 10
        CH = 128
    combinationSet = list(range(1, numberShots + 1))

    if featureSet == 1:
        # Setting variables
        Feature1 = 'logvarMatrix'
        numberFeatures = 1
        allFeatures = numberFeatures * CH
        # Getting Data
        logvarMatrix = np.genfromtxt(path + 'ExtractedData/' + Database + '/' + Feature1 + windowSize + '.csv',
                                     delimiter=',')
        if Database == 'EPN':
            dataMatrix = logvarMatrix[:, 0:]
            labelsDataMatrix = dataMatrix[:, allFeatures + 2]
        elif Database == 'Nina5':
            dataMatrix = logvarMatrix[:, 8:]
            labelsDataMatrix = dataMatrix[:, allFeatures + 1]
        elif Database == 'Cote':
            dataMatrix = logvarMatrix[:, 0:]
            dataMatrix[:, allFeatures + 1] = dataMatrix[:, allFeatures + 1] + 1
            dataMatrix[:, allFeatures + 3] = dataMatrix[:, allFeatures + 3] + 1
            labelsDataMatrix = dataMatrix[:, allFeatures + 3]
        elif Database == 'Capgmyo_dba' or Database == 'Capgmyo_dbc' or Database == 'Nina3' or Database == 'Nina1':
            dataMatrix = logvarMatrix[:, :]
            labelsDataMatrix = dataMatrix[:, allFeatures + 1]

    elif featureSet == 2:
        # Setting variables
        Feature1 = 'mavMatrix'
        Feature2 = 'wlMatrix'
        Feature3 = 'zcMatrix'
        Feature4 = 'sscMatrix'
        numberFeatures = 4
        allFeatures = numberFeatures * CH
        mavMatrix = np.genfromtxt(path + 'ExtractedData/' + Database + '/' + Feature1 + windowSize + '.csv',
                                  delimiter=',')
        wlMatrix = np.genfromtxt(path + 'ExtractedData/' + Database + '/' + Feature2 + windowSize + '.csv',
                                 delimiter=',')
        zcMatrix = np.genfromtxt(path + 'ExtractedData/' + Database + '/' + Feature3 + windowSize + '.csv',
                                 delimiter=',')
        sscMatrix = np.genfromtxt(path + 'ExtractedData/' + Database + '/' + Feature4 + windowSize + '.csv',
                                  delimiter=',')
        if Database == 'EPN':
            dataMatrix = np.hstack((mavMatrix[:, 0:CH], wlMatrix[:, 0:CH], zcMatrix[:, 0:CH], sscMatrix[:, 0:]))
            labelsDataMatrix = dataMatrix[:, allFeatures + 2]

        elif Database == 'Nina5':
            dataMatrix = np.hstack(
                (mavMatrix[:, 8:CH * 2], wlMatrix[:, 8:CH * 2], zcMatrix[:, 8:CH * 2], sscMatrix[:, 8:]))
            labelsDataMatrix = dataMatrix[:, allFeatures + 1]

        elif Database == 'Cote':
            dataMatrix = np.hstack((mavMatrix[:, 0:CH], wlMatrix[:, 0:CH], zcMatrix[:, 0:CH], sscMatrix[:, 0:]))
            dataMatrix[:, allFeatures + 1] = dataMatrix[:, allFeatures + 1] + 1
            dataMatrix[:, allFeatures + 3] = dataMatrix[:, allFeatures + 3] + 1
            labelsDataMatrix = dataMatrix[:, allFeatures + 3]
        elif Database == 'Capgmyo_dba' or Database == 'Capgmyo_dbc' or Database == 'Nina3' or Database == 'Nina1':
            dataMatrix = np.hstack((mavMatrix[:, 0:CH], wlMatrix[:, 0:CH], zcMatrix[:, 0:CH], sscMatrix[:, 0:]))
            labelsDataMatrix = dataMatrix[:, allFeatures + 1]

    elif featureSet == 3:
        # Setting variables
        Feature1 = 'lscaleMatrix'
        Feature2 = 'mflMatrix'
        Feature3 = 'msrMatrix'
        Feature4 = 'wampMatrix'
        numberFeatures = 4
        allFeatures = numberFeatures * CH
        # Getting Data
        lscaleMatrix = np.genfromtxt(path + 'ExtractedData/' + Database + '/' + Feature1 + windowSize + '.csv',
                                     delimiter=',')
        mflMatrix = np.genfromtxt(path + 'ExtractedData/' + Database + '/' + Feature2 + windowSize + '.csv',
                                  delimiter=',')
        msrMatrix = np.genfromtxt(path + 'ExtractedData/' + Database + '/' + Feature3 + windowSize + '.csv',
                                  delimiter=',')
        wampMatrix = np.genfromtxt(path + 'ExtractedData/' + Database + '/' + Feature4 + windowSize + '.csv',
                                   delimiter=',')

        if Database == 'EPN':
            dataMatrix = np.hstack((lscaleMatrix[:, 0:CH], mflMatrix[:, 0:CH], msrMatrix[:, 0:CH], wampMatrix[:, 0:]))
            labelsDataMatrix = dataMatrix[:, allFeatures + 2]

        elif Database == 'Nina5':
            dataMatrix = np.hstack(
                (lscaleMatrix[:, 8:CH * 2], mflMatrix[:, 8:CH * 2], msrMatrix[:, 8:CH * 2], wampMatrix[:, 8:]))
            labelsDataMatrix = dataMatrix[:, allFeatures + 1]

        elif Database == 'Cote':
            dataMatrix = np.hstack((lscaleMatrix[:, 0:CH], mflMatrix[:, 0:CH], msrMatrix[:, 0:CH], wampMatrix[:, 0:]))
            dataMatrix[:, allFeatures + 1] = dataMatrix[:, allFeatures + 1] + 1
            dataMatrix[:, allFeatures + 3] = dataMatrix[:, allFeatures + 3] + 1
            labelsDataMatrix = dataMatrix[:, allFeatures + 3]
        elif Database == 'Capgmyo_dba' or Database == 'Capgmyo_dbc' or Database == 'Nina3' or Database == 'Nina1':
            dataMatrix = np.hstack((lscaleMatrix[:, 0:CH], mflMatrix[:, 0:CH], msrMatrix[:, 0:CH], wampMatrix[:, 0:]))
            labelsDataMatrix = dataMatrix[:, allFeatures + 1]

    elif featureSet == 4:
        # Setting variables
        Feature1 = 'mavMatrix'
        Feature2 = 'wlMatrix'
        Feature3 = 'zcMatrix'
        Feature4 = 'sscMatrix'
        Feature5 = 'rmsMatrix'
        Feature6 = 'iavMatrix'
        Feature7 = 'dasdvMatrix'
        Feature8 = 'varMatrix'

        numberFeatures = 8
        allFeatures = numberFeatures * CH
        mavMatrix = np.genfromtxt(path + 'ExtractedData/' + Database + '/' + Feature1 + windowSize + '.csv',
                                  delimiter=',')
        wlMatrix = np.genfromtxt(path + 'ExtractedData/' + Database + '/' + Feature2 + windowSize + '.csv',
                                 delimiter=',')
        zcMatrix = np.genfromtxt(path + 'ExtractedData/' + Database + '/' + Feature3 + windowSize + '.csv',
                                 delimiter=',')
        sscMatrix = np.genfromtxt(path + 'ExtractedData/' + Database + '/' + Feature4 + windowSize + '.csv',
                                  delimiter=',')
        rmsMatrix = np.genfromtxt(path + 'ExtractedData/' + Database + '/' + Feature5 + windowSize + '.csv',
                                  delimiter=',')
        iavMatrix = np.genfromtxt(path + 'ExtractedData/' + Database + '/' + Feature6 + windowSize + '.csv',
                                  delimiter=',')
        dasdvMatrix = np.genfromtxt(path + 'ExtractedData/' + Database + '/' + Feature7 + windowSize + '.csv',
                                    delimiter=',')
        varMatrix = np.genfromtxt(path + 'ExtractedData/' + Database + '/' + Feature8 + windowSize + '.csv',
                                  delimiter=',')
        if Database == 'EPN':
            dataMatrix = np.hstack((mavMatrix[:, 0:CH], wlMatrix[:, 0:CH], zcMatrix[:, 0:CH], sscMatrix[:, 0:CH],
                                    rmsMatrix[:, 0:CH], iavMatrix[:, 0:CH], dasdvMatrix[:, 0:CH], varMatrix[:, 0:]))
            labelsDataMatrix = dataMatrix[:, allFeatures + 2]

        elif Database == 'Nina5':
            dataMatrix = np.hstack(
                (mavMatrix[:, 8:CH * 2], wlMatrix[:, 8:CH * 2], zcMatrix[:, 8:CH * 2], sscMatrix[:, 8:CH * 2],
                 rmsMatrix[:, 8:CH * 2], iavMatrix[:, 8:CH * 2], dasdvMatrix[:, 8:CH * 2], varMatrix[:, 8:]))
            labelsDataMatrix = dataMatrix[:, allFeatures + 1]

        elif Database == 'Cote':
            dataMatrix = np.hstack((mavMatrix[:, 0:CH], wlMatrix[:, 0:CH], zcMatrix[:, 0:CH], sscMatrix[:, 0:CH],
                                    rmsMatrix[:, 0:CH], iavMatrix[:, 0:CH], dasdvMatrix[:, 0:CH], varMatrix[:, 0:]))
            dataMatrix[:, allFeatures + 1] = dataMatrix[:, allFeatures + 1] + 1
            dataMatrix[:, allFeatures + 3] = dataMatrix[:, allFeatures + 3] + 1
            labelsDataMatrix = dataMatrix[:, allFeatures + 3]
        elif Database == 'Capgmyo_dba' or Database == 'Capgmyo_dbc' or Database == 'Nina3' or Database == 'Nina1':
            dataMatrix = np.hstack((mavMatrix[:, 0:CH], wlMatrix[:, 0:CH], zcMatrix[:, 0:CH], sscMatrix[:, 0:CH],
                                    rmsMatrix[:, 0:CH], iavMatrix[:, 0:CH], dasdvMatrix[:, 0:CH], varMatrix[:, 0:]))
            labelsDataMatrix = dataMatrix[:, allFeatures + 1]

    return dataMatrix, numberFeatures, CH, classes, peoplePriorK, peopleTest, numberShots, combinationSet, allFeatures, labelsDataMatrix


# %% EVALUATION OVER THE THREE DATABASES


def evaluation(dataMatrix, classes, peoplePriorK, featureSet, numberShots, nameFile, startPerson, endPerson,
               allFeatures, typeDatabase, printR):
    scaler = preprocessing.MinMaxScaler()

    if typeDatabase == 'EPN':
        evaluationEPN(dataMatrix, classes, peoplePriorK, featureSet, numberShots, nameFile, startPerson, endPerson,
                      allFeatures, printR, scaler)
    elif typeDatabase == 'Nina5':
        evaluationNina(dataMatrix, classes, peoplePriorK, featureSet, numberShots, nameFile, startPerson, endPerson,
                       allFeatures, printR, scaler)
    elif typeDatabase == 'Cote':
        evaluationCote(dataMatrix, classes, peoplePriorK, featureSet, numberShots, nameFile, startPerson, endPerson,
                       allFeatures, printR, scaler)
    elif typeDatabase == 'Capgmyo_dba' or typeDatabase == 'Capgmyo_dbc' or typeDatabase == 'Nina1' or typeDatabase == 'Nina3':
        evaluationCapgmyo(dataMatrix, classes, peoplePriorK, featureSet, numberShots, nameFile, startPerson, endPerson,
                          allFeatures, printR, scaler)


# %% Capgmyo
def evaluationCapgmyo(dataMatrix, classes, peoplePriorK, featureSet, numberShots, nameFile,
                      startPerson, endPerson, allFeatures, printR, scaler):
    # Creating Variables

    results = pd.DataFrame(
        columns=['person', 'subset', '# shots', 'Feature Set'])

    idx = 0

    for person in range(startPerson, endPerson + 1):

        trainFeaturesGenPre = dataMatrix[np.where((dataMatrix[:, allFeatures] != person)), 0:allFeatures][0]
        trainLabelsGenPre = dataMatrix[np.where((dataMatrix[:, allFeatures] != person)), allFeatures + 1][0]

        numberShots2Test = numberShots - 1
        # print(numberShots2Test)

        for shot in range(1, numberShots2Test + 1):
            for seed in range(2, 4):
                np.random.seed(seed)
                testGestures = []
                trainGestures = []

                for cla in range(1, classes + 1):
                    repetitions = np.random.choice(numberShots, numberShots, replace=False) + 1
                    testGestures += [[shot, cla] for shot in repetitions[shot:]]
                    trainGestures += [[shot, cla] for shot in repetitions[:shot]]

                trainFeatures = np.empty((0, allFeatures))
                trainLabels = []
                for rand in list(trainGestures):
                    trainFeatures = np.vstack((
                        trainFeatures, dataMatrix[(dataMatrix[:, allFeatures] == person) & (
                                dataMatrix[:, allFeatures + 1] == rand[1]) & (
                                                          dataMatrix[:, allFeatures + 2] == rand[0]),
                                       0:allFeatures]))
                    trainLabels = np.hstack((trainLabels, dataMatrix[(dataMatrix[:, allFeatures] == person) & (
                            dataMatrix[:, allFeatures + 1] == rand[1]) & (dataMatrix[:, allFeatures + 2] == rand[0]),
                                                                     allFeatures + 1].T))

                testFeatures = np.empty((0, allFeatures))
                testLabels = []
                for rand in list(testGestures):
                    testFeatures = np.vstack((
                        testFeatures, dataMatrix[(dataMatrix[:, allFeatures] == person) & (
                                dataMatrix[:, allFeatures + 1] == rand[1]) & (
                                                         dataMatrix[:, allFeatures + 2] == rand[0]),
                                      0:allFeatures]))
                    testLabels = np.hstack((testLabels, dataMatrix[(dataMatrix[:, allFeatures] == person) & (
                            dataMatrix[:, allFeatures + 1] == rand[1]) & (dataMatrix[:, allFeatures + 2] == rand[0]),
                                                                   allFeatures + 1].T))

                trainFeaturesGen = np.vstack((trainFeaturesGenPre, trainFeatures))
                trainLabelsGen = np.hstack((trainLabelsGenPre, trainLabels))

                k = 1 - (np.log(shot) / np.log(numberShots + 1))

                scaler.fit(np.vstack((trainFeatures, trainFeaturesGen)))
                trainFeatures = scaler.transform(trainFeatures)
                trainFeaturesGen = scaler.transform(trainFeaturesGen)

                # print('features before', np.size(dataMatrix[:, 0:allFeatures], axis=1))

                if np.size(dataMatrix[:, 0:allFeatures], axis=1) == 128 or np.size(dataMatrix[:, 0:allFeatures],
                                                                                   axis=1) == 4 * 128:
                    pca = PCA(n_components=0.99, svd_solver='full')
                    pca.fit(np.vstack((trainFeatures)))
                    trainFeatures = pca.transform(trainFeatures)
                    trainFeaturesGen = pca.transform(trainFeaturesGen)

                    t = time.time()
                    testFeaturesTransform = scaler.transform(testFeatures)
                    testFeaturesTransform = pca.transform(testFeaturesTransform)
                    results.at[idx, 'tPre'] = (time.time() - t) / len(testFeatures)
                else:
                    pca = 0
                    t = time.time()
                    testFeaturesTransform = scaler.transform(testFeatures)
                    results.at[idx, 'tPre'] = (time.time() - t) / len(testFeatures)

                dataPK, allFeaturesPK = preprocessingPK(dataMatrix, allFeatures, scaler, pca)
                # print('features after', allFeaturesPK)
                preTrainedDataMatrix = preTrainedDataCapgmyo(dataPK, classes, peoplePriorK, person, allFeaturesPK)
                currentValues = currentDistributionValues(trainFeatures, trainLabels, classes, allFeaturesPK)
                pkValues = currentDistributionValues(trainFeaturesGen, trainLabelsGen, classes, allFeaturesPK)
                results, idx = resultsDataframe(currentValues, preTrainedDataMatrix, trainFeatures, trainLabels,
                                                classes, allFeaturesPK, results, testFeaturesTransform, testLabels, idx,
                                                person, trainGestures, featureSet, nameFile, printR, k, pkValues)

    return results


def preTrainedDataCapgmyo(dataMatrix, classes, peoplePriorK, evaluatedPerson, allFeatures):
    preTrainedDataMatrix = pd.DataFrame(columns=['mean', 'cov', 'class', 'person'])

    indx = 0
    for cl in range(1, classes + 1):

        for person in range(1, peoplePriorK + 1):
            if evaluatedPerson != person:
                preTrainedDataMatrix.at[indx, 'cov'] = np.cov(dataMatrix[np.where(
                    (dataMatrix[:, allFeatures] == person) * (dataMatrix[:, allFeatures + 1] == cl)), 0:allFeatures][0],
                                                              rowvar=False)
                preTrainedDataMatrix.at[indx, 'mean'] = np.mean(dataMatrix[np.where(
                    (dataMatrix[:, allFeatures] == person) * (dataMatrix[:, allFeatures + 1] == cl)), 0:allFeatures][0],
                                                                axis=0)
                preTrainedDataMatrix.at[indx, 'class'] = cl
                preTrainedDataMatrix.at[indx, 'person'] = person
                indx += 1

    return preTrainedDataMatrix


# %% EPN
def evaluationEPN(dataMatrix, classes, peoplePriorK, featureSet, numberShots, nameFile, startPerson, endPerson,
                  allFeatures, printR, scaler):
    results = pd.DataFrame(
        columns=['person', 'subset', '# shots', 'Feature Set'])

    trainFeaturesGenPre = dataMatrix[np.where((dataMatrix[:, allFeatures + 1] <= peoplePriorK)), 0:allFeatures][0]
    trainLabelsGenPre = dataMatrix[np.where((dataMatrix[:, allFeatures + 1] <= peoplePriorK)), allFeatures + 2][0]

    idx = 0
    for person in range(peoplePriorK + startPerson, peoplePriorK + endPerson + 1):

        typeData = 1
        testFeatures = dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (
                dataMatrix[:, allFeatures + 1] == person)), 0:allFeatures][0]
        testLabels = dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (
                dataMatrix[:, allFeatures + 1] == person)), allFeatures + 2][0]

        typeData = 0
        numberShots2Test = 25
        for shot in range(1, numberShots2Test + 1):

            subset = tuple(range(1, shot + 1))

            trainFeatures = np.empty((0, allFeatures))
            trainLabels = []
            for auxIndex in range(np.size(subset)):
                trainFeatures = np.vstack(
                    (trainFeatures, dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (
                            dataMatrix[:, allFeatures + 1] == person) * (
                                                                dataMatrix[:, allFeatures + 3] == subset[
                                                            auxIndex])), 0:allFeatures][0]))
                trainLabels = np.hstack(
                    (trainLabels, dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (
                            dataMatrix[:, allFeatures + 1] == person) * (
                                                              dataMatrix[:, allFeatures + 3] == subset[
                                                          auxIndex])), allFeatures + 2][0].T))

            trainFeaturesGen = np.vstack((trainFeaturesGenPre, trainFeatures))
            trainLabelsGen = np.hstack((trainLabelsGenPre, trainLabels))

            k = 1 - (np.log(shot) / np.log(numberShots + 1))

            scaler.fit(trainFeatures)
            trainFeatures = scaler.transform(trainFeatures)
            trainFeaturesGen = scaler.transform(trainFeaturesGen)

            t = time.time()
            testFeaturesTransform = scaler.transform(testFeatures)
            results.at[idx, 'tPre'] = (time.time() - t) / len(testFeatures)

            dataPK, allFeaturesPK = preprocessingPK(dataMatrix, allFeatures, scaler, pca=0)

            preTrainedDataMatrix = preTrainedDataEPN(dataPK, classes, peoplePriorK, allFeaturesPK)
            currentValues = currentDistributionValues(trainFeatures, trainLabels, classes, allFeaturesPK)
            pkValues = currentDistributionValues(trainFeaturesGen, trainLabelsGen, classes, allFeaturesPK)
            results, idx = resultsDataframe(currentValues, preTrainedDataMatrix, trainFeatures, trainLabels,
                                            classes, allFeaturesPK, results, testFeaturesTransform, testLabels, idx,
                                            person, subset, featureSet, nameFile, printR, k, pkValues)

    return results


def preTrainedDataEPN(dataMatrix, classes, peoplePriorK, allFeatures):
    preTrainedDataMatrix = pd.DataFrame(columns=['mean', 'cov', 'class', 'person'])

    indx = 0
    for cl in range(1, classes + 1):

        for person in range(1, peoplePriorK + 1):
            preTrainedDataMatrix.at[indx, 'cov'] = np.cov(
                dataMatrix[np.where((dataMatrix[:, allFeatures + 1] == person) * (
                        dataMatrix[:, allFeatures + 2] == cl)), 0:allFeatures][0], rowvar=False)
            preTrainedDataMatrix.at[indx, 'mean'] = np.mean(
                dataMatrix[np.where((dataMatrix[:, allFeatures + 1] == person) * (
                        dataMatrix[:, allFeatures + 2] == cl)), 0:allFeatures][0], axis=0)
            preTrainedDataMatrix.at[indx, 'class'] = cl
            preTrainedDataMatrix.at[indx, 'person'] = person
            indx += 1

    return preTrainedDataMatrix


# %% Cote-Allard

def evaluationCote(dataMatrix, classes, peoplePriorK, featureSet, numberShots, nameFile, startPerson, endPerson,
                   allFeatures, printR, scaler):
    # Creating Variables
    results = pd.DataFrame(
        columns=['person', 'subset', '# shots', 'Feature Set'])

    idx = 0

    typeData = 0
    trainFeaturesGenPre = dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData)), 0:allFeatures][0]
    trainLabelsGenPre = dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData)), allFeatures + 3][0]

    typeData = 1
    for person in range(peoplePriorK + startPerson, peoplePriorK + endPerson + 1):

        carpet = 2
        testFeatures = \
            dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (dataMatrix[:, allFeatures + 1] == person)
                                * (dataMatrix[:, allFeatures + 2] == carpet)), 0:allFeatures][0]
        testLabels = \
            dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (dataMatrix[:, allFeatures + 1] == person)
                                * (dataMatrix[:, allFeatures + 2] == carpet)), allFeatures + 3][0]

        carpet = 1
        # 4 cycles - cross_validation for 4 cycles or shots
        numberShots2Test = 4
        for shot in range(1, numberShots2Test + 1):

            subset = tuple(range(1, shot + 1))

            trainFeatures = np.empty((0, allFeatures))
            trainLabels = []

            for auxIndex in range(np.size(subset)):
                trainFeatures = np.vstack((trainFeatures, dataMatrix[
                                                          np.where((dataMatrix[:, allFeatures] == typeData)
                                                                   * (dataMatrix[:, allFeatures + 1] == person)
                                                                   * (dataMatrix[:, allFeatures + 2] == carpet)
                                                                   * (dataMatrix[:, allFeatures + 4] == subset[
                                                              auxIndex]))
                , 0:allFeatures][0]))
                trainLabels = np.hstack((trainLabels, dataMatrix[
                    np.where((dataMatrix[:, allFeatures] == typeData)
                             * (dataMatrix[:, allFeatures + 1] == person)
                             * (dataMatrix[:, allFeatures + 2] == carpet)
                             * (dataMatrix[:, allFeatures + 4] == subset[auxIndex]))
                    , allFeatures + 3][0].T))
            trainFeaturesGen = np.vstack((trainFeaturesGenPre, trainFeatures))
            trainLabelsGen = np.hstack((trainLabelsGenPre, trainLabels))

            k = 1 - (np.log(shot) / np.log(numberShots + 1))

            scaler.fit(trainFeatures)

            scaler.fit(np.vstack((trainFeatures, trainFeaturesGen)))
            trainFeatures = scaler.transform(trainFeatures)
            trainFeaturesGen = scaler.transform(trainFeaturesGen)

            print('features before', np.size(dataMatrix[:, 0:allFeatures], axis=1))
            # pca = PCA(n_components=0.99, svd_solver='full')
            # pca.fit(np.vstack((trainFeatures, trainFeaturesGen)))
            # trainFeatures = pca.transform(trainFeatures)
            # trainFeaturesGen = pca.transform(trainFeaturesGen)
            # testFeaturesTransform = pca.transform(testFeaturesTransform)

            t = time.time()
            testFeaturesTransform = scaler.transform(testFeatures)
            # testFeaturesTransform = pca.transform(testFeaturesTransform)
            results.at[idx, 'tPre'] = (time.time() - t) / len(testFeatures)

            dataPK, allFeaturesPK = preprocessingPK(dataMatrix, allFeatures, scaler)
            print('features after', allFeaturesPK)

            preTrainedDataMatrix = preTrainedDataCote(dataPK, classes, peoplePriorK, allFeaturesPK)
            currentValues = currentDistributionValues(trainFeatures, trainLabels, classes, allFeaturesPK)
            pkValues = currentDistributionValues(trainFeaturesGen, trainLabelsGen, classes, allFeaturesPK)
            results, idx = resultsDataframe(currentValues, preTrainedDataMatrix, trainFeatures, trainLabels,
                                            classes, allFeaturesPK, results, testFeaturesTransform, testLabels, idx,
                                            person, subset, featureSet, nameFile, printR, k, pkValues)

    return results


def preTrainedDataCote(dataMatrix, classes, peoplePriorK, allFeatures):
    preTrainedDataMatrix = pd.DataFrame(columns=['mean', 'cov', 'class', 'person'])
    typeData = 0
    indx = 0
    for cl in range(1, classes + 1):

        for person in range(1, peoplePriorK + 1):
            preTrainedDataMatrix.at[indx, 'cov'] = np.cov(dataMatrix[np.where(
                (dataMatrix[:, allFeatures] == typeData) * (dataMatrix[:, allFeatures + 1] == person) * (
                        dataMatrix[:, allFeatures + 3] == cl)), 0:allFeatures][0], rowvar=False)
            preTrainedDataMatrix.at[indx, 'mean'] = np.mean(dataMatrix[np.where(
                (dataMatrix[:, allFeatures] == typeData) * (dataMatrix[:, allFeatures + 1] == person) * (
                        dataMatrix[:, allFeatures + 3] == cl)), 0:allFeatures][0], axis=0)
            preTrainedDataMatrix.at[indx, 'class'] = cl
            preTrainedDataMatrix.at[indx, 'person'] = person
            indx += 1

    return preTrainedDataMatrix


# %% Nina Pro

def evaluationNina(dataMatrix, classes, peoplePriorK, featureSet, numberShots, nameFile,
                   startPerson, endPerson, allFeatures, printR, scaler):
    # Creating Variables

    results = pd.DataFrame(
        columns=['person', 'subset', '# shots', 'Feature Set'])

    idx = 0

    for person in range(startPerson, endPerson + 1):

        trainFeaturesGenPre = dataMatrix[np.where((dataMatrix[:, allFeatures] != person)), 0:allFeatures][0]
        trainLabelsGenPre = dataMatrix[np.where((dataMatrix[:, allFeatures] != person)), allFeatures + 1][0]

        testFeatures = \
            dataMatrix[np.where((dataMatrix[:, allFeatures] == person) * (dataMatrix[:, allFeatures + 2] >= 5)),
            0:allFeatures][0]
        testLabels = dataMatrix[
            np.where((dataMatrix[:, allFeatures] == person) * (dataMatrix[:, allFeatures + 2] >= 5)), allFeatures + 1][
            0].T

        numberShots2Test = 4
        for shot in range(1, numberShots2Test + 1):
            subset = tuple(range(1, shot + 1))
            trainFeatures = np.empty((0, allFeatures))
            trainLabels = []
            for auxIndex in range(np.size(subset)):
                trainFeatures = np.vstack((trainFeatures, dataMatrix[np.where(
                    (dataMatrix[:, allFeatures] == person) * (dataMatrix[:, allFeatures + 2] == subset[auxIndex])),
                                                          0:allFeatures][0]))
                trainLabels = np.hstack((trainLabels, dataMatrix[np.where(
                    (dataMatrix[:, allFeatures] == person) * (dataMatrix[:, allFeatures + 2] == subset[auxIndex]))
                , allFeatures + 1][0].T))

            trainFeaturesGen = np.vstack((trainFeaturesGenPre, trainFeatures))
            trainLabelsGen = np.hstack((trainLabelsGenPre, trainLabels))

            k = 1 - (np.log(shot) / np.log(numberShots + 1))

            scaler.fit(np.vstack((trainFeatures, trainFeaturesGen)))
            trainFeatures = scaler.transform(trainFeatures)
            trainFeaturesGen = scaler.transform(trainFeaturesGen)

            t = time.time()
            testFeaturesTransform = scaler.transform(testFeatures)
            results.at[idx, 'tPre'] = (time.time() - t) / len(testFeatures)

            dataPK, allFeaturesPK = preprocessingPK(dataMatrix, allFeatures, scaler)

            preTrainedDataMatrix = preTrainedDataNina(dataPK, classes, peoplePriorK, person, allFeaturesPK)
            currentValues = currentDistributionValues(trainFeatures, trainLabels, classes, allFeaturesPK)
            pkValues = currentDistributionValues(trainFeaturesGen, trainLabelsGen, classes, allFeaturesPK)
            results, idx = resultsDataframe(currentValues, preTrainedDataMatrix, trainFeatures, trainLabels,
                                            classes, allFeaturesPK, results, testFeaturesTransform, testLabels, idx,
                                            person, subset, featureSet, nameFile, printR, k, pkValues)

    return results


def preTrainedDataNina(dataMatrix, classes, peoplePriorK, evaluatedPerson, allFeatures):
    preTrainedDataMatrix = pd.DataFrame(columns=['mean', 'cov', 'class', 'person'])

    indx = 0
    for cl in range(1, classes + 1):

        for person in range(1, peoplePriorK + 1):
            if evaluatedPerson != person:
                preTrainedDataMatrix.at[indx, 'cov'] = np.cov(dataMatrix[np.where(
                    (dataMatrix[:, allFeatures] == person) * (dataMatrix[:, allFeatures + 1] == cl)), 0:allFeatures][0],
                                                              rowvar=False)
                preTrainedDataMatrix.at[indx, 'mean'] = np.mean(dataMatrix[np.where(
                    (dataMatrix[:, allFeatures] == person) * (dataMatrix[:, allFeatures + 1] == cl)), 0:allFeatures][0],
                                                                axis=0)
                preTrainedDataMatrix.at[indx, 'class'] = cl
                preTrainedDataMatrix.at[indx, 'person'] = person
                indx += 1

    return preTrainedDataMatrix


# %% Auxiliar functions of the evaluation

def resultsDataframe(currentValues, preTrainedDataMatrix, trainFeatures, trainLabels, classes, allFeatures,
                     results, testFeatures, testLabels, idx, person, subset, featureSet, nameFile, printR, k, pkValues):
    # Amount of Training data
    numSamples = 20
    trainFeatures, trainLabels = subsetTraining(trainFeatures, trainLabels, numSamples, classes)

    # step = math.ceil(np.shape(trainLabels)[0] / (classes * minSamplesClass))
    step = 1
    print('step: ', np.shape(trainLabels)[0], step)

    liuModel = adaptive.LiuModel(currentValues, preTrainedDataMatrix, classes, allFeatures)
    vidovicModelL, vidovicModelQ = adaptive.VidovicModel(currentValues, preTrainedDataMatrix, classes, allFeatures)

    propModelLDA, _, results.at[idx, 'wTargetMeanLDA'], _, results.at[idx, 'wTargetCovLDA'], results.at[
        idx, 'tPropLDA'] = adaptive.OurModel(
        currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels, step, 'LDA', k)
    propModelQDA, _, results.at[idx, 'wTargetMeanQDA'], _, results.at[idx, 'wTargetCovQDA'], results.at[
        idx, 'tPropQDA'] = adaptive.OurModel(
        currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels, step, 'QDA', k)

    results.at[idx, 'person'] = person
    results.at[idx, 'subset'] = subset
    results.at[idx, '# shots'] = np.size(subset)
    results.at[idx, 'Feature Set'] = featureSet
    # LDA results
    results.at[idx, 'AccLDAInd'], results.at[idx, 'tIndL'] = DA_Classifiers.accuracyModelLDA(testFeatures, testLabels,
                                                                                             currentValues,
                                                                                             classes)
    results.at[idx, 'AccLDAMulti'], results.at[idx, 'tGenL'] = DA_Classifiers.accuracyModelLDA(testFeatures, testLabels,
                                                                                               pkValues,
                                                                                               classes)
    results.at[idx, 'AccLDAProp'], results.at[idx, 'tCLPropL'] = DA_Classifiers.accuracyModelLDA(testFeatures,
                                                                                                 testLabels,
                                                                                                 propModelLDA,
                                                                                                 classes)
    results.at[idx, 'AccLDALiu'], _ = DA_Classifiers.accuracyModelLDA(testFeatures, testLabels, liuModel, classes)
    results.at[idx, 'AccLDAVidovic'], _ = DA_Classifiers.accuracyModelLDA(testFeatures, testLabels, vidovicModelL,
                                                                          classes)

    # QDA results
    results.at[idx, 'AccQDAInd'], results.at[idx, 'tIndQ'] = DA_Classifiers.accuracyModelQDA(testFeatures, testLabels,
                                                                                             currentValues,
                                                                                             classes)
    results.at[idx, 'AccQDAMulti'], results.at[idx, 'tGenQ'] = DA_Classifiers.accuracyModelQDA(testFeatures, testLabels,
                                                                                               pkValues,
                                                                                               classes)
    results.at[idx, 'AccQDAProp'], results.at[idx, 'tCLPropQ'] = DA_Classifiers.accuracyModelQDA(testFeatures,
                                                                                                 testLabels,
                                                                                                 propModelQDA,
                                                                                                 classes)
    results.at[idx, 'AccQDALiu'], _ = DA_Classifiers.accuracyModelQDA(testFeatures, testLabels, liuModel, classes)
    results.at[idx, 'AccQDAVidovic'], _ = DA_Classifiers.accuracyModelQDA(testFeatures, testLabels, vidovicModelQ,
                                                                          classes)

    if nameFile is not None:
        results.to_csv(nameFile)
    if printR:
        print(featureSet)
        print('Results: person= ', person, ' shot set= ', subset)
        print(results.loc[idx])
        print('Samples: ', np.shape(trainLabels)[0], ' step: ', step)

    idx += 1

    return results, idx


def currentDistributionValues(trainFeatures, trainLabels, classes, allFeatures):
    currentValues = pd.DataFrame(columns=['cov', 'mean', 'class'])
    trainLabelsAux = trainLabels[np.newaxis]
    Matrix = np.hstack((trainFeatures, trainLabelsAux.T))
    for cla in range(classes):
        currentValues.at[cla, 'cov'] = np.cov(Matrix[np.where((Matrix[:, allFeatures] == cla + 1)), 0:allFeatures][0],
                                              rowvar=False)
        currentValues.at[cla, 'mean'] = np.mean(Matrix[np.where((Matrix[:, allFeatures] == cla + 1)), 0:allFeatures][0],
                                                axis=0)
        currentValues.at[cla, 'class'] = cla + 1

    return currentValues


def preprocessingPK(dataMatrix, allFeatures, scaler, pca=0):
    dataMatrixFeatures = scaler.transform(dataMatrix[:, 0:allFeatures])
    if pca == 0:
        return np.hstack((dataMatrixFeatures, dataMatrix[:, allFeatures:])), np.size(dataMatrixFeatures, axis=1)
    else:
        dataMatrixFeatures = pca.transform(dataMatrixFeatures)
        return np.hstack((dataMatrixFeatures, dataMatrix[:, allFeatures:])), np.size(dataMatrixFeatures, axis=1)


def subsetTraining(trainFeatures, trainLabels, numSamples, classes):
    idx = []
    for cla in range(classes):
        aux = np.where(trainLabels == cla + 1)[0]

        if len(aux) > numSamples:
            modNumber = np.ceil(len(aux) / numSamples)
            idxAux = []
            [idxAux.append(a) for a in aux if a % modNumber == 1 and len(idxAux) < numSamples]
            if len(idxAux) < numSamples:
                [idxAux.append(a) for a in aux if a % modNumber == 2 and len(idxAux) < numSamples]
            idx.extend(idxAux)
        else:
            idx.extend(list(aux))
    return trainFeatures[idx], trainLabels[idx]
