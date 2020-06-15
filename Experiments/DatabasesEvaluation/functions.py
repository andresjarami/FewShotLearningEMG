import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from scipy import stats
from scipy.spatial import distance

import time
import math

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.datasets import make_spd_matrix

import itertools
import random


## Reduced Daily Recalibration of Myoelectric Prosthesis Classifiers Based on Domain Adaptation
def weightDenominatorLiu(currentMean, preTrainedDataMatrix):
    weightDenominatorV = 0
    for i in range(len(preTrainedDataMatrix.index)):
        weightDenominatorV = weightDenominatorV + (
                1 / distance.mahalanobis(currentMean, preTrainedDataMatrix['mean'].loc[i],
                                         np.linalg.inv(preTrainedDataMatrix['cov'].loc[i])))
    return weightDenominatorV


def reTrainedMeanLiu(r, currentMean, preTrainedDataMatrix, weightDenominatorV, allFeatures):
    sumAllPreTrainedMean_Weighted = np.zeros((1, allFeatures))
    for i in range(len(preTrainedDataMatrix.index)):
        sumAllPreTrainedMean_Weighted = np.add(sumAllPreTrainedMean_Weighted, preTrainedDataMatrix['mean'].loc[i] * (
                1 / distance.mahalanobis(currentMean, preTrainedDataMatrix['mean'].loc[i],
                                         np.linalg.inv(preTrainedDataMatrix['cov'].loc[i]))))

    reTrainedMeanValue = np.add((1 - r) * currentMean, (r / weightDenominatorV) * sumAllPreTrainedMean_Weighted)
    return reTrainedMeanValue


def reTrainedCovLiu(r, currentMean, currentCov, preTrainedDataMatrix, weightDenominatorV, allFeatures):
    sumAllPreTrainedCov_Weighted = np.zeros((allFeatures, allFeatures))
    for i in range(len(preTrainedDataMatrix.index)):
        sumAllPreTrainedCov_Weighted = np.add(sumAllPreTrainedCov_Weighted, preTrainedDataMatrix['cov'][i] * (
                1 / distance.mahalanobis(currentMean, preTrainedDataMatrix['mean'][i],
                                         np.linalg.inv(preTrainedDataMatrix['cov'][i]))))

    reTrainedCovValue = np.add((1 - r) * currentCov, (r / weightDenominatorV) * sumAllPreTrainedCov_Weighted)
    return reTrainedCovValue


def LiuModel(currentValues, preTrainedDataMatrix, classes, allFeatures):
    trainedModel = pd.DataFrame(columns=['cov', 'mean', 'class'])
    r = 0.5
    for cla in range(0, classes):
        preTrainedMatrix_Class = pd.DataFrame(
            preTrainedDataMatrix[['cov', 'mean']].loc[(preTrainedDataMatrix['class'] == cla + 1)])
        preTrainedMatrix_Class = preTrainedMatrix_Class.reset_index(drop=True)
        currentCov = currentValues['cov'].loc[cla]
        currentMean = currentValues['mean'].loc[cla]
        weightDenominatorV = weightDenominatorLiu(currentMean, preTrainedMatrix_Class)
        trainedModel.at[cla, 'cov'] = reTrainedCovLiu(r, currentMean, currentCov, preTrainedMatrix_Class,
                                                      weightDenominatorV, allFeatures)
        trainedModel.at[cla, 'mean'] = \
            reTrainedMeanLiu(r, currentMean, preTrainedMatrix_Class, weightDenominatorV, allFeatures)[0]
        trainedModel.at[cla, 'class'] = cla + 1

    return trainedModel


# KL Divergence Distance

def KL_DivergenceDistance(mean_1, cov_1, mean_2, cov_2, k):
    partA = np.trace(np.dot(np.linalg.inv(cov_1), cov_2))
    partB = np.dot(np.dot((mean_1 - mean_2), np.linalg.inv(cov_1)), (mean_1 - mean_2).T)
    partC = np.log(np.linalg.det(cov_1) / np.linalg.det(cov_2))

    result = (1 / 2) * (partA + partB - k + partC)

    return result


# F-score LDA and QDA

def scoreModelLDA_r(testFeatures, testLabels, model, classes, currentClass, step):
    trueP = 0

    falseP = 0
    falseN = 0

    currentClass = currentClass + 1
    LDACov = LDA_Cov(model, classes)
    for i in range(0, np.size(testLabels), step):
        currentPredictor = predictedModelLDA(testFeatures[i, :], model, classes, LDACov)
        if currentPredictor == currentClass:
            if currentPredictor == testLabels[i]:
                trueP += 1
            else:
                falseP += 1
        if (testLabels[i] == currentClass) and (testLabels[i] != currentPredictor):
            falseN += 1

    if trueP == 0:
        F1 = 0
    else:
        recall = trueP / (trueP + falseN)
        precision = trueP / (trueP + falseP)
        F1 = 2 * (precision * recall) / (precision + recall)

    return F1


def scoreModelQDA_r(testFeatures, testLabels, model, classes, currentClass, step):
    trueP = 0

    falseP = 0
    falseN = 0

    currentClass = currentClass + 1
    for i in range(0, np.size(testLabels), step):
        currentPredictor = predictedModelQDA(testFeatures[i, :], model, classes)

        if currentPredictor == currentClass:
            if currentPredictor == testLabels[i]:
                trueP += 1
            else:
                falseP += 1
        if (testLabels[i] == currentClass) and (testLabels[i] != currentPredictor):
            falseN += 1

    if trueP == 0:
        F1 = 0
    else:
        recall = trueP / (trueP + falseN)
        precision = trueP / (trueP + falseP)
        F1 = 2 * (precision * recall) / (precision + recall)

    return F1


# F-score LDA and QDA for all people

def scoreModelLDA_ALL(testFeatures, testLabels, model, classes, step):
    trueP = np.zeros([classes])

    falseP = np.zeros([classes])
    falseN = np.zeros([classes])

    LDACov = LDA_Cov(model, classes)

    for i in range(0, np.size(testLabels), step):
        currentPredictor = predictedModelLDA(testFeatures[i, :], model, classes, LDACov)

        if currentPredictor == testLabels[i]:
            trueP[int(testLabels[i] - 1)] += 1
        else:
            falseN[int(testLabels[i] - 1)] += 1
            falseP[int(currentPredictor - 1)] += 1

    recall = trueP / (trueP + falseN)
    precision = trueP / (trueP + falseP)
    F1 = 2 * (precision * recall) / (precision + recall)
    F1[np.isnan(F1)] = 0
    return F1


def scoreModelQDA_ALL(testFeatures, testLabels, model, classes, step):
    trueP = np.zeros([classes])

    falseP = np.zeros([classes])
    falseN = np.zeros([classes])

    for i in range(0, np.size(testLabels), step):
        currentPredictor = predictedModelQDA(testFeatures[i, :], model, classes)

        if currentPredictor == testLabels[i]:
            trueP[int(testLabels[i] - 1)] += 1
        else:
            falseN[int(testLabels[i] - 1)] += 1
            falseP[int(currentPredictor - 1)] += 1

    recall = trueP / (trueP + falseN)
    precision = trueP / (trueP + falseP)
    F1 = 2 * (precision * recall) / (precision + recall)
    F1[np.isnan(F1)] = 0
    return F1


# Weight Calculation LDA or QDA

def rCalculatedProposed10BB_r(currentValues, personMean, personCov, currentClass, classes
                              , trainFeatures, trainLabels, F1C, step, typeModel):
    personValues = currentValues.copy()
    personValues['mean'].at[currentClass] = personMean
    personValues['cov'].at[currentClass] = personCov
    if typeModel == 'LDA':
        F1P = scoreModelLDA_r(trainFeatures, trainLabels, personValues, classes, currentClass, step)
    elif typeModel == 'QDA':
        F1P = scoreModelQDA_r(trainFeatures, trainLabels, personValues, classes, currentClass, step)
    if F1P + F1C == 0:
        r = 0
    else:
        r = F1P / (F1P + F1C)

    return r


# Adaptative Calssifier LDA or QDA
def ProposedModel(currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels, step,
                  typeModel):
    t = time.time()

    trainedModel = pd.DataFrame(columns=['cov', 'mean', 'class'])

    for cla in range(0, classes):
        trainedModel.at[cla, 'cov'] = np.zeros((allFeatures, allFeatures))
        trainedModel.at[cla, 'mean'] = np.zeros((1, allFeatures))[0]
    if typeModel == 'LDA':
        F1C = scoreModelLDA_ALL(trainFeatures, trainLabels, currentValues, classes, step)
    elif typeModel == 'QDA':
        F1C = scoreModelQDA_ALL(trainFeatures, trainLabels, currentValues, classes, step)

    rClass = np.zeros(classes)
    for cla in range(0, classes):
        preTrainedMatrix_Class = pd.DataFrame(
            preTrainedDataMatrix[['cov', 'mean']].loc[(preTrainedDataMatrix['class'] == cla + 1)])
        preTrainedMatrix_Class = preTrainedMatrix_Class.reset_index(drop=True)
        currentCov = currentValues['cov'].loc[cla]
        currentMean = currentValues['mean'].loc[cla]
        peopleClass = len(preTrainedMatrix_Class.index)

        for i in range(peopleClass):
            personMean = preTrainedMatrix_Class['mean'].loc[i]
            personCov = preTrainedMatrix_Class['cov'].loc[i]
            r = rCalculatedProposed10BB_r(currentValues, personMean, personCov, cla, classes
                                          , trainFeatures, trainLabels, F1C[cla], step, typeModel)

            rClass[cla] = (1 - r) + rClass[cla]
            trainedModel.at[cla, 'cov'] = trainedModel['cov'].loc[cla] + (
                    (1 - r) * currentCov + r * personCov) / peopleClass

            trainedModel.at[cla, 'mean'] = trainedModel['mean'].loc[cla] + (
                    (1 - r) * currentMean + r * personMean) / peopleClass

        trainedModel.at[cla, 'class'] = cla + 1
    elapsed = time.time() - t
    return trainedModel, rClass / peopleClass, rClass.mean() / peopleClass, elapsed


def reTrainedMeanProposed(r, currentMean, reTrainedCalculatedMeanValue):
    reTrainedMeanValue = np.add((1 - r) * currentMean, r * reTrainedCalculatedMeanValue)
    return reTrainedMeanValue


def reTrainedCovProposed(r, currentCov, reTrainedCalculatedCovValue):
    reTrainedCovValue = np.add((1 - r) * currentCov, r * reTrainedCalculatedCovValue)
    return reTrainedCovValue


# LDA Calssifier
def LDA_Discriminant(x, covariance, mean):
    invCov = np.linalg.inv(covariance)
    discriminant = np.dot(np.dot(x, invCov), mean) - 0.5 * np.dot(np.dot(mean, invCov), mean)
    return discriminant


def LDA_Cov(trainedModel, classes):
    LDACov = trainedModel['cov'].sum() / classes
    return LDACov


def predictedModelLDA(sample, model, classes, LDACov):
    d = np.zeros([classes])
    for cl in range(classes):
        d[cl] = LDA_Discriminant(sample, LDACov, model['mean'].loc[cl])
    return np.argmax(d) + 1


def scoreModelLDA(testFeatures, testLabels, model, classes):
    true = 0
    count = 0
    LDACov = LDA_Cov(model, classes)
    for i in range(0, np.size(testLabels)):
        currentPredictor = predictedModelLDA(testFeatures[i, :], model, classes, LDACov)
        if currentPredictor == testLabels[i]:
            true += 1
            count += 1
        else:
            count += 1

    return true / count


# QDA Classifier
def QDA_Discriminant(x, cov_k, u_k):
    det = np.linalg.det(cov_k)
    a = -.5 * np.log(det)
    b = -.5 * np.dot(np.dot((x - u_k), np.linalg.inv(cov_k)), (x - u_k).T)
    d = a + b
    return d


def predictedModelQDA(sample, model, classes):
    d = np.zeros([classes])
    for cl in range(classes):
        d[cl] = QDA_Discriminant(sample, model['cov'].loc[cl], model['mean'].loc[cl])
    return np.argmax(d) + 1


def scoreModelQDA(testFeatures, testLabels, model, classes):
    true = 0
    count = 0
    for i in range(0, np.size(testLabels)):
        actualPredictor = predictedModelQDA(testFeatures[i, :], model, classes)
        if actualPredictor == testLabels[i]:
            true += 1
            count += 1
        else:
            count += 1
    return true / count


# Accuracy LDA and QDA

def scoreModelLDA_Classification(testFeatures, testLabels, model, classes, testRepetitions):
    true = 0
    count = 0
    LDACov = LDA_Cov(model, classes)
    auxRep = testRepetitions[0]
    auxClass = testLabels[0]
    actualPredictor = np.empty(0)

    for i in range(0, np.size(testLabels)):
        if auxRep != testRepetitions[i] or auxClass != testLabels[i] or i == np.size(testLabels) - 1:
            auxRep = testRepetitions[i]
            auxClass = testLabels[i]

            if stats.mode(actualPredictor)[0][0] == testLabels[i - 1]:
                true += 1
                count += 1
            else:
                count += 1
            actualPredictor = np.empty(0)
        actualPredictor = np.append(actualPredictor, predictedModelLDA(testFeatures[i, :], model, classes, LDACov))

    return true / count


def scoreModelQDA_Classification(testFeatures, testLabels, model, classes, testRepetitions):
    true = 0
    count = 0
    auxRep = testRepetitions[0]
    auxClass = testLabels[0]
    actualPredictor = np.empty(0)

    for i in range(0, np.size(testLabels)):
        if auxRep != testRepetitions[i] or auxClass != testLabels[i] or i == np.size(testLabels) - 1:
            auxRep = testRepetitions[i]
            auxClass = testLabels[i]
            if stats.mode(actualPredictor)[0][0] == testLabels[i - 1]:
                true += 1
                count += 1
            else:
                count += 1
            actualPredictor = np.empty(0)
        actualPredictor = np.append(actualPredictor, predictedModelQDA(testFeatures[i, :], model, classes))
    return true / count


# Evaluation

def resultsDataframe(currentValues, preTrainedDataMatrix, trainFeatures, trainLabels, classes, allFeatures,
                     clfLDAInd, clfQDAInd, clfLDAGen, clfQDAGen, trainFeaturesGen, trainLabelsGen, results,
                     testFeatures, testLabels, idx, person, subset, featureSet, nameFile, printR):
    # Amount of Training data
    minSamplesClass = 10
    step = math.ceil(np.shape(trainLabels)[0] / (classes * minSamplesClass))

    liuModel = LiuModel(currentValues, preTrainedDataMatrix, classes, allFeatures)

    propModelLDA, wLDA, wMeanLDA, tPropL = ProposedModel(currentValues, preTrainedDataMatrix, classes, allFeatures,
                                                         trainFeatures, trainLabels, step, 'LDA')
    propModelQDA, wQDA, wMeanQDA, tPropQ = ProposedModel(currentValues, preTrainedDataMatrix, classes, allFeatures,
                                                         trainFeatures, trainLabels, step, 'QDA')
    t = time.time()
    clfLDAInd.fit(trainFeatures, trainLabels)
    tIndL = time.time() - t
    t = time.time()
    clfQDAInd.fit(trainFeatures, trainLabels)
    tIndQ = time.time() - t
    t = time.time()
    clfLDAGen.fit(trainFeaturesGen, trainLabelsGen)
    tGenL = time.time() - t
    t = time.time()
    clfQDAGen.fit(trainFeaturesGen, trainLabelsGen)
    tGenQ = time.time() - t

    results.at[idx, 'person'] = person
    results.at[idx, 'subset'] = subset
    results.at[idx, '# shots'] = np.size(subset)
    results.at[idx, 'Feature Set'] = featureSet
    # LDA results
    results.at[idx, 'AccLDAInd'] = clfLDAInd.score(testFeatures, testLabels)
    results.at[idx, 'AccLDAGen'] = clfLDAGen.score(testFeatures, testLabels)
    results.at[idx, 'AccLDAProp'] = scoreModelLDA(testFeatures, testLabels, propModelLDA, classes)
    results.at[idx, 'AccLDALiu'] = scoreModelLDA(testFeatures, testLabels, liuModel, classes)
    results.at[idx, 'AccLDAPropQ'] = scoreModelLDA(testFeatures, testLabels, propModelQDA, classes)
    results.at[idx, 'tPropL'] = tPropL
    results.at[idx, 'tIndL'] = tIndL
    results.at[idx, 'tGenL'] = tGenL
    results.at[idx, 'wLDA'] = wLDA
    results.at[idx, 'wMeanLDA'] = wMeanLDA
    # QDA results
    results.at[idx, 'AccQDAInd'] = clfQDAInd.score(testFeatures, testLabels)
    results.at[idx, 'AccQDAGen'] = clfQDAGen.score(testFeatures, testLabels)
    results.at[idx, 'AccQDAProp'] = scoreModelQDA(testFeatures, testLabels, propModelQDA, classes)
    results.at[idx, 'AccQDALiu'] = scoreModelQDA(testFeatures, testLabels, liuModel, classes)
    results.at[idx, 'AccQDAPropL'] = scoreModelQDA(testFeatures, testLabels, propModelLDA, classes)
    results.at[idx, 'tPropQ'] = tPropQ
    results.at[idx, 'tIndQ'] = tIndQ
    results.at[idx, 'tGenQ'] = tGenQ
    results.at[idx, 'wQDA'] = wQDA
    results.at[idx, 'wMeanQDA'] = wMeanQDA

    if nameFile is not None:
        results.to_csv(nameFile)
    if printR:
        print(featureSet)
        print('Results: person= ', person, ' shot set= ', subset)
        print(results.loc[idx])
        print('Samples: ', np.shape(trainLabels)[0], ' step: ', step)

    idx += 1

    return results, idx


def resultsDataframeNoPCA(currentValues, preTrainedDataMatrix, trainFeatures, trainLabels, classes, allFeatures,
                          clfLDAInd, clfQDAInd, clfLDAGen, clfQDAGen, trainFeaturesGen, trainLabelsGen, results,
                          testFeatures, testLabels, idx, person, subset, featureSet, nameFile, printR):
    # Amount of Training data
    minSamplesClass = 10
    step = math.ceil(np.shape(trainLabels)[0] / (classes * minSamplesClass))

    liuModel = LiuModel(currentValues, preTrainedDataMatrix, classes, allFeatures)

    propModelLDA, wLDA, wMeanLDA, tPropL = ProposedModel(currentValues, preTrainedDataMatrix, classes, allFeatures,
                                                         trainFeatures, trainLabels, step, 'LDA')
    # propModelQDA, wQDA, wMeanQDA, tPropQ = ProposedModel(currentValues, preTrainedDataMatrix, classes, allFeatures,
    #                                                      trainFeatures, trainLabels, step, 'QDA')
    t = time.time()
    clfLDAInd.fit(trainFeatures, trainLabels)
    tIndL = time.time() - t
    t = time.time()
    clfQDAInd.fit(trainFeatures, trainLabels)
    tIndQ = time.time() - t
    t = time.time()
    clfLDAGen.fit(trainFeaturesGen, trainLabelsGen)
    tGenL = time.time() - t
    t = time.time()
    clfQDAGen.fit(trainFeaturesGen, trainLabelsGen)
    tGenQ = time.time() - t

    results.at[idx, 'person'] = person
    results.at[idx, 'subset'] = subset
    results.at[idx, '# shots'] = np.size(subset)
    results.at[idx, 'Feature Set'] = featureSet
    # LDA results
    results.at[idx, 'AccLDAInd'] = clfLDAInd.score(testFeatures, testLabels)
    results.at[idx, 'AccLDAGen'] = clfLDAGen.score(testFeatures, testLabels)
    results.at[idx, 'AccLDAProp'] = scoreModelLDA(testFeatures, testLabels, propModelLDA, classes)
    results.at[idx, 'AccLDALiu'] = scoreModelLDA(testFeatures, testLabels, liuModel, classes)
    # results.at[idx, 'AccLDAPropQ'] = scoreModelLDA(testFeatures, testLabels, propModelQDA, classes)
    results.at[idx, 'tPropL'] = tPropL
    results.at[idx, 'tIndL'] = tIndL
    results.at[idx, 'tGenL'] = tGenL
    results.at[idx, 'wLDA'] = wLDA
    results.at[idx, 'wMeanLDA'] = wMeanLDA

    ## QDA results
    results.at[idx, 'AccQDAInd'] = clfQDAInd.score(testFeatures, testLabels)
    results.at[idx, 'AccQDAGen'] = clfQDAGen.score(testFeatures, testLabels)
    # results.at[idx, 'AccQDAProp'] = scoreModelQDA(testFeatures, testLabels, propModelQDA, classes)
    # results.at[idx, 'AccQDALiu'] = scoreModelQDA(testFeatures, testLabels, liuModel, classes)
    # results.at[idx, 'AccQDAPropL'] = scoreModelQDA(testFeatures, testLabels, propModelLDA, classes)
    # results.at[idx, 'tPropQ'] = tPropQ
    results.at[idx, 'tIndQ'] = tIndQ
    results.at[idx, 'tGenQ'] = tGenQ
    # results.at[idx, 'wQDA'] = wQDA
    # results.at[idx, 'wMeanQDA'] = wMeanQDA

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
    for cla in range(0, classes):
        currentValues.at[cla, 'cov'] = np.cov(Matrix[np.where((Matrix[:, allFeatures] == cla + 1)), 0:allFeatures][0],
                                              rowvar=False)
        currentValues.at[cla, 'mean'] = np.mean(Matrix[np.where((Matrix[:, allFeatures] == cla + 1)), 0:allFeatures][0],
                                                axis=0)
        currentValues.at[cla, 'class'] = cla + 1

    return currentValues


def evaluation(dataMatrix, classes, peoplePriorK, featureSet, numberShots, combinationSet, nameFile, startPerson,
               endPerson, allFeatures, typeDatabase, NoPca, printR):
    if typeDatabase == 'EPN':
        evaluationEPN(dataMatrix, classes, peoplePriorK, featureSet, numberShots, combinationSet, nameFile, startPerson,
                      endPerson, allFeatures, NoPca, printR)
    elif typeDatabase == 'Nina5':
        evaluationNina5(dataMatrix, classes, peoplePriorK, featureSet, numberShots, combinationSet, nameFile,
                        startPerson, endPerson, allFeatures, NoPca, printR)
    elif typeDatabase == 'Cote':
        evaluationCote(dataMatrix, classes, peoplePriorK, featureSet, numberShots, combinationSet,
                       nameFile, startPerson, endPerson, allFeatures, NoPca, printR)


### EPN
def evaluationEPN(dataMatrix, classes, peoplePriorK, featureSet, numberShots, combinationSet, nameFile, startPerson,
                  endPerson, allFeatures, NoPca, printR):
    clfLDAInd = LDA()
    clfQDAInd = QDA()
    clfLDAGen = LDA()
    clfQDAGen = QDA()
    # Creating Variables
    if NoPca:
        results = pd.DataFrame(columns=['person', 'subset', '# shots', 'Feature Set', 'wLDA'])
    else:
        results = pd.DataFrame(columns=['person', 'subset', '# shots', 'Feature Set', 'wLDA', 'wQDA'])
    idx = 0

    for person in range(startPerson, endPerson + 1):
        combinationSet = list(range(1, 26))
        if person == 11 or person == 44:
            combinationSet = np.setdiff1d(list(range(1, 26)), 15)

        preTrainedDataMatrix = preTrainedDataEPN(dataMatrix, classes, peoplePriorK, person, allFeatures)
        typeData = 0
        trainFeaturesGenPre = dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (
                dataMatrix[:, allFeatures + 1] != person)), 0:allFeatures][0]
        trainLabelsGenPre = dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (
                dataMatrix[:, allFeatures + 1] != person)), allFeatures + 2][0]

        typeData = 1
        testFeatures = dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (
                dataMatrix[:, allFeatures + 1] == person)), 0:allFeatures][0]
        testLabels = dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (
                dataMatrix[:, allFeatures + 1] == person)), allFeatures + 2][0]

        typeData = 0
        for shot in range(1, numberShots + 1):
            if shot == 2:
                combinationSet = list(range(1, 6))
            elif shot == 3:
                combinationSet = list(range(1, 6))
            elif shot == 4:
                combinationSet = list(range(1, 6))

            for subset in itertools.combinations(combinationSet, shot):

                trainFeatures = np.empty((0, allFeatures))
                trainLabels = []

                for auxIndex in range(np.size(subset)):
                    trainFeatures = np.vstack(
                        (trainFeatures, dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (
                                dataMatrix[:, allFeatures + 1] == person) * (dataMatrix[:, allFeatures + 3] == subset[
                            auxIndex])), 0:allFeatures][0]))
                    trainLabels = np.hstack(
                        (trainLabels, dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (
                                dataMatrix[:, allFeatures + 1] == person) * (dataMatrix[:, allFeatures + 3] == subset[
                            auxIndex])), allFeatures + 2][0].T))

                trainFeaturesGen = np.vstack((trainFeaturesGenPre, trainFeatures))
                trainLabelsGen = np.hstack((trainLabelsGenPre, trainLabels))

                currentValues = currentDistributionValues(trainFeatures, trainLabels, classes, allFeatures)

                if NoPca:
                    results, idx = resultsDataframeNoPCA(currentValues, preTrainedDataMatrix, trainFeatures,
                                                         trainLabels,
                                                         classes, allFeatures, clfLDAInd, clfQDAInd, clfLDAGen,
                                                         clfQDAGen,
                                                         trainFeaturesGen, trainLabelsGen, results, testFeatures,
                                                         testLabels,
                                                         idx, person, subset, featureSet, nameFile, printR)
                else:
                    results, idx = resultsDataframe(currentValues, preTrainedDataMatrix, trainFeatures, trainLabels,
                                                    classes, allFeatures, clfLDAInd, clfQDAInd, clfLDAGen, clfQDAGen,
                                                    trainFeaturesGen, trainLabelsGen, results, testFeatures, testLabels,
                                                    idx, person, subset, featureSet, nameFile, printR)

    return results


def preTrainedDataEPN(dataMatrix, classes, peoplePreTrain, evaluatedPerson, allFeatures):
    preTrainedDataMatrix = pd.DataFrame(columns=['mean', 'cov', 'class', 'person'])
    typeData = 0
    indx = 0
    for cl in range(1, classes + 1):

        for person in range(1, peoplePreTrain + 1):
            if evaluatedPerson != person:
                preTrainedDataMatrix.at[indx, 'cov'] = np.cov(dataMatrix[np.where(
                    (dataMatrix[:, allFeatures] == typeData) * (dataMatrix[:, allFeatures + 1] == person) * (
                            dataMatrix[:, allFeatures + 2] == cl)), 0:allFeatures][0], rowvar=False)
                preTrainedDataMatrix.at[indx, 'mean'] = np.mean(dataMatrix[np.where(
                    (dataMatrix[:, allFeatures] == typeData) * (dataMatrix[:, allFeatures + 1] == person) * (
                            dataMatrix[:, allFeatures + 2] == cl)), 0:allFeatures][0], axis=0)
                preTrainedDataMatrix.at[indx, 'class'] = cl
                preTrainedDataMatrix.at[indx, 'person'] = person
                indx += 1

    return preTrainedDataMatrix


### Cote

def evaluationCote(dataMatrix, classes, peoplePriorK, featureSet, numberShots, combinationSet,
                   nameFile, startPerson, endPerson, allFeatures, NoPca, printR):
    clfLDAInd = LDA()
    clfQDAInd = QDA()
    clfLDAGen = LDA()
    clfQDAGen = QDA()
    # Creating Variables
    if NoPca:
        results = pd.DataFrame(columns=['person', 'subset', '# shots', 'Feature Set', 'wLDA'])
    else:
        results = pd.DataFrame(columns=['person', 'subset', '# shots', 'Feature Set', 'wLDA', 'wQDA'])
    idx = 0

    preTrainedDataMatrix = preTrainedDataCote(dataMatrix, classes, peoplePriorK, allFeatures)
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
        for shot in range(1, numberShots + 1):
            for subset in itertools.combinations(combinationSet, shot):
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
                currentValues = currentDistributionValues(trainFeatures, trainLabels, classes, allFeatures)

                if NoPca:
                    results, idx = resultsDataframeNoPCA(currentValues, preTrainedDataMatrix, trainFeatures,
                                                         trainLabels,
                                                         classes, allFeatures, clfLDAInd, clfQDAInd, clfLDAGen,
                                                         clfQDAGen,
                                                         trainFeaturesGen, trainLabelsGen, results, testFeatures,
                                                         testLabels,
                                                         idx, person, subset, featureSet, nameFile, printR)
                else:

                    results, idx = resultsDataframe(currentValues, preTrainedDataMatrix, trainFeatures, trainLabels,
                                                    classes, allFeatures, clfLDAInd, clfQDAInd, clfLDAGen, clfQDAGen,
                                                    trainFeaturesGen, trainLabelsGen, results, testFeatures, testLabels,
                                                    idx, person, subset, featureSet, nameFile, printR)

    return results


def preTrainedDataCote(dataMatrix, classes, peoplePreTrain, allFeatures):
    preTrainedDataMatrix = pd.DataFrame(columns=['mean', 'cov', 'class', 'person'])
    typeData = 0
    indx = 0
    for cl in range(1, classes + 1):

        for person in range(1, peoplePreTrain + 1):
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


### Nina Pro 5

def evaluationNina5(dataMatrix, classes, peoplePriorK, featureSet, numberShots, combinationSet, nameFile, startPerson,
                    endPerson, allFeatures, NoPca, printR):
    clfLDAInd = LDA()
    clfQDAInd = QDA()
    clfLDAGen = LDA()
    clfQDAGen = QDA()
    # Creating Variables
    if NoPca:
        results = pd.DataFrame(columns=['person', 'subset', '# shots', 'Feature Set', 'wLDA'])
    else:
        results = pd.DataFrame(columns=['person', 'subset', '# shots', 'Feature Set', 'wLDA', 'wQDA'])
    idx = 0

    for person in range(startPerson, endPerson + 1):

        preTrainedDataMatrix = preTrainedDataNina5(dataMatrix, classes, peoplePriorK, person, allFeatures)
        trainFeaturesGenPre = dataMatrix[np.where((dataMatrix[:, allFeatures] != person)), 0:allFeatures][0]
        trainLabelsGenPre = dataMatrix[np.where((dataMatrix[:, allFeatures] != person)), allFeatures + 1][0]

        Set = np.arange(1, 7)

        # 4 cycles - cross_validation for 4 cycles or shots
        for shot in range(1, numberShots + 1):
            for subset in itertools.combinations(combinationSet, shot):
                trainFeatures = np.empty((0, allFeatures))
                trainLabels = []
                testFeatures = np.empty((0, allFeatures))
                testLabels = []
                auxSubset = np.asarray(subset)
                diffSubset = np.setdiff1d(Set, auxSubset)
                for auxIndex in range(np.size(subset)):
                    trainFeatures = np.vstack((trainFeatures, dataMatrix[np.where(
                        (dataMatrix[:, allFeatures] == person) * (dataMatrix[:, allFeatures + 2] == subset[auxIndex])),
                                                              0:allFeatures][0]))
                    trainLabels = np.hstack((trainLabels, dataMatrix[np.where(
                        (dataMatrix[:, allFeatures] == person) * (dataMatrix[:, allFeatures + 2] == subset[auxIndex]))
                    , allFeatures + 1][0].T))

                for auxIndex in range(np.size(diffSubset)):
                    testFeatures = np.vstack((testFeatures, dataMatrix[np.where(
                        (dataMatrix[:, allFeatures] == person) * (
                                dataMatrix[:, allFeatures + 2] == diffSubset[auxIndex]))
                    , 0:allFeatures][0]))
                    testLabels = np.hstack((testLabels, dataMatrix[
                        np.where((dataMatrix[:, allFeatures] == person)
                                 * (dataMatrix[:, allFeatures + 2] == diffSubset[auxIndex]))
                        , allFeatures + 1][0].T))

                trainFeaturesGen = np.vstack((trainFeaturesGenPre, trainFeatures))
                trainLabelsGen = np.hstack((trainLabelsGenPre, trainLabels))

                currentValues = currentDistributionValues(trainFeatures, trainLabels, classes, allFeatures)

                if NoPca:
                    results, idx = resultsDataframeNoPCA(currentValues, preTrainedDataMatrix, trainFeatures,
                                                         trainLabels,
                                                         classes, allFeatures, clfLDAInd, clfQDAInd, clfLDAGen,
                                                         clfQDAGen,
                                                         trainFeaturesGen, trainLabelsGen, results, testFeatures,
                                                         testLabels,
                                                         idx, person, subset, featureSet, nameFile, printR)
                else:
                    results, idx = resultsDataframe(currentValues, preTrainedDataMatrix, trainFeatures, trainLabels,
                                                    classes, allFeatures, clfLDAInd, clfQDAInd, clfLDAGen, clfQDAGen,
                                                    trainFeaturesGen, trainLabelsGen, results, testFeatures, testLabels,
                                                    idx, person, subset, featureSet, nameFile, printR)

    return results


def preTrainedDataNina5(dataMatrix, classes, peoplePreTrain, evaluatedPerson, allFeatures):
    preTrainedDataMatrix = pd.DataFrame(columns=['mean', 'cov', 'class', 'person'])

    indx = 0
    for cl in range(1, classes + 1):

        for person in range(1, peoplePreTrain + 1):
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


# Upload Databases

def uploadDatabases(typeDatabase, featureSet=1):
    # Setting general variables
    numberShots = 4
    CH = 8
    if typeDatabase == 'EPN':
        carpet = 'ExtractedDataCollectedData'
        classes = 5
        peoplePriorK = 60
        peopleTrain = 60
        people = 60
        combinationSet = list(range(1, 26))
    elif typeDatabase == 'Nina5':
        carpet = 'ExtractedDataNinaDB5'
        classes = 18
        peoplePriorK = 10
        peopleTrain = 10
        people = 10
        combinationSet = list(range(1, 5))
    elif typeDatabase == 'Cote':
        carpet = 'ExtractedDataCoteAllard'
        classes = 7
        peoplePriorK = 19
        peopleTrain = 17
        people = peoplePriorK + peopleTrain
        combinationSet = list(range(1, 5))

    if featureSet == 1:
        # Setting variables
        Feature1 = 'mavMatrix'
        segment = ''
        numberFeatures = 1
        allFeatures = numberFeatures * CH
        # Getting Data
        mavMatrix = np.genfromtxt('../../ExtractedData/' + carpet + '/' + Feature1 + segment + '.csv', delimiter=',')
        if typeDatabase == 'EPN':
            dataMatrix = mavMatrix[:, 0:]
        elif typeDatabase == 'Nina5':
            dataMatrix = mavMatrix[:, 8:]
        elif typeDatabase == 'Cote':
            dataMatrix = mavMatrix[:, 0:]
            dataMatrix[:, allFeatures + 1] = dataMatrix[:, allFeatures + 1] + 1
            dataMatrix[:, allFeatures + 3] = dataMatrix[:, allFeatures + 3] + 1



    elif featureSet == 2:
        # Setting variables
        Feature1 = 'mavMatrix'
        Feature2 = 'wlMatrix'
        Feature3 = 'zcMatrix'
        Feature4 = 'sscMatrix'
        segment = ''
        numberFeatures = 4
        allFeatures = numberFeatures * CH
        mavMatrix = np.genfromtxt('../../ExtractedData/' + carpet + '/' + Feature1 + segment + '.csv', delimiter=',')
        wlMatrix = np.genfromtxt('../../ExtractedData/' + carpet + '/' + Feature2 + segment + '.csv', delimiter=',')
        zcMatrix = np.genfromtxt('../../ExtractedData/' + carpet + '/' + Feature3 + segment + '.csv', delimiter=',')
        sscMatrix = np.genfromtxt('../../ExtractedData/' + carpet + '/' + Feature4 + segment + '.csv', delimiter=',')
        if typeDatabase == 'EPN':
            dataMatrix = np.hstack((mavMatrix[:, 0:CH], wlMatrix[:, 0:CH], zcMatrix[:, 0:CH], sscMatrix[:, 0:]))

        elif typeDatabase == 'Nina5':
            dataMatrix = np.hstack(
                (mavMatrix[:, 8:CH * 2], wlMatrix[:, 8:CH * 2], zcMatrix[:, 8:CH * 2], sscMatrix[:, 8:]))

        elif typeDatabase == 'Cote':
            dataMatrix = np.hstack((mavMatrix[:, 0:CH], wlMatrix[:, 0:CH], zcMatrix[:, 0:CH], sscMatrix[:, 0:]))
            dataMatrix[:, allFeatures + 1] = dataMatrix[:, allFeatures + 1] + 1
            dataMatrix[:, allFeatures + 3] = dataMatrix[:, allFeatures + 3] + 1


    elif featureSet == 3:
        # Setting variables
        Feature1 = 'lscaleMatrix'
        Feature2 = 'mflMatrix'
        Feature3 = 'msrMatrix'
        Feature4 = 'wampMatrix'
        segment = ''
        numberFeatures = 4
        allFeatures = numberFeatures * CH
        # Getting Data
        lscaleMatrix = np.genfromtxt('../../ExtractedData/' + carpet + '/' + Feature1 + segment + '.csv', delimiter=',')
        mflMatrix = np.genfromtxt('../../ExtractedData/' + carpet + '/' + Feature2 + segment + '.csv', delimiter=',')
        msrMatrix = np.genfromtxt('../../ExtractedData/' + carpet + '/' + Feature3 + segment + '.csv', delimiter=',')
        wampMatrix = np.genfromtxt('../../ExtractedData/' + carpet + '/' + Feature4 + segment + '.csv', delimiter=',')

        if typeDatabase == 'EPN':
            dataMatrix = np.hstack((lscaleMatrix[:, 0:CH], mflMatrix[:, 0:CH], msrMatrix[:, 0:CH], wampMatrix[:, 0:]))

        elif typeDatabase == 'Nina5':
            dataMatrix = np.hstack(
                (lscaleMatrix[:, 8:CH * 2], mflMatrix[:, 8:CH * 2], msrMatrix[:, 8:CH * 2], wampMatrix[:, 8:]))

        elif typeDatabase == 'Cote':
            dataMatrix = np.hstack((lscaleMatrix[:, 0:CH], mflMatrix[:, 0:CH], msrMatrix[:, 0:CH], wampMatrix[:, 0:]))
            dataMatrix[:, allFeatures + 1] = dataMatrix[:, allFeatures + 1] + 1
            dataMatrix[:, allFeatures + 3] = dataMatrix[:, allFeatures + 3] + 1

    return dataMatrix, numberFeatures, CH, classes, people, peoplePriorK, peopleTrain, numberShots, combinationSet, allFeatures


#####################################################################

# Graph a ellipse of a normal distribution
# Taken from https://matplotlib.org/3.1.0/gallery/statistics/confidence_ellipse.html

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def get_correlated_dataset(n, dependency, mu, scale):
    latent = np.random.randn(n, 2)
    dependent = latent.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    # return x and y of the new, correlated dataset
    return scaled_with_offset[:, 0], scaled_with_offset[:, 1]


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
        fig1, ax = plt.subplots(1, 1, figsize=(5, 5))

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
                    confidence_ellipse(x1, x2, ax, edgecolor=color, label=label)
                else:
                    ax.scatter(x1, x2, s=0.5, color=color)
                    confidence_ellipse(x1, x2, ax, edgecolor=color)

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


def ResultsSyntheticData(DataFrame, nameFile, shots, people, samples, Features, classes, times, Graph):
    iSample = 3

    shot = np.arange(shots)
    clfLDA = LDA()
    yLDA = np.zeros(shots)
    clfQDA = QDA()
    yQDA = np.zeros(shots)
    yPropQ = np.zeros(shots)
    yPropL = np.zeros(shots)
    yLiu = np.zeros(shots)
    yLiuQDA = np.zeros(shots)
    wPropL = np.zeros(shots)
    wPropQ = np.zeros(shots)
    wLiu = np.ones(shots) * 0.5

    resultsData = pd.DataFrame(
        columns=['yLDA', 'yQDA', 'y10BBQ', 'y10BBL', 'yLiu', 'yLiuQDA', 'r10BBQ', 'r10BBL', 'rLiu',
                 'person', 'times', 'shots'])
    idx = 0

    for per in range(people):

        currentPerson = DataFrame[DataFrame['person'] == per].reset_index(drop=True)
        lenTest = int(samples / 2)
        preTrainedDataMatrix = DataFrame[DataFrame['person'] != per].reset_index(drop=True)
        preTrainedDataMatrix.at[:, 'class'] = preTrainedDataMatrix.loc[:, 'class'] + 1

        currentPersonTrain = pd.DataFrame(columns=['data', 'class'])
        currentValues = pd.DataFrame(columns=['data', 'mean', 'cov', 'class'])

        x_test = currentPerson.loc[0, 'data'][0:Features, 0:lenTest]
        y_test = np.ones(lenTest)
        currentPersonTrain.at[0, 'data'] = currentPerson.loc[0, 'data'][0:Features, lenTest:samples]
        currentPersonTrain.at[0, 'class'] = 1
        for cl in range(1, classes):
            currentPersonTrain.at[cl, 'data'] = currentPerson.loc[cl, 'data'][0:Features, lenTest:samples]
            currentPersonTrain.at[cl, 'class'] = cl + 1
            x_test = np.hstack((x_test, currentPerson.loc[cl, 'data'][0:Features, 0:lenTest]))
            y_test = np.hstack((y_test, np.ones(lenTest) * cl + 1))
        x_test = x_test.T

        for t in range(times):

            for i in range(shots):
                x_train = currentPersonTrain.loc[0, 'data'][:, t:i + t + iSample]
                y_train = np.ones(i + iSample)
                currentValues.at[0, 'data'] = currentPersonTrain.loc[0, 'data'][:, t:i + t + iSample]
                currentValues.at[0, 'mean'] = np.mean(x_train, axis=1)
                currentValues.at[0, 'cov'] = np.cov(x_train, rowvar=True)
                currentValues.at[0, 'class'] = 1

                for cl in range(1, classes):
                    x_train = np.hstack((x_train, currentPersonTrain.loc[cl, 'data'][:, t:i + t + iSample]))
                    y_train = np.hstack((y_train, np.ones(i + iSample) * cl + 1))
                    currentValues.at[cl, 'data'] = currentPersonTrain.loc[cl, 'data'][:, t:i + t + iSample]
                    currentValues.at[cl, 'mean'] = np.mean(currentValues.loc[cl, 'data'], axis=1)
                    currentValues.at[cl, 'cov'] = np.cov(currentValues.loc[cl, 'data'], rowvar=True)
                    currentValues.at[cl, 'class'] = cl + 1
                x_train = x_train.T

                clfLDA.fit(x_train, y_train)

                clfQDA.fit(x_train, y_train)

                step = 1
                propModelQDA, wQDA, wMeanQDA, tPropQ = ProposedModel(currentValues,
                                                                     preTrainedDataMatrix,
                                                                     classes, Features, x_train,
                                                                     y_train, step,
                                                                     'QDA')
                propModelLDA, wLDA, wMeanLDA, tPropL = ProposedModel(currentValues,
                                                                     preTrainedDataMatrix,
                                                                     classes, Features,
                                                                     x_train,
                                                                     y_train, step, 'LDA')

                liuModel = LiuModel(currentValues, preTrainedDataMatrix, classes, Features)

                resultsData.at[idx, 'yLDA'] = clfLDA.score(x_test, y_test)
                resultsData.at[idx, 'yQDA'] = clfQDA.score(x_test, y_test)
                resultsData.at[idx, 'yPropL'] = scoreModelLDA(x_test, y_test, propModelLDA, classes)
                resultsData.at[idx, 'yPropQ'] = scoreModelQDA(x_test, y_test, propModelQDA, classes)
                resultsData.at[idx, 'yLiu'] = scoreModelLDA(x_test, y_test, liuModel, classes)
                resultsData.at[idx, 'yLiuQ'] = scoreModelQDA(x_test, y_test, liuModel, classes)

                resultsData.at[idx, 'wPropQ'] = wMeanQDA
                resultsData.at[idx, 'wPropL'] = wMeanLDA
                resultsData.at[idx, 'rLiu'] = 0.5
                resultsData.at[idx, 'person'] = per
                resultsData.at[idx, 'times'] = t
                resultsData.at[idx, 'shots'] = i + iSample

                if Graph:
                    yLDA[i] += resultsData.loc[idx, 'yLDA']
                    yQDA[i] += resultsData.loc[idx, 'yQDA']
                    yPropL[i] += resultsData.loc[idx, 'yPropL']
                    yPropQ[i] += resultsData.loc[idx, 'yPropQ']
                    yLiu[i] += resultsData.loc[idx, 'yLiu']
                    yLiuQDA[i] += resultsData.loc[idx, 'yLiuQ']
                    wPropL[i] += resultsData.loc[idx, 'wPropL']
                    wPropQ[i] += resultsData.loc[idx, 'wPropQ']

                # print(resultsData.loc[idx])
                if nameFile is not None:
                    resultsData.to_csv(nameFile)
                idx += 1

                # print(per, t, i)

    if Graph:
        fig2, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6))

        ax1.plot(shot + iSample, yPropL / (times * people), label='PropoLDA', color='green')
        ax1.plot(shot + iSample, yLiu / (times * people), label='Liu', color='blue')
        ax1.plot(shot + iSample, yLDA / (times * people), label='BL LDA')

        ax1.set_title('ACC of models vs Samples (LDA)')
        ax1.grid()
        ax1.legend(loc='lower right', prop={'size': 8})

        ax1.set_ylabel('Acc')

        ax2.plot(shot + iSample, 1 - (wPropL / (times * people)), label='W PropoLDA', color='green')
        ax2.plot(shot + iSample, wLiu, label='W Liu', color='blue')
        ax2.set_title('The weight of the current distribution (1-r) vs Shots')
        ax2.grid()
        ax2.legend(loc='lower right', prop={'size': 8})
        ax2.set_xlabel('Shots')
        ax2.set_ylabel('W')

        fig3, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6))

        ax1.plot(shot + iSample, yPropQ / (times * people), label='PropoQDA', color='green')
        ax1.plot(shot + iSample, yLiuQDA / (times * people), label='Liu QDA', color='blue')
        ax1.plot(shot + iSample, yQDA / (times * people), label='BL QDA')

        ax1.set_title('Accuracy vs Samples')
        ax1.grid()
        ax1.legend(loc='lower right', prop={'size': 8})

        ax1.set_ylabel('Accuracy')

        ax2.plot(shot + iSample, 1 - (wPropQ / (times * people)), label='wPropQDA', color='green')
        ax2.plot(shot + iSample, wLiu, label='W LiuQDA', color='blue')
        ax2.set_title('Weight vs Shots')
        ax2.grid()
        ax2.legend(loc='lower right', prop={'size': 8})
        ax2.set_xlabel('Shots')
        ax2.set_ylabel('W')

        plt.show()
