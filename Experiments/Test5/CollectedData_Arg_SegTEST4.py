#!/usr/bin/env python
# coding: utf-8

# In[388]:


import numpy as np
import pandas as pd
import itertools
import sys
# import matplotlib.pyplot as plt
# import pickle

from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import GridSearchCV
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# from scipy import stats
from scipy.spatial import distance
from sklearn.decomposition import PCA

# from sklearn.preprocessing import RobustScaler


# In[523]:
## Input Variables 

featureSet=int(sys.argv[1])
startPerson=int(sys.argv[2])
endPerson=int(sys.argv[3])
place=str(sys.argv[4])
nameFile=place+'_FeatureSet_'+sys.argv[1]+'_startPerson_'+sys.argv[2]+'_endPerson_'+sys.argv[3]+'.csv'

# featureSet = 3
# startPerson = 1
# endPerson = 1
# nameFile = 'CollectpruebaNewMaha1.csv'


#Data Preparation
def preTrainedData(dataMatrix, classes, peoplePreTrain, evaluatedPerson, allFeatures):
    preTrainedDataMatrix = pd.DataFrame(columns=['mean', 'cov', 'class', 'person', 'covLDA'])
    typeData = 0
    indx = 0
    for cl in range(1, classes + 1):

        for person in range(1, peoplePreTrain + 1):
            if evaluatedPerson != person:
                preTrainedDataMatrix.at[indx, 'cov'] = np.cov(dataMatrix[np.where(
                    (dataMatrix[:, allFeatures] == typeData) * (
                            dataMatrix[:, allFeatures + 1] == person) * (
                            dataMatrix[:, allFeatures + 2] == cl)), 0:allFeatures][0], rowvar=False)
                preTrainedDataMatrix.at[indx, 'mean'] = np.mean(dataMatrix[np.where(
                    (dataMatrix[:, allFeatures] == typeData) * (
                            dataMatrix[:, allFeatures + 1] == person) * (
                            dataMatrix[:, allFeatures + 2] == cl)), 0:allFeatures][0], axis=0)
                preTrainedDataMatrix.at[indx, 'class'] = cl
                preTrainedDataMatrix.at[indx, 'person'] = person
                indx += 1
    # print(preTrainedDataMatrix)       
    return preTrainedDataMatrix


def currentDistributionValues(trainFeatures, trainLabels, classes, evaluatedPerson, allFeatures):
    currentValues = pd.DataFrame(columns=['cov', 'mean', 'class', 'person', 'covLDA'])
    trainLabelsAux = trainLabels[np.newaxis]
    Matrix = np.hstack((trainFeatures, trainLabelsAux.T))
    for cla in range(0, classes):
        currentValues.at[cla, 'cov'] = np.cov(
            Matrix[np.where((Matrix[:, allFeatures] == cla + 1)), 0:allFeatures][0], rowvar=False)
        currentValues.at[cla, 'mean'] = np.mean(
            Matrix[np.where((Matrix[:, allFeatures] == cla + 1)), 0:allFeatures][0], axis=0)
        currentValues.at[cla, 'class'] = cla + 1
        currentValues.at[cla, 'person'] = evaluatedPerson
    #     print(currentValues,trainFeatures,np.shape(trainFeatures))

    covAux = currentValues['cov'].sum() / classes
    for cla in range(0, classes):
        currentValues.at[cla, 'covLDA'] = covAux
    return currentValues


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


def reTrainedCovLiu(r, currentMean, currentCov, preTrainedDataMatrix, weightDenominatorV,allFeatures):
    sumAllPreTrainedCov_Weighted = np.zeros((allFeatures, allFeatures))
    for i in range(len(preTrainedDataMatrix.index)):
        sumAllPreTrainedCov_Weighted = np.add(sumAllPreTrainedCov_Weighted, preTrainedDataMatrix['cov'][i] * (
                1 / distance.mahalanobis(currentMean, preTrainedDataMatrix['mean'][i],
                                         np.linalg.inv(preTrainedDataMatrix['cov'][i]))))

    reTrainedCovValue = np.add((1 - r) * currentCov, (r / weightDenominatorV) * sumAllPreTrainedCov_Weighted)
    return reTrainedCovValue


def reTrainigLiu(currentValues, preTrainedDataMatrix, classes, allFeatures):
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


# Proposed Models

def weightDenominatorProposed7(currentMean, currentCov, preTrainedDataMatrix, allFeatures):
    weightDenominatorV = 0
    for i in range(len(preTrainedDataMatrix.index)):
        weightDenominatorV = weightDenominatorV + (
                1 / KL_DivergenceDistance(currentMean, currentCov, preTrainedDataMatrix['mean'].loc[i],
                                          preTrainedDataMatrix['cov'].loc[i], allFeatures))
    return weightDenominatorV



def reTrainedCalculatedMeanProposed7(currentMean, currentCov, preTrainedDataMatrix, weightDenominatorV, allFeatures):
    sumAllPreTrainedMean_Weighted = np.zeros((1, allFeatures))
    for i in range(len(preTrainedDataMatrix.index)):
        sumAllPreTrainedMean_Weighted = np.add(sumAllPreTrainedMean_Weighted, preTrainedDataMatrix['mean'].loc[i] * (
                1 / KL_DivergenceDistance(currentMean, currentCov, preTrainedDataMatrix['mean'].loc[i],
                                          preTrainedDataMatrix['cov'].loc[i], allFeatures)))

    reTrainedCalculatedMeanValue = sumAllPreTrainedMean_Weighted / weightDenominatorV
    return reTrainedCalculatedMeanValue


def reTrainedCalculatedCovProposed7(currentMean, currentCov, preTrainedDataMatrix, weightDenominatorV, allFeatures):
    sumAllPreTrainedCov_Weighted = np.zeros((allFeatures, allFeatures))
    for i in range(len(preTrainedDataMatrix.index)):
        sumAllPreTrainedCov_Weighted = np.add(sumAllPreTrainedCov_Weighted, preTrainedDataMatrix['cov'][i] * (
                1 / KL_DivergenceDistance(currentMean, currentCov, preTrainedDataMatrix['mean'][i],
                                          preTrainedDataMatrix['cov'][i], allFeatures)))

    reTrainedCalculatedCovValue = sumAllPreTrainedCov_Weighted / weightDenominatorV
    return reTrainedCalculatedCovValue


def KL_DivergenceDistance(mean_1, cov_1, mean_2, cov_2, k):
    det1 = np.linalg.det(cov_1)
    det2 = np.linalg.det(cov_2)

    if det1 == 0 or det2 == 0:

        partA = 0
        partB = np.dot(np.dot((mean_1 - mean_2), np.linalg.pinv(cov_1)), (mean_1 - mean_2).T)
        partC = 0

    else:
        partA = np.trace(np.dot(np.linalg.inv(cov_1), cov_2))
        partB = np.dot(np.dot((mean_1 - mean_2), np.linalg.inv(cov_1)), (mean_1 - mean_2).T)
        partC = np.log(np.linalg.det(cov_1) / np.linalg.det(cov_2))

    result = (1 / 2) * (partA + partB - k + partC)
    #     print(result)
    return result

def rCalculatedProposed10BB(currentValues, personMean, personCov, currentClass, classes, allFeatures):
    # # aux=np.zeros((2,classes))
    # d = np.zeros((1, classes))
    # countr = 0
    # count = 0
    # for i in range(0, classes):
    #
    #     if i != currentClass:
    #
    #         a = distance.mahalanobis(currentValues['mean'].loc[i],currentValues['mean'].loc[currentClass],
    #                                  currentValues['cov'].loc[currentClass])
    #         a1 = distance.mahalanobis(currentValues['mean'].loc[currentClass], currentValues['mean'].loc[i],
    #                                  currentValues['cov'].loc[i])
    #         b = distance.mahalanobis( currentValues['mean'].loc[i],personMean,
    #                                   personCov)
    #         b1 = distance.mahalanobis(personMean, currentValues['mean'].loc[i],
    #                                   currentValues['cov'].loc[i])
    #         d[0, i] = b
    #         if a < b:
    #             countr += abs(a - b)
    #             count += abs(a - b)
    #         else:
    #             count += abs(a - b)
    #     else:
    #
    #         a = distance.mahalanobis(currentValues['mean'].loc[i], currentValues['mean'].loc[currentClass],
    #                                  currentValues['cov'].loc[currentClass])
    #         a1 = distance.mahalanobis(currentValues['mean'].loc[currentClass], currentValues['mean'].loc[i],
    #                                   currentValues['cov'].loc[i])
    #         b = distance.mahalanobis(currentValues['mean'].loc[i], personMean,
    #                                  personCov)
    #         b1 = distance.mahalanobis(personMean, currentValues['mean'].loc[i],
    #                                   currentValues['cov'].loc[i])
    #         d[0, i] = b
    #         if a > b:
    #             countr += (classes - 1) * abs(a - b)
    #             count += (classes - 1) * abs(a - b)
    #         else:
    #             count += (classes - 1) * abs(a - b)
    #
    # if np.where(d == np.amin(d))[1] == currentClass:
    #     r = countr / count
    # #         peopleNumb=1
    # else:
    #     r = 0



    b = np.zeros((1, classes))
    c = np.zeros((1, classes))
    b1 = np.zeros((1, classes))
    c1 = np.zeros((1, classes))
    for i in range(0, classes):

        c1[0, i] = distance.mahalanobis(currentValues['mean'].loc[i], currentValues['mean'].loc[currentClass],
                                 currentValues['cov'].loc[currentClass])
        c[0, i] = distance.mahalanobis(currentValues['mean'].loc[currentClass], currentValues['mean'].loc[i],
                                  currentValues['cov'].loc[i])
        b1[0, i] = distance.mahalanobis(currentValues['mean'].loc[i], personMean,
                                 personCov)
        b[0, i] = distance.mahalanobis(personMean, currentValues['mean'].loc[i],
                                  currentValues['cov'].loc[i])
    b = (b + b1)
    c = (c + c1)

    x = b[b != b[0, currentClass]]
    y = c[c != c[0, currentClass]]
    # r1 = ((y - b[0, currentClass])[(y - b[0, currentClass]) >= 0]).sum() / np.abs(y - b[0, currentClass]).sum()
    P1 = (x / (x + b[0, currentClass])).sum()
    P2 = (x / (x + y)).sum()
    Current1 = classes - 1
    Current2 = (y / (x + y)).sum()
    C = (Current1 + Current2)
    P = (P1 + P2)
    r = P / (C + P)


    #         peopleNumb=1

    #     r=countr/count



    # b = np.zeros((1, classes))
    # c = np.zeros((1, classes))
    #
    # for i in range(0, classes):
    #     b[0, i] = LDA_Discriminant(currentValues['mean'].loc[i], personCov, personMean)
    #
    #     c[0, i] = LDA_Discriminant(currentValues['mean'].loc[i], currentValues['cov'].loc[currentClass]
    #                                    ,currentValues['mean'].loc[currentClass])
    #
    #
    #
    # b = b - np.min(b)
    # c = c - np.min(c)
    #
    # x = b[b != b[0, currentClass]]
    # y = c[c != c[0, currentClass]]
    # r1 = ((- x + b[0, currentClass])[(-x + b[0, currentClass]) >= 0]).sum() / np.abs(-x + b[0, currentClass]).sum()
    # r2 = ((- y + c[0, currentClass])[(-y + c[0, currentClass]) >= 0]).sum() / np.abs(-y + c[0, currentClass]).sum()
    #
    # if r1==0 and r2==0:
    #     r=0
    # else:
    #     r = r1 / (r1+r2)








    # b = np.zeros((1, classes))
    # c = np.zeros((1, classes))
    #
    # for i in range(0, classes):
    #     b[0, i] = distance.mahalanobis(currentValues['mean'].loc[i], personMean, personCov)
    #
    #     c[0, i] = distance.mahalanobis(currentValues['mean'].loc[i],
    #                                    currentValues['mean'].loc[currentClass], currentValues['cov'].loc[currentClass])
    #
    # x = b[0, np.where(b != b[0, currentClass])[1]]
    # y = c[0, np.where(c != c[0, currentClass])[1]]
    # r1 = ((x - b[0, currentClass])[(x - b[0, currentClass]) >= 0]).sum() / np.abs(x - b[0, currentClass]).sum()
    # r2 = (x / (x + y)).mean()
    # r = r1 * r2
    return r


def reTrainigProposed10BB(currentValues, preTrainedDataMatrix, classes, allFeatures):
    trainedModel = pd.DataFrame(columns=['cov', 'mean', 'class'])
    # MeansCovs= pd.DataFrame(columns=['currentCov','currentMean','calculatedCov','calculatedMean','class'])

    for cla in range(0, classes):
        trainedModel.at[cla, 'cov'] = np.zeros((allFeatures, allFeatures))
        trainedModel.at[cla, 'mean'] = np.zeros((1, allFeatures))[0]

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
            r = rCalculatedProposed10BB(currentValues, personMean, personCov, cla, classes, allFeatures)

            trainedModel.at[cla, 'cov'] = trainedModel['cov'].loc[cla] + (
                    (1 - r) * currentCov + r * personCov) / peopleClass
            trainedModel.at[cla, 'mean'] = trainedModel['mean'].loc[cla] + (
                    (1 - r) * currentMean + r * personMean) / peopleClass

        trainedModel.at[cla, 'class'] = cla + 1
    return trainedModel


def reTrainigProposed11BB(currentValues, preTrainedDataMatrix, classes, allFeatures):
    trainedModel = pd.DataFrame(columns=['cov', 'mean', 'class'])
    MeansCovs = pd.DataFrame(columns=['currentCov', 'currentMean', 'calculatedCov', 'calculatedMean', 'class'])

    for cla in range(0, classes):
        preTrainedMatrix_Class = pd.DataFrame(
            preTrainedDataMatrix[['cov', 'mean']].loc[(preTrainedDataMatrix['class'] == cla + 1)])
        preTrainedMatrix_Class = preTrainedMatrix_Class.reset_index(drop=True)
        currentCov = currentValues['cov'].loc[cla]
        currentMean = currentValues['mean'].loc[cla]
        weightDenominatorV = weightDenominatorLiu(currentMean, preTrainedMatrix_Class)
        MeansCovs.at[cla, 'currentCov'] = currentCov
        MeansCovs.at[cla, 'currentMean'] = currentMean
        MeansCovs.at[cla, 'calculatedCov'] = reTrainedCovLiu(1, currentMean, currentCov,
                                                             preTrainedMatrix_Class, weightDenominatorV, allFeatures)
        MeansCovs.at[cla, 'calculatedMean'] = reTrainedMeanLiu(1, currentMean,
                                                               preTrainedMatrix_Class,
                                                               weightDenominatorV, allFeatures)[0]
        MeansCovs.at[cla, 'class'] = cla + 1

    for cla in range(0, classes):
        r = rCalculatedProposed10BB(currentValues, MeansCovs['calculatedMean'].loc[cla],
                                    MeansCovs['calculatedCov'].loc[cla], cla, classes, allFeatures)

        print(r)
        trainedModel.at[cla, 'cov'] = reTrainedCovProposed(r, MeansCovs['currentCov'].loc[cla],
                                                           MeansCovs['calculatedCov'].loc[cla])
        trainedModel.at[cla, 'mean'] = reTrainedMeanProposed(r, MeansCovs['currentMean'].loc[cla],
                                                           MeansCovs['calculatedMean'].loc[cla])

        trainedModel.at[cla, 'class'] = cla + 1

    return trainedModel


def reTrainedMeanProposed(r, currentMean, reTrainedCalculatedMeanValue):
    reTrainedMeanValue = np.add((1 - r) * currentMean, r * reTrainedCalculatedMeanValue)
    return reTrainedMeanValue


def reTrainedCovProposed(r, currentCov, reTrainedCalculatedCovValue):
    reTrainedCovValue = np.add((1 - r) * currentCov, r * reTrainedCalculatedCovValue)
    return reTrainedCovValue


# LDA

def LDA_Discriminant(x, covariance, mean):
    invCov = np.linalg.inv(covariance)
    discriminant = np.dot(np.dot(x, invCov), mean) - 0.5 * np.dot(np.dot(mean, invCov), mean)
    return discriminant


def LDA_Cov(trainedModel, classes, allFeatures):
    LDACov = np.zeros((allFeatures, allFeatures))
    factor = 1 / classes
    for cl in range(0, classes):
        LDACov = LDACov + factor * trainedModel['cov'].loc[cl]

    return LDACov


def predictedModelLDA(sample, model, classes, LDACov):
    # pi_k = 1 / classes
    cl = 0
    predictedClass = cl + 1

    mean = model['mean'].loc[cl]
    d = LDA_Discriminant(sample, LDACov, mean)
    # print(dist,cl)
    for cl in range(1, classes):

        mean = model['mean'].loc[cl]
        dActual = LDA_Discriminant(sample, LDACov, mean)
        # print(distActual,cl)
        if dActual > d:
            predictedClass = cl + 1
            d = dActual
    return predictedClass


def scoreModelLDA(testFeatures, testLabels, model, classes):
    true = 0
    count = 0
    LDACov = LDA_Cov(model, classes, allFeatures)
    for i in range(0, np.size(testLabels)):
        currentPredictor = predictedModelLDA(testFeatures[i, :], model, classes, LDACov)
        if currentPredictor == testLabels[i]:
            true += 1
            count += 1
        else:
            count += 1
    #     print(actualPredictor)
    return true / count


# QDA
def scaledposterior(x, cov_k, u_k, pi_k):
    #PseudoQuadratic
    # det = np.linalg.det(cov_k)
    # a = -.5 * np.log(det)
    b = -.5 * np.dot(np.dot((x - u_k), np.linalg.pinv(cov_k)), (x - u_k).T)
    # d = a + b
    return b


def predictedModelQDA(sample, model, classes):
    pi_k = 1 / classes
    cl = 0
    predictedClass = cl + 1
    cov = model['cov'].loc[cl]
    mean = model['mean'].loc[cl]
    d = scaledposterior(sample, cov, mean, pi_k)
    # print(dist,cl)
    for cl in range(1, classes):

        cov = model['cov'].loc[cl]
        mean = model['mean'].loc[cl]
        dActual = scaledposterior(sample, cov, mean, pi_k)
        # print(distActual,cl)
        if dActual > d:
            predictedClass = cl + 1
            d = dActual
    return predictedClass


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
    #     print(actualPredictor)
    return true / count


def SamplesFew_ShotTrainedModels(model, samples, classes):
    xTrain = []
    yTrain = []
    for cl in range(0, classes):
        meanDistribution = model['mean'].loc[cl]
        covDistribution = model['cov'].loc[cl]
        xTrain.extend(np.random.multivariate_normal(meanDistribution, covDistribution, samples, check_valid='ignore'))
        yTrain.extend((cl + 1) * np.ones(samples))
    return xTrain, yTrain


def evaluation(dataMatrix, segment, classes, people, CH, numberFeatures, featureSet, numberShots, combinationSet
               , generalClassifierKNN, pBest, n_neighborsBest, weightsBest
               , generalClassifierSVM, CBest, kernelBest, gammaBest, nameFile, startPerson, endPerson, allFeatures):


    ClassifierKNNIndividual = KNeighborsClassifier(n_neighbors=n_neighborsBest, weights=weightsBest, p=pBest)
    ClassifierKNNIndividualGenerated = KNeighborsClassifier(n_neighbors=n_neighborsBest, weights=weightsBest, p=pBest)
    ClassifierKNNLiu = KNeighborsClassifier(n_neighbors=n_neighborsBest, weights=weightsBest, p=pBest)
    ClassifierKNN10BB = KNeighborsClassifier(n_neighbors=n_neighborsBest, weights=weightsBest, p=pBest)
    ClassifierKNN11BB = KNeighborsClassifier(n_neighbors=n_neighborsBest, weights=weightsBest, p=pBest)

    ClassifierSVMIndividual = SVC(C=CBest, kernel=kernelBest, gamma=gammaBest)
    ClassifierSVMIndividualGenerated = SVC(C=CBest, kernel=kernelBest, gamma=gammaBest)
    ClassifierSVMLiu = SVC(C=CBest, kernel=kernelBest, gamma=gammaBest)
    ClassifierSVM10BB = SVC(C=CBest, kernel=kernelBest, gamma=gammaBest)
    ClassifierSVM11BB = SVC(C=CBest, kernel=kernelBest, gamma=gammaBest)

    # Creating Variables
    resultsFeatureSet = pd.DataFrame(columns=['person', 'subset', '# shots', 'Feature Set'
        , 'AccLDAInd', 'AccLDALiu', 'AccLDA10BB', 'AccLDA11BB'
        , 'AccQDAInd', 'AccQDALiu', 'AccQDA10BB', 'AccQDA11BB'
        , 'AccKNNInd', 'AccKNNIndGen', 'AccKNNLiu', 'AccKNN10BB', 'AccKNN11BB'
        , 'AccSVMInd', 'AccSVMGIndGen', 'AccSVMLiu', 'AccSVM10BB', 'AccSVM11BB'])

    auxIndex1 = 0

    samples = 1000

    for person in range(startPerson, endPerson + 1):

        preTrainedDataMatrix = preTrainedData(dataMatrix, classes, people, person, allFeatures)
        typeData = 1
        testFeatures = dataMatrix[
                       np.where((dataMatrix[:, allFeatures] == typeData) * (
                               dataMatrix[:, allFeatures + 1] == person)
                                ), 0:allFeatures][0]
        testLabels = dataMatrix[
            np.where(
                (dataMatrix[:, allFeatures] == typeData) * (dataMatrix[:, allFeatures + 1] == person)
            ), allFeatures + 2][0]

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
                    trainFeatures = np.vstack((trainFeatures, dataMatrix[
                                                              np.where(
                                                                  (dataMatrix[:, allFeatures] == typeData) * (
                                                                          dataMatrix[:,
                                                                          allFeatures + 1] == person)
                                                                  * (dataMatrix[:, allFeatures + 3] == subset[
                                                                      auxIndex]))
                    , 0:allFeatures][0]))
                    trainLabels = np.hstack((trainLabels, dataMatrix[
                        np.where((dataMatrix[:, allFeatures] == typeData) * (
                                dataMatrix[:, allFeatures + 1] == person)
                                 * (dataMatrix[:, allFeatures + 3] == subset[auxIndex]))
                        , allFeatures + 2][0].T))

                currentValues = currentDistributionValues(trainFeatures, trainLabels, classes,
                                                          person,allFeatures)

                trainedModelLiu = reTrainigLiu(currentValues, preTrainedDataMatrix, classes,allFeatures)
                trainedModel10BB = reTrainigProposed10BB(currentValues, preTrainedDataMatrix,
                                                         classes,allFeatures)
                trainedModel11BB = reTrainigProposed11BB(currentValues, preTrainedDataMatrix,
                                                         classes,allFeatures)

                xTrainIndividualGenerated, yTrainIndividualGenerated = SamplesFew_ShotTrainedModels(currentValues,
                                                                                                    samples, classes)
                xTrainLiu, yTrainLiu = SamplesFew_ShotTrainedModels(trainedModelLiu, samples, classes)
                xTrain10BB, yTrain10BB = SamplesFew_ShotTrainedModels(trainedModel10BB, samples, classes)
                xTrain11BB, yTrain11BB = SamplesFew_ShotTrainedModels(trainedModel11BB, samples, classes)

                ClassifierKNNIndividual.fit(trainFeatures, trainLabels)
                ClassifierKNNIndividualGenerated.fit(xTrainIndividualGenerated, yTrainIndividualGenerated)
                ClassifierKNNLiu.fit(xTrainLiu, yTrainLiu)
                ClassifierKNN10BB.fit(xTrain10BB, yTrain10BB)
                ClassifierKNN11BB.fit(xTrain11BB, yTrain11BB)

                ClassifierSVMIndividual.fit(trainFeatures, trainLabels)
                ClassifierSVMIndividualGenerated.fit(xTrainIndividualGenerated, yTrainIndividualGenerated)
                ClassifierSVMLiu.fit(xTrainLiu, yTrainLiu)
                ClassifierSVM10BB.fit(xTrain10BB, yTrain10BB)
                ClassifierSVM11BB.fit(xTrain11BB, yTrain11BB)

                resultsFeatureSet.loc[auxIndex1] = [person, subset, np.size(subset), featureSet
                    , scoreModelLDA(testFeatures, testLabels, currentValues, classes)
                    , scoreModelLDA(testFeatures, testLabels, trainedModelLiu, classes)
                    , scoreModelLDA(testFeatures, testLabels, trainedModel10BB, classes)
                    , scoreModelLDA(testFeatures, testLabels, trainedModel11BB, classes)
                    , scoreModelQDA(testFeatures, testLabels, currentValues, classes)
                    , scoreModelQDA(testFeatures, testLabels, trainedModelLiu, classes)
                    , scoreModelQDA(testFeatures, testLabels, trainedModel10BB, classes)
                    , scoreModelQDA(testFeatures, testLabels, trainedModel11BB, classes)
                    , ClassifierKNNIndividual.score(testFeatures, testLabels)
                    , ClassifierKNNIndividualGenerated.score(testFeatures, testLabels)
                    , ClassifierKNNLiu.score(testFeatures, testLabels)
                    , ClassifierKNN10BB.score(testFeatures, testLabels)
                    , ClassifierKNN11BB.score(testFeatures, testLabels)
                    , ClassifierSVMIndividual.score(testFeatures, testLabels)
                    , ClassifierSVMIndividualGenerated.score(testFeatures, testLabels)
                    , ClassifierSVMLiu.score(testFeatures, testLabels)
                    , ClassifierSVM10BB.score(testFeatures, testLabels)
                    , ClassifierSVM11BB.score(testFeatures, testLabels)]
                resultsFeatureSet.to_csv(nameFile)
                print(featureSet)
                print('Results: person= ', person, ' shot set= ', subset)
                print(resultsFeatureSet.loc[auxIndex1])
                auxIndex1 += 1

    return resultsFeatureSet


## Collected Data
resultsParameters = pd.DataFrame(
    columns=['pBest', 'n_neighborsBest', 'weightsBest', 'CBest', 'kernelBest', 'gammaBest'])
Index = 0
# ## Feature Set 1 (MAV)

# ### Setting variables

# In[512]:
if featureSet == 1:
    # Setting variables
    Feature1 = 'mavMatrix'
    carpet = 'ExtractedDataCollectedData'
    segment = ''
    classes = 5
    people = 60
    CH = 8
    numberFeatures = 1
    numberShots = 4
    combinationSet = list(range(1, 26))
    # Getting Data
    mavMatrix = np.genfromtxt('../../ExtractedData/' + carpet + '/' + Feature1 + segment + '.csv', delimiter=',')
    dataMatrix = mavMatrix[:, 0:CH]
    # Preprocessing
    dataMatrix=preprocessing.scale(dataMatrix)
    pca = PCA(n_components='mle',svd_solver = 'full')
    principalComponents = pca.fit_transform(dataMatrix)
    dataMatrix = np.hstack((principalComponents, mavMatrix[:, CH:]))
    allFeatures=np.size(principalComponents, axis=1)
    print('Features: ',allFeatures)
    ### Evaluation

    resultsParameters = pd.read_csv('HPT_Parameters_Seg_CollectedData_1.csv')
    n_neighborsBest = resultsParameters['n_neighborsBest'].loc[0]
    weightsBest = resultsParameters['weightsBest'].loc[0]
    pBest = resultsParameters['pBest'].loc[0]
    CBest = resultsParameters['CBest'].loc[0]
    kernelBest = resultsParameters['kernelBest'].loc[0]
    gammaBest = resultsParameters['gammaBest'].loc[0]
    generalClassifierKNN = 0
    generalClassifierSVM = 0

    resultsFeatureSet11 = evaluation(dataMatrix, segment, classes, people, CH, numberFeatures, featureSet, numberShots,
                                     combinationSet
                                     , generalClassifierKNN, pBest, n_neighborsBest, weightsBest, generalClassifierSVM,
                                     CBest, kernelBest, gammaBest
                                     , nameFile, startPerson, endPerson,allFeatures)


# In[520]:

elif featureSet == 2:
    # Setting variables 
    Feature1 = 'mavMatrix'
    Feature2 = 'wlMatrix'
    Feature3 = 'zcMatrix'
    Feature4 = 'sscMatrix'
    carpet = 'ExtractedDataCollectedData'
    segment = ''
    classes = 5
    people = 60
    CH = 8
    numberFeatures = 4
    numberShots = 4
    combinationSet = list(range(1, 26))

    mavMatrix = np.genfromtxt('../../ExtractedData/' + carpet + '/' + Feature1 + segment + '.csv', delimiter=',')
    wlMatrix = np.genfromtxt('../../ExtractedData/' + carpet + '/' + Feature2 + segment + '.csv', delimiter=',')
    zcMatrix = np.genfromtxt('../../ExtractedData/' + carpet + '/' + Feature3 + segment + '.csv', delimiter=',')
    sscMatrix = np.genfromtxt('../../ExtractedData/' + carpet + '/' + Feature4 + segment + '.csv', delimiter=',')
    dataMatrix = np.hstack((mavMatrix[:, 0:CH], wlMatrix[:, 0:CH], zcMatrix[:, 0:CH], sscMatrix[:, 0:CH]))
    # Preprocessing
    dataMatrix = preprocessing.scale(dataMatrix)
    pca = PCA(n_components='mle',svd_solver = 'full')
    principalComponents = pca.fit_transform(dataMatrix)
    dataMatrix = np.hstack((principalComponents, sscMatrix[:, CH:]))
    allFeatures = np.size(principalComponents, axis=1)
    print('Features: ', allFeatures)


    ### Evaluation

    resultsParameters = pd.read_csv('HPT_Parameters_Seg_CollectedData_2.csv')
    n_neighborsBest = resultsParameters['n_neighborsBest'].loc[0]
    weightsBest = resultsParameters['weightsBest'].loc[0]
    pBest = resultsParameters['pBest'].loc[0]
    CBest = resultsParameters['CBest'].loc[0]
    kernelBest = resultsParameters['kernelBest'].loc[0]
    gammaBest = resultsParameters['gammaBest'].loc[0]
    generalClassifierKNN = 0
    generalClassifierSVM = 0

    resultsFeatureSet21 = evaluation(dataMatrix, segment, classes, people, CH, numberFeatures, featureSet, numberShots,
                                     combinationSet
                                     , generalClassifierKNN, pBest, n_neighborsBest, weightsBest, generalClassifierSVM,
                                     CBest, kernelBest, gammaBest
                                     , nameFile, startPerson, endPerson,allFeatures)

elif featureSet == 3:
    # Setting variables
    Feature1 = 'lscaleMatrix'
    Feature2 = 'mflMatrix'
    Feature3 = 'msrMatrix'
    Feature4 = 'wampMatrix'
    carpet = 'ExtractedDataCollectedData'
    segment = ''
    classes = 5
    people = 60
    CH = 8
    numberFeatures = 4
    numberShots = 4
    combinationSet = list(range(1, 26))
    # Getting Data
    lscaleMatrix = np.genfromtxt('../../ExtractedData/' + carpet + '/' + Feature1 + segment + '.csv', delimiter=',')
    mflMatrix = np.genfromtxt('../../ExtractedData/' + carpet + '/' + Feature2 + segment + '.csv', delimiter=',')
    msrMatrix = np.genfromtxt('../../ExtractedData/' + carpet + '/' + Feature3 + segment + '.csv', delimiter=',')
    wampMatrix = np.genfromtxt('../../ExtractedData/' + carpet + '/' + Feature4 + segment + '.csv', delimiter=',')
    dataMatrix = np.hstack((lscaleMatrix[:, 0:CH], mflMatrix[:, 0:CH], msrMatrix[:, 0:CH], wampMatrix[:, 0:CH]))

    # Preprocessing
    dataMatrix = preprocessing.scale(dataMatrix)
    pca = PCA(n_components='mle',svd_solver = 'full')
    principalComponents = pca.fit_transform(dataMatrix)
    dataMatrix = np.hstack((principalComponents, wampMatrix[:, CH:]))
    allFeatures = np.size(principalComponents, axis=1)
    print('Features: ', allFeatures)


    # ### Evaluation

    resultsParameters = pd.read_csv('HPT_Parameters_Seg_CollectedData_3.csv')
    n_neighborsBest = resultsParameters['n_neighborsBest'].loc[0]
    weightsBest = resultsParameters['weightsBest'].loc[0]
    pBest = resultsParameters['pBest'].loc[0]
    CBest = resultsParameters['CBest'].loc[0]
    kernelBest = resultsParameters['kernelBest'].loc[0]
    gammaBest = resultsParameters['gammaBest'].loc[0]
    generalClassifierKNN = 0
    generalClassifierSVM = 0

    resultsFeatureSet31 = evaluation(dataMatrix, segment, classes, people, CH, numberFeatures, featureSet, numberShots,
                                     combinationSet
                                     , generalClassifierKNN, pBest, n_neighborsBest, weightsBest, generalClassifierSVM,
                                     CBest, kernelBest, gammaBest
                                     , nameFile, startPerson, endPerson,allFeatures)