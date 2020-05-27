#!/usr/bin/env python
# coding: utf-8

# In[388]:


import numpy as np
import pandas as pd
import itertools
import sys
# import time
import math
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from scipy.spatial import distance
from sklearn.decomposition import PCA


## Input Variables 

featureSet=int(sys.argv[1])
startPerson=int(sys.argv[2])
endPerson=int(sys.argv[3])
place=str(sys.argv[4])
nameFile=place+'_FeatureSet_'+sys.argv[1]+'_startPerson_'+sys.argv[2]+'_endPerson_'+sys.argv[3]+'.csv'

# featureSet = 1
# startPerson = 11
# endPerson = 11
# nameFile = 'CollectpruebaNewMaha1.csv'



pca=True
pcaComp=0.999


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

    partA = np.trace(np.dot(np.linalg.inv(cov_1), cov_2))
    partB = np.dot(np.dot((mean_1 - mean_2), np.linalg.inv(cov_1)), (mean_1 - mean_2).T)
    partC = np.log(np.linalg.det(cov_1) / np.linalg.det(cov_2))

    result = (1 / 2) * (partA + partB - k + partC)

    return result


def scoreModelLDA_r(testFeatures, testLabels, model, classes,currentClass,step):
    trueP = 0

    falseP = 0
    falseN = 0

    currentClass=currentClass+1
    LDACov = LDA_Cov(model, classes)
    for i in range(0, np.size(testLabels),step):
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


def scoreModelQDA_r(testFeatures, testLabels, model, classes,currentClass,step):
    trueP = 0

    falseP = 0
    falseN = 0

    currentClass = currentClass + 1
    for i in range(0, np.size(testLabels),step):
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

def scoreModelLDA_ALL(testFeatures, testLabels, model, classes,step):
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

    return F1

def scoreModelQDA_ALL(testFeatures, testLabels, model, classes,step):
    trueP = np.zeros([classes])

    falseP = np.zeros([classes])
    falseN = np.zeros([classes])



    for i in range(0, np.size(testLabels),step):
        currentPredictor = predictedModelQDA(testFeatures[i, :], model, classes)

        if currentPredictor == testLabels[i]:
            trueP[int(testLabels[i] - 1)] += 1
        else:
            falseN[int(testLabels[i] - 1)] += 1
            falseP[int(currentPredictor - 1)] += 1


    recall = trueP / (trueP + falseN)
    precision = trueP / (trueP + falseP)
    F1 = 2 * (precision * recall) / (precision + recall)

    return F1


def rCalculatedProposed10BB_r(currentValues, personMean, personCov, currentClass, classes
                              ,trainFeatures,trainLabels,F1C,step,typeModel):
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




def reTrainigProposed10BB(currentValues, preTrainedDataMatrix, classes, allFeatures,trainFeatures,trainLabels,step,typeModel):
    trainedModel = pd.DataFrame(columns=['cov', 'mean', 'class'])

    for cla in range(0, classes):
        trainedModel.at[cla, 'cov'] = np.zeros((allFeatures, allFeatures))
        trainedModel.at[cla, 'mean'] = np.zeros((1, allFeatures))[0]
    if typeModel == 'LDA':
        F1C = scoreModelLDA_ALL(trainFeatures, trainLabels, currentValues, classes,step)
    elif typeModel == 'QDA':
        F1C = scoreModelQDA_ALL(trainFeatures, trainLabels, currentValues, classes,step)

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
                                                    , trainFeatures, trainLabels, F1C[cla],step,typeModel)

            rClass[cla] = (1 - r) + rClass[cla]
            trainedModel.at[cla, 'cov'] = trainedModel['cov'].loc[cla] + (
                    (1 - r) * currentCov + r * personCov) / peopleClass

            trainedModel.at[cla, 'mean'] = trainedModel['mean'].loc[cla]+ (
                    (1 - r) * currentMean + r * personMean) / peopleClass

        trainedModel.at[cla, 'class'] = cla + 1
    return trainedModel, rClass/peopleClass , rClass.mean()/peopleClass


def reTrainigProposed11BB(currentValues, preTrainedDataMatrix, classes, allFeatures,trainFeatures,trainLabels,step,typeModel):
    trainedModel = pd.DataFrame(columns=['cov', 'mean', 'class'])
    MeansCovs = pd.DataFrame(columns=['currentCov', 'currentMean', 'calculatedCov', 'calculatedMean', 'class'])

    for cla in range(0, classes):
        preTrainedMatrix_Class = pd.DataFrame(
            preTrainedDataMatrix[['cov', 'mean']].loc[(preTrainedDataMatrix['class'] == cla + 1)])
        preTrainedMatrix_Class = preTrainedMatrix_Class.reset_index(drop=True)
        currentCov = currentValues['cov'].loc[cla]
        currentMean = currentValues['mean'].loc[cla]
        weightDenominatorV = weightDenominatorProposed7(currentMean, currentCov, preTrainedMatrix_Class, allFeatures)
        MeansCovs.at[cla, 'currentCov'] = currentCov
        MeansCovs.at[cla, 'currentMean'] = currentMean
        MeansCovs.at[cla, 'calculatedCov'] = reTrainedCalculatedCovProposed7(currentMean, currentCov,
                                                                             preTrainedMatrix_Class, weightDenominatorV,
                                                                             allFeatures)
        MeansCovs.at[cla, 'calculatedMean'] = reTrainedCalculatedMeanProposed7(currentMean, currentCov,
                                                                               preTrainedMatrix_Class,
                                                                               weightDenominatorV, allFeatures)[0]

        MeansCovs.at[cla, 'class'] = cla + 1
    if typeModel == 'LDA':
        F1C = scoreModelLDA_ALL(trainFeatures, trainLabels, currentValues, classes,step)
    elif typeModel == 'QDA':
        F1C = scoreModelQDA_ALL(trainFeatures, trainLabels, currentValues, classes,step)

    rClass=np.zeros(classes)
    for cla in range(0, classes):

        r = rCalculatedProposed10BB_r(currentValues, MeansCovs['calculatedMean'].loc[cla],
                                                MeansCovs['calculatedCov'].loc[cla], cla, classes
                                                , trainFeatures, trainLabels,F1C[cla],step,typeModel)
        rClass[cla] = 1 - r
        trainedModel.at[cla, 'cov'] = reTrainedCovProposed(r, MeansCovs['currentCov'].loc[cla],
                                                           MeansCovs['calculatedCov'].loc[cla])

        trainedModel.at[cla, 'mean'] = reTrainedMeanProposed(r, MeansCovs['currentMean'].loc[cla],
                                                             MeansCovs['calculatedMean'].loc[cla])

        trainedModel.at[cla, 'class'] = cla + 1

    return trainedModel, rClass, rClass.mean()


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


def LDA_Cov(trainedModel, classes):
    LDACov = trainedModel['cov'].sum()/ classes
    return LDACov


def predictedModelLDA(sample, model, classes, LDACov):
    d=np.zeros([classes])
    for cl in range(classes):
        d[cl] = LDA_Discriminant(sample, LDACov, model['mean'].loc[cl])
    return np.argmax(d)+1



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


# QDA
def QDA_Discriminant(x, cov_k, u_k):
    #PseudoQuadratic
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
    #     print(actualPredictor)
    return true / count

def scoreModelQDA_Classification(testFeatures, testLabels, model, classes,testRepetitions):
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


def scoreModelLDA_Classification(testFeatures, testLabels, model, classes,testRepetitions):
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
    ClassifierKNN10BBLDA = KNeighborsClassifier(n_neighbors=n_neighborsBest, weights=weightsBest, p=pBest)
    ClassifierKNN11BBLDA = KNeighborsClassifier(n_neighbors=n_neighborsBest, weights=weightsBest, p=pBest)
    ClassifierKNN10BBQDA = KNeighborsClassifier(n_neighbors=n_neighborsBest, weights=weightsBest, p=pBest)
    ClassifierKNN11BBQDA = KNeighborsClassifier(n_neighbors=n_neighborsBest, weights=weightsBest, p=pBest)

    ClassifierSVMIndividual = SVC(C=CBest, kernel=kernelBest, gamma=gammaBest)
    ClassifierSVMIndividualGenerated = SVC(C=CBest, kernel=kernelBest, gamma=gammaBest)
    ClassifierSVMLiu = SVC(C=CBest, kernel=kernelBest, gamma=gammaBest)
    ClassifierSVM10BBLDA = SVC(C=CBest, kernel=kernelBest, gamma=gammaBest)
    ClassifierSVM11BBLDA = SVC(C=CBest, kernel=kernelBest, gamma=gammaBest)
    ClassifierSVM10BBQDA = SVC(C=CBest, kernel=kernelBest, gamma=gammaBest)
    ClassifierSVM11BBQDA = SVC(C=CBest, kernel=kernelBest, gamma=gammaBest)

    # Creating Variables
    resultsFeatureSet = pd.DataFrame(columns=['person', 'subset', '# shots', 'Feature Set'
        , 'AccLDAInd', 'AccLDALiu', 'AccLDA10BB', 'AccLDA11BB', 'AccLDA10BBcl', 'AccLDA11BBcl'
        , 'AccQDAInd', 'AccQDALiu', 'AccQDA10BB', 'AccQDA11BB', 'AccQDA10BBcl', 'AccQDA11BBcl'
        , 'AccKNNInd', 'AccKNNIndGen', 'AccKNNLiu', 'AccKNN10BBl', 'AccKNN11BBl', 'AccKNN10BBq', 'AccKNN11BBq'
        , 'AccSVMInd', 'AccSVMGIndGen', 'AccSVMLiu', 'AccSVM10BBl', 'AccSVM11BBl', 'AccSVM10BBq', 'AccSVM11BBq'
        , 'r10BBl', 'r11BBl', 'r10BBq', 'r11BBq', 'r10BBlmean', 'r11BBlmean', 'r10BBqmean', 'r11BBqmean'])

    auxIndex1 = 0

    samples = 1000

    for person in range(startPerson, endPerson + 1):
        if person == 11 or person == 44:
            combinationSet = np.setdiff1d(list(range(1, 26)), 15)


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

        testRepetitions = dataMatrix[
            np.where(
                (dataMatrix[:, allFeatures] == typeData) * (dataMatrix[:, allFeatures + 1] == person)
            ), allFeatures + 3][0]

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
                # Amount of Training data
                minSamplesClass = 50
                step = math.ceil(np.shape(trainLabels)[0] / (classes * minSamplesClass))
                print(np.shape(trainLabels)[0],step)

                trainedModelLiu = reTrainigLiu(currentValues, preTrainedDataMatrix, classes, allFeatures)

                trainedModel10BBLDA, rClass10BBLDA, rClass10BBmeanLDA = reTrainigProposed10BB(currentValues,
                                                                                              preTrainedDataMatrix,
                                                                                              classes, allFeatures,
                                                                                              trainFeatures,
                                                                                              trainLabels, step, 'LDA')
                trainedModel10BBQDA, rClass10BBQDA, rClass10BBmeanQDA = reTrainigProposed10BB(currentValues,
                                                                                              preTrainedDataMatrix,
                                                                                              classes, allFeatures,
                                                                                              trainFeatures,
                                                                                              trainLabels, step, 'QDA')
                trainedModel11BBLDA, rClass11BBLDA, rClass11BBmeanLDA = reTrainigProposed11BB(currentValues,
                                                                                              preTrainedDataMatrix,
                                                                                              classes, allFeatures,
                                                                                              trainFeatures,
                                                                                              trainLabels, step,
                                                                                              'LDA')
                trainedModel11BBQDA, rClass11BBQDA, rClass11BBmeanQDA = reTrainigProposed11BB(currentValues,
                                                                                              preTrainedDataMatrix,
                                                                                              classes, allFeatures,
                                                                                              trainFeatures,
                                                                                              trainLabels, step,
                                                                                              'QDA')

                xTrainIndividualGenerated, yTrainIndividualGenerated = SamplesFew_ShotTrainedModels(currentValues,
                                                                                                    samples, classes)
                xTrainLiu, yTrainLiu = SamplesFew_ShotTrainedModels(trainedModelLiu, samples, classes)
                xTrain10BBL, yTrain10BBL = SamplesFew_ShotTrainedModels(trainedModel10BBLDA, samples, classes)
                xTrain11BBL, yTrain11BBL = SamplesFew_ShotTrainedModels(trainedModel10BBQDA, samples, classes)
                xTrain10BBQ, yTrain10BBQ = SamplesFew_ShotTrainedModels(trainedModel11BBLDA, samples, classes)
                xTrain11BBQ, yTrain11BBQ = SamplesFew_ShotTrainedModels(trainedModel11BBQDA, samples, classes)

                ClassifierKNNIndividual.fit(trainFeatures, trainLabels)
                ClassifierKNNIndividualGenerated.fit(xTrainIndividualGenerated, yTrainIndividualGenerated)
                ClassifierKNNLiu.fit(xTrainLiu, yTrainLiu)
                ClassifierKNN10BBLDA.fit(xTrain10BBL, yTrain10BBL)
                ClassifierKNN11BBLDA.fit(xTrain11BBL, yTrain11BBL)
                ClassifierKNN10BBQDA.fit(xTrain10BBQ, yTrain10BBQ)
                ClassifierKNN11BBQDA.fit(xTrain11BBQ, yTrain11BBQ)

                ClassifierSVMIndividual.fit(trainFeatures, trainLabels)
                ClassifierSVMIndividualGenerated.fit(xTrainIndividualGenerated, yTrainIndividualGenerated)
                ClassifierSVMLiu.fit(xTrainLiu, yTrainLiu)
                ClassifierSVM10BBLDA.fit(xTrain10BBL, yTrain10BBL)
                ClassifierSVM11BBLDA.fit(xTrain11BBL, yTrain11BBL)
                ClassifierSVM10BBQDA.fit(xTrain10BBQ, yTrain10BBQ)
                ClassifierSVM11BBQDA.fit(xTrain11BBQ, yTrain11BBQ)

                resultsFeatureSet.loc[auxIndex1] = [person, subset, np.size(subset), featureSet
                    , scoreModelLDA(testFeatures, testLabels, currentValues, classes)
                    , scoreModelLDA(testFeatures, testLabels, trainedModelLiu, classes)
                    , scoreModelLDA(testFeatures, testLabels, trainedModel10BBLDA, classes)
                    , scoreModelLDA(testFeatures, testLabels, trainedModel11BBLDA, classes)
                    , scoreModelLDA_Classification(testFeatures, testLabels, trainedModel10BBLDA
                                                   , classes, testRepetitions)
                    , scoreModelLDA_Classification(testFeatures, testLabels, trainedModel11BBLDA
                                                   , classes, testRepetitions)
                    , scoreModelQDA(testFeatures, testLabels, currentValues, classes)
                    , scoreModelQDA(testFeatures, testLabels, trainedModelLiu, classes)
                    , scoreModelQDA(testFeatures, testLabels, trainedModel10BBQDA, classes)
                    , scoreModelQDA(testFeatures, testLabels, trainedModel11BBQDA, classes)
                    , scoreModelQDA_Classification(testFeatures, testLabels, trainedModel10BBQDA
                                                   , classes, testRepetitions)
                    , scoreModelQDA_Classification(testFeatures, testLabels, trainedModel11BBQDA
                                                   , classes, testRepetitions)
                    , ClassifierKNNIndividual.score(testFeatures, testLabels)
                    , ClassifierKNNIndividualGenerated.score(testFeatures, testLabels)
                    , ClassifierKNNLiu.score(testFeatures, testLabels)
                    , ClassifierKNN10BBLDA.score(testFeatures, testLabels)
                    , ClassifierKNN11BBLDA.score(testFeatures, testLabels)
                    , ClassifierKNN10BBQDA.score(testFeatures, testLabels)
                    , ClassifierKNN11BBQDA.score(testFeatures, testLabels)
                    , ClassifierSVMIndividual.score(testFeatures, testLabels)
                    , ClassifierSVMIndividualGenerated.score(testFeatures, testLabels)
                    , ClassifierSVMLiu.score(testFeatures, testLabels)
                    , ClassifierSVM10BBLDA.score(testFeatures, testLabels)
                    , ClassifierSVM11BBLDA.score(testFeatures, testLabels)
                    , ClassifierSVM10BBQDA.score(testFeatures, testLabels)
                    , ClassifierSVM11BBQDA.score(testFeatures, testLabels)
                    , rClass10BBLDA, rClass11BBLDA, rClass10BBQDA, rClass11BBQDA
                    , rClass10BBmeanLDA, rClass11BBmeanLDA, rClass10BBmeanQDA, rClass11BBmeanQDA]

                resultsFeatureSet.to_csv(nameFile)
                # print(featureSet)
                # print('Results: person= ', person, ' shot set= ', subset)
                # print(resultsFeatureSet.loc[auxIndex1])
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
    if pca == True:
        pca = PCA(n_components=pcaComp)
        principalComponents = pca.fit_transform(dataMatrix)
        dataMatrix = np.hstack((principalComponents, mavMatrix[:, CH:]))
        allFeatures = np.size(principalComponents, axis=1)
        print('Features: ', allFeatures)
    else:
        dataMatrix = np.hstack((dataMatrix, mavMatrix[:, CH:]))
        allFeatures = numberFeatures * CH
        print('Features: ', allFeatures)

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
    if pca == True:
        pca = PCA(n_components=pcaComp)
        principalComponents = pca.fit_transform(dataMatrix)
        dataMatrix = np.hstack((principalComponents, sscMatrix[:, CH:]))
        allFeatures = np.size(principalComponents, axis=1)
        print('Features: ', allFeatures)
    else:
        dataMatrix = np.hstack((dataMatrix, sscMatrix[:, CH:]))
        allFeatures = numberFeatures * CH
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
    if pca == True:
        pca = PCA(n_components=pcaComp)
        principalComponents = pca.fit_transform(dataMatrix)
        dataMatrix = np.hstack((principalComponents, wampMatrix[:, CH:]))
        allFeatures = np.size(principalComponents, axis=1)
        print('Features: ', allFeatures)
    else:
        dataMatrix = np.hstack((dataMatrix, wampMatrix[:, CH:]))
        allFeatures = numberFeatures * CH
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