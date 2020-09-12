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
from sklearn.svm import SVC as SVM
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.datasets import make_spd_matrix

import itertools
import random

from sklearn.model_selection import GridSearchCV

from sklearn import preprocessing
from sklearn.decomposition import PCA


#
# import warnings
# warnings.filterwarnings("ignore")


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


def VidovicModel(currentValues, preTrainedDataMatrix, classes, allFeatures):
    trainedModelL = pd.DataFrame(columns=['cov', 'mean', 'class'])
    trainedModelQ = pd.DataFrame(columns=['cov', 'mean', 'class'])

    preTrainedCov = np.zeros((allFeatures, allFeatures))
    preTrainedMean = np.zeros((1, allFeatures))

    for cla in range(0, classes):
        preTrainedMatrix_Class = pd.DataFrame(
            preTrainedDataMatrix[['cov', 'mean']].loc[(preTrainedDataMatrix['class'] == cla + 1)])
        preTrainedMatrix_Class = preTrainedMatrix_Class.reset_index(drop=True)
        for i in range(len(preTrainedMatrix_Class.index)):
            preTrainedCov += preTrainedDataMatrix['cov'][i]
            preTrainedMean += preTrainedDataMatrix['mean'][i]
        preTrainedCov = preTrainedCov / len(preTrainedMatrix_Class.index)
        preTrainedMean = preTrainedMean / len(preTrainedMatrix_Class.index)
        currentCov = currentValues['cov'].loc[cla]
        currentMean = currentValues['mean'].loc[cla]
        trainedModelL.at[cla, 'cov'] = (1 - 0.8) * preTrainedCov + 0.8 * currentCov
        trainedModelL.at[cla, 'mean'] = (1 - 0.8) * preTrainedMean[0] + 0.8 * currentMean
        trainedModelQ.at[cla, 'cov'] = (1 - 0.9) * preTrainedCov + 0.9 * currentCov
        trainedModelQ.at[cla, 'mean'] = (1 - 0.7) * preTrainedMean[0] + 0.7 * currentMean

        trainedModelL.at[cla, 'class'] = cla + 1
        trainedModelQ.at[cla, 'class'] = cla + 1

    return trainedModelL, trainedModelQ


# Matthews correlation coefficient
def MCC(TP, TN, FP, FN):
    mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    if np.isscalar(mcc):
        if np.isnan(mcc) or mcc < 0:
            mcc = 0
    else:
        mcc[np.isnan(mcc)] = 0
        mcc[mcc < 0] = 0

    return mcc


# F-score LDA and QDA

def scoreModelLDA_r(testFeatures, testLabels, model, classes, currentClass, step):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    currentClass = currentClass + 1
    LDACov = LDA_Cov(model, classes)
    for i in range(0, np.size(testLabels), step):
        currentPredictor = predictedModelLDA(testFeatures[i, :], model, classes, LDACov)
        if currentPredictor == testLabels[i]:
            if currentPredictor == currentClass:
                TP += 1
            else:
                TN += 1
        else:
            if testLabels[i] == currentClass:
                FN += 1
            else:
                FP += 1
    W = MCC(TP, TN, FP, FN)

    return W


def scoreModelQDA_r(testFeatures, testLabels, model, classes, currentClass, step):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    currentClass = currentClass + 1
    for i in range(0, np.size(testLabels), step):
        currentPredictor = predictedModelQDA(testFeatures[i, :], model, classes)

        if currentPredictor == testLabels[i]:
            if currentPredictor == currentClass:
                TP += 1
            else:
                TN += 1
        else:
            if testLabels[i] == currentClass:
                FN += 1
            else:
                FP += 1

    W = MCC(TP, TN, FP, FN)

    return W


# F-score LDA and QDA for all people

def scoreModelLDA_ALL(testFeatures, testLabels, model, classes, step):
    TP = np.zeros([classes])
    TN = np.zeros([classes])
    FP = np.zeros([classes])
    FN = np.zeros([classes])

    LDACov = LDA_Cov(model, classes)

    for i in range(0, np.size(testLabels), step):
        currentPredictor = predictedModelLDA(testFeatures[i, :], model, classes, LDACov)

        if currentPredictor == testLabels[i]:
            TP[int(testLabels[i] - 1)] += 1
            for j in range(classes):
                if j != int(testLabels[i] - 1):
                    TN[j] += 1
        else:
            FN[int(testLabels[i] - 1)] += 1
            for j in range(classes):
                if j != int(testLabels[i] - 1):
                    FP[j] += 1
    W = MCC(TP, TN, FP, FN)

    return W


def scoreModelQDA_ALL(testFeatures, testLabels, model, classes, step):
    TP = np.zeros([classes])
    TN = np.zeros([classes])
    FP = np.zeros([classes])
    FN = np.zeros([classes])

    for i in range(0, np.size(testLabels), step):
        currentPredictor = predictedModelQDA(testFeatures[i, :], model, classes)

        if currentPredictor == testLabels[i]:
            TP[int(testLabels[i] - 1)] += 1
            for j in range(classes):
                if j != int(testLabels[i] - 1):
                    TN[j] += 1
        else:
            FN[int(testLabels[i] - 1)] += 1
            for j in range(classes):
                if j != int(testLabels[i] - 1):
                    FP[j] += 1

    W = MCC(TP, TN, FP, FN)

    return W


# Weight Calculation LDA or QDA

def wPeopleProposedModelMean(currentValues, personMean, currentClass, classes, trainFeatures, trainLabels, step,
                             typeModel):
    personValues = currentValues.copy()
    personValues['mean'].at[currentClass] = personMean
    if typeModel == 'LDA':
        W = scoreModelLDA_r(trainFeatures, trainLabels, personValues, classes, currentClass, step)
    elif typeModel == 'QDA':
        W = scoreModelQDA_r(trainFeatures, trainLabels, personValues, classes, currentClass, step)

    return W


def wPeopleProposedModelCov(currentValues, personCov, currentClass, classes, trainFeatures, trainLabels, step,
                            typeModel):
    personValues = currentValues.copy()
    personValues['cov'].at[currentClass] = personCov
    if typeModel == 'LDA':
        W = scoreModelLDA_r(trainFeatures, trainLabels, personValues, classes, currentClass, step)
    elif typeModel == 'QDA':
        W = scoreModelQDA_r(trainFeatures, trainLabels, personValues, classes, currentClass, step)
    return W


def wPeopleProposedModel(currentValues, personMean, personCov, currentClass, classes, trainFeatures, trainLabels, step,
                         typeModel):
    personValues = currentValues.copy()
    personValues['cov'].at[currentClass] = personCov
    personValues['mean'].at[currentClass] = personMean
    if typeModel == 'LDA':
        W = scoreModelLDA_r(trainFeatures, trainLabels, personValues, classes, currentClass, step)
    elif typeModel == 'QDA':
        W = scoreModelQDA_r(trainFeatures, trainLabels, personValues, classes, currentClass, step)
    return W


# Adaptative Calssifier LDA or QDA
def ProposedModel(currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels, step,
                  typeModel, k):
    t = time.time()

    trainedModel = pd.DataFrame(columns=['cov', 'mean', 'class'])

    for cla in range(classes):
        trainedModel.at[cla, 'cov'] = np.zeros((allFeatures, allFeatures))
        trainedModel.at[cla, 'mean'] = np.zeros((1, allFeatures))[0]

    if typeModel == 'LDA':
        wTarget = scoreModelLDA_ALL(trainFeatures, trainLabels, currentValues, classes, step)
    elif typeModel == 'QDA':
        wTarget = scoreModelQDA_ALL(trainFeatures, trainLabels, currentValues, classes, step)
    wTargetCov = wTarget.copy()
    wTargetMean = wTarget.copy()

    for cla in range(classes):
        preTrainedMatrix_Class = pd.DataFrame(
            preTrainedDataMatrix[['cov', 'mean']].loc[(preTrainedDataMatrix['class'] == cla + 1)])
        preTrainedMatrix_Class = preTrainedMatrix_Class.reset_index(drop=True)
        currentCov = currentValues['cov'].loc[cla]
        currentMean = currentValues['mean'].loc[cla]
        peopleClass = len(preTrainedMatrix_Class.index)

        wPeopleMean = np.zeros(peopleClass)
        wPeopleCov = np.zeros(peopleClass)

        for i in range(peopleClass):
            personMean = preTrainedMatrix_Class['mean'].loc[i]
            personCov = preTrainedMatrix_Class['cov'].loc[i]
            wPeopleMean[i] = wPeopleProposedModelMean(currentValues, personMean, cla, classes
                                                      , trainFeatures, trainLabels, step, typeModel)
            wPeopleCov[i] = wPeopleProposedModelCov(currentValues, personCov, cla, classes
                                                    , trainFeatures, trainLabels, step, typeModel)

        # print('before', cla, k, wTargetMean[cla], wPeopleMean)
        sumWMean = np.sum(wPeopleMean)

        if (sumWMean != 0) and (sumWMean + wTargetMean[cla] != 0):
            wTargetMean[cla] = wTargetMean[cla] / (wTargetMean[cla] + np.mean(wPeopleMean[wPeopleMean != 0]) * k)
            wPeopleMean = (wPeopleMean / sumWMean) * (1 - wTargetMean[cla])

        else:
            wTargetMean[cla] = 1
            wPeopleMean = np.zeros(peopleClass)

        # print('after', cla, k, wTargetMean[cla], wPeopleMean)

        sumWCov = np.sum(wPeopleCov)
        if (sumWCov != 0) and (sumWCov + wTargetCov[cla] != 0):

            wTargetCov[cla] = wTargetCov[cla] / (wTargetCov[cla] + np.mean(wPeopleCov[wPeopleCov != 0]) * k)
            wPeopleCov = (wPeopleCov / sumWCov) * (1 - wTargetCov[cla])

        else:
            wTargetCov[cla] = 1
            wPeopleCov = np.zeros(peopleClass)

        trainedModel.at[cla, 'cov'] = np.sum(preTrainedMatrix_Class['cov'] * wPeopleCov) + currentCov * wTargetCov[cla]
        trainedModel.at[cla, 'mean'] = np.sum(preTrainedMatrix_Class['mean'] * wPeopleMean) + currentMean * wTargetMean[
            cla]
        trainedModel.at[cla, 'class'] = cla + 1

    elapsed = time.time() - t
    return trainedModel, wTargetMean, wTargetMean.mean(), wTargetCov, wTargetCov.mean(), elapsed


def reTrainedCovProposed(r, currentCov, reTrainedCalculatedCovValue):
    reTrainedCovValue = np.add((1 - r) * currentCov, r * reTrainedCalculatedCovValue)
    return reTrainedCovValue


# LDA Calssifier
def LDA_Discriminant(x, covariance, mean):
    det = np.linalg.det(covariance)
    if det > 0:
        invCov = np.linalg.inv(covariance)
        discriminant = np.dot(np.dot(x, invCov), mean) - 0.5 * np.dot(np.dot(mean, invCov), mean)
    else:
        discriminant = float('NaN')
    return discriminant


def LDA_Discriminant_pseudo(x, covariance, mean):
    invCov = np.linalg.pinv(covariance)
    return np.dot(np.dot(x, invCov), mean) - 0.5 * np.dot(np.dot(mean, invCov), mean)


def LDA_Cov(trainedModel, classes):
    LDACov = trainedModel['cov'].sum() / classes
    return LDACov


def predictedModelLDA(sample, model, classes, LDACov):
    d = np.zeros([classes])
    for cl in range(classes):
        d[cl] = LDA_Discriminant(sample, LDACov, model['mean'].loc[cl])
        if math.isnan(d[cl]):
            return predictedModelLDA_pseudo(sample, model, classes, LDACov)
    return np.argmax(d) + 1


def predictedModelLDA_pseudo(sample, model, classes, LDACov):
    d = np.zeros([classes])
    for cl in range(classes):
        d[cl] = LDA_Discriminant_pseudo(sample, LDACov, model['mean'].loc[cl])
    return np.argmax(d) + 1


def scoreModelLDA(testFeatures, testLabels, model, classes):
    t = 0
    true = 0
    count = 0
    LDACov = LDA_Cov(model, classes)
    for i in range(0, np.size(testLabels)):
        auxt = time.time()
        currentPredictor = predictedModelLDA(testFeatures[i, :], model, classes, LDACov)
        t += (time.time() - auxt)
        if currentPredictor == testLabels[i]:
            true += 1
            count += 1
        else:
            count += 1

    return true / count, t / np.size(testLabels)


# QDA Classifier
def QDA_Discriminant(x, covariance, mean):
    det = np.linalg.det(covariance)
    if det > 0:
        discriminant = -.5 * np.log(det) - .5 * np.dot(np.dot((x - mean), np.linalg.inv(covariance)), (x - mean).T)
    else:
        discriminant = float('NaN')
    return discriminant


def QDA_Discriminant_pseudo(x, covariance, mean):
    return -.5 * np.dot(np.dot((x - mean), np.linalg.pinv(covariance)), (x - mean).T)


def predictedModelQDA(sample, model, classes):
    d = np.zeros([classes])
    for cl in range(classes):
        d[cl] = QDA_Discriminant(sample, model['cov'].loc[cl], model['mean'].loc[cl])
        if math.isnan(d[cl]):
            return predictedModelQDA_pseudo(sample, model, classes)
    return np.argmax(d) + 1


def predictedModelQDA_pseudo(sample, model, classes):
    d = np.zeros([classes])
    for cl in range(classes):
        d[cl] = QDA_Discriminant_pseudo(sample, model['cov'].loc[cl], model['mean'].loc[cl])
    return np.argmax(d) + 1


def scoreModelQDA(testFeatures, testLabels, model, classes):
    t = 0
    true = 0
    count = 0
    for i in range(0, np.size(testLabels)):
        auxt = time.time()
        actualPredictor = predictedModelQDA(testFeatures[i, :], model, classes)
        t += (time.time() - auxt)
        if actualPredictor == testLabels[i]:
            true += 1
            count += 1
        else:
            count += 1
    # print('QDA', (time.time() - t) / np.size(testLabels))
    return true / count, t / np.size(testLabels)


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


def scoreModel_SVM_KNN_Classification(testFeatures, testLabels, model, testRepetitions):
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
        actualPredictor = np.append(actualPredictor, model.predict([testFeatures[i, :]]))

    return true / count


# Evaluation

def SamplesProposedModel(model, samples, classes):
    xTrain = []
    yTrain = []
    for cl in range(classes):
        meanDistribution = model['mean'].loc[cl]
        covDistribution = model['cov'].loc[cl]
        xTrain.extend(
            np.random.multivariate_normal(meanDistribution, covDistribution, samples, check_valid='ignore'))
        yTrain.extend((cl + 1) * np.ones(samples))
    return xTrain, yTrain


def resultsDataframe(currentValues, preTrainedDataMatrix, trainFeatures, trainLabels, classes, allFeatures,
                     trainFeaturesGen, trainLabelsGen, results, testFeatures, testLabels, idx, person, subset,
                     featureSet, nameFile, printR, clfKNNInd, clfKNNMulti, clfSVMInd, clfSVMMulti, clfLDAInd, clfQDAInd,
                     clfLDAMulti, clfQDAMulti, k, testRep, tPre,pkValues):
    # Amount of Training data
    minSamplesClass = 20
    step = math.ceil(np.shape(trainLabels)[0] / (classes * minSamplesClass))
    print('step: ', np.shape(trainLabels)[0], step)

    liuModel = LiuModel(currentValues, preTrainedDataMatrix, classes, allFeatures)
    vidovicModelL, vidovicModelQ = VidovicModel(currentValues, preTrainedDataMatrix, classes, allFeatures)

    # propModelLDA, wTargetMeanLDA, wTargetMeanLDAm, wTargetCovLDA, wTargetCovLDAm, tPropL = ProposedModel(
    #     currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels, step, 'LDA', k)

    propModelQDA, wTargetMeanQDA, wTargetMeanQDAm, wTargetCovQDA, wTargetCovQDAm, tPropQ = ProposedModel(
        currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels, step, 'QDA', k)

    # t = time.time()
    # clfLDAInd.fit(trainFeatures, trainLabels)
    # tIndL = time.time() - t
    # t = time.time()
    # clfQDAInd.fit(trainFeatures, trainLabels)
    # tIndQ = time.time() - t
    # t = time.time()
    # clfLDAMulti.fit(trainFeaturesGen, trainLabelsGen)
    # tGenL = time.time() - t
    # t = time.time()
    # clfQDAMulti.fit(trainFeaturesGen, trainLabelsGen)
    # tGenQ = time.time() - t

    results.at[idx, 'person'] = person
    results.at[idx, 'subset'] = subset
    results.at[idx, '# shots'] = np.size(subset)
    results.at[idx, 'Feature Set'] = featureSet
    # LDA results
    results.at[idx, 'AccLDAInd'], tIndL = scoreModelLDA(testFeatures, testLabels, currentValues, classes)
    results.at[idx, 'AccLDAMulti'] , tGenL = scoreModelLDA(testFeatures, testLabels, pkValues, classes)
    # results.at[idx, 'AccLDAProp'], _ = scoreModelLDA(testFeatures, testLabels, propModelLDA,
    #                                                  classes)
    results.at[idx, 'AccLDAPropQ'], results.at[idx, 'tCLPropL'] = scoreModelLDA(testFeatures, testLabels, propModelQDA,
                                                                                classes)
    results.at[idx, 'AccLDALiu'], _ = scoreModelLDA(testFeatures, testLabels, liuModel, classes)
    results.at[idx, 'AccLDAVidovic'], _ = scoreModelLDA(testFeatures, testLabels, vidovicModelL, classes)

    results.at[idx, 'AccClLDAInd'] = scoreModelLDA_Classification(testFeatures, testLabels, currentValues, classes,
                                                                  testRep)
    # results.at[idx, 'AccClLDAProp'] = scoreModelLDA_Classification(testFeatures, testLabels, propModelLDA, classes,
    #                                                                testRep)
    results.at[idx, 'AccClLDAPropQ'] = scoreModelLDA_Classification(testFeatures, testLabels, propModelQDA, classes,
                                                                    testRep)
    results.at[idx, 'AccClLDALiu'] = scoreModelLDA_Classification(testFeatures, testLabels, liuModel, classes,
                                                                  testRep)
    results.at[idx, 'AccClLDAVidovic'] = scoreModelLDA_Classification(testFeatures, testLabels, vidovicModelL, classes,
                                                                      testRep)
    # results.at[idx, 'tPropL'] = tPropL
    results.at[idx, 'tIndL'] = tIndL
    results.at[idx, 'tGenL'] = tGenL
    # results.at[idx, 'wTargetMeanLDA'] = wTargetMeanLDA
    # results.at[idx, 'wTargetMeanLDAm'] = wTargetMeanLDAm
    # results.at[idx, 'wTargetCovLDA'] = wTargetCovLDA
    # results.at[idx, 'wTargetCovLDAm'] = wTargetCovLDAm
    ## QDA results
    results.at[idx, 'AccQDAInd'], tIndQ = scoreModelQDA(testFeatures, testLabels, currentValues, classes)
    results.at[idx, 'AccQDAMulti'] , tGenQ = scoreModelQDA(testFeatures, testLabels, pkValues, classes)
    results.at[idx, 'AccQDAProp'], results.at[idx, 'tCLPropQ'] = scoreModelQDA(testFeatures, testLabels, propModelQDA,
                                                                               classes)
    results.at[idx, 'AccQDALiu'], _ = scoreModelQDA(testFeatures, testLabels, liuModel, classes)
    results.at[idx, 'AccQDAVidovic'], _ = scoreModelQDA(testFeatures, testLabels, vidovicModelQ, classes)
    # results.at[idx, 'AccQDAPropL'], _ = scoreModelQDA(testFeatures, testLabels, propModelLDA, classes)
    results.at[idx, 'AccClQDAInd'] = scoreModelQDA_Classification(testFeatures, testLabels, currentValues, classes,
                                                                  testRep)
    results.at[idx, 'AccClQDAProp'] = scoreModelQDA_Classification(testFeatures, testLabels, propModelQDA, classes,
                                                                   testRep)
    # results.at[idx, 'AccClQDAPropL'] = scoreModelQDA_Classification(testFeatures, testLabels, propModelLDA, classes,
    #                                                                 testRep)
    results.at[idx, 'AccClQDALiu'] = scoreModelQDA_Classification(testFeatures, testLabels, liuModel, classes,
                                                                  testRep)
    results.at[idx, 'AccClQDAVidovic'] = scoreModelQDA_Classification(testFeatures, testLabels, vidovicModelQ, classes,
                                                                      testRep)
    results.at[idx, 'tPropQ'] = tPropQ
    results.at[idx, 'tIndQ'] = tIndQ
    results.at[idx, 'tGenQ'] = tGenQ
    results.at[idx, 'tPre'] = tPre
    results.at[idx, 'wTargetMeanQDA'] = wTargetMeanQDA
    results.at[idx, 'wTargetMeanQDAm'] = wTargetMeanQDAm
    results.at[idx, 'wTargetCovQDA'] = wTargetCovQDA
    results.at[idx, 'wTargetCovQDAm'] = wTargetCovQDAm
    # SVM KNN

    # clfSVMInd.fit(trainFeatures, trainLabels)
    # results.at[idx, 'AccSVMInd'] = clfSVMInd.score(testFeatures, testLabels)
    # clfKNNInd.fit(trainFeatures, trainLabels)
    # results.at[idx, 'AccKNNInd'] = clfKNNInd.score(testFeatures, testLabels)
    #
    # results.at[idx, 'AccClSVMInd'] = scoreModel_SVM_KNN_Classification(testFeatures, testLabels, clfSVMInd, testRep)
    # results.at[idx, 'AccClKNNInd'] = scoreModel_SVM_KNN_Classification(testFeatures, testLabels, clfKNNInd, testRep)

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


def evaluation(dataMatrix, classes, peoplePriorK, peopleTest, featureSet, numberShots, combinationSet, nameFile,
               startPerson, endPerson, allFeatures, typeDatabase, printR):
    hyperKNN = pd.read_csv('KNN_hyper' + typeDatabase + str(featureSet) + '.csv')
    hyperSVM = pd.read_csv('SVM_hyper' + typeDatabase + str(featureSet) + '.csv')

    p = hyperKNN['p'].mode()[0]
    n_neighbors = hyperKNN['n_neighbors'].mode()[0]
    weights = hyperKNN['weights'].mode()[0]
    clfKNNMulti = KNN(n_neighbors=n_neighbors, p=p, weights=weights)

    C = hyperSVM['C'].mode()[0]
    degree = hyperSVM['degree'].mode()[0]
    kernel = hyperSVM['kernel'].mode()[0]
    gamma = hyperSVM['gamma'].mode()[0]
    clfSVMMulti = SVM(C=C, kernel=kernel, degree=degree, gamma=gamma)

    clfLDAInd = LDA()
    clfQDAInd = QDA()
    clfLDAMulti = LDA()
    clfQDAMulti = QDA()

    scaler = preprocessing.MinMaxScaler()

    if typeDatabase == 'EPN':
        evaluationEPN(dataMatrix, classes, peoplePriorK, peopleTest, featureSet, numberShots, combinationSet, nameFile,
                      startPerson, endPerson, allFeatures, printR, hyperKNN, clfKNNMulti, hyperSVM, clfSVMMulti,
                      clfLDAInd, clfQDAInd, clfLDAMulti, clfQDAMulti, scaler)
    elif typeDatabase == 'Nina5':
        evaluationNina5(dataMatrix, classes, peoplePriorK, peopleTest, featureSet, numberShots, combinationSet,
                        nameFile, startPerson, endPerson, allFeatures, printR, hyperKNN, clfKNNMulti, hyperSVM,
                        clfSVMMulti, clfLDAInd, clfQDAInd, clfLDAMulti, clfQDAMulti, scaler)
    elif typeDatabase == 'Cote':
        evaluationCote(dataMatrix, classes, peoplePriorK, peopleTest, featureSet, numberShots, combinationSet,
                       nameFile, startPerson, endPerson, allFeatures, printR, hyperKNN, clfKNNMulti, hyperSVM,
                       clfSVMMulti, clfLDAInd, clfQDAInd, clfLDAMulti, clfQDAMulti, scaler)


### EPN
def evaluationEPN(dataMatrix, classes, peoplePriorK, peopleTest, featureSet, numberShots, combinationSet, nameFile,
                  startPerson, endPerson, allFeatures, printR, hyperKNN, clfKNNMulti, hyperSVM, clfSVMMulti,
                  clfLDAInd, clfQDAInd, clfLDAMulti, clfQDAMulti, scaler):
    results = pd.DataFrame(
        columns=['person', 'subset', '# shots', 'Feature Set', 'wTargetMeanQDA', 'wTargetCovQDA'])

    trainFeaturesGenPre = dataMatrix[np.where((dataMatrix[:, allFeatures + 1] <= peoplePriorK)), 0:allFeatures][0]
    trainLabelsGenPre = dataMatrix[np.where((dataMatrix[:, allFeatures + 1] <= peoplePriorK)), allFeatures + 2][0]
    # pca.fit_transform(scaler.fit(trainFeaturesGenPre))

    idx = 0
    for person in range(peoplePriorK + startPerson, peoplePriorK + endPerson + 1):

        p = hyperKNN['p'].loc[hyperKNN['person'] == person].reset_index(drop=True)[0]
        n_neighbors = hyperKNN['n_neighbors'].loc[hyperKNN['person'] == person].reset_index(drop=True)[0]
        weights = hyperKNN['weights'].loc[hyperKNN['person'] == person].reset_index(drop=True)[0]
        clfKNNInd = KNN(n_neighbors=n_neighbors, p=p, weights=weights)

        C = hyperSVM['C'].loc[hyperKNN['person'] == person].reset_index(drop=True)[0]
        degree = hyperSVM['degree'].loc[hyperKNN['person'] == person].reset_index(drop=True)[0]
        kernel = hyperSVM['kernel'].loc[hyperKNN['person'] == person].reset_index(drop=True)[0]
        gamma = hyperSVM['gamma'].loc[hyperKNN['person'] == person].reset_index(drop=True)[0]
        clfSVMInd = SVM(C=C, kernel=kernel, degree=degree, gamma=gamma)

        # combinationSet = list(range(1, 26))
        # if person == 44:
        #     combinationSet = np.setdiff1d(list(range(1, 26)), 15)

        typeData = 1
        testFeatures = dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (
                dataMatrix[:, allFeatures + 1] == person)), 0:allFeatures][0]
        testLabels = dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (
                dataMatrix[:, allFeatures + 1] == person)), allFeatures + 2][0]
        testRep = dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (
                dataMatrix[:, allFeatures + 1] == person)), allFeatures + 3][0]

        typeData = 0

        for shot in range(1, numberShots + 1):
            # for shot in range(1, 2):

            if shot >= 0:
                randSubsets = range(1)
            else:

                random.seed(1)
                subsets = []
                for subset in itertools.combinations(combinationSet, shot):
                    subsets.append(subset)

                if len(subsets) <= 25:
                    randSubsets = subsets
                else:
                    randSubsets = random.sample(subsets, 25)

            auxSh = 0
            for subset in randSubsets:
                if auxSh == 0:
                    subset = tuple(range(1, shot + 1))
                    auxSh = 1

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
                print(k)

                scaler.fit(trainFeatures)
                t = time.time()
                trainFeatures = scaler.transform(trainFeatures)
                tPre = (time.time() - t) / len(trainFeatures)

                trainFeaturesGen = scaler.transform(trainFeaturesGen)
                testFeaturesTransform = scaler.transform(testFeatures)

                dataPK, allFeaturesPK = preprocessingPK(dataMatrix, allFeatures, scaler)

                preTrainedDataMatrix = preTrainedDataEPN(dataPK, classes, peoplePriorK, allFeaturesPK)
                currentValues = currentDistributionValues(trainFeatures, trainLabels, classes, allFeaturesPK)
                pkValues = currentDistributionValues(trainFeaturesGen, trainLabelsGen, classes, allFeaturesPK)
                results, idx = resultsDataframe(currentValues, preTrainedDataMatrix, trainFeatures, trainLabels,
                                                classes, allFeaturesPK, trainFeaturesGen, trainLabelsGen, results,
                                                testFeaturesTransform, testLabels, idx, person, subset, featureSet,
                                                nameFile, printR, clfKNNInd, clfKNNMulti, clfSVMInd, clfSVMMulti,
                                                clfLDAInd, clfQDAInd, clfLDAMulti, clfQDAMulti, k, testRep, tPre,pkValues)

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


### Cote

def evaluationCote(dataMatrix, classes, peoplePriorK, peopleTest, featureSet, numberShots, combinationSet,
                   nameFile, startPerson, endPerson, allFeatures, printR, hyperKNN, clfKNNMulti, hyperSVM,
                   clfSVMMulti, clfLDAInd, clfQDAInd, clfLDAMulti, clfQDAMulti, scaler):
    # Creating Variables
    results = pd.DataFrame(
        columns=['person', 'subset', '# shots', 'Feature Set', 'wTargetMeanQDA', 'wTargetCovQDA'])

    idx = 0

    typeData = 0
    trainFeaturesGenPre = dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData)), 0:allFeatures][0]
    trainLabelsGenPre = dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData)), allFeatures + 3][0]
    # pca.fit_transform(scaler.fit_transform(trainFeaturesGenPre))

    typeData = 1
    for person in range(peoplePriorK + startPerson, peoplePriorK + endPerson + 1):

        p = hyperKNN['p'].loc[hyperKNN['person'] == person].reset_index(drop=True)[0]
        n_neighbors = hyperKNN['n_neighbors'].loc[hyperKNN['person'] == person].reset_index(drop=True)[0]
        weights = hyperKNN['weights'].loc[hyperKNN['person'] == person].reset_index(drop=True)[0]
        clfKNNInd = KNN(n_neighbors=n_neighbors, p=p, weights=weights)

        C = hyperSVM['C'].loc[hyperKNN['person'] == person].reset_index(drop=True)[0]
        degree = hyperSVM['degree'].loc[hyperKNN['person'] == person].reset_index(drop=True)[0]
        kernel = hyperSVM['kernel'].loc[hyperKNN['person'] == person].reset_index(drop=True)[0]
        gamma = hyperSVM['gamma'].loc[hyperKNN['person'] == person].reset_index(drop=True)[0]
        clfSVMInd = SVM(C=C, kernel=kernel, degree=degree, gamma=gamma)

        carpet = 2
        testFeatures = \
            dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (dataMatrix[:, allFeatures + 1] == person)
                                * (dataMatrix[:, allFeatures + 2] == carpet)), 0:allFeatures][0]
        testLabels = \
            dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (dataMatrix[:, allFeatures + 1] == person)
                                * (dataMatrix[:, allFeatures + 2] == carpet)), allFeatures + 3][0]
        testRep = \
            dataMatrix[np.where((dataMatrix[:, allFeatures] == typeData) * (dataMatrix[:, allFeatures + 1] == person)
                                * (dataMatrix[:, allFeatures + 2] == carpet)), allFeatures + 4][0]

        carpet = 1
        # 4 cycles - cross_validation for 4 cycles or shots
        for shot in range(1, numberShots + 1):

            subset = tuple(range(1, shot + 1))
            # for subset in itertools.combinations(combinationSet, shot):

            ####
            # subset = list(range(1, shot + 1))
            ###

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
            print(k)

            scaler.fit(trainFeatures)
            t = time.time()
            trainFeatures = scaler.transform(trainFeatures)
            tPre = (time.time() - t) / len(trainFeatures)

            trainFeaturesGen = scaler.transform(trainFeaturesGen)
            testFeaturesTransform = scaler.transform(testFeatures)

            dataPK, allFeaturesPK = preprocessingPK(dataMatrix, allFeatures, scaler)

            preTrainedDataMatrix = preTrainedDataCote(dataPK, classes, peoplePriorK, allFeaturesPK)
            currentValues = currentDistributionValues(trainFeatures, trainLabels, classes, allFeaturesPK)
            pkValues = currentDistributionValues(trainFeaturesGen, trainLabelsGen, classes, allFeaturesPK)
            results, idx = resultsDataframe(currentValues, preTrainedDataMatrix, trainFeatures, trainLabels,
                                            classes, allFeaturesPK, trainFeaturesGen, trainLabelsGen, results,
                                            testFeaturesTransform, testLabels, idx, person, subset, featureSet,
                                            nameFile, printR, clfKNNInd, clfKNNMulti, clfSVMInd, clfSVMMulti,
                                            clfLDAInd, clfQDAInd, clfLDAMulti, clfQDAMulti, k, testRep, tPre,pkValues)

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


### Nina Pro 5

def evaluationNina5(dataMatrix, classes, peoplePriorK, peopleTest, featureSet, numberShots, combinationSet, nameFile,
                    startPerson, endPerson, allFeatures, printR, hyperKNN, clfKNNMulti, hyperSVM, clfSVMMulti,
                    clfLDAInd, clfQDAInd, clfLDAMulti, clfQDAMulti, scaler):
    # Creating Variables

    results = pd.DataFrame(
        columns=['person', 'subset', '# shots', 'Feature Set', 'wTargetMeanQDA', 'wTargetCovQDA'])

    idx = 0

    for person in range(startPerson, endPerson + 1):

        p = hyperKNN['p'].loc[hyperKNN['person'] == person].reset_index(drop=True)[0]
        n_neighbors = hyperKNN['n_neighbors'].loc[hyperKNN['person'] == person].reset_index(drop=True)[0]
        weights = hyperKNN['weights'].loc[hyperKNN['person'] == person].reset_index(drop=True)[0]
        clfKNNInd = KNN(n_neighbors=n_neighbors, p=p, weights=weights)

        C = hyperSVM['C'].loc[hyperKNN['person'] == person].reset_index(drop=True)[0]
        degree = hyperSVM['degree'].loc[hyperKNN['person'] == person].reset_index(drop=True)[0]
        kernel = hyperSVM['kernel'].loc[hyperKNN['person'] == person].reset_index(drop=True)[0]
        gamma = hyperSVM['gamma'].loc[hyperKNN['person'] == person].reset_index(drop=True)[0]
        clfSVMInd = SVM(C=C, kernel=kernel, degree=degree, gamma=gamma)

        trainFeaturesGenPre = dataMatrix[np.where((dataMatrix[:, allFeatures] != person)), 0:allFeatures][0]
        trainLabelsGenPre = dataMatrix[np.where((dataMatrix[:, allFeatures] != person)), allFeatures + 1][0]
        # pca.fit_transform(scaler.fit_transform(trainFeaturesGenPre))

        testFeatures = \
            dataMatrix[np.where((dataMatrix[:, allFeatures] == person) * (dataMatrix[:, allFeatures + 2] >= 5)),
            0:allFeatures][0]
        testLabels = dataMatrix[
            np.where((dataMatrix[:, allFeatures] == person) * (dataMatrix[:, allFeatures + 2] >= 5)), allFeatures + 1][
            0].T
        testRep = dataMatrix[
            np.where((dataMatrix[:, allFeatures] == person) * (dataMatrix[:, allFeatures + 2] >= 5)), allFeatures + 2][
            0].T

        Set = np.arange(1, 7)

        # 4 cycles - cross_validation for 4 cycles
        for shot in range(1, numberShots + 1):
            # shot = 4
            subset = tuple(range(1, shot + 1))
            # for subset in itertools.combinations(combinationSet, shot):

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
            # k = 1
            print(k)

            scaler.fit(trainFeatures)
            t = time.time()
            trainFeatures = scaler.transform(trainFeatures)
            tPre = (time.time() - t) / len(trainFeatures)

            trainFeaturesGen = scaler.transform(trainFeaturesGen)
            testFeaturesTransform = scaler.transform(testFeatures)

            dataPK, allFeaturesPK = preprocessingPK(dataMatrix, allFeatures, scaler)

            preTrainedDataMatrix = preTrainedDataNina5(dataPK, classes, peopleTest, person, allFeaturesPK)
            currentValues = currentDistributionValues(trainFeatures, trainLabels, classes, allFeaturesPK)
            pkValues = currentDistributionValues(trainFeaturesGen, trainLabelsGen, classes, allFeaturesPK)
            results, idx = resultsDataframe(currentValues, preTrainedDataMatrix, trainFeatures, trainLabels,
                                            classes, allFeaturesPK, trainFeaturesGen, trainLabelsGen, results,
                                            testFeaturesTransform, testLabels, idx, person, subset, featureSet,
                                            nameFile, printR, clfKNNInd, clfKNNMulti, clfSVMInd, clfSVMMulti,
                                            clfLDAInd, clfQDAInd, clfLDAMulti, clfQDAMulti, k, testRep, tPre,pkValues)

    return results


def preTrainedDataNina5(dataMatrix, classes, peoplePriorK, evaluatedPerson, allFeatures):
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


# Upload Databases

def uploadDatabases(typeDatabase, featureSet=1):
    # Setting general variables

    CH = 8
    windowFile = '295'
    if typeDatabase == 'EPN':
        carpet = 'ExtractedDataCollectedData'
        classes = 5
        peoplePriorK = 30
        peopleTest = 30
        combinationSet = list(range(1, 26))
        numberShots = 25
    elif typeDatabase == 'Nina5':
        carpet = 'ExtractedDataNinaDB5'
        classes = 18
        peoplePriorK = 0
        peopleTest = 10
        combinationSet = list(range(1, 5))
        numberShots = 4
    elif typeDatabase == 'Cote':
        carpet = 'ExtractedDataCoteAllard'
        classes = 7
        peoplePriorK = 19
        peopleTest = 17
        combinationSet = list(range(1, 5))
        numberShots = 4

    if featureSet == 1:
        # Setting variables
        Feature1 = 'logvarMatrix' + windowFile
        segment = ''
        numberFeatures = 1
        allFeatures = numberFeatures * CH
        # Getting Data
        logvarMatrix = np.genfromtxt('../../ExtractedData/' + carpet + '/' + Feature1 + segment + '.csv', delimiter=',')
        if typeDatabase == 'EPN':
            dataMatrix = logvarMatrix[:, 0:]
            labelsDataMatrix = dataMatrix[:, allFeatures + 2]
        elif typeDatabase == 'Nina5':
            dataMatrix = logvarMatrix[:, 8:]
            labelsDataMatrix = dataMatrix[:, allFeatures + 1]
        elif typeDatabase == 'Cote':
            dataMatrix = logvarMatrix[:, 0:]
            dataMatrix[:, allFeatures + 1] = dataMatrix[:, allFeatures + 1] + 1
            dataMatrix[:, allFeatures + 3] = dataMatrix[:, allFeatures + 3] + 1
            labelsDataMatrix = dataMatrix[:, allFeatures + 3]



    elif featureSet == 2:
        # Setting variables
        Feature1 = 'mavMatrix' + windowFile
        Feature2 = 'wlMatrix' + windowFile
        Feature3 = 'zcMatrix' + windowFile
        Feature4 = 'sscMatrix' + windowFile
        segment = ''
        numberFeatures = 4
        allFeatures = numberFeatures * CH
        mavMatrix = np.genfromtxt('../../ExtractedData/' + carpet + '/' + Feature1 + segment + '.csv', delimiter=',')
        wlMatrix = np.genfromtxt('../../ExtractedData/' + carpet + '/' + Feature2 + segment + '.csv', delimiter=',')
        zcMatrix = np.genfromtxt('../../ExtractedData/' + carpet + '/' + Feature3 + segment + '.csv', delimiter=',')
        sscMatrix = np.genfromtxt('../../ExtractedData/' + carpet + '/' + Feature4 + segment + '.csv', delimiter=',')
        if typeDatabase == 'EPN':
            dataMatrix = np.hstack((mavMatrix[:, 0:CH], wlMatrix[:, 0:CH], zcMatrix[:, 0:CH], sscMatrix[:, 0:]))
            labelsDataMatrix = dataMatrix[:, allFeatures + 2]

        elif typeDatabase == 'Nina5':
            dataMatrix = np.hstack(
                (mavMatrix[:, 8:CH * 2], wlMatrix[:, 8:CH * 2], zcMatrix[:, 8:CH * 2], sscMatrix[:, 8:]))
            labelsDataMatrix = dataMatrix[:, allFeatures + 1]

        elif typeDatabase == 'Cote':
            dataMatrix = np.hstack((mavMatrix[:, 0:CH], wlMatrix[:, 0:CH], zcMatrix[:, 0:CH], sscMatrix[:, 0:]))
            dataMatrix[:, allFeatures + 1] = dataMatrix[:, allFeatures + 1] + 1
            dataMatrix[:, allFeatures + 3] = dataMatrix[:, allFeatures + 3] + 1
            labelsDataMatrix = dataMatrix[:, allFeatures + 3]


    elif featureSet == 3:
        # Setting variables
        Feature1 = 'lscaleMatrix' + windowFile
        Feature2 = 'mflMatrix' + windowFile
        Feature3 = 'msrMatrix' + windowFile
        Feature4 = 'wampMatrix' + windowFile
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
            labelsDataMatrix = dataMatrix[:, allFeatures + 2]

        elif typeDatabase == 'Nina5':
            dataMatrix = np.hstack(
                (lscaleMatrix[:, 8:CH * 2], mflMatrix[:, 8:CH * 2], msrMatrix[:, 8:CH * 2], wampMatrix[:, 8:]))
            labelsDataMatrix = dataMatrix[:, allFeatures + 1]

        elif typeDatabase == 'Cote':
            dataMatrix = np.hstack((lscaleMatrix[:, 0:CH], mflMatrix[:, 0:CH], msrMatrix[:, 0:CH], wampMatrix[:, 0:]))
            dataMatrix[:, allFeatures + 1] = dataMatrix[:, allFeatures + 1] + 1
            dataMatrix[:, allFeatures + 3] = dataMatrix[:, allFeatures + 3] + 1
            labelsDataMatrix = dataMatrix[:, allFeatures + 3]

    return dataMatrix, numberFeatures, CH, classes, peoplePriorK, peopleTest, numberShots, combinationSet, allFeatures, labelsDataMatrix


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


def DataGenerator_TwoCL_TwoFeat(seed=None, samples=100, people=5, peopleSame=0, Graph=True, covRandom=True,
                                covRandomPK=True, nameFile=None, mix=False, same=True):
    peopleDiff = people - (peopleSame + 1)
    if not mix:
        peopleDiff = 0
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
        fig1, ax = plt.subplots(1, 1, figsize=(9, 12))

    for person in range(people):
        if covRandomPK:
            covSet1 = make_spd_matrix(Features)
            covSet2 = make_spd_matrix(Features)
            covSet = np.vstack((covSet1, covSet2))
        else:
            covSet = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])

        if person == people - 1:
            auxPoint = np.zeros(2)
            covSet = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
            if covRandom:
                covSet1 = make_spd_matrix(Features)
                covSet2 = make_spd_matrix(Features)
                covSet = np.vstack((covSet1, covSet2))
        elif mix:
            if person < peopleDiff:
                classCovFactor = 10
                # auxPoint = np.random.uniform(low=5, high=classCovFactor * 2 + 5, size=(1, Features))[0]
                # only for the graph
                auxPoint = np.array([0, 7])
            else:
                auxPoint = np.zeros(2)
        elif same:
            auxPoint = np.zeros(2)
        elif not same:
            classCovFactor = 10
            auxPoint = np.random.uniform(low=5, high=classCovFactor * 2 + 5, size=(1, Features))[0]

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
                sizeM = 7

                if person == people - 1:
                    color = colors_list[cl]
                    label = 'clase ' + str(cl) + ' Target person'
                    if cl == 0:
                        markerCL = 'o'
                    else:
                        markerCL = '^'
                    ax.scatter(x1, x2, s=sizeM, color=color, marker=markerCL)
                    confidence_ellipse(x1, x2, ax, edgecolor=color, label=label)
                elif mix:
                    if person < peopleDiff:
                        color = colors_list[cl + 4]
                        label = 'clase ' + str(cl) + ' PK: ' + str(
                            people - 1 - peopleSame) + ' people distinct to the target user'
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
                elif same:
                    color = colors_list[cl + 2]

                    label = 'clase ' + str(cl) + ' PK: ' + str(people - 1) + ' person similar to the target user'
                    lineStyle = ':'
                    if cl == 0:
                        markerCL = '*'
                    else:
                        markerCL = 'x'
                elif not same:
                    color = colors_list[cl + 4]
                    label = 'clase ' + str(cl) + ' PK: ' + str(people - 1) + ' people distinct to the target user'
                    lineStyle = '-.'
                    if cl == 0:
                        markerCL = 's'
                    else:
                        markerCL = 'p'

                if person != people - 1:
                    if person == people - 2 or person == peopleDiff - 1:
                        ax.scatter(x1, x2, s=sizeM, color=color, marker=markerCL)
                        confidence_ellipse(x1, x2, ax, edgecolor=color, label=label, linestyle=lineStyle)
                    else:
                        ax.scatter(x1, x2, s=sizeM, color=color, marker=markerCL)
                        confidence_ellipse(x1, x2, ax, edgecolor=color, linestyle=lineStyle)

            idx += 1

    if Graph:
        if people == 1:
            ax.set_title(
                'Traget Person and Prior Knowledge \n (' + str(classes) + ' Clases - ' + str(Features) + ' Features)')
        else:
            ax.set_title(
                'Traget Person and Prior Knowledge \n (' + str(classes) + ' Clases - ' + str(Features) + ' Features)')

        plt.grid()
        plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0., prop={'size': 8})
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        fig1.tight_layout(pad=0.1)
        plt.savefig("distr.png", bbox_inches='tight', dpi=600)
        plt.show()

    if nameFile is not None:
        DataFrame.to_pickle(nameFile + '.pkl')
    return DataFrame


def ResultsSyntheticData(DataFrame, nameFile, numberShots=30, peoplePK=0, peopleEvaluated=0, samples=500, Features=2,
                         classes=2, times=10, Graph=False, printValues=False, peopleSame=1):
    iSample = 3
    people = 1
    clfLDAInd = LDA()
    clfQDAInd = QDA()
    clfLDAMulti = LDA()
    clfQDAMulti = QDA()

    resultsData = pd.DataFrame(
        columns=['yLDAInd', 'yQDAInd', 'yLDAMulti', 'yQDAMulti', 'yPropL', 'yPropQ', 'yLiu', 'yLiuQDA',
                 'wTargetMeanLDA', 'wTargetCovLDA', 'wTargetMeanQDA', 'wTargetCovQDA', 'wLiu', 'person', 'times',
                 'shots'])

    idx = 0

    for per in range(peopleEvaluated - 1, peopleEvaluated):

        currentPerson = DataFrame[DataFrame['person'] == per].reset_index(drop=True)
        lenTest = int(samples / 2)
        preTrainedDataMatrix = DataFrame[DataFrame['person'] != per].reset_index(drop=True)
        preTrainedDataMatrix.at[:, 'class'] = preTrainedDataMatrix.loc[:, 'class'] + 1

        currentPersonTrain = pd.DataFrame(columns=['data', 'class'])
        currentValues = pd.DataFrame(columns=['data', 'mean', 'cov', 'class'])

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

        if peopleSame == 0:

            hyperKNN = hyperParameterTuning_KNN(x_test, y_test, peopleEvaluated)
            hyperSVM = hyperParameterTuning_SVM(x_test, y_test, peopleEvaluated)

            p = hyperKNN['p'].loc[hyperKNN['person'] == peopleEvaluated].reset_index(drop=True)[0]
            n_neighbors = hyperKNN['n_neighbors'].loc[hyperKNN['person'] == peopleEvaluated].reset_index(drop=True)[0]
            if n_neighbors > (iSample * classes):
                n_neighbors = (iSample * classes)
            weights = hyperKNN['weights'].loc[hyperKNN['person'] == peopleEvaluated].reset_index(drop=True)[0]
            clfKNNInd = KNN(n_neighbors=n_neighbors, p=p, weights=weights)
            clfKNNMulti = KNN(n_neighbors=n_neighbors, p=p, weights=weights)

            C = hyperSVM['C'].loc[hyperKNN['person'] == peopleEvaluated].reset_index(drop=True)[0]
            degree = hyperSVM['degree'].loc[hyperKNN['person'] == peopleEvaluated].reset_index(drop=True)[0]
            kernel = hyperSVM['kernel'].loc[hyperKNN['person'] == peopleEvaluated].reset_index(drop=True)[0]
            gamma = hyperSVM['gamma'].loc[hyperKNN['person'] == peopleEvaluated].reset_index(drop=True)[0]
            clfSVMInd = SVM(C=C, kernel=kernel, degree=degree, gamma=gamma)
            clfSVMMulti = SVM(C=C, kernel=kernel, degree=degree, gamma=gamma)

        # for t in range(times):
        t = times

        for i in range(numberShots):
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

            x_Multi = np.hstack((x_PK, x_train))
            y_Multi = np.hstack((y_PK, y_train))
            x_train = x_train.T
            x_Multi = x_Multi.T

            clfLDAInd.fit(x_train, y_train)
            clfQDAInd.fit(x_train, y_train)
            clfLDAMulti.fit(x_Multi, y_Multi)
            clfQDAMulti.fit(x_Multi, y_Multi)

            if peopleSame == 0:
                clfKNNInd.fit(x_train, y_train)
                clfSVMInd.fit(x_train, y_train)
                clfKNNMulti.fit(x_Multi, y_Multi)
                clfSVMMulti.fit(x_Multi, y_Multi)
                resultsData.at[idx, 'yKNNInd'] = clfKNNInd.score(x_test, y_test)
                resultsData.at[idx, 'ySVMInd'] = clfSVMInd.score(x_test, y_test)
                resultsData.at[idx, 'yKNNMulti'] = clfKNNMulti.score(x_test, y_test)
                resultsData.at[idx, 'ySVMMulti'] = clfSVMMulti.score(x_test, y_test)

            step = 1
            k = 1 - (np.log(i + 1) / np.log(samples + 1))
            print(k)

            propModelQDA, wTargetMeanQDA, wTargetMeanQDAm, wTargetCovQDA, wTargetCovQDAm, tPropQ = ProposedModel(
                currentValues, preTrainedDataMatrix, classes, Features, x_train, y_train, step, 'QDA', k)
            propModelLDA, wTargetMeanLDA, wTargetMeanLDAm, wTargetCovLDA, wTargetCovLDAm, tPropL = ProposedModel(
                currentValues, preTrainedDataMatrix, classes, Features, x_train, y_train, step, 'LDA', k)

            liuModel = LiuModel(currentValues, preTrainedDataMatrix, classes, Features)

            resultsData.at[idx, 'yLDAInd'] = clfLDAInd.score(x_test, y_test)
            resultsData.at[idx, 'yQDAInd'] = clfQDAInd.score(x_test, y_test)

            resultsData.at[idx, 'yLDAMulti'] = clfLDAMulti.score(x_test, y_test)
            resultsData.at[idx, 'yQDAMulti'] = clfQDAMulti.score(x_test, y_test)

            resultsData.at[idx, 'yPropL'], _ = scoreModelLDA(x_test, y_test, propModelLDA, classes)
            resultsData.at[idx, 'yPropQ'], _ = scoreModelQDA(x_test, y_test, propModelQDA, classes)
            resultsData.at[idx, 'yLiu'], _ = scoreModelLDA(x_test, y_test, liuModel, classes)
            resultsData.at[idx, 'yLiuQDA'], _ = scoreModelQDA(x_test, y_test, liuModel, classes)

            resultsData.at[idx, 'wTargetMeanLDA'] = wTargetMeanLDA
            resultsData.at[idx, 'wTargetMeanLDAm'] = wTargetMeanLDAm
            resultsData.at[idx, 'wTargetCovLDA'] = wTargetCovLDA
            resultsData.at[idx, 'wTargetCovLDAm'] = wTargetCovLDAm

            resultsData.at[idx, 'wTargetMeanQDA'] = wTargetMeanQDA
            resultsData.at[idx, 'wTargetMeanQDAm'] = wTargetMeanQDAm
            resultsData.at[idx, 'wTargetCovQDA'] = wTargetCovQDA
            resultsData.at[idx, 'wTargetCovQDAm'] = wTargetCovQDAm

            resultsData.at[idx, 'wLiu'] = 0.5
            resultsData.at[idx, 'person'] = per
            resultsData.at[idx, 'times'] = t
            resultsData.at[idx, 'shots'] = i + iSample

            if nameFile is not None:
                resultsData.to_csv(nameFile)
            idx += 1
            if printValues:
                print(per + 1, t + 1, i + 1)

    if Graph:
        clAdap = True
        title = 'Accuracy vs Samples'
        graphSyntheticData(None, resultsData, numberShots, iSample, clAdap, title)


def graphSyntheticData(resultsDataCL, resultsData, numberShots, iSample, clAdap, title):
    if resultsDataCL is None:
        resultsDataCL = resultsData.copy()
    shot = np.arange(numberShots)
    yLDAInd = np.zeros(numberShots)
    yQDAInd = np.zeros(numberShots)
    yKNNInd = np.zeros(numberShots)
    ySVMInd = np.zeros(numberShots)
    yLDAMulti = np.zeros(numberShots)
    yQDAMulti = np.zeros(numberShots)
    yKNNMulti = np.zeros(numberShots)
    ySVMMulti = np.zeros(numberShots)
    yPropQ = np.zeros(numberShots)
    yPropL = np.zeros(numberShots)
    yLiu = np.zeros(numberShots)
    yLiuQDA = np.zeros(numberShots)
    wTargetLm = np.zeros(numberShots)
    wTargetQm = np.zeros(numberShots)
    wTargetLc = np.zeros(numberShots)
    wTargetQc = np.zeros(numberShots)
    wLiu = np.ones(numberShots) * 0.5

    for i in range(numberShots):
        yLDAInd[i] = resultsData['yLDAInd'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        yQDAInd[i] = resultsData['yQDAInd'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        yKNNInd[i] = resultsDataCL['yKNNInd'][
            (resultsDataCL['shots'] == i + iSample) & (resultsDataCL['shots'] == i + iSample)].mean()
        ySVMInd[i] = resultsDataCL['ySVMInd'][
            (resultsDataCL['shots'] == i + iSample) & (resultsDataCL['shots'] == i + iSample)].mean()
        yLDAMulti[i] = resultsData['yLDAMulti'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        yQDAMulti[i] = resultsData['yQDAMulti'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        yKNNMulti[i] = resultsDataCL['yKNNMulti'][
            (resultsDataCL['shots'] == i + iSample) & (resultsDataCL['shots'] == i + iSample)].mean()
        ySVMMulti[i] = resultsDataCL['ySVMMulti'][
            (resultsDataCL['shots'] == i + iSample) & (resultsDataCL['shots'] == i + iSample)].mean()
        yPropL[i] = resultsData['yPropL'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        yPropQ[i] = resultsData['yPropQ'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        yLiu[i] = resultsData['yLiu'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        yLiuQDA[i] = resultsData['yLiuQDA'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        wTargetLm[i] = resultsData['wTargetMeanLDAm'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        wTargetQm[i] = resultsData['wTargetMeanQDAm'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        wTargetLc[i] = resultsData['wTargetCovLDAm'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        wTargetQc[i] = resultsData['wTargetCovQDAm'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()

        # confidence = 0.1
        #
        # T_test = stats.ttest_ind(
        #     resultsData['yPropL'][(resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].values,
        #     LDAInd.values)[1]
        # if T_test <= confidence:
        #     results.at[idx, 'T-test (LDA_Ind)'] = T_test
        # else:
        #     results.at[idx, 'T-test (LDA_Ind)'] = 1

    fig1, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(9, 3))
    sizeM = 5
    if clAdap:
        ax1.plot(shot + iSample, yLDAInd, label='LDA', markersize=sizeM, marker='s', color='tab:orange')
        ax1.plot(shot + iSample, yQDAInd, label='QDA', markersize=sizeM, marker='p', color='tab:green')
        ax1.plot(shot + iSample, yPropL, label='Adaptive LDA', markersize=sizeM, marker='^', color='tab:blue')
        ax1.plot(shot + iSample, yPropQ, label='Adaptive QDA', markersize=sizeM, marker='v', color='tab:gray')
        ax1.plot(shot + iSample, yLiu, label='Liu', markersize=sizeM, marker='.', color='tab:red')

    else:
        ax1.plot(shot + iSample, yLDAInd, label='Adaptive LDA', markersize=sizeM, marker='s', color='tab:orange')
        ax1.plot(shot + iSample, yQDAInd, label='QDA', markersize=sizeM, marker='p', color='tab:green')
        ax1.plot(shot + iSample, yKNNInd, label='KNN', markersize=sizeM, marker='*', color='tab:brown')
        ax1.plot(shot + iSample, ySVMInd, label='SVM', markersize=sizeM, marker='x', color='tab:purple')
        # ax1.plot(shot + iSample, yPropL, label='Adaptive LDA', markersize=sizeM, marker='^', color='black')
        # ax1.plot(shot + iSample, yPropQ, label='Adaptive QDA', markersize=sizeM, marker='v', color='blue')
        # ax1.plot(shot + iSample, yLiu, label='Liu', markersize=sizeM, marker='.', color='red')

    # ax1.plot(shot + iSample, yLDAMulti, label='Mutli-user LDA',markersize=sizeM, marker='o')
    # ax1.plot(shot + iSample, yQDAMulti, label='Mutli-user QDA', markersize=sizeM, marker='^')
    # ax1.plot(shot + iSample, yKNNMulti, label='Mutli-user KNN', markersize=sizeM, marker='v')
    # ax1.plot(shot + iSample, ySVMMulti, label='Mutli-user SVM', markersize=sizeM, marker='1')

    ax1.set_title(title)
    ax1.grid()
    ax1.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0.)

    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('samples')

    # ax2.plot(shot + iSample, wTargetLm, label='W Adaptive LDA mean',markersize=sizeM, marker='^')
    # ax2.plot(shot + iSample, wTargetQm, label='W Adaptive QDA mean',markersize=sizeM, marker='v')
    # ax2.plot(shot + iSample, wTargetLc, label='W Adaptive LDA cov',markersize=sizeM, marker='^')
    # ax2.plot(shot + iSample, wTargetQc, label='W Adaptive QDA cov',markersize=sizeM, marker='v')
    # ax2.plot(shot + iSample, wLiu, label='W Liu',markersize=sizeM, marker='.')
    # ax2.set_title('Target Weight  vs Samples')
    # ax2.grid()
    # ax2.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0.)
    # ax2.set_xlabel('Samples')
    # ax2.set_ylabel('Weight')

    # fig2, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(9, 6))
    #
    # ax1.plot(shot + iSample, yPropQ, label='Adaptive QDA', color='green', marker='^')
    # ax1.plot(shot + iSample, yLiuQDA, label='Liu QDA', color='orange', marker='v')
    # ax1.plot(shot + iSample, yQDAInd, label='Individual QDA', color='red', marker='x')
    # # ax1.plot(shot + iSample, yQDAMulti, label='Mutli-user QDA', color='blue', marker='o')
    #
    # ax1.set_title('Accuracy vs Samples')
    # ax1.grid()
    # ax1.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0.)
    #
    # ax1.set_ylabel('QDA \n Accuracy')
    #
    # ax2.plot(shot + iSample, wTargetQ, label='W Adaptive QDA', color='green', marker='^')
    # ax2.plot(shot + iSample, wLiu, label='W Liu QDA', color='red', marker='x')
    # ax2.set_title('Target Weight vs Samples')
    # ax2.grid()
    # ax2.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0.)
    # ax2.set_xlabel('Shots')
    # ax2.set_ylabel('QDA \n Weight')
    fig1.tight_layout()
    plt.savefig("synthetic.png", bbox_inches='tight', dpi=600)
    # fig2.tight_layout()
    plt.show()


# Hyper-Parameters Tuning

def hyperParameterTuning_KNN(x, y, per):
    parameters = {'p': [1, 2, 3], 'n_neighbors': [1, 3, 5, 7], 'weights': ('uniform', 'distance')}
    # Create new KNN object
    knn = KNN()
    # Use GridSearch
    generalClassifierKNN = GridSearchCV(knn, parameters, cv=3)
    # Fit the model
    best_model = generalClassifierKNN.fit(x, y)
    # Print The value of best Hyperparameters
    print('PERSON:', per)
    print('Best p:', best_model.best_estimator_.get_params()['p'])
    print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
    print('Best weights:', best_model.best_estimator_.get_params()['weights'])
    # Best parameters
    parameters = pd.DataFrame(columns=['p', 'n_neighbors', 'weights', 'person'])
    parameters.at[0, 'p'] = best_model.best_estimator_.get_params()['p']
    parameters.at[0, 'n_neighbors'] = best_model.best_estimator_.get_params()['n_neighbors']
    parameters.at[0, 'weights'] = best_model.best_estimator_.get_params()['weights']
    parameters.at[0, 'person'] = per

    return parameters


def hyperParameterTuning_SVM(x, y, per):
    parameters = {'C': [0.1, 1, 10], 'degree': [2, 3], 'kernel': ('rbf', 'poly', 'linear'), 'gamma': ('scale', 'auto')}
    # Create new SVM object
    svm = SVM()
    # Use GridSearch
    generalClassifierSVM = GridSearchCV(svm, parameters, cv=3)
    # Fit the model
    best_model = generalClassifierSVM.fit(x, y)
    # Print The value of best Hyperparameters
    print('PERSON:', per)
    print('Best C:', best_model.best_estimator_.get_params()['C'])
    print('Best degree:', best_model.best_estimator_.get_params()['degree'])
    print('Best kernel:', best_model.best_estimator_.get_params()['kernel'])
    print('Best gamma:', best_model.best_estimator_.get_params()['gamma'])
    # Best parameters
    parameters = pd.DataFrame(columns=['C', 'degree', 'kernel', 'gamma', 'person'])
    parameters.at[0, 'C'] = best_model.best_estimator_.get_params()['C']
    parameters.at[0, 'degree'] = best_model.best_estimator_.get_params()['degree']
    parameters.at[0, 'kernel'] = best_model.best_estimator_.get_params()['kernel']
    parameters.at[0, 'gamma'] = best_model.best_estimator_.get_params()['gamma']
    parameters.at[0, 'person'] = per

    return parameters


def hyperParameterTuning(dataMatrix, labelsDataMatrix, featureSet, typeDatabase, allFeatures, peoplePriorK, peopleTest):
    parametersKNN = pd.DataFrame(columns=['p', 'n_neighbors', 'weights', 'person'])
    parametersSVM = pd.DataFrame(columns=['C', 'degree', 'kernel', 'gamma', 'person'])
    if typeDatabase == 'Cote':
        idxPer = allFeatures + 1
    elif typeDatabase == 'Nina5':
        idxPer = allFeatures
    elif typeDatabase == 'EPN':
        idxPer = allFeatures + 1

    for per in range(peoplePriorK + 1, peoplePriorK + peopleTest + 1):
        x = dataMatrix[np.where((dataMatrix[:, idxPer] == per)), 0:allFeatures][0]
        y = labelsDataMatrix[np.where((dataMatrix[:, idxPer] == per))]
        parametersKNN = parametersKNN.append(hyperParameterTuning_KNN(x, y, per), ignore_index=True)
        parametersSVM = parametersSVM.append(hyperParameterTuning_SVM(x, y, per), ignore_index=True)

    parametersKNN.to_csv('KNN_hyper' + typeDatabase + str(featureSet) + '.csv')
    parametersSVM.to_csv('SVM_hyper' + typeDatabase + str(featureSet) + '.csv')
    print('PRINT PARAMETERS' + typeDatabase + str(featureSet))


def preprocessingData(dataMatrix, allFeatures, peoplePriorK, peopleTest, typeDatabase):
    scaler = preprocessing.MinMaxScaler()

    if typeDatabase == 'Cote':
        idxPer = allFeatures + 1
    elif typeDatabase == 'Nina5':
        idxPer = allFeatures
    elif typeDatabase == 'EPN':
        idxPer = allFeatures + 1

    for per in range(1, peoplePriorK + peopleTest + 1):
        dataMatrix[np.where((dataMatrix[:, idxPer] == per)), 0:allFeatures] = scaler.fit_transform(
            dataMatrix[np.where((dataMatrix[:, idxPer] == per)), 0:allFeatures][0])
    return dataMatrix


def pcaData(dataMatrix, allFeatures):
    pcaComp = 0.99
    pca = PCA(n_components=pcaComp)
    dataMatrixPca = pca.fit_transform(dataMatrix[:, 0:allFeatures])
    dataMatrix = np.hstack((dataMatrixPca, dataMatrix[:, allFeatures:]))
    allFeatures = np.size(dataMatrixPca, axis=1)
    return dataMatrix, allFeatures


def preprocessingPK(dataMatrix, allFeatures, scaler):
    dataMatrixFeatures = scaler.transform(dataMatrix[:, 0:allFeatures])
    return np.hstack((dataMatrixFeatures, dataMatrix[:, allFeatures:])), np.size(dataMatrixFeatures, axis=1)
