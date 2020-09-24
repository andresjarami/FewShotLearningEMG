import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.ticker as mtick


from scipy import stats
from scipy.spatial import distance

import time
import math

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.svm import SVC as SVM
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.decomposition import PCA

import itertools
import random
import ast




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
                     clfLDAMulti, clfQDAMulti, k, testRep, tPre, pkValues):
    # Amount of Training data
    minSamplesClass = 20
    step = math.ceil(np.shape(trainLabels)[0] / (classes * minSamplesClass))
    print('step: ', np.shape(trainLabels)[0], step)

    liuModel = LiuModel(currentValues, preTrainedDataMatrix, classes, allFeatures)
    vidovicModelL, vidovicModelQ = VidovicModel(currentValues, preTrainedDataMatrix, classes, allFeatures)

    propModelQDA, _, results.at[idx, 'wTargetMeanQDA'], _, results.at[idx, 'wTargetCovQDA'], results.at[
        idx, 'tPropQDA'] = ProposedModel(
        currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels, step, 'QDA', k)

    results.at[idx, 'person'] = person
    results.at[idx, 'subset'] = subset
    results.at[idx, '# shots'] = np.size(subset)
    results.at[idx, 'Feature Set'] = featureSet
    # LDA results
    results.at[idx, 'AccLDAInd'], results.at[idx, 'tIndL'] = scoreModelLDA(testFeatures, testLabels, currentValues,
                                                                           classes)
    results.at[idx, 'AccLDAMulti'], results.at[idx, 'tGenL'] = scoreModelLDA(testFeatures, testLabels, pkValues,
                                                                             classes)
    results.at[idx, 'AccLDAProp'], results.at[idx, 'tCLPropL'] = scoreModelLDA(testFeatures, testLabels, propModelQDA,
                                                                               classes)
    results.at[idx, 'AccLDALiu'], _ = scoreModelLDA(testFeatures, testLabels, liuModel, classes)
    results.at[idx, 'AccLDAVidovic'], _ = scoreModelLDA(testFeatures, testLabels, vidovicModelL, classes)

    ## QDA results
    results.at[idx, 'AccQDAInd'], results.at[idx, 'tIndQ'] = scoreModelQDA(testFeatures, testLabels, currentValues,
                                                                           classes)
    results.at[idx, 'AccQDAMulti'], results.at[idx, 'tGenQ'] = scoreModelQDA(testFeatures, testLabels, pkValues,
                                                                             classes)
    results.at[idx, 'AccQDAProp'], results.at[idx, 'tCLPropQ'] = scoreModelQDA(testFeatures, testLabels, propModelQDA,
                                                                               classes)
    results.at[idx, 'AccQDALiu'], _ = scoreModelQDA(testFeatures, testLabels, liuModel, classes)
    results.at[idx, 'AccQDAVidovic'], _ = scoreModelQDA(testFeatures, testLabels, vidovicModelQ, classes)

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
                                                clfLDAInd, clfQDAInd, clfLDAMulti, clfQDAMulti, k, testRep, tPre,
                                                pkValues)

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
                                            clfLDAInd, clfQDAInd, clfLDAMulti, clfQDAMulti, k, testRep, tPre, pkValues)

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
                                            clfLDAInd, clfQDAInd, clfLDAMulti, clfQDAMulti, k, testRep, tPre, pkValues)

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
    windowFile = ''
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


def DataGenerator_TwoCL_TwoFeat(seed=None, samples=100, people=5, peopleSame=0, Graph=True):
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
                # only for the graph
                # auxPoint = np.array([0, 7])
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
                        confidence_ellipse(x1, x2, ax, edgecolor=color, label=label, linestyle=lineStyle)
                    else:
                        ax.scatter(x1, x2, s=sizeM, color=color, marker=markerCL)
                        confidence_ellipse(x1, x2, ax, edgecolor=color, linestyle=lineStyle)

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

    # for t in range(times):
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

        step = 1
        k = 1 - (np.log(i + 1) / np.log(samples + 1))

        propModelQDA, _, resultsData.at[idx, 'wTargetMeanQDA'], _, resultsData.at[idx, 'wTargetCovQDA'], resultsData.at[
            idx, 'tPropQDA'] = ProposedModel(currentValues, preTrainedDataMatrix, classes, Features, x_train.T, y_train,
                                             step, 'LDA', k)

        liuModel = LiuModel(currentValues, preTrainedDataMatrix, classes, Features)
        vidovicModelL, vidovicModelQ = VidovicModel(currentValues, preTrainedDataMatrix, classes, Features)

        resultsData.at[idx, 'AccLDAInd'], _ = scoreModelLDA(x_test, y_test, currentValues, classes)
        resultsData.at[idx, 'AccQDAInd'], _ = scoreModelQDA(x_test, y_test, currentValues, classes)

        resultsData.at[idx, 'AccLDAMulti'], _ = scoreModelLDA(x_test, y_test, pkValues, classes)
        resultsData.at[idx, 'AccQDAMulti'], _ = scoreModelQDA(x_test, y_test, pkValues, classes)

        resultsData.at[idx, 'AccLDAProp'], _ = scoreModelLDA(x_test, y_test, propModelQDA, classes)
        resultsData.at[idx, 'AccQDAProp'], _ = scoreModelQDA(x_test, y_test, propModelQDA, classes)
        resultsData.at[idx, 'AccLDALiu'], _ = scoreModelLDA(x_test, y_test, liuModel, classes)
        resultsData.at[idx, 'AccQDALiu'], _ = scoreModelQDA(x_test, y_test, liuModel, classes)
        resultsData.at[idx, 'AccLDAVidovic'], _ = scoreModelLDA(x_test, y_test, vidovicModelL, classes)
        resultsData.at[idx, 'AccQDAVidovic'], _ = scoreModelQDA(x_test, y_test, vidovicModelQ, classes)

        resultsData.at[idx, 'person'] = per
        resultsData.at[idx, 'times'] = t
        resultsData.at[idx, 'shots'] = i + iSample

        if nameFile is not None:
            resultsData.to_csv(nameFile)
        idx += 1
        if printValues:
            print(per + 1, t + 1, i + 1)

    if Graph:
        title = 'Accuracy vs Samples'
        graphSyntheticData(resultsData, numberShots, iSample)


def graphSyntheticData(resultsData, numberShots, iSample):
    shot = np.arange(numberShots)
    AccLDAInd = np.zeros(numberShots)
    AccQDAInd = np.zeros(numberShots)

    AccLDAMulti = np.zeros(numberShots)
    AccQDAMulti = np.zeros(numberShots)

    AccLDAProp = np.zeros(numberShots)
    AccQDAProp = np.zeros(numberShots)
    AccLDALiu = np.zeros(numberShots)
    AccQDALiu = np.zeros(numberShots)
    AccLDAVidovic = np.zeros(numberShots)
    AccQDAVidovic = np.zeros(numberShots)

    for i in range(numberShots):
        AccLDAInd[i] = resultsData['AccLDAInd'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        AccQDAInd[i] = resultsData['AccQDAInd'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()

        AccLDAMulti[i] = resultsData['AccLDAMulti'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        AccQDAMulti[i] = resultsData['AccQDAMulti'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()

        AccLDAProp[i] = resultsData['AccLDAProp'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        AccQDAProp[i] = resultsData['AccQDAProp'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        AccLDALiu[i] = resultsData['AccLDALiu'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        AccQDALiu[i] = resultsData['AccQDALiu'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        AccLDAVidovic[i] = resultsData['AccLDAVidovic'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        AccQDAVidovic[i] = resultsData['AccQDAVidovic'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()

    fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(9, 3))
    sizeM = 5
    ax1.plot(shot + iSample, AccLDAInd, label='Individual', markersize=sizeM, color='tab:orange', linestyle='--')
    ax1.plot(shot + iSample, AccLDAMulti, label='Multi-user', markersize=sizeM, color='tab:purple',
             linestyle=(0, (3, 3, 1, 3, 1, 3)))
    ax1.plot(shot + iSample, AccLDALiu, label='Liu', markersize=sizeM, color='tab:green', linestyle=':')
    ax1.plot(shot + iSample, AccLDAVidovic, label='Vidovic', markersize=sizeM, color='tab:red',
             linestyle=(0, (3, 3, 1, 3)))
    ax1.plot(shot + iSample, AccLDAProp, label='Adaptive', color='tab:blue')
    ax1.set_title('LDA')
    ax1.grid()
    ax1.set_ylabel('accuracy')
    ax2.plot(shot + iSample, AccQDAInd, label='Individual', markersize=sizeM, color='tab:orange', linestyle='--')
    ax2.plot(shot + iSample, AccQDAMulti, label='Multi-user', markersize=sizeM, color='tab:purple',
             linestyle=(0, (3, 3, 1, 3, 1, 3)))
    ax2.plot(shot + iSample, AccQDALiu, label='Liu', markersize=sizeM, color='tab:green', linestyle=':')
    ax2.plot(shot + iSample, AccQDAVidovic, label='Vidovic', markersize=sizeM, color='tab:red',
             linestyle=(0, (3, 3, 1, 3)))
    ax2.plot(shot + iSample, AccQDAProp, label='Adaptive', color='tab:blue')
    ax2.set_title('QDA')
    ax2.grid()
    ax2.set_ylabel('accuracy')
    ax2.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0.)
    ax2.set_xlabel('samples')

    fig1.tight_layout()
    # plt.savefig("synthetic.png", bbox_inches='tight', dpi=600)
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


#### Functions used in the Jupyters Notebooks


def DataGenerator_TwoCL_TwoFeatEXAMPLE(seed=1, samples=50, people=3, peopleSame=1):
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

    colors_list = list(['blue', 'red', 'deepskyblue', 'lightcoral', 'orange', 'green'])
    fig1, ax = plt.subplots(1, 1, figsize=(9, 4))

    for person in range(people):

        covSet = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])

        if person == people - 1:
            auxPoint = np.zeros(2)
            covSet = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])

        else:
            if person < peopleDiff:

                auxPoint = np.array([10, 0])
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

            x1 = DataFrame.loc[idx, 'data'][0, :]
            x2 = DataFrame.loc[idx, 'data'][1, :]

            sizeM = 7

            if person == people - 1:
                color = colors_list[cl]
                label = 'clase ' + str(cl) + ': target user'
                if cl == 0:
                    markerCL = 'o'
                else:
                    markerCL = '^'
                ax.scatter(x1, x2, s=sizeM, color=color, marker=markerCL)
                confidence_ellipse(x1, x2, ax, edgecolor=color, label=label)
            else:
                if person < peopleDiff:
                    color = colors_list[cl + 4]
                    label = 'clase ' + str(cl) + ': different user'
                    lineStyle = '-.'
                    if cl == 0:
                        markerCL = 's'
                    else:
                        markerCL = 'p'
                else:
                    color = colors_list[cl + 2]

                    label = 'clase ' + str(cl) + ': similar user'
                    lineStyle = ':'
                    if cl == 0:
                        markerCL = '*'
                    else:
                        markerCL = 'x'

            if person != people - 1:
                if person == people - 2 or person == peopleDiff - 1:
                    ax.scatter(x1, x2, s=sizeM, color=color, marker=markerCL)
                    confidence_ellipse(x1, x2, ax, edgecolor=color, label=label, linestyle=lineStyle)
                else:
                    ax.scatter(x1, x2, s=sizeM, color=color, marker=markerCL)
                    confidence_ellipse(x1, x2, ax, edgecolor=color, linestyle=lineStyle)

            idx += 1

    plt.grid()
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3, prop={'size': 6.5})
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    fig1.tight_layout(pad=0.1)
    # plt.savefig("distr.png", bbox_inches='tight', dpi=600)
    plt.show()
    return


def SyntheticData(resultsData, numberShots, iSample):
    shot = np.arange(numberShots)
    AccLDAInd = np.zeros(numberShots)
    AccQDAInd = np.zeros(numberShots)

    AccLDAMulti = np.zeros(numberShots)
    AccQDAMulti = np.zeros(numberShots)

    AccLDAProp = np.zeros(numberShots)
    AccQDAProp = np.zeros(numberShots)
    AccLDALiu = np.zeros(numberShots)
    AccQDALiu = np.zeros(numberShots)
    AccLDAVidovic = np.zeros(numberShots)
    AccQDAVidovic = np.zeros(numberShots)

    for i in range(numberShots):
        AccLDAInd[i] = resultsData['AccLDAInd'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        AccQDAInd[i] = resultsData['AccQDAInd'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()

        AccLDAMulti[i] = resultsData['AccLDAMulti'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        AccQDAMulti[i] = resultsData['AccQDAMulti'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()

        AccLDAProp[i] = resultsData['AccLDAProp'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        AccQDAProp[i] = resultsData['AccQDAProp'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        AccLDALiu[i] = resultsData['AccLDALiu'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        AccQDALiu[i] = resultsData['AccQDALiu'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        AccLDAVidovic[i] = resultsData['AccLDAVidovic'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        AccQDAVidovic[i] = resultsData['AccQDAVidovic'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()

    return shot, AccLDAInd * 100, AccQDAInd * 100, AccLDAMulti * 100, AccQDAMulti * 100, AccLDAProp * 100, AccQDAProp * 100, AccLDALiu * 100, AccQDALiu * 100, AccLDAVidovic * 100, AccQDAVidovic * 100


def graphSyntheticDataALL(place):
    samples = 50
    # place = 'Experiments/Experiment1_2/ResultsExp2/results'
    for j in [0, 1, 3, 5, 10, 15, 20]:
        frame = pd.read_csv(place + 'Synthetic_peopleSimilar_' + str(j) + 'time_0' + '.csv')

        for i in range(1, 100):
            auxFrame = pd.read_csv(place + 'Synthetic_peopleSimilar_' + str(j) + 'time_' + str(i) + '.csv')
            frame = pd.concat([frame, auxFrame], ignore_index=True)
            if len(auxFrame) != samples:
                print('error' + ' 0 ' + str(i))
                print(len(auxFrame))
        frame.to_csv(place + 'Synthetic_peopleSimilar_' + str(j) + '.csv')



    fig, ax = plt.subplots(nrows=4, ncols=2, sharex=True, sharey=True, figsize=(11, 6))
    sizeM = 5

    numberShots = 48
    iSample = 3
    idx = 0
    for peopleSimilar in [0, 1, 3, 5]:
        resultsData = pd.read_csv(place + 'Synthetic_peopleSimilar_' + str(peopleSimilar) + '.csv')

        shot, AccLDAInd, AccQDAInd, AccLDAMulti, AccQDAMulti, AccLDAProp, AccQDAProp, AccLDALiu, AccQDALiu, AccLDAVidovic, AccQDAVidovic = SyntheticData(
            resultsData, numberShots, iSample)

        ax[idx, 0].plot(shot + iSample, AccLDAInd, label='Individual', markersize=sizeM, color='tab:orange',
                        linestyle='--')
        #         ax[idx,0].plot(shot + iSample, AccLDAMulti, label='Multi-user', markersize=sizeM, color='tab:purple',
        #                  linestyle=(0, (3, 3, 1, 3, 1, 3)))
        ax[idx, 0].plot(shot + iSample, AccLDALiu, label='Liu', markersize=sizeM, color='tab:green', linestyle=':')
        ax[idx, 0].plot(shot + iSample, AccLDAVidovic, label='Vidovic', markersize=sizeM, color='tab:red',
                        linestyle=(0, (3, 3, 1, 3)))
        ax[idx, 0].plot(shot + iSample, AccLDAProp, label='Our technique', color='tab:blue')
        ax[idx, 0].grid()
        ax[idx, 0].set_ylabel(
            str(20 - peopleSimilar) + ' different\n and \n' + str(peopleSimilar) + ' similar\nusers\n\naccuracy')
        ax[idx, 0].xaxis.set_ticks([3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50])
        ax[idx, 0].yaxis.set_ticks(np.arange(50, 90, 10))

        ax[idx, 1].plot(shot + iSample, AccQDAInd, label='Individual', markersize=sizeM, color='tab:orange',
                        linestyle='--')
        ax[idx, 1].plot(shot + iSample, AccQDALiu, label='Liu', markersize=sizeM, color='tab:green', linestyle=':')
        ax[idx, 1].plot(shot + iSample, AccQDAVidovic, label='Vidovic', markersize=sizeM, color='tab:red',
                        linestyle=(0, (3, 3, 1, 3)))
        ax[idx, 1].plot(shot + iSample, AccQDAProp, label='Our technique', color='tab:blue')
        ax[idx, 1].grid()
        ax[idx, 1].xaxis.set_ticks([3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50])
        ax[idx, 1].yaxis.set_ticks(np.arange(50, 90, 10))

        idx += 1

    ax[3, 1].legend(loc='lower center', bbox_to_anchor=(0.8, -0.9), ncol=4, prop={'size': 9})

    ax[0, 0].set_title('LDA')
    ax[0, 1].set_title('QDA')
    ax[3, 0].set_xlabel('samples')
    ax[3, 1].set_xlabel('samples')

    fig.tight_layout()
    # plt.savefig("synthetic.png", bbox_inches='tight', dpi=600)
    plt.show()


def uploadData(place, samples, people, shots):
    resultsTest = pd.read_csv(place + "_FeatureSet_1_startPerson_1_endPerson_1.csv")
    if len(resultsTest) != samples:
        print('error' + ' 1' + ' 1')
        print(len(resultsTest))

    for i in range(2, people + 1):
        auxFrame = pd.read_csv(place + "_FeatureSet_1_startPerson_" + str(i) + "_endPerson_" + str(i) + ".csv")
        resultsTest = pd.concat([resultsTest, auxFrame], ignore_index=True)
        if len(auxFrame) != samples:
            print('error' + ' 1 ' + str(i))
            print(len(auxFrame))
    for j in range(2, 4):
        for i in range(1, people + 1):
            auxFrame = pd.read_csv(
                place + "_FeatureSet_" + str(j) + "_startPerson_" + str(i) + "_endPerson_" + str(i) + ".csv")
            resultsTest = pd.concat([resultsTest, auxFrame], ignore_index=True)

            if len(auxFrame) != samples:
                print('error' + ' ' + str(j) + ' ' + str(i))
                print(len(auxFrame))

    return resultsTest.drop(columns='Unnamed: 0')


def uploadDatabase(place, samples, people, shots, Classification=False):
    resultsTest = pd.read_csv(place + "_FeatureSet_1_startPerson_1_endPerson_1.csv")
    if len(resultsTest) != samples:
        print('error' + ' 1' + ' 1')
        print(len(resultsTest))

    for i in range(2, people + 1):
        auxFrame = pd.read_csv(place + "_FeatureSet_1_startPerson_" + str(i) + "_endPerson_" + str(i) + ".csv")
        resultsTest = pd.concat([resultsTest, auxFrame], ignore_index=True)
        if len(auxFrame) != samples:
            print('error' + ' 1 ' + str(i))
            print(len(auxFrame))
    for j in range(2, 4):
        for i in range(1, people + 1):
            auxFrame = pd.read_csv(
                place + "_FeatureSet_" + str(j) + "_startPerson_" + str(i) + "_endPerson_" + str(i) + ".csv")
            resultsTest = pd.concat([resultsTest, auxFrame], ignore_index=True)

            if len(auxFrame) != samples:
                print('error' + ' ' + str(j) + ' ' + str(i))
                print(len(auxFrame))

    return analysisResults(resultsTest.drop(columns='Unnamed: 0'), shots, Classification)


def analysisResults(resultDatabase, shots, Classification=False):
    results = pd.DataFrame(columns=['Feature Set', '# shots'])
    timeM = pd.DataFrame(columns=[])

    idx = 0
    for j in range(1, 4):
        for i in range(1, shots + 1):
            results.at[idx, 'Feature Set'] = j
            results.at[idx, '# shots'] = i

            subset = str(tuple(range(1, i + 1)))

            LDAmulti = resultDatabase['AccLDAMulti'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
            QDAmulti = resultDatabase['AccQDAMulti'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]

            if Classification:

                LDAInd = resultDatabase['AccClLDAInd'].loc[
                    (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
                QDAInd = resultDatabase['AccClQDAInd'].loc[
                    (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]

                PropQ = resultDatabase['AccClQDAProp'].loc[
                    (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
                PropQ_L = resultDatabase['AccClLDAPropQ'].loc[
                    (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]

                LiuL = resultDatabase['AccClLDALiu'].loc[
                    (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
                LiuQ = resultDatabase['AccClQDALiu'].loc[
                    (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
                VidL = resultDatabase['AccClLDAVidovic'].loc[
                    (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
                VidQ = resultDatabase['AccClQDAVidovic'].loc[
                    (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]

            else:
                LDAInd = resultDatabase['AccLDAInd'].loc[
                    (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
                QDAInd = resultDatabase['AccQDAInd'].loc[
                    (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
                PropQ = resultDatabase['AccQDAProp'].loc[
                    (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
                PropQ_L = resultDatabase['AccLDAPropQ'].loc[
                    (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]

                LiuL = resultDatabase['AccLDALiu'].loc[
                    (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
                LiuQ = resultDatabase['AccQDALiu'].loc[
                    (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
                VidL = resultDatabase['AccLDAVidovic'].loc[
                    (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
                VidQ = resultDatabase['AccQDAVidovic'].loc[
                    (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]

            wmQ = resultDatabase['wTargetMeanQDAm'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
            wcQ = resultDatabase['wTargetCovQDAm'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]

            trainingPropT = resultDatabase['tPropQ'].loc[
                (resultDatabase['subset'] == subset) & (resultDatabase['Feature Set'] == j)]
            classificationPropQDAT = resultDatabase['tCLPropQ'].loc[(resultDatabase['Feature Set'] == j)]
            classificationPropLDAT = resultDatabase['tCLPropL'].loc[(resultDatabase['Feature Set'] == j)]

            results.at[idx, 'LDA_Ind'] = LDAInd.median(axis=0)
            results.at[idx, 'QDA_Ind'] = QDAInd.median(axis=0)
            results.at[idx, 'LDA_Multi'] = LDAmulti.median(axis=0)
            results.at[idx, 'QDA_Multi'] = QDAmulti.median(axis=0)
            results.at[idx, 'LiuL'] = LiuL.median(axis=0)
            results.at[idx, 'LiuQ'] = LiuQ.median(axis=0)
            results.at[idx, 'VidL'] = VidL.median(axis=0)
            results.at[idx, 'VidQ'] = VidQ.median(axis=0)
            results.at[idx, 'PropQ'] = PropQ.median(axis=0)
            results.at[idx, 'PropQ_L'] = PropQ_L.median(axis=0)
            results.at[idx, 'wmQ'] = wmQ.mean(axis=0)
            results.at[idx, 'wcQ'] = wcQ.mean(axis=0)

            results.at[idx, 'trainingPropT'] = trainingPropT.mean(axis=0)
            results.at[idx, 'classificationPropQDAT'] = classificationPropQDAT.mean(axis=0)
            results.at[idx, 'classificationPropLDAT'] = classificationPropLDAT.mean(axis=0)
            results.at[idx, 'std_tPropQ'] = trainingPropT.std(axis=0)
            results.at[idx, 'std_tIndQ'] = classificationPropQDAT.std(axis=0)
            results.at[idx, 'std_tIndL'] = classificationPropLDAT.std(axis=0)

            results.at[idx, 'stdLDA_Ind'] = LDAInd.std(axis=0)
            results.at[idx, 'stdQDA_Ind'] = QDAInd.std(axis=0)
            results.at[idx, 'stdPropQ_L'] = PropQ_L.std(axis=0)
            results.at[idx, 'stdPropQ'] = PropQ.std(axis=0)
            results.at[idx, 'stdLiuL'] = LiuL.std(axis=0)
            results.at[idx, 'stdLiuQ'] = LiuQ.std(axis=0)

            confidence = 0.05

            p = stats.wilcoxon(PropQ_L.values, LDAInd.values, alternative='greater', zero_method='zsplit')[1]
            if p < confidence:
                results.at[idx, 'T-test (LDA_Ind)'] = p
            else:
                results.at[idx, 'T-test (LDA_Ind)'] = 1

            p = stats.wilcoxon(PropQ.values, QDAInd.values, alternative='greater', zero_method='zsplit')[1]
            if p < confidence:
                results.at[idx, 'T-test (QDA_Ind)'] = p
            else:
                results.at[idx, 'T-test (QDA_Ind)'] = 1

            p = stats.wilcoxon(PropQ_L.values, LDAmulti.values, alternative='greater', zero_method='zsplit')[1]
            if p < confidence:
                results.at[idx, 'T-test (LDA_Multi)'] = p
            else:
                results.at[idx, 'T-test (LDA_Multi)'] = 1

            p = stats.wilcoxon(PropQ.values, QDAmulti.values, alternative='greater', zero_method='zsplit')[1]
            if p < confidence:
                results.at[idx, 'T-test (QDA_Multi)'] = p
            else:
                results.at[idx, 'T-test (QDA_Multi)'] = 1

            idx += 1

        timeM.at[j, 'meanLDA'] = round(classificationPropLDAT.mean(axis=0) * 1000, 2)
        timeM.at[j, 'stdLDA'] = round(classificationPropLDAT.std(axis=0) * 1000, 2)
        timeM.at[j, 'varLDA'] = round(classificationPropLDAT.var(axis=0) * 1000, 2)
        timeM.at[j, 'meanQDA'] = round(classificationPropQDAT.mean(axis=0) * 1000, 2)
        timeM.at[j, 'stdQDA'] = round(classificationPropQDAT.std(axis=0) * 1000, 2)
        timeM.at[j, 'varQDA'] = round(classificationPropQDAT.var(axis=0) * 1000, 2)

        print('Training Time Proposed (Feature set:' + str(j) + ')', round(trainingPropT.mean(axis=0), 2), '+-',
              round(trainingPropT.std(axis=0), 2))

    return results, timeM


def graphACC(resultsNina5T,resultsCoteT,resultsEPNT):
    FeatureSetM = 3
    fig, ax = plt.subplots(nrows=3, ncols=6, sharey='row', figsize=(13, 6))
    #     shot=np.arange(1,5)
    for classifier in range(2):
        for FeatureSet in range(FeatureSetM):
            for Data in range(FeatureSetM):
                if Data == 0:
                    shot = np.arange(1, 5)
                    results = resultsNina5T

                    ax[Data, FeatureSet].yaxis.set_ticks(np.arange(0.30, 1, .06))
                elif Data == 1:
                    shot = np.arange(1, 5)
                    results = resultsCoteT

                    ax[Data, FeatureSet].yaxis.set_ticks(np.arange(0.62, 1, 0.06))
                elif Data == 2:
                    results = resultsEPNT
                    shot = np.arange(1, 26)

                    ax[Data, FeatureSet].yaxis.set_ticks(np.arange(0.50, 1, 0.06))



                if classifier == 0:

                    #                     Model='T-test (LDA_Ind)'
                    #                     a=np.array(results[Model].loc[results['Feature Set']==FeatureSet+1])
                    #                     markers_on = list(np.where(a <= value)[0])

                    Model = 'LDA_Ind'

                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1])
                    ax[Data, FeatureSet].plot(shot, Y, label='Individual', color='tab:orange', linestyle='--')

                    Model = 'LDA_Multi'

                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1])
                    ax[Data, FeatureSet].plot(shot, Y, label='Multi-user', color='tab:purple',
                                              linestyle=(0, (3, 3, 1, 3, 1, 3)))

                    Model = 'LiuL'

                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1])
                    ax[Data, FeatureSet].plot(shot, Y, label='Liu', color='tab:green', linestyle=':')

                    Model = 'VidL'

                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1])
                    ax[Data, FeatureSet].plot(shot, Y, label='Vidovic', color='tab:red', linestyle=(0, (3, 3, 1, 3)))

                    Model = 'PropQ_L'
                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1])
                    ax[Data, FeatureSet].plot(shot, Y, label='Adaptive', color='tab:blue')

                    ax[Data, FeatureSet].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

                    if len(shot) == 25:
                        ax[Data, FeatureSet].xaxis.set_ticks([1, 5, 10, 15, 20, 25])
                    else:

                        ax[Data, FeatureSet].xaxis.set_ticks(np.arange(1, len(shot) + .2, 1))
                    ax[Data, FeatureSet].grid()




                elif classifier == 1:


                    Model = 'QDA_Ind'

                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1])
                    ax[Data, FeatureSet + 3].plot(shot, Y, label='Individual', color='tab:orange', linestyle='--')

                    Model = 'QDA_Multi'

                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1])
                    ax[Data, FeatureSet + 3].plot(shot, Y, label='Multi-user', color='tab:purple',
                                                  linestyle=(0, (3, 3, 1, 3, 1, 3)))

                    Model = 'LiuQ'

                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1])
                    ax[Data, FeatureSet + 3].plot(shot, Y, label='Liu', color='tab:green', linestyle=':')

                    Model = 'VidQ'

                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1])
                    ax[Data, FeatureSet + 3].plot(shot, Y, label='Vidovic', color='tab:red',
                                                  linestyle=(0, (3, 3, 1, 3)))

                    Model = 'PropQ'

                    Y = np.array(results[Model].loc[results['Feature Set'] == FeatureSet + 1])

                    ax[Data, FeatureSet + 3].plot(shot, Y, label='Adaptive', color='tab:blue')

                    ax[Data, FeatureSet + 3].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

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
    ax[0, 0].set_ylabel('NinaPro5\n\naccuracy')
    ax[1, 0].set_ylabel('Cote Allard\n\naccuracy')
    ax[2, 0].set_ylabel('EPN \n\naccuracy')

    ax[2, 5].legend(loc='lower center', bbox_to_anchor=(2, -0.7), ncol=5)

    fig.tight_layout(pad=0.1)
    # plt.savefig("databaseACC.png", bbox_inches='tight', dpi=600)
    plt.show()


def graphWeights(resultsNina5T,resultsCoteT,resultsEPNT):
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey='row', sharex='col', figsize=(9, 5))
    for Data in range(3):

        if Data == 0:
            shot = np.arange(1, 5)
            shots = 4
            results = resultsNina5T

            ax[Data].xaxis.set_ticks(np.arange(1, 4.1, 1))
            title = 'NinaPro5'
        elif Data == 1:
            shot = np.arange(1, 5)
            shots = 4
            results = resultsCoteT
            ax[Data].xaxis.set_ticks(np.arange(1, 4.1, 1))
            title = 'Cote-Allard'
        elif Data == 2:

            results = resultsEPNT
            shot = np.arange(1, 26)
            shots = 25

            ax[Data].xaxis.set_ticks([1, 5, 10, 15, 20, 25])
            title = 'EPN'

        wm = np.array(results['wmQ'])
        wc = np.array(results['wcQ'])

        ax[Data].plot(shot, np.mean(wm.reshape((3, shots)), axis=0), label='$\hat{\omega}_c$', marker='.',
                      color='tab:blue', markersize=5)
        ax[Data].plot(shot, np.mean(wc.reshape((3, shots)), axis=0), label='$\hat{\lambda}_c$', marker='^',
                      color='tab:blue', markersize=5)

        ax[Data].set_title(title)
        ax[Data].grid()

    ax[0].set_xlabel('repetitions')
    ax[1].set_xlabel('repetitions')
    ax[2].set_xlabel('repetitions')
    ax[0].set_ylabel('Weight value')
    lgd = ax[2].legend(loc='lower center', bbox_to_anchor=(1.2, -0.8), ncol=2)
    fig.tight_layout(pad=1)
    # plt.savefig("weights.png", bbox_inches='tight', dpi=600, bbox_extra_artists=(lgd,))
    plt.show()


def Analysis():

    bases = ['NinaPro5', 'Cote', 'EPN']
    confidence = 0.05
    results = pd.DataFrame()

    w = 'Total'

    methodCL = 0
    idxD = 0
    print('\n\n', 'Method ' + str(methodCL) + ' ' + w)
    for base in bases:
        print('\n\n', base)
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
        for f in range(1, 4):

            for s in range(1, shots):

                place = "Experiments/Experiment1_2/ResultsExp1/" + base
                DataFrame = uploadData(place, samples, people, shots)



                if methodCL == 0:
                    propQ = DataFrame['AccQDAProp'].loc[
                                (DataFrame['Feature Set'] == f) & (DataFrame['# shots'] == s)].values * 100
                    propL = DataFrame['AccLDAPropQ'].loc[
                                (DataFrame['Feature Set'] == f) & (DataFrame['# shots'] == s)].values * 100
                    indQ = DataFrame['AccQDAInd'].loc[
                               (DataFrame['Feature Set'] == f) & (DataFrame['# shots'] == s)].values * 100
                    indL = DataFrame['AccLDAInd'].loc[
                               (DataFrame['Feature Set'] == f) & (DataFrame['# shots'] == s)].values * 100
                    liuL = DataFrame['AccLDALiu'].loc[
                               (DataFrame['Feature Set'] == f) & (DataFrame['# shots'] == s)].values * 100
                    liuQ = DataFrame['AccQDALiu'].loc[
                               (DataFrame['Feature Set'] == f) & (DataFrame['# shots'] == s)].values * 100
                    vidL = DataFrame['AccLDAVidovic'].loc[
                               (DataFrame['Feature Set'] == f) & (DataFrame['# shots'] == s)].values * 100
                    vidQ = DataFrame['AccQDAVidovic'].loc[
                               (DataFrame['Feature Set'] == f) & (DataFrame['# shots'] == s)].values * 100



                iL = np.median(indL)
                iQ = np.median(indQ)
                pL = np.median(propL)
                pQ = np.median(propQ)
                lL = np.median(liuL)
                lQ = np.median(liuQ)
                vL = np.median(vidL)
                vQ = np.median(vidQ)




                WilcoxonMethod = 'wilcox'
                alternativeMethod = 'greater'

                results.at['propL LDA' + base, idx] = round(pL - iL, 2)
                results.at['propL LDA (p)' + base, idx] = 1
                if pL > iL:
                    p = stats.wilcoxon(propL, indL, alternative=alternativeMethod, zero_method=WilcoxonMethod)[1]
                    if p < confidence:

                        results.at['propL LDA (p)' + base, idx] = p
                elif iL > pL:
                    p = stats.wilcoxon(indL, propL, alternative=alternativeMethod, zero_method=WilcoxonMethod)[1]
                    if p < confidence:

                        results.at['propL LDA (p)' + base, idx] = p
                        print(1)

                results.at['propL LiuL' + base, idx] = round(pL - lL, 2)
                results.at['propL LiuL (p)' + base, idx] = 1
                if pL > lL:
                    p = stats.wilcoxon(propL, liuL, alternative=alternativeMethod, zero_method=WilcoxonMethod)[1]
                    if p < confidence:

                        results.at['propL LiuL (p)' + base, idx] = p
                elif lL > pL:
                    p = stats.wilcoxon(liuL, propL, alternative=alternativeMethod, zero_method=WilcoxonMethod)[1]
                    if p < confidence:

                        results.at['propL LiuL (p)' + base, idx] = p
                        print(1)

                results.at['propL VidovicL' + base, idx] = round(pL - vL, 2)
                results.at['propL VidovicL (p)' + base, idx] = 1
                if pL > vL:
                    p = stats.wilcoxon(propL, vidL, alternative=alternativeMethod, zero_method=WilcoxonMethod)[1]
                    if p < confidence:

                        results.at['propL VidovicL (p)' + base, idx] = p
                elif vL > pL:
                    p = stats.wilcoxon(vidL, propL, alternative=alternativeMethod, zero_method=WilcoxonMethod)[1]
                    if p < confidence:

                        results.at['propL VidovicL (p)' + base, idx] = p
                        print(1)


                results.at['propQ QDA BL' + base, idx] = round(pQ - iQ, 2)
                results.at['propQ QDA BL (p)' + base, idx] = 1
                if pQ > iQ:
                    p = stats.wilcoxon(propQ, indQ, alternative=alternativeMethod, zero_method=WilcoxonMethod)[1]
                    if p < confidence:

                        results.at['propQ QDA BL (p)' + base, idx] = p

                elif iQ > pQ:
                    p = stats.wilcoxon(indQ, propQ, alternative=alternativeMethod, zero_method=WilcoxonMethod)[1]
                    if p < confidence:

                        results.at['propQ QDA BL (p)' + base, idx] = p
                        print(1)

                results.at['propQ LiuQ' + base, idx] = round(pQ - lQ, 2)
                results.at['propQ LiuQ (p)' + base, idx] = 1
                if pQ > lQ:
                    p = stats.wilcoxon(propQ, liuQ, alternative=alternativeMethod, zero_method=WilcoxonMethod)[1]
                    if p < confidence:

                        results.at['propQ LiuQ (p)' + base, idx] = p
                elif lQ > pQ:
                    p = stats.wilcoxon(liuQ, propQ, alternative=alternativeMethod, zero_method=WilcoxonMethod)[1]
                    if p < confidence:

                        results.at['propQ LiuQ (p)' + base, idx] = p
                        print(1)


                results.at['propQ Vidovic QDA' + base, idx] = round(pQ - vQ, 2)
                results.at['propQ Vidovic QDA (p)' + base, idx] = 1
                if pQ > vQ:
                    p = stats.wilcoxon(propQ, vidQ, alternative=alternativeMethod, zero_method=WilcoxonMethod)[1]
                    if p < confidence:

                        results.at['propQ Vidovic QDA (p)' + base, idx] = p
                elif vQ > pQ:
                    p = stats.wilcoxon(vidQ, propQ, alternative=alternativeMethod, zero_method=WilcoxonMethod)[1]
                    if p < confidence:

                        results.at['propQ Vidovic QDA (p)' + base, idx] = p

                idx += 1
        idxD += 12
    return results





def friedman_test(*args):
    """
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

    chi2 = ((12 * n) / float((k * (k + 1)))) * (
                (np.sum(r ** 2 for r in rankings_avg)) - ((k * (k + 1) ** 2) / float(4)))
    iman_davenport = ((n - 1) * chi2) / float((n * (k - 1) - chi2))

    p_value = 1 - stats.f.cdf(iman_davenport, k - 1, (k - 1) * (n - 1))

    return iman_davenport, p_value, rankings_avg, rankings_cmp


def holm_test(ranks, control=None):
    """
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


def AnalysisFriedman():
    dataFrame = pd.DataFrame()

    base = 'NinaPro5'
    samples = 4
    people = 10
    shots = 5
    place = "Experiments/Experiment1_2/ResultsExp1/" + base
    DataFrameN = uploadData(place, samples, people, shots)
    base = 'Cote'
    samples = 4
    people = 17
    shots = 5
    place = "Experiments/Experiment1_2/ResultsExp1/" + base
    DataFrameC = uploadData(place, samples, people, shots)
    base = 'EPN'
    samples = 25
    people = 30
    shots = 5
    place = "Experiments/Experiment1_2/ResultsExp1/" + base
    DataFrameE = uploadData(place, samples, people, shots)

    TotalDataframe = pd.concat([DataFrameN, DataFrameC, DataFrameE])

    dataFrame["propL"] = TotalDataframe['AccLDAPropQ'].values
    dataFrame["indL"] = TotalDataframe['AccLDAInd'].values
    dataFrame["liuL"] = TotalDataframe['AccLDALiu'].values
    dataFrame["vidL"] = TotalDataframe['AccLDAVidovic'].values

    dataFrame["propQ"] = TotalDataframe['AccQDAProp'].values
    dataFrame["indQ"] = TotalDataframe['AccQDAInd'].values
    dataFrame["liuQ"] = TotalDataframe['AccQDALiu'].values
    dataFrame["vidQ"] = TotalDataframe['AccQDAVidovic'].values

    data = np.asarray(dataFrame)
    num_datasets, num_methods = data.shape
    print("Methods:", num_methods, "People:", num_datasets)

    alpha = 0.05  # Set this to the desired alpha/signifance level

    stat, p = stats.friedmanchisquare(*data)

    reject = p <= alpha
    print("Should we reject H0 (i.e. is there a difference in the means) at the", (1 - alpha) * 100,
          "% confidence level?", reject)

    if not reject:
        print(
            "Exiting early. The rankings are only relevant if there was a difference in the means i.e. if we rejected h0 above")
    else:
        statistic, p_value, ranking, rank_cmp = friedman_test(*np.transpose(data))
        ranks = {key: rank_cmp[i] for i, key in enumerate(list(dataFrame.columns))}

        comparisons, z_values, p_values, adj_p_values, best = holm_test(ranks)

        adj_p_values = np.asarray(adj_p_values)

        for method, rank in ranks.items():
            print(method + ":", "%.2f" % rank)
        print('\nthe best method is: ', best)
        holm_scores = pd.DataFrame({"p": adj_p_values, "sig": adj_p_values < alpha}, index=comparisons)
        print(holm_scores)

    return


def AnalysisCote():
    bases = ['NinaPro5', 'Cote', 'EPN']
    confidence = 0.05
    results = pd.DataFrame()

    idxD = 0
    for base in bases:
        print('\n\n', base)
        if base == 'NinaPro5':
            samples = 4
            people = 10
            shots = 5
            f = 1
        elif base == 'Cote':
            samples = 4
            people = 17
            shots = 5
            f = 1
        elif base == 'EPN':
            samples = 25
            people = 30
            shots = 5
            f = 1
        idx = 0

        for s in range(1, shots):

            place = "Experiments/CotePyTorchImplementation/Cote_CWT_" + base + "/"
            cote = pd.read_csv(place + "Pytorch_results_" + str(s) + "_cycles.csv", header=None)
            c = []

            if base == 'NinaPro5':
                for i in range(people):
                    c.append(np.array(ast.literal_eval(cote.loc[0][i])).T[0].mean())
                c = np.array(c)
            elif base == 'EPN':
                for i in range(20):
                    c.append(ast.literal_eval(cote.loc[0][i]))
                c = np.mean(np.array(c), axis=0)
            elif base == 'Cote':
                for i in range(20):
                    c.append(ast.literal_eval(cote.loc[0][i]))
                    c.append(ast.literal_eval(cote.loc[1][i]))
                c = np.mean(np.array(c), axis=0)
            place = "Experiments/Experiment1_2/ResultsExp1/" + base
            DataFrame = uploadData(place, samples, people, shots)

            propQ = DataFrame['AccQDAProp'].loc[
                        (DataFrame['Feature Set'] == f) & (DataFrame['# shots'] == s)].values * 100
            propL = DataFrame['AccLDAPropQ'].loc[
                        (DataFrame['Feature Set'] == f) & (DataFrame['# shots'] == s)].values * 100



            pQ = np.median(propQ)
            co = np.median(c)



            WilcoxonMethod = 'wilcox'
            alternativeMethod = 'greater'

            results.at['AccQDAProp' + base, idx] = round(pQ, 2)
            results.at['AccCote' + base, idx] = round(co, 2)
            results.at['prop QDA' + base, idx] = round(pQ - co, 2)
            results.at['prop QDA (p)' + base, idx] = 1
            if pQ > co:
                p = stats.wilcoxon(propQ, c, alternative=alternativeMethod, zero_method=WilcoxonMethod)[1]
                if p < confidence:
                    results.at['prop QDA (p)' + base, idx] = p
            elif co > pQ:
                p = stats.wilcoxon(c, propQ, alternative=alternativeMethod, zero_method=WilcoxonMethod)[1]
                if p < confidence:
                    results.at['prop QDA (p)' + base, idx] = p

            idx += 1
        idxD += 2
    return results