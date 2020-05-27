
import numpy as np
import pandas as pd
import math

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from scipy import stats
from scipy.spatial import distance


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
def reTrainigProposed12(currentValues, preTrainedDataMatrix, classes, trainFeatures, trainLabels, people, per, maxCL,
                        features):
    beta = np.zeros(people)

    for i in range(people):
        if i != per:
            error = 1 - scoreModelLDA(trainFeatures, trainLabels,
                                      preTrainedDataMatrix[preTrainedDataMatrix['person'] == i].reset_index(
                                          drop=True), classes)
            if 0 < error < 1 / classes:
                beta[i] = math.log((1 - error) / error)
            elif error == 0:
                beta[i] = 1
        else:
            error = 1 - scoreModelLDA(trainFeatures, trainLabels, currentValues, classes)
            # print('errorTraining', error)
            if error == 0:
                beta[i] = 2
            elif error >= 1 / classes:
                beta[i] = 0
            else:
                beta[i] = math.log((1 - error) / error)
    if (len(beta[beta != 0]) < 2 and beta[per] != 0) or np.isin(beta, 0).all():
        beta[per] = 1
        Model = currentValues
    else:
        Model = pd.DataFrame(columns=['mean', 'cov', 'class', 'beta', '#model', 'covLDA'])
        idxBeta = np.argsort(-beta)
        idx = 0
        idxM = 0
        covLDA = np.zeros([features, features])
        while beta[idxBeta[idxM]] != 0 and idx <= maxCL - 1:
            if idxBeta[idxM] != per:
                for cl in range(classes):
                    auxFrame = preTrainedDataMatrix[preTrainedDataMatrix['person'] == idxBeta[idxM]].reset_index(
                        drop=True)
                    Model.at[idx, 'mean'] = auxFrame.loc[cl, 'mean']
                    Model.at[idx, 'cov'] = auxFrame.loc[cl, 'cov']
                    covLDA = covLDA + Model.loc[idx, 'cov']
                    Model.at[idx, 'class'] = cl + 1
                    Model.at[idx, 'beta'] = beta[idxBeta[idxM]]
                    Model.at[idx, '#model'] = idxM
                    idx += 1
                idx = idx - classes
                for cl in range(classes):
                    Model.at[idx, 'covLDA'] = covLDA / classes
                    idx += 1

            else:
                for cl in range(classes):
                    Model.at[idx, 'mean'] = currentValues.loc[cl, 'mean']
                    Model.at[idx, 'cov'] = currentValues.loc[cl, 'cov']
                    covLDA = covLDA + Model.loc[idx, 'cov']
                    Model.at[idx, 'class'] = cl + 1
                    Model.at[idx, 'beta'] = beta[idxBeta[idxM]]
                    Model.at[idx, '#model'] = idxM
                    idx += 1
                idx = idx - classes
                for cl in range(classes):
                    Model.at[idx, 'covLDA'] = covLDA / classes
                    idx += 1

            idxM += 1
            maxCL += 1

    return Model

def reTrainigProposed12Q(currentValues, preTrainedDataMatrix, classes, trainFeatures, trainLabels, people, per, maxCL,
                        features):
    beta = np.zeros(people)

    for i in range(people):
        if i != per:
            error = 1 - scoreModelQDA(trainFeatures, trainLabels,
                                      preTrainedDataMatrix[preTrainedDataMatrix['person'] == i].reset_index(
                                          drop=True), classes)
            if 0 < error < 1 / classes:
                beta[i] = math.log((1 - error) / error)
            elif error == 0:
                beta[i] = 1
        else:
            error = 1 - scoreModelQDA(trainFeatures, trainLabels, currentValues, classes)
            # print('errorTraining', error)
            if error == 0:
                beta[i] = 2
            elif error >= 1 / classes:
                beta[i] = 0
            else:
                beta[i] = math.log((1 - error) / error)
    if (len(beta[beta != 0]) < 2 and beta[per] != 0) or np.isin(beta, 0).all():
        beta[per] = 1
        Model = currentValues
    else:
        Model = pd.DataFrame(columns=['mean', 'cov', 'class', 'beta', '#model'])
        idxBeta = np.argsort(-beta)
        idx = 0
        idxM = 0
        # covLDA = np.zeros([features, features])
        while beta[idxBeta[idxM]] != 0 and idx <= maxCL - 1:
            if idxBeta[idxM] != per:
                for cl in range(classes):
                    auxFrame = preTrainedDataMatrix[preTrainedDataMatrix['person'] == idxBeta[idxM]].reset_index(
                        drop=True)
                    Model.at[idx, 'mean'] = auxFrame.loc[cl, 'mean']
                    Model.at[idx, 'cov'] = auxFrame.loc[cl, 'cov']
                    # covLDA = covLDA + Model.loc[idx, 'cov']
                    Model.at[idx, 'class'] = cl + 1
                    Model.at[idx, 'beta'] = beta[idxBeta[idxM]]
                    Model.at[idx, '#model'] = idxM
                    idx += 1
                # idx = idx - classes
                # for cl in range(classes):
                #     Model.at[idx, 'covLDA'] = covLDA / classes
                #     idx += 1

            else:
                for cl in range(classes):
                    Model.at[idx, 'mean'] = currentValues.loc[cl, 'mean']
                    Model.at[idx, 'cov'] = currentValues.loc[cl, 'cov']
                    # covLDA = covLDA + Model.loc[idx, 'cov']
                    Model.at[idx, 'class'] = cl + 1
                    Model.at[idx, 'beta'] = beta[idxBeta[idxM]]
                    Model.at[idx, '#model'] = idxM
                    idx += 1
                # idx = idx - classes
                # for cl in range(classes):
                #     Model.at[idx, 'covLDA'] = covLDA / classes
                #     idx += 1

            idxM += 1
            maxCL += 1

    return Model


def scoreModelLDA12(testFeatures, testLabels, peopleModels, classes):
    true = 0
    count = 0
    if int(peopleModels['mean'].count()) == classes:
        acc = scoreModelLDA(testFeatures, testLabels, peopleModels, classes)
    else:
        vote = np.zeros(classes)
        for i in range(0, np.size(testLabels)):
            for m in range(peopleModels['#model'].max() + 1):
                auxFrame = peopleModels[peopleModels['#model'] == m].reset_index(drop=True)
                idxCL = predictedModelLDA(testFeatures[i, :], auxFrame, classes, auxFrame.loc[0, 'covLDA']) - 1
                vote[int(idxCL)] = vote[int(idxCL)] + auxFrame.loc[0, 'beta']
            currentPredictor = np.argmax(vote) + 1
            if currentPredictor == testLabels[i]:
                true += 1
                count += 1
            else:
                count += 1

        acc = true / count
    return acc

def scoreModelQDA12(testFeatures, testLabels, peopleModels, classes):
    true = 0
    count = 0
    if int(peopleModels['mean'].count()) == classes:
        acc = scoreModelQDA(testFeatures, testLabels, peopleModels, classes)
    else:
        vote = np.zeros(classes)
        for i in range(0, np.size(testLabels)):
            for m in range(peopleModels['#model'].max() + 1):
                auxFrame = peopleModels[peopleModels['#model'] == m].reset_index(drop=True)
                idxCL = predictedModelQDA(testFeatures[i, :], auxFrame, classes) - 1
                vote[int(idxCL)] = vote[int(idxCL)] + auxFrame.loc[0, 'beta']
            currentPredictor = np.argmax(vote) + 1
            if currentPredictor == testLabels[i]:
                true += 1
                count += 1
            else:
                count += 1

        acc = true / count
    return acc


def reTrainigProposed10BB(currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels, step,
                          typeModel):
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
    return trainedModel, rClass / peopleClass, rClass.mean() / peopleClass


def reTrainigProposed11BB(currentValues, preTrainedDataMatrix, classes, allFeatures, trainFeatures, trainLabels, step,
                          typeModel):
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
        F1C = scoreModelLDA_ALL(trainFeatures, trainLabels, currentValues, classes, step)
    elif typeModel == 'QDA':
        F1C = scoreModelQDA_ALL(trainFeatures, trainLabels, currentValues, classes, step)

    rClass = np.zeros(classes)
    for cla in range(0, classes):
        r = rCalculatedProposed10BB_r(currentValues, MeansCovs['calculatedMean'].loc[cla],
                                      MeansCovs['calculatedCov'].loc[cla], cla, classes
                                      , trainFeatures, trainLabels, F1C[cla], step, typeModel)
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


# QDA
def QDA_Discriminant(x, cov_k, u_k):
    # PseudoQuadratic
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


def SamplesFew_ShotTrainedModels(model, samples, classes):
    xTrain = []
    yTrain = []
    for cl in range(0, classes):
        meanDistribution = model['mean'].loc[cl]
        covDistribution = model['cov'].loc[cl]
        xTrain.extend(np.random.multivariate_normal(meanDistribution, covDistribution, samples, check_valid='ignore'))
        yTrain.extend((cl + 1) * np.ones(samples))
    return xTrain, yTrain


#####################################################################

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
