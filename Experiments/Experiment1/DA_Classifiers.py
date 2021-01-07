import numpy as np
import time
import math


#%% LDA Calssifier
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


def accuracyModelLDA(testFeatures, testLabels, model, classes):
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


#%% QDA Classifier
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


def accuracyModelQDA(testFeatures, testLabels, model, classes):
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
    return true / count, t / np.size(testLabels)
