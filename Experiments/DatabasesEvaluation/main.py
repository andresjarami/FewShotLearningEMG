import numpy as np
import sys
from sklearn import preprocessing
from sklearn.decomposition import PCA
import functions as F

## Input Variables

featureSet=int(sys.argv[1])
startPerson=int(sys.argv[2])
endPerson=int(sys.argv[3])
place=str(sys.argv[4])
typeDatabase=str(sys.argv[5])
NoPca = bool(int(sys.argv[6]))
printR = bool(int(sys.argv[7]))
nameFile=place+'_FeatureSet_'+sys.argv[1]+'_startPerson_'+sys.argv[2]+'_endPerson_'+sys.argv[3]+'.csv'

# featureSet = 3
# startPerson = 1
# endPerson = 1
# typeDatabase = 'Cote'
# NoPca = bool(1)
# printR = bool(1)
# nameFile = None
# place = 'Results/noPCA_EPN'
# nameFile = place + '_FeatureSet_' + str(featureSet) + '_startPerson_' + str(startPerson) + '_endPerson_' + str(
#     endPerson) + '.csv'

if NoPca:
    pca = False
    preProcessing = True
else:
    pca = True
    preProcessing = True
    pcaComp = 0.99

# Upload Data
dataMatrix, numberFeatures, CH, classes, people, peoplePriorK, peopleTrain, numberShots, combinationSet, allFeatures = F.uploadDatabases(
    typeDatabase, featureSet)

# Preprocessing
if preProcessing:
    dataMatrixPre = preprocessing.scale(dataMatrix[:, 0:allFeatures])
    dataMatrix = np.hstack((dataMatrixPre, dataMatrix[:, allFeatures:]))

# PCA
if pca:
    pca = PCA(n_components=pcaComp)
    dataMatrixPca = pca.fit_transform(dataMatrix[:, 0:allFeatures])
    dataMatrix = np.hstack((dataMatrixPca, dataMatrix[:, allFeatures:]))
    allFeatures = np.size(dataMatrixPca, axis=1)

### Evaluation

F.evaluation(dataMatrix, classes, peoplePriorK, featureSet, numberShots, combinationSet, nameFile, startPerson,
             endPerson, allFeatures, typeDatabase, NoPca, printR)
