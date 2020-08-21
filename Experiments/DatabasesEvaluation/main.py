
import sys
import functions as F
# import warnings
# warnings.filterwarnings("ignore")

## Input Variables

# featureSet=int(sys.argv[1])
# startPerson=int(sys.argv[2])
# endPerson=int(sys.argv[3])
# place=str(sys.argv[4])
# typeDatabase=str(sys.argv[5])
# pca = bool(int(sys.argv[6]))
# printR = bool(int(sys.argv[7]))
# hyper = bool(int(sys.argv[8]))
# eval = bool(int(sys.argv[9]))
# nameFile=place+'_FeatureSet_'+sys.argv[1]+'_startPerson_'+sys.argv[2]+'_endPerson_'+sys.argv[3]+'.csv'

featureSet = 3
startPerson = 3
endPerson = 3
typeDatabase = 'EPN'
pca = bool(1)
printR = bool(1)
hyper = bool(0)
eval = bool(1)
nameFile = 'prueba.csv'

# Upload Data
dataMatrix, numberFeatures, CH, classes, peoplePriorK, peopleTest, numberShots, combinationSet, allFeatures, \
labelsDataMatrix = F.uploadDatabases(typeDatabase, featureSet)

# Scale




# Hyper-Parameters tuning
if hyper:
    #Preprocessing
    dataMatrix = F.preprocessingData(dataMatrix, allFeatures, peoplePriorK, peopleTest, typeDatabase)
    # PCA
    dataMatrix, allFeatures = F.pcaData(dataMatrix, allFeatures)

    F.hyperParameterTuning(dataMatrix, labelsDataMatrix, featureSet, typeDatabase, allFeatures, peoplePriorK,
                           peopleTest)

# Evaluation
# dataMatrix = F.preprocessingData(dataMatrix, allFeatures, peoplePriorK, peopleTest, typeDatabase)
# dataMatrix, allFeatures = F.pcaData(dataMatrix, allFeatures)

if eval:
    F.evaluation(dataMatrix, classes, peoplePriorK, peopleTest, featureSet, numberShots, combinationSet, nameFile,
             startPerson, endPerson, allFeatures, typeDatabase, printR)
