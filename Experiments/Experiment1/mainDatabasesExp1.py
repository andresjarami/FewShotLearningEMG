import sys
import Experiments.Experiment1.functionsExp1 as functionsExp1

## Input Variables

# featureSet = int(sys.argv[1])
# startPerson = int(sys.argv[2])
# endPerson = int(sys.argv[3])
# place = str(sys.argv[4])
# typeDatabase = str(sys.argv[5])
# printR = bool(int(sys.argv[6]))
# nameFile = place + '_FeatureSet_' + sys.argv[1] + '_startPerson_' + sys.argv[2] + '_endPerson_' + sys.argv[3] + '.csv'

featureSet = 3
# for featureSet in [1, 3]:
startPerson = 1
endPerson = 30
typeDatabase = 'EPN'
printR = bool(1)
nameFile = 'results260/' + typeDatabase + '_FeatureSet_' + str(featureSet) + '_startPerson_' + str(
    startPerson) + '_endPerson_' + str(endPerson) + '.csv'

# Upload Data
# windowFileName: '260' for a window of 260ms with overlap of 235ms or '295' for a window of 295ms with overlap of 290ms
windowFileName = '260'
dataMatrix, _, _, classes, peoplePriorK, _, numberShots, _, allFeatures, _ = functionsExp1.uploadDatabases(
    typeDatabase, featureSet, windowFileName)

# Evaluation

functionsExp1.evaluation(dataMatrix, classes, peoplePriorK, featureSet, numberShots, nameFile, startPerson,
                         endPerson,
                         allFeatures, typeDatabase, printR)
