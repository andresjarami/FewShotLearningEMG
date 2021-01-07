# %% Libraries
import sys
import Experiments.Experiment1.functionsExp1 as functionsExp1

# %%## Input Variables

featureSet = int(sys.argv[1])
startPerson = int(sys.argv[2])
endPerson = int(sys.argv[3])
place = str(sys.argv[4])
typeDatabase = str(sys.argv[5])
printR = bool(int(sys.argv[6]))
windowSize = str(sys.argv[7])
nameFile = place + typeDatabase + '_FeatureSet_' + sys.argv[1] + '_startPerson_' + sys.argv[2] + '_endPerson_' + \
           sys.argv[3] + '_windowSize_' + windowSize + '.csv'

## Example of the parameters
'''
featureSet = 1
startPerson = 1
endPerson = 1
## typeDatabase: 'Nina5' or 'Cote' or 'EPN'
typeDatabase = 'Cote'
## printR: Print results (True or False)
printR = bool(1)
## windowSize: '260' for a window of 260ms with overlap of 235ms or '295' for a window of 295ms with overlap of 290ms
windowSize = '295'
## nameFile: Put in string the name of the file where you want to save the results or None to not save a file
nameFile = None
# nameFile = 'results260/' + typeDatabase + '_FeatureSet_' + str(featureSet) + '_startPerson_' + str(
#     startPerson) + '_endPerson_' + str(endPerson) + '.csv'
'''
# %% Upload Data
dataMatrix, _, _, classes, peoplePriorK, _, numberShots, _, allFeatures, _ = functionsExp1.uploadDatabases(
    typeDatabase, featureSet, windowSize)

# %% Evaluation
functionsExp1.evaluation(dataMatrix, classes, peoplePriorK, featureSet, numberShots, nameFile, startPerson,
                         endPerson, allFeatures, typeDatabase, printR)
