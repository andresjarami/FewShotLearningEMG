# %% Libraries
import sys
import functionsExp1 as functionsExp1

# %%## Input Variables

# featureSet = int(sys.argv[1])
# startPerson = int(sys.argv[2])
# endPerson = int(sys.argv[3])
# place = str(sys.argv[4])
# typeDatabase = str(sys.argv[5])
# printR = bool(int(sys.argv[6]))
# windowSize = str(sys.argv[7])
# nameFile = place + typeDatabase + '_FeatureSet_' + sys.argv[1] + '_startPerson_' + sys.argv[2] + '_endPerson_' + \
#            sys.argv[3] + '_windowSize_' + windowSize + '.csv'

# %%## Example of the parameters
# featureSet 1 (logVar) 2 (MAV,WL,ZC,SSC) or 3(LS,MFL,MSR,WAMP)
featureSet = 2
## people Nina5 (1-10) Cote(1-17) EPN(1-30) Capgmyo_dba(1-18) Capgmyo_dbc(1-10) Nina3 (1-9) Nina1 (1-27)
startPerson = 1
endPerson = 9
## typeDatabase: 'Nina5' or 'Cote' or 'EPN' or 'Capgmyo_dba' or 'Capgmyo_dbc'
typeDatabase = 'Nina3'
## printR: Print results (True or False)
printR = True
## windowSize: '260' for a window of 260ms with overlap of 235ms or '295' for a window of 295ms with overlap of 290ms
windowSize = '295'
## nameFile: Put in string the name of the file where you want to save the results or None to not save a file
# nameFile = None
nameFile = 'examples/esperanza4' + typeDatabase + '_FeatureSet_' + str(featureSet) + '_startPerson_' + str(
    startPerson) + '_endPerson_' + str(endPerson) + '_windowSize_' + windowSize + '.csv'

# %% Upload Data
dataMatrix, _, _, classes, peoplePriorK, _, numberShots, _, allFeatures, _ = functionsExp1.uploadDatabases(
    typeDatabase, featureSet, windowSize)

# %% Evaluation
functionsExp1.evaluation(dataMatrix, classes, peoplePriorK, featureSet, numberShots, nameFile, startPerson,
                         endPerson, allFeatures, typeDatabase, printR)
