import sys
import Experiments.Experiment2.functionsExp2 as functionsExp2


## Input Variables

peopleSame = int(sys.argv[1])
place = str(sys.argv[2])
times = int(sys.argv[3])
nameFileR = place + 'resultsSynthetic_peopleSimilar_' + str(peopleSame) + 'time_' + str(times) + '.csv'

################EXAMPLE
# peopleSame = 0
# place = 'ResultsExp2/'
# times = 25
# nameFileR = None

seed = 1
samples = 1000
people = 21
Graph = False

generatedData = functionsExp2.DataGenerator_TwoCL_TwoFeat(seed, samples, people, peopleSame, Graph)

shots = 50
peoplePK = 20
Features = 2
classes = 2
Graph = False
printValues = False

functionsExp2.ResultsSyntheticData(generatedData, nameFileR, shots, peoplePK, samples, Features, classes, times, Graph, printValues)
