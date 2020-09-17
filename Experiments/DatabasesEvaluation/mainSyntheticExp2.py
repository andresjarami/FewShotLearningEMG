import sys
import functions as F
import warnings

warnings.simplefilter('ignore')

## Input Variables

peopleSame = int(sys.argv[1])
place = str(sys.argv[2])
times = int(sys.argv[3])
nameFileR = place + 'resultsSynthetic_peopleSimilar_' + str(peopleSame) + 'time_' + str(times) + '.csv'

# peopleSame = 0
# place = 'ResultsExp2/'
# times = 1
# nameFileR = place + 'resultsSynthetic_peopleSimilar_' + str(peopleSame) + str(times) + '.csv'

seed = 1
samples = 1000
people = 21
Graph = False

generatedData = F.DataGenerator_TwoCL_TwoFeat(seed, samples, people, peopleSame, Graph)

shots = 50
peoplePK = 20
Features = 2
classes = 2
Graph = False
printValues = False

F.ResultsSyntheticData(generatedData, nameFileR, shots, peoplePK, samples, Features, classes, times, Graph, printValues)
