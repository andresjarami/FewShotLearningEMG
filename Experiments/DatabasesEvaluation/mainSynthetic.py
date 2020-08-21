import sys
import functions as F
import warnings

warnings.simplefilter('ignore')

## Input Variables

# peopleSame = int(sys.argv[1])
# covRandom = bool(int(sys.argv[2]))
# place = str(sys.argv[3])
# times = int(sys.argv[4])
# nameFileR = place + 'resultsSynthetic_peopleSame_' + str(peopleSame) + 'covRand' + str(covRandom) + str(times) + '.csv'

peopleSame = 1
covRandom = False
place='ResultsSynthetic/'
times = 12
nameFileR = place+'prueba.csv'


seed = 1
samples = 100
people = 21
Graph = False
nameFile = None
mixDistribution = True
sameDistribution = False

DataFrameSame = F.DataGenerator_TwoCL_TwoFeat(seed, samples, people, peopleSame, Graph, covRandom, nameFile,
                                              mixDistribution, sameDistribution)

shots = 10
peopleEvaluated = 21
peoplePK=20
Features = 2
classes = 2
Graph = True
printValues = False

F.ResultsSyntheticData(DataFrameSame, nameFileR, shots, peoplePK, peopleEvaluated, samples, Features, classes, times,
                       Graph, printValues,peopleSame)
