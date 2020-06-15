import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import Experiments.DatabasesEvaluation.functions as F


# classes = int(sys.argv[1])
# Features = int(sys.argv[2])
# seed = int(sys.argv[3])
# samples = int(sys.argv[4])
# people = int(sys.argv[5])
# shots = int(sys.argv[6])
# Graph = bool(int(sys.argv[7]))
# covRandom = bool(int(sys.argv[8]))
# meanRandom = bool(int(sys.argv[9]))
# times = int(sys.argv[10])
# mix = bool(int(sys.argv[11]))
# per = int(sys.argv[12])
# t = int(sys.argv[13])
# place = str(sys.argv[14])

classes = 2
Features = 2
seed = 1
samples = 500
peopleG = 10
people = 1
shots = 30
Graph = True
covRandom = False
meanRandom = False
times = 5
mix = True
per = 0
t = 5
place = 'resultsMix/'

nameFileG = None
DataFrame = F.DataGenerator_TwoCL_TwoFeat(seed, samples, peopleG, Graph, covRandom,nameFileG,
                              mix)

# nameFileData = 'DATA' + 'C' + str(classes) + 'F' + str(Features) + 's' + str(seed) + 'S' + str(
#     samples) + 'P' + str(people) + 'sh' + str(shots) + 'cR' + str(covRandom) + 'mR' + str(meanRandom) + 't' + str(
#     times) + 'mix' + str(mix)
#
# DataFrame = pd.read_pickle(nameFileData + '.pkl')

iSample = 3
# nameFile = place + 'C' + str(classes) + 'F' + str(Features) + 's' + str(seed) + 'S' + str(samples) + 'P' + str(
#     people) + 'sh' + str(shots) + 'cR' + str(covRandom) + 'mR' + str(meanRandom) + 't' + str(times) + 'mix' + str(
#     mix) + '_' + str(per) + '_' + str(t) + '.csv'

nameFile = None

shot = np.arange(shots)

clfLDA = LDA()
yLDA = np.zeros(shots)

clfQDA = QDA()
yQDA = np.zeros(shots)

yPropQ = np.zeros(shots)
yPropL = np.zeros(shots)
# y12L = np.zeros(shots)
# y12Q = np.zeros(shots)
yLiu = np.zeros(shots)
yLiuQDA = np.zeros(shots)

rPropL = np.zeros(shots)
rPropQ = np.zeros(shots)
rLiu = np.ones(shots) * 0.5

resultsData = pd.DataFrame(
    columns=['yLDA', 'yQDA', 'y10BBQ', 'y10BBL', 'yLiu', 'yLiuQDA', 'r10BBQ', 'r10BBL', 'rLiu',
             'person', 'times', 'shots'])
idx = 0

for per in range(people):

    currentPerson = DataFrame[DataFrame['person'] == per].reset_index(drop=True)
    lenTest = int(samples / 2)
    preTrainedDataMatrix = DataFrame[DataFrame['person'] != per].reset_index(drop=True)
    preTrainedDataMatrix.at[:, 'class'] = preTrainedDataMatrix.loc[:, 'class'] + 1

    currentPersonTrain = pd.DataFrame(columns=['data', 'class'])
    currentValues = pd.DataFrame(columns=['data', 'mean', 'cov', 'class'])

    x_test = currentPerson.loc[0, 'data'][0:Features, 0:lenTest]
    y_test = np.ones(lenTest)
    currentPersonTrain.at[0, 'data'] = currentPerson.loc[0, 'data'][0:Features, lenTest:samples]
    currentPersonTrain.at[0, 'class'] = 1
    for cl in range(1, classes):
        currentPersonTrain.at[cl, 'data'] = currentPerson.loc[cl, 'data'][0:Features, lenTest:samples]
        currentPersonTrain.at[cl, 'class'] = cl + 1
        x_test = np.hstack((x_test, currentPerson.loc[cl, 'data'][0:Features, 0:lenTest]))
        y_test = np.hstack((y_test, np.ones(lenTest) * cl + 1))
    x_test = x_test.T

    for t in range(times):

        for i in range(shots):
            x_train = currentPersonTrain.loc[0, 'data'][:, t:i + t + iSample]
            # print(t, i + t + iSample)
            y_train = np.ones(i + iSample)
            currentValues.at[0, 'data'] = currentPersonTrain.loc[0, 'data'][:, t:i + t + iSample]
            currentValues.at[0, 'mean'] = np.mean(x_train, axis=1)
            currentValues.at[0, 'cov'] = np.cov(x_train, rowvar=True)
            currentValues.at[0, 'class'] = 1

            for cl in range(1, classes):
                x_train = np.hstack((x_train, currentPersonTrain.loc[cl, 'data'][:, t:i + t + iSample]))
                y_train = np.hstack((y_train, np.ones(i + iSample) * cl + 1))
                currentValues.at[cl, 'data'] = currentPersonTrain.loc[cl, 'data'][:, t:i + t + iSample]
                currentValues.at[cl, 'mean'] = np.mean(currentValues.loc[cl, 'data'], axis=1)
                currentValues.at[cl, 'cov'] = np.cov(currentValues.loc[cl, 'data'], rowvar=True)
                currentValues.at[cl, 'class'] = cl + 1
            x_train = x_train.T

            clfLDA.fit(x_train, y_train)
            # yLDA[i] += clfLDA.score(x_test, y_test)

            clfQDA.fit(x_train, y_train)
            # yQDA[i] += clfQDA.score(x_test, y_test)

            step = 1
            trainedModel10BBQDA, rClass10BBQDA, rClass10BBmeanQDA, tm = F.ProposedModel(currentValues,
                                                                                        preTrainedDataMatrix,
                                                                                        classes, Features, x_train,
                                                                                        y_train, step,
                                                                                        'QDA')
            trainedModel10BBLDA, rClass10BBLDA, rClass10BBmeanLDA, tm = F.ProposedModel(currentValues,
                                                                                        preTrainedDataMatrix,
                                                                                        classes, Features,
                                                                                        x_train,
                                                                                        y_train, step, 'LDA')

            trainedModelLiu = F.LiuModel(currentValues, preTrainedDataMatrix, classes, Features)

            # Model12L = CF.reTrainigProposed12(currentValues, preTrainedDataMatrix, classes, x_train, y_train, people,
            #                                   per, maxCL, Features)
            #
            # Model12Q = CF.reTrainigProposed12Q(currentValues, preTrainedDataMatrix, classes, x_train, y_train, people,
            #                                    per, maxCL, Features)

            # y10BBQ[i] += CF.scoreModelQDA(x_test, y_test, trainedModel10BBQDA, classes)
            # y10BBL[i] += CF.scoreModelLDA(x_test, y_test, trainedModel10BBLDA, classes)
            # yLiu[i] += CF.scoreModelLDA(x_test, y_test, trainedModelLiu, classes)
            # yLiuQDA[i] += CF.scoreModelQDA(x_test, y_test, trainedModelLiu, classes)

            # y12L[i] += CF.scoreModelLDA12(x_test, y_test, Model12L, classes)
            # y12Q[i] += CF.scoreModelQDA12(x_test, y_test, Model12Q, classes)

            # print('person', per, 'time', t, 'shot', i, 'LDA', yLDA[i], y10BBL[i], y12L[i], 'QDA', yQDA[i], y10BBQ[i],
            #       y12Q[i])
            # print('LDA', yLDA[i], y10BBL[i], 'QDA', yQDA[i], y10BBQ[i])

            # r10BBQ[i] += rClass10BBmeanQDA
            # r10BBL[i] += rClass10BBmeanLDA

            resultsData.at[idx, 'yLDA'] = clfLDA.score(x_test, y_test)
            resultsData.at[idx, 'yQDA'] = clfQDA.score(x_test, y_test)
            resultsData.at[idx, 'yPropL'] = F.scoreModelLDA(x_test, y_test, trainedModel10BBLDA, classes)
            resultsData.at[idx, 'yPropQ'] = F.scoreModelQDA(x_test, y_test, trainedModel10BBQDA, classes)
            resultsData.at[idx, 'yLiu'] = F.scoreModelLDA(x_test, y_test, trainedModelLiu, classes)
            resultsData.at[idx, 'yLiuQ'] = F.scoreModelQDA(x_test, y_test, trainedModelLiu, classes)

            resultsData.at[idx, 'rPropQ'] = rClass10BBmeanQDA
            resultsData.at[idx, 'rPropL'] = rClass10BBmeanLDA
            resultsData.at[idx, 'rLiu'] = 0.5
            resultsData.at[idx, 'person'] = per
            resultsData.at[idx, 'times'] = t
            resultsData.at[idx, 'shots'] = i + iSample

            if Graph:
                yLDA[i] += resultsData.loc[idx, 'yLDA']
                yQDA[i] += resultsData.loc[idx, 'yQDA']
                yPropL[i] += resultsData.loc[idx, 'yPropL']
                yPropQ[i] += resultsData.loc[idx, 'yPropQ']
                yLiu[i] += resultsData.loc[idx, 'yLiu']
                yLiuQDA[i] += resultsData.loc[idx, 'yLiuQ']
                rPropL[i] += resultsData.loc[idx, 'rPropL']
                rPropQ[i] += resultsData.loc[idx, 'rPropQ']

            # print(resultsData.loc[idx])
            if nameFile is not None:
                resultsData.to_csv(nameFile)
            idx += 1

            # print(per, t, i)

if Graph:
    fig2, ((ax1, ax2)) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6))

    ax1.plot(shot + iSample, yPropL / (times * people), label='PropoLDA',color='green')
    # ax1.plot(shot + iSample, resultsData['y12L'].mean(), label='12L')
    ax1.plot(shot + iSample, yLiu / (times * people), label='Liu',color='blue')
    ax1.plot(shot + iSample, yLDA / (times * people), label='BL LDA')

    ax1.set_title('ACC of models vs Samples (LDA)')
    ax1.grid()
    ax1.legend(loc='lower right', prop={'size': 8})

    ax1.set_ylabel('Acc')

    ax2.plot(shot + iSample, 1-(rPropL / (times * people)), label='W PropoLDA',color='green')
    ax2.plot(shot + iSample, rLiu, label='W Liu',color='blue')
    ax2.set_title('The weight of the current distribution (1-r) vs Shots')
    ax2.grid()
    ax2.legend(loc='lower right', prop={'size': 8})
    ax2.set_xlabel('Shots')
    ax2.set_ylabel('W')

    fig3, ((ax1, ax2)) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6))

    ax1.plot(shot + iSample, yPropQ / (times * people), label='PropoQDA',color='green')
    # ax1.plot(shot + iSample, resultsData['y12Q'].mean(), label='12Q')
    ax1.plot(shot + iSample, yLiuQDA / (times * people), label='Liu QDA',color='blue')
    ax1.plot(shot + iSample, yQDA / (times * people), label='BL QDA')

    ax1.set_title('Accuracy vs Samples')
    ax1.grid()
    ax1.legend(loc='lower right', prop={'size': 8})

    ax1.set_ylabel('Accuracy')

    ax2.plot(shot + iSample, 1-(rPropQ / (times * people)), label='wPropQDA',color='green')
    ax2.plot(shot + iSample, rLiu, label='W LiuQDA',color='blue')
    ax2.set_title('Weight vs Shots')
    ax2.grid()
    ax2.legend(loc='lower right', prop={'size': 8})
    ax2.set_xlabel('Shots')
    ax2.set_ylabel('W')

    plt.show()

# # pickle.dump(fig1, open(fig1Label, 'wb'))
# # pickle.dump(fig2, open(fig2Label, 'wb'))
# plt.show()
#
# with open(nameFile + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump([fig1, fig2, fig3, yLDA, y10BBL, yLiu, y12L, yQDA, y10BBQ, yLiuQDA, y12Q], f)

# Getting back the objects:
# with open(nameFile+'.pkl', 'rb') as f:
#     fig1, fig2, fig3, yLDA,y10BBL,yLiu,y12L,yQDA,y10BBQ,yLiuQDA,y12Q= pickle.load(f)
# ax_master = fig2.axes[0]
# for ax in fig2.axes:
#     if ax is not ax_master:
#         ax_master.get_shared_y_axes().join(ax_master, ax)
#
# plt.show()

# pickle.dump(fig1, file('fig1'+ nameFile + '.pkl', 'wb'))
#
#
#
# fig2 = pickle.load(file('fig1.pkl','rb'))
# ax_master = fig2.axes[0]
# for ax in fig2.axes:
#     if ax is not ax_master:
#         ax_master.get_shared_y_axes().join(ax_master, ax)
#
# plt.show()
