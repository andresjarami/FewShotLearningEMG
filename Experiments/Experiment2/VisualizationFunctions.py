import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Experiments.Experiment1.VisualizationFunctions as VF1
import random


def DataGenerator_TwoCL_TwoFeatEXAMPLE(seed=1, samples=50, people=3, peopleSame=1):
    peopleDiff = people - (peopleSame + 1)

    classes = 2
    Features = 2
    meansDistance = 2
    np.random.seed(seed)
    random.seed(seed)

    meanSet = np.zeros([classes, Features])
    meanSet[0, :] = np.array([0, 0])
    meanSet[1, :] = np.array([meansDistance, 0])

    DataFrame = pd.DataFrame(columns=['data', 'mean', 'cov', 'person', 'class'])
    idx = 0

    colors_list = list(['blue', 'red', 'deepskyblue', 'lightcoral', 'orange', 'green'])
    fig1, ax = plt.subplots(1, 1, figsize=(4.5, 3))

    for person in range(people):

        covSet = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])

        if person == people - 1:
            auxPoint = np.zeros(2)
            covSet = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])

        else:
            if person < peopleDiff:

                auxPoint = np.array([10, 0])
            else:
                auxPoint = np.zeros(2)

        for cl in range(classes):

            DataFrame.at[idx, 'data'] = np.random.multivariate_normal(meanSet[cl, :] + auxPoint,
                                                                      covSet[(cl + 1) * Features -
                                                                             Features:(cl + 1) * Features,
                                                                      0:Features], samples).T
            DataFrame.at[idx, 'mean'] = np.mean(DataFrame.loc[idx, 'data'], axis=1)
            DataFrame.at[idx, 'cov'] = np.cov(DataFrame.loc[idx, 'data'])

            DataFrame.at[idx, 'class'] = cl
            DataFrame.at[idx, 'person'] = person

            x1 = DataFrame.loc[idx, 'data'][0, :]
            x2 = DataFrame.loc[idx, 'data'][1, :]

            sizeM = 7

            if person == people - 1:
                color = colors_list[cl]
                label = 'clase ' + str(cl) + ': target user'
                if cl == 0:
                    markerCL = 'o'
                else:
                    markerCL = '^'
                ax.scatter(x1, x2, s=sizeM, color=color, marker=markerCL)
                VF1.confidence_ellipse(x1, x2, ax, edgecolor=color, label=label)
            else:
                if person < peopleDiff:
                    color = colors_list[cl + 4]
                    label = 'clase ' + str(cl) + ': different user'
                    lineStyle = '-.'
                    if cl == 0:
                        markerCL = 's'
                    else:
                        markerCL = 'p'
                else:
                    color = colors_list[cl + 2]

                    label = 'clase ' + str(cl) + ': similar user'
                    lineStyle = ':'
                    if cl == 0:
                        markerCL = '*'
                    else:
                        markerCL = 'x'

            if person != people - 1:
                if person == people - 2 or person == peopleDiff - 1:
                    ax.scatter(x1, x2, s=sizeM, color=color, marker=markerCL)
                    VF1.confidence_ellipse(x1, x2, ax, edgecolor=color, label=label, linestyle=lineStyle)
                else:
                    ax.scatter(x1, x2, s=sizeM, color=color, marker=markerCL)
                    VF1.confidence_ellipse(x1, x2, ax, edgecolor=color, linestyle=lineStyle)

            idx += 1

    plt.grid()
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.8), ncol=3, prop={'size': 6.5})
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    fig1.tight_layout(pad=0.1)
    plt.savefig("fig2.png", bbox_inches='tight', dpi=600)
    plt.show()
    return


def graphSyntheticDataALL(place):
    samples = 50
    for j in [0, 1, 3, 5, 10, 15, 20]:
        frame = pd.read_csv(place + 'Synthetic_peopleSimilar_' + str(j) + 'time_0' + '.csv')

        for i in range(1, 100):
            auxFrame = pd.read_csv(place + 'Synthetic_peopleSimilar_' + str(j) + 'time_' + str(i) + '.csv')
            frame = pd.concat([frame, auxFrame], ignore_index=True)
            if len(auxFrame) != samples:
                print('error' + ' 0 ' + str(i))
                print(len(auxFrame))
        frame.to_csv(place + 'Synthetic_peopleSimilar_' + str(j) + '.csv')

    fig, ax = plt.subplots(nrows=4, ncols=2, sharex='all', sharey='all', figsize=(4, 6))
    sizeM = 40

    numberShots = 13
    iSample = 3
    idx = 0
    for peopleSimilar in [0, 1, 3, 5]:
        resultsData = pd.read_csv(place + 'Synthetic_peopleSimilar_' + str(peopleSimilar) + '.csv')

        shot, AccLDAInd, AccQDAInd, AccLDAMulti, AccQDAMulti, AccLDAProp, AccQDAProp, AccLDALiu, AccQDALiu, AccLDAVidovic, AccQDAVidovic = SyntheticData(
            resultsData, numberShots, iSample)

        ax[idx, 0].scatter(shot + iSample, AccLDAInd,marker='x', label='Individual',  color='tab:orange',s=sizeM)
        #         ax[idx,0].plot(shot + iSample, AccLDAMulti, label='Multi-user', markersize=sizeM, color='tab:purple',
        #                  linestyle=(0, (3, 3, 1, 3, 1, 3)))
        ax[idx, 0].scatter(shot + iSample, AccLDALiu, marker='o', label='Liu', color='tab:green',s=sizeM)
        ax[idx, 0].scatter(shot + iSample, AccLDAVidovic,marker='^', label='Vidovic', color='tab:red',s=sizeM)
        ax[idx, 0].plot(shot + iSample, AccLDAProp, label='Our technique', color='tab:blue')
        ax[idx, 0].grid()
        ax[idx, 0].set_ylabel('accuracy')
        ax[idx, 0].xaxis.set_ticks([3, 5, 7, 10, 15])
        ax[idx, 0].yaxis.set_ticks(np.arange(50, 90, 10))

        ax[idx, 1].scatter(shot + iSample, AccQDAInd,marker='x', label='Individual',  color='tab:orange',s=sizeM)
        ax[idx, 1].scatter(shot + iSample, AccQDALiu, marker='o', label='Liu', color='tab:green',s=sizeM)
        ax[idx, 1].scatter(shot + iSample, AccQDAVidovic,marker='^', label='Vidovic', color='tab:red',s=sizeM)
        ax[idx, 1].plot(shot + iSample, AccQDAProp, label='Our technique', color='tab:blue')
        ax[idx, 1].grid()
        ax[idx, 1].xaxis.set_ticks([3, 5, 7, 10, 15])
        ax[idx, 1].yaxis.set_ticks(np.arange(50, 90, 10))

        idx += 1

    # ax[3, 1].legend(loc='lower center', bbox_to_anchor=(1.5, -1.5), ncol=4, prop={'size': 9})

    ax[0, 0].set_title('LDA')
    ax[0, 1].set_title('QDA')
    ax[3, 0].set_xlabel('samples')
    ax[3, 1].set_xlabel('samples')

    fig.tight_layout()
    plt.savefig("fig3.png", bbox_inches='tight', dpi=600)
    plt.show()


def SyntheticData(resultsData, numberShots, iSample):
    shot = np.arange(numberShots)
    AccLDAInd = np.zeros(numberShots)
    AccQDAInd = np.zeros(numberShots)

    AccLDAMulti = np.zeros(numberShots)
    AccQDAMulti = np.zeros(numberShots)

    AccLDAProp = np.zeros(numberShots)
    AccQDAProp = np.zeros(numberShots)
    AccLDALiu = np.zeros(numberShots)
    AccQDALiu = np.zeros(numberShots)
    AccLDAVidovic = np.zeros(numberShots)
    AccQDAVidovic = np.zeros(numberShots)

    for i in range(numberShots):
        AccLDAInd[i] = resultsData['AccLDAInd'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        AccQDAInd[i] = resultsData['AccQDAInd'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()

        AccLDAMulti[i] = resultsData['AccLDAMulti'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        AccQDAMulti[i] = resultsData['AccQDAMulti'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()

        AccLDAProp[i] = resultsData['AccLDAProp'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        AccQDAProp[i] = resultsData['AccQDAProp'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        AccLDALiu[i] = resultsData['AccLDALiu'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        AccQDALiu[i] = resultsData['AccQDALiu'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        AccLDAVidovic[i] = resultsData['AccLDAVidovic'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()
        AccQDAVidovic[i] = resultsData['AccQDAVidovic'][
            (resultsData['shots'] == i + iSample) & (resultsData['shots'] == i + iSample)].mean()

    return shot, AccLDAInd * 100, AccQDAInd * 100, AccLDAMulti * 100, AccQDAMulti * 100, AccLDAProp * 100, AccQDAProp * 100, AccLDALiu * 100, AccQDALiu * 100, AccLDAVidovic * 100, AccQDAVidovic * 100
