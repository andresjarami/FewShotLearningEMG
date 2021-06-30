# %% Libraries
import scipy.io
import numpy as np
import scipy.linalg
import csv
import time
from statsmodels.tsa.ar_model import AutoReg


# %% Features

def MAVch(EMG, ch):
    EMG = abs(EMG)
    mavVector = EMG.mean(axis=0)
    return mavVector


def WLch(EMG, ch):
    wlVector = np.zeros([1, ch])
    for i in range(ch):
        x_f = EMG[1:np.size(EMG, 0), i]
        x = EMG[0:np.size(EMG, 0) - 1, i]
        wlVector[0, i] = np.sum(abs(x_f - x))
    return wlVector


def ZCch(EMG, ch):
    zcVector = np.zeros([1, ch])
    for i in range(ch):
        zcVector[0, i] = np.size(np.where(np.diff(np.sign(EMG[:, i]))), 1)
    return zcVector


def SSCch(EMG, ch):
    sscVector = np.zeros([1, ch])
    for i in range(ch):
        x = EMG[1:np.size(EMG, 0) - 1, i]
        x_b = EMG[0:np.size(EMG, 0) - 2, i]
        x_f = EMG[2:np.size(EMG, 0), i]
        sscVector[0, i] = np.sum(abs((x - x_b) * (x - x_f)))
    return sscVector


def Lscalech(EMG, ch):
    LscaleVector = np.zeros([1, ch])
    for i in range(ch):
        lengthAux = np.size(EMG[:, i], 0)
        Matrix = np.sort(np.transpose(EMG[:, i]))
        aux = (1 / lengthAux) * np.sum((np.arange(1, lengthAux, 1) / (lengthAux - 1)) * Matrix[1:lengthAux + 1])
        aux3 = np.array([[aux], [np.mean(EMG[:, i])]])
        aux5 = np.array([[2], [-1]])
        LscaleVector[0, i] = np.sum(aux5 * aux3)
    return LscaleVector


def MFLch(EMG, ch):
    mflVector = np.zeros([1, ch])
    for i in range(ch):
        x_f = EMG[1:np.size(EMG, 0), i]
        x = EMG[0:np.size(EMG, 0) - 1, i]
        try:
            mflVector[0, i] = np.log10(np.sqrt(np.sum((x_f - x) ** 2)))
        except:
            mflVector[0, i] = 0
    return mflVector


def MSRch(EMG, ch):
    msrVector = np.zeros([1, ch])
    for i in range(ch):
        msrVector[0, i] = np.sum(np.sqrt(abs(EMG[:, i]))) / np.size(EMG[:, i], 0)
    return msrVector


def WAMPch(EMG, ch):
    wampVector = np.zeros([1, ch])
    for i in range(ch):
        x_f = EMG[1:np.size(EMG, 0), i]
        x = EMG[0:np.size(EMG, 0) - 1, i]
        wampVector[0, i] = np.sum((np.sign(x - x_f) + 1) / 2)
    return wampVector


def logVARch(EMG, ch):
    logVarVector = np.zeros([1, ch])
    for i in range(ch):
        try:
            logVarVector[0, i] = np.log(np.sum((EMG[:, i] - np.mean(EMG[:, i])) ** 2) / (np.size(EMG[:, i], 0) - 1))
        except:
            logVarVector[0, i] = 0
    return logVarVector


# %% Segmentation, and Feature Extraction
# for database in ['Nina5', 'Cote', 'EPN', 'Capgmyo_dba','Capgmyo_dbc','Nina3','Nina1']:
for database in ['Nina3']:
    # Our interface: window=295 and overlap=290
    # Cote: window=260 and overlap=235
    # window = [260,295,100]
    # overlap = [235,290,50]
    for window in [295]:
        if window == 260:
            overlap = 235
        elif window == 295:
            overlap = 290
        elif window == 100:
            overlap = 50
        elif window == 280:
            overlap = 50
        t1 = []
        t2 = []
        t3 = []

        logvarMatrix = []
        mavMatrix = []
        wlMatrix = []
        zcMatrix = []
        sscMatrix = []
        lscaleMatrix = []
        mflMatrix = []
        msrMatrix = []
        wampMatrix = []

        windowFileName = str(window)

        if database == 'Nina1':
            np.seterr(divide='raise')
            sampleRate = 100
            rpt = 10
            ch = 10
            classes = 12
            people = 27

            windowSamples = int(window * sampleRate / 1000)
            incrmentSamples = windowSamples - int(overlap * sampleRate / 1000)
            if incrmentSamples == 0:
                incrmentSamples = 1

            for person in range(1, people + 1):
                aux = scipy.io.loadmat('../Databases/ninaDB1/s' + str(person) + '/S' + str(person) + '_A1_E1.mat')
                auxEMG = aux['emg']
                auxRestimulus = aux['restimulus']

                stack = 0
                rp = 1
                auxIdx = 0
                for i in range(np.size(auxRestimulus)):
                    if auxRestimulus[i] != 0 and stack == 0:
                        aux1 = i
                        stack = 1
                        cl = int(auxRestimulus[i])

                    elif auxRestimulus[i] == 0 and stack == 1:
                        aux2 = i
                        stack = 0
                        wi = aux1
                        segments = int((aux2 - aux1 - windowSamples) / incrmentSamples + 1)
                        for w in range(segments):
                            wf = wi + windowSamples

                            t = time.time()
                            logvarMatrix.append(
                                np.hstack((logVARch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            t1.append(time.time() - t)
                            t = time.time()
                            mavMatrix.append(np.hstack((MAVch(auxEMG[wi:wf], ch), np.array([person, cl, rp]))))
                            wlMatrix.append(np.hstack((WLch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            zcMatrix.append(np.hstack((ZCch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            sscMatrix.append(np.hstack((SSCch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            t2.append(time.time() - t)
                            t = time.time()
                            lscaleMatrix.append(
                                np.hstack((Lscalech(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            mflMatrix.append(np.hstack((MFLch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            msrMatrix.append(np.hstack((MSRch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            wampMatrix.append(np.hstack((WAMPch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            t3.append(time.time() - t)

                            wi += incrmentSamples

                        rp = rp + 1
                        if rp == 11:
                            rp = 1

            timesFeatures = np.vstack((t1, t2, t3))
            auxName = 'timesFeatures' + windowFileName
            myFile = database + '/' + auxName + '.csv'
            np.savetxt(myFile, timesFeatures, delimiter=',')

            auxName = 'mavMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(mavMatrix)
            auxName = 'wlMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(wlMatrix)
            auxName = 'zcMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(zcMatrix)
            auxName = 'sscMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(sscMatrix)
            auxName = 'lscaleMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(lscaleMatrix)
            auxName = 'mflMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(mflMatrix)
            auxName = 'msrMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(msrMatrix)
            auxName = 'wampMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(wampMatrix)
            auxName = 'logvarMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(logvarMatrix)

        elif database == 'Nina3':
            np.seterr(divide='raise')
            sampleRate = 2000
            ## NINA PRO 3 DATABASE
            rpt = 6
            ch = 12
            classes = 17

            windowSamples = int(window * sampleRate / 1000)
            incrmentSamples = windowSamples - int(overlap * sampleRate / 1000)
            per = 0
            people = 11
            for person in range(1, people + 1):
                if person != 6 and person != 7:
                    per += 1
                    print(person, per)
                    aux = scipy.io.loadmat(
                        '../Databases/ninaDB3/s' + str(person) + '_0/DB3_s' + str(person) + '/S' + str(
                            person) + '_E1_A1.mat')
                    auxEMG = aux['emg']
                    auxRestimulus = aux['restimulus']

                    stack = 0
                    rp = 1
                    # stackR = 0
                    # rpR = 0
                    auxIdx = 0
                    for i in range(np.size(auxRestimulus)):
                        if auxRestimulus[i] != 0 and stack == 0:
                            aux1 = i
                            stack = 1
                            cl = int(auxRestimulus[i])

                        elif auxRestimulus[i] == 0 and stack == 1:
                            aux2 = i
                            stack = 0
                            wi = aux1
                            segments = int((aux2 - aux1 - windowSamples) / incrmentSamples + 1)
                            for w in range(segments):
                                wf = wi + windowSamples

                                t = time.time()
                                logvarMatrix.append(
                                    np.hstack((logVARch(auxEMG[wi:wf], ch)[0], np.array([per, cl, rp]))))
                                t1.append(time.time() - t)
                                t = time.time()
                                mavMatrix.append(np.hstack((MAVch(auxEMG[wi:wf], ch), np.array([per, cl, rp]))))
                                wlMatrix.append(np.hstack((WLch(auxEMG[wi:wf], ch)[0], np.array([per, cl, rp]))))
                                zcMatrix.append(np.hstack((ZCch(auxEMG[wi:wf], ch)[0], np.array([per, cl, rp]))))
                                sscMatrix.append(np.hstack((SSCch(auxEMG[wi:wf], ch)[0], np.array([per, cl, rp]))))
                                t2.append(time.time() - t)
                                t = time.time()
                                lscaleMatrix.append(
                                    np.hstack((Lscalech(auxEMG[wi:wf], ch)[0], np.array([per, cl, rp]))))
                                mflMatrix.append(np.hstack((MFLch(auxEMG[wi:wf], ch)[0], np.array([per, cl, rp]))))
                                msrMatrix.append(np.hstack((MSRch(auxEMG[wi:wf], ch)[0], np.array([per, cl, rp]))))
                                wampMatrix.append(np.hstack((WAMPch(auxEMG[wi:wf], ch)[0], np.array([per, cl, rp]))))
                                t3.append(time.time() - t)

                                wi += incrmentSamples

                            rp = rp + 1
                            if rp == 7:
                                rp = 1

                        if rpR <= rpt:
                            if auxRestimulus[i] == 0 and stackR == 0:
                                aux1R = i
                                stackR = 1
                                clR = 18

                            elif auxRestimulus[i] != 0 and stackR == 1:
                                aux2R = i
                                stackR = 0
                                wiR = aux1R
                                if rpR != 0:
                                    segments = int(((aux2R - aux1R) - windowSamples) / (incrmentSamples) + 1)
                                    for w in range(segments):
                                        wfR = wiR + windowSamples

                                        t = time.time()
                                        logvarMatrix.append(
                                            np.hstack((logVARch(auxEMG[wiR:wfR], ch)[0], np.array([per, clR, rpR]))))
                                        t1.append(time.time() - t)
                                        t = time.time()
                                        mavMatrix.append(
                                            np.hstack((MAVch(auxEMG[wiR:wfR], ch), np.array([per, clR, rpR]))))
                                        wlMatrix.append(
                                            np.hstack((WLch(auxEMG[wiR:wfR], ch)[0], np.array([per, clR, rpR]))))
                                        zcMatrix.append(
                                            np.hstack((ZCch(auxEMG[wiR:wfR], ch)[0], np.array([per, clR, rpR]))))
                                        sscMatrix.append(
                                            np.hstack((SSCch(auxEMG[wiR:wfR], ch)[0], np.array([per, clR, rpR]))))
                                        t2.append(time.time() - t)
                                        t = time.time()
                                        lscaleMatrix.append(
                                            np.hstack((Lscalech(auxEMG[wiR:wfR], ch)[0], np.array([per, clR, rpR]))))
                                        mflMatrix.append(
                                            np.hstack((MFLch(auxEMG[wiR:wfR], ch)[0], np.array([per, clR, rpR]))))
                                        msrMatrix.append(
                                            np.hstack((MSRch(auxEMG[wiR:wfR], ch)[0], np.array([per, clR, rpR]))))
                                        wampMatrix.append(
                                            np.hstack((WAMPch(auxEMG[wiR:wfR], ch)[0], np.array([per, clR, rpR]))))
                                        t3.append(time.time() - t)

                                        wiR += incrmentSamples
                                rpR += 1



            timesFeatures = np.vstack((t1, t2, t3))
            auxName = 'timesFeatures' + windowFileName
            myFile = database + '/' + auxName + '.csv'
            np.savetxt(myFile, timesFeatures, delimiter=',')

            auxName = 'mavMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(mavMatrix)
            auxName = 'wlMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(wlMatrix)
            auxName = 'zcMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(zcMatrix)
            auxName = 'sscMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(sscMatrix)
            auxName = 'lscaleMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(lscaleMatrix)
            auxName = 'mflMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(mflMatrix)
            auxName = 'msrMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(msrMatrix)
            auxName = 'wampMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(wampMatrix)
            auxName = 'logvarMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(logvarMatrix)

        elif database == 'Nina5':
            sampleRate = 200
            ## NINA PRO 5 DATABASE
            rpt = 6
            ch = 16
            classes = 18
            people = 10

            windowSamples = int(window * sampleRate / 1000)
            incrmentSamples = windowSamples - int(overlap * sampleRate / 1000)

            for person in range(1, people + 1):
                print(person)
                aux = scipy.io.loadmat('../Databases/ninaDB5/s' + str(person) + '/S' + str(person) + '_E2_A1.mat')
                auxEMG = aux['emg']
                auxRestimulus = aux['restimulus']

                stack = 0
                rp = 1
                stackR = 0
                rpR = 0
                auxIdx = 0
                for i in range(np.size(auxRestimulus)):
                    if auxRestimulus[i] != 0 and stack == 0:
                        aux1 = i
                        stack = 1
                        cl = int(auxRestimulus[i])

                    elif auxRestimulus[i] == 0 and stack == 1:
                        aux2 = i
                        stack = 0
                        wi = aux1
                        segments = int((aux2 - aux1 - windowSamples) / incrmentSamples + 1)
                        for w in range(segments):
                            wf = wi + windowSamples

                            t = time.time()
                            logvarMatrix.append(np.hstack((logVARch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            t1.append(time.time() - t)
                            t = time.time()
                            mavMatrix.append(np.hstack((MAVch(auxEMG[wi:wf], ch), np.array([person, cl, rp]))))
                            wlMatrix.append(np.hstack((WLch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            zcMatrix.append(np.hstack((ZCch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            sscMatrix.append(np.hstack((SSCch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            t2.append(time.time() - t)
                            t = time.time()
                            lscaleMatrix.append(np.hstack((Lscalech(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            mflMatrix.append(np.hstack((MFLch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            msrMatrix.append(np.hstack((MSRch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            wampMatrix.append(np.hstack((WAMPch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            t3.append(time.time() - t)

                            # LS, MFL, MSR, WAMP, ZC, RMS, IAV, DASDV, and VAR

                            wi += incrmentSamples

                        rp = rp + 1
                        if rp == 7:
                            rp = 1

                    if rpR <= rpt:
                        if auxRestimulus[i] == 0 and stackR == 0:
                            aux1R = i
                            stackR = 1
                            clR = 18

                        elif auxRestimulus[i] != 0 and stackR == 1:
                            aux2R = i
                            stackR = 0
                            wiR = aux1R
                            if rpR != 0:
                                segments = int(((aux2R - aux1R) - windowSamples) / (incrmentSamples) + 1)
                                for w in range(segments):
                                    wfR = wiR + windowSamples

                                    t = time.time()
                                    logvarMatrix.append(
                                        np.hstack((logVARch(auxEMG[wiR:wfR], ch)[0], np.array([person, clR, rpR]))))
                                    t1.append(time.time() - t)
                                    t = time.time()
                                    mavMatrix.append(
                                        np.hstack((MAVch(auxEMG[wiR:wfR], ch), np.array([person, clR, rpR]))))
                                    wlMatrix.append(
                                        np.hstack((WLch(auxEMG[wiR:wfR], ch)[0], np.array([person, clR, rpR]))))
                                    zcMatrix.append(
                                        np.hstack((ZCch(auxEMG[wiR:wfR], ch)[0], np.array([person, clR, rpR]))))
                                    sscMatrix.append(
                                        np.hstack((SSCch(auxEMG[wiR:wfR], ch)[0], np.array([person, clR, rpR]))))
                                    t2.append(time.time() - t)
                                    t = time.time()
                                    lscaleMatrix.append(
                                        np.hstack((Lscalech(auxEMG[wiR:wfR], ch)[0], np.array([person, clR, rpR]))))
                                    mflMatrix.append(
                                        np.hstack((MFLch(auxEMG[wiR:wfR], ch)[0], np.array([person, clR, rpR]))))
                                    msrMatrix.append(
                                        np.hstack((MSRch(auxEMG[wiR:wfR], ch)[0], np.array([person, clR, rpR]))))
                                    wampMatrix.append(
                                        np.hstack((WAMPch(auxEMG[wiR:wfR], ch)[0], np.array([person, clR, rpR]))))
                                    t3.append(time.time() - t)

                                    wiR += incrmentSamples
                            rpR += 1

            timesFeatures = np.vstack((t1, t2, t3))
            auxName = 'timesFeatures' + windowFileName
            myFile = database + '/' + auxName + '.csv'
            np.savetxt(myFile, timesFeatures, delimiter=',')

            auxName = 'mavMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(mavMatrix)
            auxName = 'wlMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(wlMatrix)
            auxName = 'zcMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(zcMatrix)
            auxName = 'sscMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(sscMatrix)
            auxName = 'lscaleMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(lscaleMatrix)
            auxName = 'mflMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(mflMatrix)
            auxName = 'msrMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(msrMatrix)
            auxName = 'wampMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(wampMatrix)
            auxName = 'logvarMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(logvarMatrix)

            print('Finished')

        elif database == 'Cote':
            sampleRate = 200
            # COTE ALLARD DATABASE
            ch = 8
            classes = 7
            peopleFemalePT = 7
            peopleMalePT = 12
            peopleFemaleE = 2
            peopleMaleE = 15
            types = 2
            genders = 2
            filesPerFolder = 28

            windowSamples = int(window * sampleRate / 1000)
            incrmentSamples = windowSamples - int(overlap * sampleRate / 1000)


            def emgMatrix(ty, gender, person, carpet, number):
                myarray = np.fromfile(
                    '../Databases/MyoArmbandDataset-master/' + ty + '/' + gender + str(
                        person) + '/' + carpet + '/classe_' + str(
                        number) + '.dat', dtype=np.int16)
                myarray = np.array(myarray, dtype=np.float32)
                emg = np.reshape(myarray, (int(len(myarray) / 8), 8))
                return emg


            per = 0
            for tyi in range(0, types):

                for genderi in range(0, genders):
                    if tyi == 0 and genderi == 0:
                        ty = 'PreTrainingDataset'
                        gender = 'Female'
                        carpets = np.array(['training0'])
                        people = peopleFemalePT
                    elif tyi == 0 and genderi == 1:
                        ty = 'PreTrainingDataset'
                        gender = 'Male'
                        carpets = np.array(['training0'])
                        people = peopleMalePT
                    elif tyi == 1 and genderi == 0:
                        ty = 'EvaluationDataset'
                        gender = 'Female'
                        carpets = np.array(['training0', 'Test0', 'Test1'])
                        people = peopleFemaleE
                    elif tyi == 1 and genderi == 1:
                        ty = 'EvaluationDataset'
                        gender = 'Male'
                        carpets = np.array(['training0', 'Test0', 'Test1'])
                        people = peopleMaleE

                    for person in range(people):
                        print(tyi, genderi, person)

                        rp = 1
                        for carpet in carpets:
                            if carpet == 'training0':
                                carp = 1
                            else:
                                carp = 2

                            for number in range(filesPerFolder):
                                cl = number
                                while cl > 6:
                                    cl = cl - 7

                                auxEMG = emgMatrix(ty, gender, person, carpet, number)

                                wi = 0

                                segments = int((len(auxEMG) - windowSamples) / incrmentSamples + 1)
                                for w in range(segments):
                                    wf = wi + windowSamples

                                    t = time.time()
                                    logvarMatrix.append(
                                        np.hstack((logVARch(auxEMG[wi:wf], ch)[0], np.array([tyi, per, carp, cl, rp]))))
                                    t1.append(time.time() - t)
                                    t = time.time()
                                    mavMatrix.append(
                                        np.hstack((MAVch(auxEMG[wi:wf], ch), np.array([tyi, per, carp, cl, rp]))))
                                    wlMatrix.append(
                                        np.hstack((WLch(auxEMG[wi:wf], ch)[0], np.array([tyi, per, carp, cl, rp]))))
                                    zcMatrix.append(
                                        np.hstack((ZCch(auxEMG[wi:wf], ch)[0], np.array([tyi, per, carp, cl, rp]))))
                                    sscMatrix.append(
                                        np.hstack((SSCch(auxEMG[wi:wf], ch)[0], np.array([tyi, per, carp, cl, rp]))))
                                    t2.append(time.time() - t)
                                    t = time.time()
                                    lscaleMatrix.append(
                                        np.hstack((Lscalech(auxEMG[wi:wf], ch)[0], np.array([tyi, per, carp, cl, rp]))))
                                    mflMatrix.append(
                                        np.hstack((MFLch(auxEMG[wi:wf], ch)[0], np.array([tyi, per, carp, cl, rp]))))
                                    msrMatrix.append(
                                        np.hstack((MSRch(auxEMG[wi:wf], ch)[0], np.array([tyi, per, carp, cl, rp]))))
                                    wampMatrix.append(
                                        np.hstack((WAMPch(auxEMG[wi:wf], ch)[0], np.array([tyi, per, carp, cl, rp]))))
                                    t3.append(time.time() - t)

                                    wi += incrmentSamples

                                if cl == 6:
                                    rp = rp + 1
                        per += 1

            timesFeatures = np.vstack((t1, t2, t3))
            auxName = 'timesFeatures' + windowFileName
            myFile = database + '/' + auxName + '.csv'
            np.savetxt(myFile, timesFeatures, delimiter=',')

            auxName = 'mavMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(mavMatrix)
            auxName = 'wlMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(wlMatrix)
            auxName = 'zcMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(zcMatrix)
            auxName = 'sscMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(sscMatrix)
            auxName = 'lscaleMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(lscaleMatrix)
            auxName = 'mflMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(mflMatrix)
            auxName = 'msrMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(msrMatrix)
            auxName = 'wampMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(wampMatrix)
            auxName = 'logvarMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(logvarMatrix)

            print('Finished')

        elif database == 'EPN':
            sampleRate = 200
            # EPN DATABASE

            rpt = 25
            ch = 8
            classes = 5
            people = 60
            types = 2
            windowSamples = int(window * sampleRate / 1000)
            incrmentSamples = windowSamples - int(overlap * sampleRate / 1000)

            for ty in range(0, types):
                for person in range(1, people + 1):
                    for cl in range(1, classes + 1):
                        print(ty, person, cl)
                        for rp in range(1, rpt + 1):
                            aux = scipy.io.loadmat(
                                '../Databases/CollectedData/allUsers_data/detectedData/emg_person' + str(
                                    person) + '_class' + str(
                                    cl) + '_rpt' + str(
                                    rp) + '_type' + str(ty) + '.mat')
                            auxEMG = aux['emg']

                            wi = 0
                            segments = int((len(auxEMG) - windowSamples) / incrmentSamples + 1)
                            for w in range(segments):
                                wf = wi + windowSamples

                                t = time.time()
                                logvarMatrix.append(
                                    np.hstack((logVARch(auxEMG[wi:wf], ch)[0], np.array([ty, person, cl, rp]))))
                                t1.append(time.time() - t)
                                t = time.time()
                                mavMatrix.append(np.hstack((MAVch(auxEMG[wi:wf], ch), np.array([ty, person, cl, rp]))))
                                wlMatrix.append(np.hstack((WLch(auxEMG[wi:wf], ch)[0], np.array([ty, person, cl, rp]))))
                                zcMatrix.append(np.hstack((ZCch(auxEMG[wi:wf], ch)[0], np.array([ty, person, cl, rp]))))
                                sscMatrix.append(
                                    np.hstack((SSCch(auxEMG[wi:wf], ch)[0], np.array([ty, person, cl, rp]))))
                                t2.append(time.time() - t)
                                t = time.time()
                                lscaleMatrix.append(
                                    np.hstack((Lscalech(auxEMG[wi:wf], ch)[0], np.array([ty, person, cl, rp]))))
                                mflMatrix.append(
                                    np.hstack((MFLch(auxEMG[wi:wf], ch)[0], np.array([ty, person, cl, rp]))))
                                msrMatrix.append(
                                    np.hstack((MSRch(auxEMG[wi:wf], ch)[0], np.array([ty, person, cl, rp]))))
                                wampMatrix.append(
                                    np.hstack((WAMPch(auxEMG[wi:wf], ch)[0], np.array([ty, person, cl, rp]))))
                                t3.append((time.time() - t))

                                wi += incrmentSamples

            timesFeatures = np.vstack((t1, t2, t3))
            auxName = 'timesFeatures' + windowFileName
            myFile = database + '/' + auxName + '.csv'
            np.savetxt(myFile, timesFeatures, delimiter=',')

            auxName = 'mavMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(mavMatrix)
            auxName = 'wlMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(wlMatrix)
            auxName = 'zcMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(zcMatrix)
            auxName = 'sscMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(sscMatrix)
            auxName = 'lscaleMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(lscaleMatrix)
            auxName = 'mflMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(mflMatrix)
            auxName = 'msrMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(msrMatrix)
            auxName = 'wampMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(wampMatrix)
            auxName = 'logvarMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(logvarMatrix)

            print('Finished')

        elif database == 'Capgmyo_dba':
            sampleRate = 1000
            rpt = 10
            ch = 128
            classes = 8
            people = 18

            windowSamples = int(window * sampleRate / 1000)
            incrmentSamples = windowSamples - int(overlap * sampleRate / 1000)

            for person in ["%.2d" % i for i in range(1, people + 1)]:
                for cl in ["%.2d" % i for i in range(1, classes + 1)]:
                    for rp in ["%.2d" % i for i in range(1, rpt + 1)]:
                        print(person, cl, rp)
                        aux = scipy.io.loadmat(
                            '../Databases/capgmyo_dba/dba-preprocessed-0' + person + '/0' + person + '-0' + cl + '-0' + rp + '.mat')
                        auxEMG = aux['data']

                        wi = 0
                        segments = int((len(auxEMG) - windowSamples) / incrmentSamples + 1)
                        for w in range(segments):
                            wf = wi + windowSamples

                            t = time.time()
                            logvarMatrix.append(
                                np.hstack((logVARch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            t1.append(time.time() - t)
                            t = time.time()
                            zcMatrix.append(np.hstack((ZCch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            mavMatrix.append(np.hstack((MAVch(auxEMG[wi:wf], ch), np.array([person, cl, rp]))))
                            wlMatrix.append(np.hstack((WLch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            sscMatrix.append(np.hstack((SSCch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            t2.append(time.time() - t)
                            t = time.time()
                            lscaleMatrix.append(
                                np.hstack((Lscalech(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            mflMatrix.append(np.hstack((MFLch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            msrMatrix.append(np.hstack((MSRch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            wampMatrix.append(np.hstack((WAMPch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            t3.append(time.time() - t)

                            # t = time.time()
                            # logvarMatrix.append(
                            #     np.hstack((logVARch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            # t1.append(time.time() - t)
                            # t = time.time()
                            # zcMatrix.append(np.hstack((ZCch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            # t2.append(time.time() - t)
                            # t = time.time()
                            # mavMatrix.append(np.hstack((MAVch(auxEMG[wi:wf], ch), np.array([person, cl, rp]))))
                            # wlMatrix.append(np.hstack((WLch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            # sscMatrix.append(np.hstack((SSCch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            # t3.append(time.time() - t)
                            # t = time.time()
                            # lscaleMatrix.append(
                            #     np.hstack((Lscalech(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            # mflMatrix.append(np.hstack((MFLch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            # msrMatrix.append(np.hstack((MSRch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            # wampMatrix.append(np.hstack((WAMPch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            # t4.append(time.time() - t)
                            # t = time.time()
                            # rmsMatrix.append(np.hstack((RMSch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            # t5.append(time.time() - t)
                            # t = time.time()
                            # iavMatrix.append(np.hstack((IAVch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            # dasdvMatrix.append(np.hstack((DASDVch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            # varMatrix.append(np.hstack((VARch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            # t6.append(time.time() - t)
                            # t = time.time()
                            # ar1C, ar2C, ar3C, ar4C, ar5C, ar6C = AR6ch(auxEMG[wi:wf], ch)
                            # ar1.append(np.hstack((ar1C, np.array([person, cl, rp]))))
                            # ar2.append(np.hstack((ar2C, np.array([person, cl, rp]))))
                            # ar3.append(np.hstack((ar3C, np.array([person, cl, rp]))))
                            # ar4.append(np.hstack((ar4C, np.array([person, cl, rp]))))
                            # ar5.append(np.hstack((ar5C, np.array([person, cl, rp]))))
                            # ar6.append(np.hstack((ar6C, np.array([person, cl, rp]))))
                            # t7.append(time.time() - t)

                            wi += incrmentSamples

                            # LS, MFL, MSR, WAMP, ZC, RMS, IAV, DASDV, and VAR

                timesFeatures = np.vstack((t1, t2, t3))
                # timesFeatures = np.vstack((t1, t2, t3, t4, t5, t6, t7))
                auxName = 'timesFeatures' + windowFileName
                myFile = database + '/' + auxName + '.csv'
                np.savetxt(myFile, timesFeatures, delimiter=',')

                auxName = 'mavMatrix' + windowFileName
                myFile = open(database + '/' + auxName + '.csv', 'w')
                with myFile:
                    writer = csv.writer(myFile)
                    writer.writerows(mavMatrix)
                auxName = 'wlMatrix' + windowFileName
                myFile = open(database + '/' + auxName + '.csv', 'w')
                with myFile:
                    writer = csv.writer(myFile)
                    writer.writerows(wlMatrix)
                auxName = 'zcMatrix' + windowFileName
                myFile = open(database + '/' + auxName + '.csv', 'w')
                with myFile:
                    writer = csv.writer(myFile)
                    writer.writerows(zcMatrix)
                auxName = 'sscMatrix' + windowFileName
                myFile = open(database + '/' + auxName + '.csv', 'w')
                with myFile:
                    writer = csv.writer(myFile)
                    writer.writerows(sscMatrix)
                auxName = 'lscaleMatrix' + windowFileName
                myFile = open(database + '/' + auxName + '.csv', 'w')
                with myFile:
                    writer = csv.writer(myFile)
                    writer.writerows(lscaleMatrix)
                auxName = 'mflMatrix' + windowFileName
                myFile = open(database + '/' + auxName + '.csv', 'w')
                with myFile:
                    writer = csv.writer(myFile)
                    writer.writerows(mflMatrix)
                auxName = 'msrMatrix' + windowFileName
                myFile = open(database + '/' + auxName + '.csv', 'w')
                with myFile:
                    writer = csv.writer(myFile)
                    writer.writerows(msrMatrix)
                auxName = 'wampMatrix' + windowFileName
                myFile = open(database + '/' + auxName + '.csv', 'w')
                with myFile:
                    writer = csv.writer(myFile)
                    writer.writerows(wampMatrix)
                auxName = 'logvarMatrix' + windowFileName
                myFile = open(database + '/' + auxName + '.csv', 'w')
                with myFile:
                    writer = csv.writer(myFile)
                    writer.writerows(logvarMatrix)

        elif database == 'Capgmyo_dbc':
            sampleRate = 1000
            rpt = 10
            ch = 128
            classes = 12
            people = 10

            windowSamples = int(window * sampleRate / 1000)
            incrmentSamples = windowSamples - int(overlap * sampleRate / 1000)

            for person in ["%.2d" % i for i in range(1, people + 1)]:
                for cl in ["%.2d" % i for i in range(1, classes + 1)]:
                    for rp in ["%.2d" % i for i in range(1, rpt + 1)]:
                        print(person, cl, rp)
                        aux = scipy.io.loadmat(
                            '../Databases/capgmyo_dbc/dbc-preprocessed-0' + person + '/0' + person + '-0' + cl + '-0' + rp + '.mat')
                        auxEMG = aux['data']

                        wi = 0
                        segments = int((len(auxEMG) - windowSamples) / incrmentSamples + 1)
                        for w in range(segments):
                            wf = wi + windowSamples

                            t = time.time()
                            logvarMatrix.append(
                                np.hstack((logVARch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            t1.append(time.time() - t)
                            t = time.time()
                            zcMatrix.append(np.hstack((ZCch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            mavMatrix.append(np.hstack((MAVch(auxEMG[wi:wf], ch), np.array([person, cl, rp]))))
                            wlMatrix.append(np.hstack((WLch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            sscMatrix.append(np.hstack((SSCch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            t2.append(time.time() - t)
                            t = time.time()
                            lscaleMatrix.append(
                                np.hstack((Lscalech(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            mflMatrix.append(np.hstack((MFLch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            msrMatrix.append(np.hstack((MSRch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            wampMatrix.append(np.hstack((WAMPch(auxEMG[wi:wf], ch)[0], np.array([person, cl, rp]))))
                            t3.append(time.time() - t)

                            wi += incrmentSamples

            timesFeatures = np.vstack((t1, t2, t3))
            auxName = 'timesFeatures' + windowFileName
            myFile = database + '/' + auxName + '.csv'
            np.savetxt(myFile, timesFeatures, delimiter=',')

            auxName = 'mavMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(mavMatrix)
            auxName = 'wlMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(wlMatrix)
            auxName = 'zcMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(zcMatrix)
            auxName = 'sscMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(sscMatrix)
            auxName = 'lscaleMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(lscaleMatrix)
            auxName = 'mflMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(mflMatrix)
            auxName = 'msrMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(msrMatrix)
            auxName = 'wampMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(wampMatrix)
            auxName = 'logvarMatrix' + windowFileName
            myFile = open(database + '/' + auxName + '.csv', 'w')
            with myFile:
                writer = csv.writer(myFile)
                writer.writerows(logvarMatrix)

