import numpy as np
import pandas as pd
import csv

windowFileName = '295'
database = 'Capgmyo_dba'

for nameVar in ['mavMatrix', 'wlMatrix', 'zcMatrix', 'sscMatrix', 'lscaleMatrix', 'mflMatrix', 'msrMatrix',
                'wampMatrix', 'logvarMatrix']:
    a = np.genfromtxt('ExtractedDataCapgmyo_dba_old' + '/' + nameVar + '295' + '.csv', delimiter=',')
    b = np.genfromtxt('ExtractedDataCapgmyo_dba_new' + '/' + nameVar + '295' + '.csv', delimiter=',')
    c = np.vstack((a, b))
    auxName = nameVar + windowFileName
    myFile = open('ExtractedData' + database + '_join/' + auxName + '.csv', 'w')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(c)
