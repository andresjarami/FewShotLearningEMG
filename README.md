# FewShotLearningEMG

This repository describe the three experiments performed to the paper *A Multi-Source Domain Adaptation for Real-Time Hand Gesture Classification using EMGs* using [python 3.7](https://www.python.org/downloads/release/python-377/)

## Required libraries 
Numpy [https://numpy.org/install/](https://numpy.org/install/)

Pandas [https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html/](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html/)

Matplotlib [https://matplotlib.org/3.1.1/users/installing.html](https://matplotlib.org/3.1.1/users/installing.html)

Scipy [https://www.scipy.org/install.html](https://www.scipy.org/install.html)

Time [https://docs.python.org/3/library/time.html](https://docs.python.org/3/library/time.html)

Math [https://docs.python.org/3/library/math.html](https://docs.python.org/3/library/math.html)

Sklearn [https://scikit-learn.org/stable/install.html](https://scikit-learn.org/stable/install.html)

Itertools [https://pypi.org/project/more-itertools/](https://pypi.org/project/more-itertools/)

Random [https://pypi.org/project/random2/](https://pypi.org/project/random2/)

## Databases

1. NinaPro5 http://ninaweb.hevs.ch/
2. CoteAllard https://github.com/UlysseCoteAllard/MyoArmbandDataset
3. EPN  https://ieeexplore.ieee.org/abstract/document/8903136/?casa_token=RYo5viuh6S8AAAAA:lizIpEqM4rK5eeo1Wxm-aPuDB20da2PngeRRnrC7agqSK1j26mqmtq5YJFLive7uW083m9tT

## Experiments
To reproduce our experiments, please perform the following steps:

1. We extracted the data from the three databases using the [DataExtraction](ExtractedData/DataExtraction.py) python file. This file was run over a personal computer (Intel® Core™ i5-8250U processor and 8GB of RAM).

The next stepts were perfromed over a supercomputer [Gadi](http://nci.org.au/our-services/supercomputing). The characteristics of Gadi are: Intel Xeon Platinum 8274 (Cascade Lake), Two physical processors per node, 3.2 GHz clock speed, and 48 cores per node.

2. For the experiment 1 and 2, we run over Gadi the batch files : 
 
The three experiments were performed over a 

## Visualization of the three experiments:
[Experiment 1](Experiment1_3.ipynb)

[Experiment 2](Experiment2.ipynb)

[Experiment 3](Experiment1_3.ipynb)

