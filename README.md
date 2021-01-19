# FewShotLearningEMG

This repository describes the three experiments performed in the paper *A Multi-Source Domain Adaptation for Real-Time Hand Gesture Classification using EMGs*. We implemented the experiments using [python 3.7](https://www.python.org/downloads/release/python-377/).

*NOTE:* This [link](https://anonymous.4open.science/r/21aef992-6a78-4b3b-8622-50b56c257659) is an anonymous git version of this repository.


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

1. [NinaPro5](http://ninaweb.hevs.ch/). This database should import to the folder [ninaDB5](Databases/ninaDB5).
2. [Côté-Allard](https://github.com/UlysseCoteAllard/MyoArmbandDataset). This database should import to the folder [MyoArmbandDataset-master](Databases/MyoArmbandDataset-master).
3. [EPN](https://ieeexplore.ieee.org/abstract/document/8903136/?casa_token=RYo5viuh6S8AAAAA:lizIpEqM4rK5eeo1Wxm-aPuDB20da2PngeRRnrC7agqSK1j26mqmtq5YJFLive7uW083m9tT). 
This database should import to the folder [rawData](Databases/CollectedData/allUsers_data/rawData). In addition, the file [main](Databases/CollectedData/Detect_muscle_activity/main.m) 
should be executed to only segment the gesture data using the detect-muscle-activity's technique proposed by the authors of 
this [work](https://ieeexplore.ieee.org/document/8247458). 

## Experiments

To reproduce our experiments, please perform the following steps:

1. We extracted the data from the three databases using the [DataExtraction](ExtractedData/DataExtraction.py) python file. This file was run over a personal computer (Intel® Core™ i5-8250U processor and 8GB of RAM).

The next three steps were performed over a supercomputer in [NCI Gadi](http://nci.org.au/our-services/supercomputing). The characteristics of Gadi are Intel Xeon Platinum 8274 (Cascade Lake), two physical processors per node, 3.2 GHz clock speed, and 48 cores per node. Documentation for NCI Gadi can be found [here](https://opus.nci.org.au/display/Help/Gadi+User+Guide)

2. For the experiment 1, we run over Gadi three batch files (one per database): [NinaPro5](Experiments/Experiment1/Nina5_PBS.sh), [CoteAllard](Experiments/Experiment1/Cote_PBS.sh), and [EPN](Experiments/Experiment1/EPN_PBS.sh).

3. For the experiment 2, we run over Gadi one batch files ([Synthetic Data](Experiments/Experiment2/SyntheticRand_script.sh))

4. For the experiment 3, we run over Gadi three batch files (one per database): [NinaPro5](Experiments/Experiment3/Cote_CWT_Nina5/EVALUATE.sh), [CoteAllard](Experiments/Experiment3/Cote_CWT_Cote/EVALUATE.sh), and [EPN](Experiments/Experiment3/Cote_CWT_EPN/EVALUATE.sh).

#### NOTE

The implementations of our adaptation technique, and the [Liu](https://ieeexplore.ieee.org/abstract/document/6985518/?casa_token=H9vZpl9IcF8AAAAA:Iom6Q55n9FSn-G9CqqS6bxQzzho7vvb0OtQPdgZMQBOuNo5HwCHZSh0wddgdSp6V3q_pFsSJ) and [Vidovic](https://ieeexplore.ieee.org/abstract/document/7302056/?casa_token=3KVFZed5PzoAAAAA:rQJutibAYMQ_Za4ZSNEee6VIR59ZlWlt9o6_MKLFY2GKq2_zgYBkFPqs5UhrFCvMyP41SBbJ) methods are in the [DA_BasedAdaptiveModels](Experiments/Experiment1/DA_BasedAdaptiveModels.py) python file. In addition, the [CoteAllard implementation](Experiments/Experiment3) used in experiment 3 is a version of the original one, which you can find [here](https://github.com/UlysseCoteAllard/MyoArmbandDataset). We only modified the files used to load the data in order to use the three databases and evaluate using the Friedman rank and Holm Post-Hoc tests (the convolutional neural networks and the transfer learning technique did not modify).

## Visualization of the three experiments:
After the experiments' execution, we use Jupyter notebooks to analyze and develop the graphs presented in the paper. In the anonymous git repository, the Jupyter notebooks may not render correctly, so we include markdown files of the three notebooks.

[Experiment 1](Experiment1.ipynb) (jupyter notebook) or [Experiment 1](MarkdownExperiments/Experiment1.md) (markdown file)

[Experiment 2](Experiment2.ipynb) (jupyter notebook) or [Experiment 2](MarkdownExperiments/Experiment2.md) (markdown file)

[Experiment 3](Experiment3.ipynb) (jupyter notebook) or [Experiment 3](MarkdownExperiments/Experiment3.md) (markdown file)
