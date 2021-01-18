# Experiment 1

In this experiment, we compare the performance of the following LDA and QDA classifiers within the myoelectric
interface setups from section 3 (in the manuscript):
1. A non-adaptive DA classifier trained over a small-labeled target training set Tˆ (individual classifier).
2. A non-adaptive DA classifier trained over a small-labeled target training set Tˆ and training sets from the source
users (multi-user classifier).
3. Our adaptive DA classifier from section 2 (In the manuscript).
4. [Liu’s adaptive DA classifier](https://ieeexplore.ieee.org/abstract/document/6985518/?casa_token=H9vZpl9IcF8AAAAA:Iom6Q55n9FSn-G9CqqS6bxQzzho7vvb0OtQPdgZMQBOuNo5HwCHZSh0wddgdSp6V3q_pFsSJ).
5. [Vidovic’s adaptive DA classifier](https://ieeexplore.ieee.org/abstract/document/7302056/?casa_token=3KVFZed5PzoAAAAA:rQJutibAYMQ_Za4ZSNEee6VIR59ZlWlt9o6_MKLFY2GKq2_zgYBkFPqs5UhrFCvMyP41SBbJ).

We use three databases are: [NinaPro5](http://ninaweb.hevs.ch/),
[Côté-Allard](https://github.com/UlysseCoteAllard/MyoArmbandDataset),
[EPN](https://ieeexplore.ieee.org/abstract/document/8903136/?casa_token=RYo5viuh6S8AAAAA:lizIpEqM4rK5eeo1Wxm-aPuDB20da2PngeRRnrC7agqSK1j26mqmtq5YJFLive7uW083m9tT).
All of them contain EMG data of hand gestures acquired by a Myo armband.
The three feature sets extracted from the databases are Hahne, Hudgins, and
Phinyomark sets.

*Feature Set 1 (Hahne):*
1. Logarithm of the variance (logVAR)

*Feature Set 2 (Hudgins):*
1. Mean absolute value (MAV)
2. Waveform length (WL)
3. Zero Crossing (ZC)
4. Slope sign change (SSC)

*Feature Set 3 (Phinyomark):*
1. L-scale (LS)
2. Maximum fractal length (MFL)
3. Mean of the square root (MSR)
4. Willison amplitude (WAMP)



## Load Results

First: Import the library developed to visualize the results


```python
import Experiments.Experiment1.VisualizationFunctions as VF1
import pandas as pd
```

Second: Load of the extraction times over the three databases


```python
place='ExtractedData/'
windowSize='295'

extractionTimeEPN=pd.read_csv(place+'ExtractedDataCollectedData/timesFeatures'+windowSize+'.csv',header=None)
extractionTimeCote=pd.read_csv(place+'ExtractedDataCoteAllard/timesFeatures'+windowSize+'.csv',header=None)
extractionTimeNina5=pd.read_csv(place+'ExtractedDataNinaDB5/timesFeatures'+windowSize+'.csv',header=None)
```

Third: Load of the DA-based adaptation techniques' results over the three databases using the three feature sets.

NINA PRO 5 database


```python
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
place='Experiments/Experiment1/results/'
windowSize='295'

database='Nina5'
resultsNina5,timeNina5=VF1.uploadResultsDatabases(place,database,windowSize)
resultsNina5
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature Set</th>
      <th># shots</th>
      <th>IndLDA</th>
      <th>IndLDAstd</th>
      <th>IndQDA</th>
      <th>IndQDAstd</th>
      <th>MultiLDA</th>
      <th>MultiLDAstd</th>
      <th>MultiQDA</th>
      <th>MultiQDAstd</th>
      <th>LiuLDA</th>
      <th>LiuLDAstd</th>
      <th>LiuQDA</th>
      <th>LiuQDAstd</th>
      <th>VidLDA</th>
      <th>VidLDAstd</th>
      <th>VidQDA</th>
      <th>VidQDAstd</th>
      <th>OurLDA</th>
      <th>OurLDAstd</th>
      <th>OurQDA</th>
      <th>OurQDAstd</th>
      <th>wLDA</th>
      <th>lLDA</th>
      <th>wQDA</th>
      <th>lQDA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0.469496</td>
      <td>0.060965</td>
      <td>0.422803</td>
      <td>0.065441</td>
      <td>0.312546</td>
      <td>0.070921</td>
      <td>0.311988</td>
      <td>0.077936</td>
      <td>0.453851</td>
      <td>0.049627</td>
      <td>0.475283</td>
      <td>0.083744</td>
      <td>0.461022</td>
      <td>0.056204</td>
      <td>0.425570</td>
      <td>0.065878</td>
      <td>0.471202</td>
      <td>0.064744</td>
      <td>0.505954</td>
      <td>0.072623</td>
      <td>0.921058</td>
      <td>0.569883</td>
      <td>0.866013</td>
      <td>0.624340</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>0.521554</td>
      <td>0.051513</td>
      <td>0.545461</td>
      <td>0.051622</td>
      <td>0.323491</td>
      <td>0.070998</td>
      <td>0.341456</td>
      <td>0.080460</td>
      <td>0.486173</td>
      <td>0.055212</td>
      <td>0.525321</td>
      <td>0.078764</td>
      <td>0.508808</td>
      <td>0.045406</td>
      <td>0.506843</td>
      <td>0.060067</td>
      <td>0.520875</td>
      <td>0.054835</td>
      <td>0.583879</td>
      <td>0.050606</td>
      <td>0.951825</td>
      <td>0.694085</td>
      <td>0.918691</td>
      <td>0.796141</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>0.549970</td>
      <td>0.053120</td>
      <td>0.601473</td>
      <td>0.050863</td>
      <td>0.334441</td>
      <td>0.070649</td>
      <td>0.369956</td>
      <td>0.079270</td>
      <td>0.499268</td>
      <td>0.055004</td>
      <td>0.546030</td>
      <td>0.073568</td>
      <td>0.537133</td>
      <td>0.049720</td>
      <td>0.553669</td>
      <td>0.048776</td>
      <td>0.549063</td>
      <td>0.053389</td>
      <td>0.621923</td>
      <td>0.046196</td>
      <td>0.966316</td>
      <td>0.795921</td>
      <td>0.952335</td>
      <td>0.887036</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>4</td>
      <td>0.575601</td>
      <td>0.049358</td>
      <td>0.637058</td>
      <td>0.044869</td>
      <td>0.344707</td>
      <td>0.070109</td>
      <td>0.395753</td>
      <td>0.077023</td>
      <td>0.509878</td>
      <td>0.056163</td>
      <td>0.559818</td>
      <td>0.075246</td>
      <td>0.558708</td>
      <td>0.050394</td>
      <td>0.575438</td>
      <td>0.044523</td>
      <td>0.575461</td>
      <td>0.049439</td>
      <td>0.644743</td>
      <td>0.043549</td>
      <td>0.980614</td>
      <td>0.900574</td>
      <td>0.976988</td>
      <td>0.946159</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>1</td>
      <td>0.450385</td>
      <td>0.066422</td>
      <td>0.377597</td>
      <td>0.065214</td>
      <td>0.314821</td>
      <td>0.069246</td>
      <td>0.287393</td>
      <td>0.075524</td>
      <td>0.438975</td>
      <td>0.079625</td>
      <td>0.412820</td>
      <td>0.071902</td>
      <td>0.453741</td>
      <td>0.058942</td>
      <td>0.448725</td>
      <td>0.061835</td>
      <td>0.446346</td>
      <td>0.068083</td>
      <td>0.443212</td>
      <td>0.064879</td>
      <td>0.922305</td>
      <td>0.565100</td>
      <td>0.720362</td>
      <td>0.579360</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>2</td>
      <td>0.518605</td>
      <td>0.051153</td>
      <td>0.486443</td>
      <td>0.046113</td>
      <td>0.326292</td>
      <td>0.069567</td>
      <td>0.315136</td>
      <td>0.083908</td>
      <td>0.476249</td>
      <td>0.073608</td>
      <td>0.449254</td>
      <td>0.078110</td>
      <td>0.517411</td>
      <td>0.042640</td>
      <td>0.525773</td>
      <td>0.047754</td>
      <td>0.509499</td>
      <td>0.054990</td>
      <td>0.520432</td>
      <td>0.050189</td>
      <td>0.952623</td>
      <td>0.702729</td>
      <td>0.834392</td>
      <td>0.804553</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>3</td>
      <td>0.549527</td>
      <td>0.042896</td>
      <td>0.553536</td>
      <td>0.045772</td>
      <td>0.336380</td>
      <td>0.070198</td>
      <td>0.335177</td>
      <td>0.091110</td>
      <td>0.494329</td>
      <td>0.069708</td>
      <td>0.469267</td>
      <td>0.089110</td>
      <td>0.548532</td>
      <td>0.040565</td>
      <td>0.564091</td>
      <td>0.045635</td>
      <td>0.542387</td>
      <td>0.048025</td>
      <td>0.567437</td>
      <td>0.052317</td>
      <td>0.973279</td>
      <td>0.793065</td>
      <td>0.904453</td>
      <td>0.906679</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2</td>
      <td>4</td>
      <td>0.576946</td>
      <td>0.038291</td>
      <td>0.590562</td>
      <td>0.051153</td>
      <td>0.346937</td>
      <td>0.072060</td>
      <td>0.354230</td>
      <td>0.097943</td>
      <td>0.508217</td>
      <td>0.069746</td>
      <td>0.482522</td>
      <td>0.092739</td>
      <td>0.569556</td>
      <td>0.039128</td>
      <td>0.587607</td>
      <td>0.049158</td>
      <td>0.572686</td>
      <td>0.040539</td>
      <td>0.598014</td>
      <td>0.050961</td>
      <td>0.987553</td>
      <td>0.888022</td>
      <td>0.956657</td>
      <td>0.962839</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3</td>
      <td>1</td>
      <td>0.472683</td>
      <td>0.059148</td>
      <td>0.416586</td>
      <td>0.060825</td>
      <td>0.331466</td>
      <td>0.083437</td>
      <td>0.326010</td>
      <td>0.074069</td>
      <td>0.462606</td>
      <td>0.065280</td>
      <td>0.459427</td>
      <td>0.066855</td>
      <td>0.472736</td>
      <td>0.050524</td>
      <td>0.438043</td>
      <td>0.065883</td>
      <td>0.464953</td>
      <td>0.064747</td>
      <td>0.476532</td>
      <td>0.061354</td>
      <td>0.910186</td>
      <td>0.544462</td>
      <td>0.737368</td>
      <td>0.553144</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3</td>
      <td>2</td>
      <td>0.534814</td>
      <td>0.044637</td>
      <td>0.507842</td>
      <td>0.045683</td>
      <td>0.342576</td>
      <td>0.082135</td>
      <td>0.361956</td>
      <td>0.076631</td>
      <td>0.501084</td>
      <td>0.059337</td>
      <td>0.511081</td>
      <td>0.067064</td>
      <td>0.527383</td>
      <td>0.041397</td>
      <td>0.510812</td>
      <td>0.055721</td>
      <td>0.526619</td>
      <td>0.050901</td>
      <td>0.544687</td>
      <td>0.052423</td>
      <td>0.951020</td>
      <td>0.672006</td>
      <td>0.829844</td>
      <td>0.755531</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3</td>
      <td>3</td>
      <td>0.569079</td>
      <td>0.039084</td>
      <td>0.573354</td>
      <td>0.042671</td>
      <td>0.352524</td>
      <td>0.080874</td>
      <td>0.389776</td>
      <td>0.074978</td>
      <td>0.523541</td>
      <td>0.057881</td>
      <td>0.544269</td>
      <td>0.065405</td>
      <td>0.562485</td>
      <td>0.037394</td>
      <td>0.564398</td>
      <td>0.050583</td>
      <td>0.567555</td>
      <td>0.042870</td>
      <td>0.589050</td>
      <td>0.049858</td>
      <td>0.968869</td>
      <td>0.780558</td>
      <td>0.905711</td>
      <td>0.887547</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3</td>
      <td>4</td>
      <td>0.591195</td>
      <td>0.041341</td>
      <td>0.613604</td>
      <td>0.049484</td>
      <td>0.362348</td>
      <td>0.079519</td>
      <td>0.414453</td>
      <td>0.081734</td>
      <td>0.534895</td>
      <td>0.060130</td>
      <td>0.564787</td>
      <td>0.069401</td>
      <td>0.585903</td>
      <td>0.041815</td>
      <td>0.598595</td>
      <td>0.049920</td>
      <td>0.592331</td>
      <td>0.043188</td>
      <td>0.623596</td>
      <td>0.049006</td>
      <td>0.986577</td>
      <td>0.883779</td>
      <td>0.960939</td>
      <td>0.958099</td>
    </tr>
  </tbody>
</table>
</div>



Cote-Allard database


```python
database='Cote'
resultsCote,timeCote=VF1.uploadResultsDatabases(place,database,windowSize)
resultsCote
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature Set</th>
      <th># shots</th>
      <th>IndLDA</th>
      <th>IndLDAstd</th>
      <th>IndQDA</th>
      <th>IndQDAstd</th>
      <th>MultiLDA</th>
      <th>MultiLDAstd</th>
      <th>MultiQDA</th>
      <th>MultiQDAstd</th>
      <th>LiuLDA</th>
      <th>LiuLDAstd</th>
      <th>LiuQDA</th>
      <th>LiuQDAstd</th>
      <th>VidLDA</th>
      <th>VidLDAstd</th>
      <th>VidQDA</th>
      <th>VidQDAstd</th>
      <th>OurLDA</th>
      <th>OurLDAstd</th>
      <th>OurQDA</th>
      <th>OurQDAstd</th>
      <th>wLDA</th>
      <th>lLDA</th>
      <th>wQDA</th>
      <th>lQDA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0.943640</td>
      <td>0.053431</td>
      <td>0.914393</td>
      <td>0.086532</td>
      <td>0.701505</td>
      <td>0.238031</td>
      <td>0.732781</td>
      <td>0.199898</td>
      <td>0.940876</td>
      <td>0.048241</td>
      <td>0.904222</td>
      <td>0.099606</td>
      <td>0.942476</td>
      <td>0.056406</td>
      <td>0.872334</td>
      <td>0.071262</td>
      <td>0.962150</td>
      <td>0.039487</td>
      <td>0.953676</td>
      <td>0.042555</td>
      <td>0.604792</td>
      <td>0.500250</td>
      <td>0.618323</td>
      <td>0.500109</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>0.957327</td>
      <td>0.049524</td>
      <td>0.959355</td>
      <td>0.052859</td>
      <td>0.715593</td>
      <td>0.232228</td>
      <td>0.773073</td>
      <td>0.170238</td>
      <td>0.952183</td>
      <td>0.050239</td>
      <td>0.925510</td>
      <td>0.087370</td>
      <td>0.952811</td>
      <td>0.048167</td>
      <td>0.915026</td>
      <td>0.043341</td>
      <td>0.966847</td>
      <td>0.044018</td>
      <td>0.970128</td>
      <td>0.047870</td>
      <td>0.731396</td>
      <td>0.638199</td>
      <td>0.728286</td>
      <td>0.639465</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>0.977673</td>
      <td>0.022352</td>
      <td>0.976573</td>
      <td>0.021488</td>
      <td>0.728691</td>
      <td>0.226035</td>
      <td>0.825591</td>
      <td>0.118134</td>
      <td>0.962241</td>
      <td>0.042114</td>
      <td>0.943855</td>
      <td>0.065457</td>
      <td>0.971266</td>
      <td>0.021481</td>
      <td>0.938019</td>
      <td>0.048435</td>
      <td>0.981359</td>
      <td>0.018941</td>
      <td>0.983824</td>
      <td>0.019367</td>
      <td>0.829534</td>
      <td>0.760006</td>
      <td>0.829197</td>
      <td>0.762887</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>4</td>
      <td>0.981882</td>
      <td>0.020309</td>
      <td>0.977272</td>
      <td>0.022945</td>
      <td>0.741107</td>
      <td>0.218632</td>
      <td>0.857591</td>
      <td>0.092089</td>
      <td>0.965628</td>
      <td>0.042584</td>
      <td>0.948689</td>
      <td>0.068327</td>
      <td>0.978001</td>
      <td>0.015898</td>
      <td>0.933895</td>
      <td>0.051514</td>
      <td>0.984138</td>
      <td>0.017804</td>
      <td>0.983066</td>
      <td>0.019779</td>
      <td>0.916569</td>
      <td>0.878500</td>
      <td>0.918251</td>
      <td>0.881166</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>1</td>
      <td>0.909874</td>
      <td>0.109100</td>
      <td>0.818818</td>
      <td>0.108093</td>
      <td>0.727513</td>
      <td>0.212953</td>
      <td>0.715136</td>
      <td>0.187280</td>
      <td>0.915733</td>
      <td>0.075071</td>
      <td>0.906386</td>
      <td>0.078004</td>
      <td>0.919391</td>
      <td>0.108805</td>
      <td>0.855717</td>
      <td>0.105036</td>
      <td>0.935331</td>
      <td>0.080158</td>
      <td>0.937519</td>
      <td>0.067299</td>
      <td>0.620740</td>
      <td>0.502726</td>
      <td>0.641069</td>
      <td>0.541929</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>2</td>
      <td>0.945417</td>
      <td>0.084247</td>
      <td>0.895282</td>
      <td>0.073392</td>
      <td>0.748305</td>
      <td>0.196651</td>
      <td>0.760803</td>
      <td>0.150154</td>
      <td>0.923891</td>
      <td>0.084314</td>
      <td>0.927974</td>
      <td>0.075004</td>
      <td>0.945227</td>
      <td>0.096295</td>
      <td>0.912096</td>
      <td>0.079487</td>
      <td>0.951304</td>
      <td>0.082973</td>
      <td>0.954426</td>
      <td>0.059298</td>
      <td>0.755172</td>
      <td>0.640638</td>
      <td>0.731757</td>
      <td>0.698164</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>3</td>
      <td>0.969438</td>
      <td>0.038156</td>
      <td>0.922262</td>
      <td>0.050130</td>
      <td>0.767207</td>
      <td>0.182514</td>
      <td>0.797529</td>
      <td>0.118060</td>
      <td>0.937713</td>
      <td>0.068967</td>
      <td>0.940033</td>
      <td>0.066551</td>
      <td>0.966775</td>
      <td>0.064826</td>
      <td>0.934636</td>
      <td>0.052413</td>
      <td>0.969840</td>
      <td>0.046105</td>
      <td>0.962754</td>
      <td>0.048529</td>
      <td>0.835955</td>
      <td>0.761898</td>
      <td>0.824419</td>
      <td>0.811239</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2</td>
      <td>4</td>
      <td>0.978541</td>
      <td>0.025791</td>
      <td>0.943979</td>
      <td>0.030199</td>
      <td>0.784083</td>
      <td>0.171070</td>
      <td>0.822530</td>
      <td>0.100187</td>
      <td>0.947048</td>
      <td>0.060803</td>
      <td>0.945668</td>
      <td>0.054501</td>
      <td>0.982800</td>
      <td>0.018878</td>
      <td>0.952765</td>
      <td>0.028002</td>
      <td>0.980452</td>
      <td>0.026000</td>
      <td>0.971176</td>
      <td>0.026728</td>
      <td>0.919393</td>
      <td>0.880085</td>
      <td>0.914261</td>
      <td>0.909220</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3</td>
      <td>1</td>
      <td>0.914089</td>
      <td>0.102151</td>
      <td>0.901808</td>
      <td>0.085922</td>
      <td>0.720824</td>
      <td>0.235403</td>
      <td>0.759881</td>
      <td>0.201857</td>
      <td>0.938754</td>
      <td>0.053922</td>
      <td>0.910927</td>
      <td>0.083738</td>
      <td>0.915828</td>
      <td>0.115327</td>
      <td>0.871927</td>
      <td>0.100809</td>
      <td>0.947870</td>
      <td>0.047134</td>
      <td>0.952748</td>
      <td>0.041036</td>
      <td>0.620492</td>
      <td>0.500203</td>
      <td>0.608017</td>
      <td>0.505827</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3</td>
      <td>2</td>
      <td>0.942654</td>
      <td>0.085124</td>
      <td>0.944160</td>
      <td>0.051821</td>
      <td>0.739606</td>
      <td>0.224899</td>
      <td>0.811444</td>
      <td>0.154473</td>
      <td>0.947834</td>
      <td>0.060809</td>
      <td>0.942863</td>
      <td>0.057258</td>
      <td>0.939704</td>
      <td>0.102129</td>
      <td>0.929486</td>
      <td>0.065380</td>
      <td>0.962325</td>
      <td>0.057756</td>
      <td>0.964754</td>
      <td>0.042666</td>
      <td>0.744600</td>
      <td>0.638405</td>
      <td>0.726225</td>
      <td>0.660818</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3</td>
      <td>3</td>
      <td>0.971714</td>
      <td>0.040443</td>
      <td>0.958511</td>
      <td>0.037047</td>
      <td>0.757362</td>
      <td>0.213818</td>
      <td>0.854229</td>
      <td>0.111295</td>
      <td>0.963698</td>
      <td>0.042172</td>
      <td>0.958804</td>
      <td>0.051514</td>
      <td>0.962653</td>
      <td>0.076876</td>
      <td>0.950510</td>
      <td>0.038543</td>
      <td>0.978323</td>
      <td>0.029097</td>
      <td>0.974812</td>
      <td>0.029833</td>
      <td>0.835027</td>
      <td>0.760600</td>
      <td>0.818772</td>
      <td>0.785824</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3</td>
      <td>4</td>
      <td>0.983464</td>
      <td>0.018633</td>
      <td>0.969912</td>
      <td>0.026316</td>
      <td>0.774881</td>
      <td>0.200310</td>
      <td>0.879035</td>
      <td>0.088150</td>
      <td>0.970301</td>
      <td>0.037570</td>
      <td>0.961497</td>
      <td>0.049482</td>
      <td>0.982214</td>
      <td>0.016972</td>
      <td>0.961951</td>
      <td>0.028846</td>
      <td>0.985311</td>
      <td>0.017090</td>
      <td>0.977453</td>
      <td>0.025441</td>
      <td>0.920651</td>
      <td>0.879100</td>
      <td>0.910683</td>
      <td>0.896291</td>
    </tr>
  </tbody>
</table>
</div>



EPN database


```python
database='EPN'
resultsEPN,timeEPN=VF1.uploadResultsDatabases(place,database,windowSize)
resultsEPN
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature Set</th>
      <th># shots</th>
      <th>IndLDA</th>
      <th>IndLDAstd</th>
      <th>IndQDA</th>
      <th>IndQDAstd</th>
      <th>MultiLDA</th>
      <th>MultiLDAstd</th>
      <th>MultiQDA</th>
      <th>MultiQDAstd</th>
      <th>LiuLDA</th>
      <th>LiuLDAstd</th>
      <th>LiuQDA</th>
      <th>LiuQDAstd</th>
      <th>VidLDA</th>
      <th>VidLDAstd</th>
      <th>VidQDA</th>
      <th>VidQDAstd</th>
      <th>OurLDA</th>
      <th>OurLDAstd</th>
      <th>OurQDA</th>
      <th>OurQDAstd</th>
      <th>wLDA</th>
      <th>lLDA</th>
      <th>wQDA</th>
      <th>lQDA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0.686165</td>
      <td>0.131710</td>
      <td>0.630271</td>
      <td>0.107627</td>
      <td>0.533857</td>
      <td>0.175196</td>
      <td>0.580119</td>
      <td>0.144893</td>
      <td>0.697570</td>
      <td>0.132179</td>
      <td>0.752834</td>
      <td>0.118613</td>
      <td>0.704752</td>
      <td>0.129903</td>
      <td>0.735736</td>
      <td>0.120711</td>
      <td>0.703437</td>
      <td>0.131320</td>
      <td>0.757732</td>
      <td>0.116762</td>
      <td>0.685148</td>
      <td>0.534075</td>
      <td>0.688496</td>
      <td>0.521165</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>0.735432</td>
      <td>0.132918</td>
      <td>0.739476</td>
      <td>0.098510</td>
      <td>0.534449</td>
      <td>0.175118</td>
      <td>0.581342</td>
      <td>0.144714</td>
      <td>0.718829</td>
      <td>0.130946</td>
      <td>0.780134</td>
      <td>0.115667</td>
      <td>0.735477</td>
      <td>0.127296</td>
      <td>0.776397</td>
      <td>0.113106</td>
      <td>0.736877</td>
      <td>0.126350</td>
      <td>0.796296</td>
      <td>0.108029</td>
      <td>0.747729</td>
      <td>0.590799</td>
      <td>0.711717</td>
      <td>0.600898</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>0.748962</td>
      <td>0.134980</td>
      <td>0.779787</td>
      <td>0.104635</td>
      <td>0.535097</td>
      <td>0.174918</td>
      <td>0.582900</td>
      <td>0.143875</td>
      <td>0.727982</td>
      <td>0.132072</td>
      <td>0.787558</td>
      <td>0.119224</td>
      <td>0.742000</td>
      <td>0.130462</td>
      <td>0.795827</td>
      <td>0.113339</td>
      <td>0.744134</td>
      <td>0.132586</td>
      <td>0.806938</td>
      <td>0.110266</td>
      <td>0.781939</td>
      <td>0.626063</td>
      <td>0.747019</td>
      <td>0.652729</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>4</td>
      <td>0.761555</td>
      <td>0.127999</td>
      <td>0.801943</td>
      <td>0.104483</td>
      <td>0.535648</td>
      <td>0.174711</td>
      <td>0.584319</td>
      <td>0.143217</td>
      <td>0.729871</td>
      <td>0.128714</td>
      <td>0.795766</td>
      <td>0.114037</td>
      <td>0.752873</td>
      <td>0.121666</td>
      <td>0.808157</td>
      <td>0.109975</td>
      <td>0.757044</td>
      <td>0.119768</td>
      <td>0.819691</td>
      <td>0.106110</td>
      <td>0.806083</td>
      <td>0.647845</td>
      <td>0.777102</td>
      <td>0.690075</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>5</td>
      <td>0.762991</td>
      <td>0.128283</td>
      <td>0.813814</td>
      <td>0.103458</td>
      <td>0.536169</td>
      <td>0.174669</td>
      <td>0.585422</td>
      <td>0.142791</td>
      <td>0.729272</td>
      <td>0.129066</td>
      <td>0.796465</td>
      <td>0.114393</td>
      <td>0.752144</td>
      <td>0.122658</td>
      <td>0.812133</td>
      <td>0.106486</td>
      <td>0.758197</td>
      <td>0.122508</td>
      <td>0.824210</td>
      <td>0.103172</td>
      <td>0.826429</td>
      <td>0.682228</td>
      <td>0.801990</td>
      <td>0.722072</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>6</td>
      <td>0.766549</td>
      <td>0.127004</td>
      <td>0.820638</td>
      <td>0.104015</td>
      <td>0.536694</td>
      <td>0.174529</td>
      <td>0.586754</td>
      <td>0.142140</td>
      <td>0.730997</td>
      <td>0.128723</td>
      <td>0.798831</td>
      <td>0.113214</td>
      <td>0.755750</td>
      <td>0.121235</td>
      <td>0.814328</td>
      <td>0.104439</td>
      <td>0.761093</td>
      <td>0.123556</td>
      <td>0.826786</td>
      <td>0.102590</td>
      <td>0.842285</td>
      <td>0.703175</td>
      <td>0.814777</td>
      <td>0.746995</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>7</td>
      <td>0.769889</td>
      <td>0.125337</td>
      <td>0.826622</td>
      <td>0.103587</td>
      <td>0.537267</td>
      <td>0.174425</td>
      <td>0.588119</td>
      <td>0.141618</td>
      <td>0.733064</td>
      <td>0.127086</td>
      <td>0.802123</td>
      <td>0.111162</td>
      <td>0.759254</td>
      <td>0.119632</td>
      <td>0.819496</td>
      <td>0.100847</td>
      <td>0.765537</td>
      <td>0.119708</td>
      <td>0.831924</td>
      <td>0.101766</td>
      <td>0.851530</td>
      <td>0.736724</td>
      <td>0.836752</td>
      <td>0.768052</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>8</td>
      <td>0.772781</td>
      <td>0.122992</td>
      <td>0.833954</td>
      <td>0.097421</td>
      <td>0.537830</td>
      <td>0.174261</td>
      <td>0.589448</td>
      <td>0.140926</td>
      <td>0.734392</td>
      <td>0.125085</td>
      <td>0.806034</td>
      <td>0.106868</td>
      <td>0.761411</td>
      <td>0.117178</td>
      <td>0.825673</td>
      <td>0.096316</td>
      <td>0.766484</td>
      <td>0.119886</td>
      <td>0.836770</td>
      <td>0.098654</td>
      <td>0.871771</td>
      <td>0.749445</td>
      <td>0.847529</td>
      <td>0.788737</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>9</td>
      <td>0.772835</td>
      <td>0.123537</td>
      <td>0.838562</td>
      <td>0.098573</td>
      <td>0.538266</td>
      <td>0.174182</td>
      <td>0.590562</td>
      <td>0.140583</td>
      <td>0.733314</td>
      <td>0.127012</td>
      <td>0.806688</td>
      <td>0.108459</td>
      <td>0.761853</td>
      <td>0.117168</td>
      <td>0.827644</td>
      <td>0.098264</td>
      <td>0.768138</td>
      <td>0.120037</td>
      <td>0.839406</td>
      <td>0.099434</td>
      <td>0.882251</td>
      <td>0.770153</td>
      <td>0.860167</td>
      <td>0.807957</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>10</td>
      <td>0.773819</td>
      <td>0.123246</td>
      <td>0.839854</td>
      <td>0.099086</td>
      <td>0.538769</td>
      <td>0.174086</td>
      <td>0.591751</td>
      <td>0.139960</td>
      <td>0.732833</td>
      <td>0.127821</td>
      <td>0.806543</td>
      <td>0.108881</td>
      <td>0.763406</td>
      <td>0.116440</td>
      <td>0.828187</td>
      <td>0.099558</td>
      <td>0.770356</td>
      <td>0.118916</td>
      <td>0.841204</td>
      <td>0.099511</td>
      <td>0.894422</td>
      <td>0.786362</td>
      <td>0.872532</td>
      <td>0.821683</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>11</td>
      <td>0.776445</td>
      <td>0.122868</td>
      <td>0.841099</td>
      <td>0.098916</td>
      <td>0.539416</td>
      <td>0.173928</td>
      <td>0.593632</td>
      <td>0.138937</td>
      <td>0.731926</td>
      <td>0.125761</td>
      <td>0.806330</td>
      <td>0.107273</td>
      <td>0.765038</td>
      <td>0.115930</td>
      <td>0.827066</td>
      <td>0.097549</td>
      <td>0.772371</td>
      <td>0.120196</td>
      <td>0.842853</td>
      <td>0.098182</td>
      <td>0.902993</td>
      <td>0.787624</td>
      <td>0.885914</td>
      <td>0.840083</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>12</td>
      <td>0.776941</td>
      <td>0.121486</td>
      <td>0.844236</td>
      <td>0.098766</td>
      <td>0.539900</td>
      <td>0.173792</td>
      <td>0.594831</td>
      <td>0.138391</td>
      <td>0.733338</td>
      <td>0.125470</td>
      <td>0.807863</td>
      <td>0.107121</td>
      <td>0.765973</td>
      <td>0.115498</td>
      <td>0.829749</td>
      <td>0.097586</td>
      <td>0.774413</td>
      <td>0.117592</td>
      <td>0.845434</td>
      <td>0.098287</td>
      <td>0.910509</td>
      <td>0.814430</td>
      <td>0.895639</td>
      <td>0.854433</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>13</td>
      <td>0.777486</td>
      <td>0.122354</td>
      <td>0.845469</td>
      <td>0.097756</td>
      <td>0.540423</td>
      <td>0.173680</td>
      <td>0.595823</td>
      <td>0.138030</td>
      <td>0.733312</td>
      <td>0.126547</td>
      <td>0.808849</td>
      <td>0.106790</td>
      <td>0.766112</td>
      <td>0.116066</td>
      <td>0.831941</td>
      <td>0.097668</td>
      <td>0.774284</td>
      <td>0.119142</td>
      <td>0.846257</td>
      <td>0.097330</td>
      <td>0.917403</td>
      <td>0.839686</td>
      <td>0.903921</td>
      <td>0.868113</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>14</td>
      <td>0.778543</td>
      <td>0.121466</td>
      <td>0.846764</td>
      <td>0.097379</td>
      <td>0.541023</td>
      <td>0.173471</td>
      <td>0.597136</td>
      <td>0.137479</td>
      <td>0.733936</td>
      <td>0.126083</td>
      <td>0.808767</td>
      <td>0.106147</td>
      <td>0.766328</td>
      <td>0.115995</td>
      <td>0.833044</td>
      <td>0.096754</td>
      <td>0.774824</td>
      <td>0.119319</td>
      <td>0.847064</td>
      <td>0.097783</td>
      <td>0.926443</td>
      <td>0.846139</td>
      <td>0.913125</td>
      <td>0.879223</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>15</td>
      <td>0.778179</td>
      <td>0.123308</td>
      <td>0.848025</td>
      <td>0.097582</td>
      <td>0.541655</td>
      <td>0.173400</td>
      <td>0.598420</td>
      <td>0.136740</td>
      <td>0.732994</td>
      <td>0.127554</td>
      <td>0.809356</td>
      <td>0.105963</td>
      <td>0.766156</td>
      <td>0.117493</td>
      <td>0.833934</td>
      <td>0.096525</td>
      <td>0.773586</td>
      <td>0.123478</td>
      <td>0.848252</td>
      <td>0.098001</td>
      <td>0.928749</td>
      <td>0.853293</td>
      <td>0.922402</td>
      <td>0.890927</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>16</td>
      <td>0.779150</td>
      <td>0.122339</td>
      <td>0.849950</td>
      <td>0.096151</td>
      <td>0.542227</td>
      <td>0.173202</td>
      <td>0.599895</td>
      <td>0.135823</td>
      <td>0.732933</td>
      <td>0.126593</td>
      <td>0.809805</td>
      <td>0.105658</td>
      <td>0.766681</td>
      <td>0.116600</td>
      <td>0.835037</td>
      <td>0.095401</td>
      <td>0.775316</td>
      <td>0.121470</td>
      <td>0.850068</td>
      <td>0.096208</td>
      <td>0.942054</td>
      <td>0.873569</td>
      <td>0.931664</td>
      <td>0.904417</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>17</td>
      <td>0.780041</td>
      <td>0.123234</td>
      <td>0.851446</td>
      <td>0.095295</td>
      <td>0.542658</td>
      <td>0.173199</td>
      <td>0.600836</td>
      <td>0.135408</td>
      <td>0.733883</td>
      <td>0.126713</td>
      <td>0.810052</td>
      <td>0.106018</td>
      <td>0.767192</td>
      <td>0.117092</td>
      <td>0.835949</td>
      <td>0.094850</td>
      <td>0.777900</td>
      <td>0.119096</td>
      <td>0.851380</td>
      <td>0.095023</td>
      <td>0.937361</td>
      <td>0.881253</td>
      <td>0.939809</td>
      <td>0.915556</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>18</td>
      <td>0.779762</td>
      <td>0.123017</td>
      <td>0.851511</td>
      <td>0.094602</td>
      <td>0.543257</td>
      <td>0.172999</td>
      <td>0.602323</td>
      <td>0.134773</td>
      <td>0.734157</td>
      <td>0.126664</td>
      <td>0.810147</td>
      <td>0.105141</td>
      <td>0.767051</td>
      <td>0.116541</td>
      <td>0.835558</td>
      <td>0.094443</td>
      <td>0.777794</td>
      <td>0.120095</td>
      <td>0.851343</td>
      <td>0.095111</td>
      <td>0.945055</td>
      <td>0.888360</td>
      <td>0.945875</td>
      <td>0.924933</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>19</td>
      <td>0.780699</td>
      <td>0.123055</td>
      <td>0.853112</td>
      <td>0.094358</td>
      <td>0.543815</td>
      <td>0.172802</td>
      <td>0.604948</td>
      <td>0.135125</td>
      <td>0.734473</td>
      <td>0.126754</td>
      <td>0.810668</td>
      <td>0.105734</td>
      <td>0.767984</td>
      <td>0.116448</td>
      <td>0.837606</td>
      <td>0.095184</td>
      <td>0.778350</td>
      <td>0.121243</td>
      <td>0.852971</td>
      <td>0.094523</td>
      <td>0.961867</td>
      <td>0.902539</td>
      <td>0.955128</td>
      <td>0.937001</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>20</td>
      <td>0.781275</td>
      <td>0.123613</td>
      <td>0.854134</td>
      <td>0.094141</td>
      <td>0.544351</td>
      <td>0.172543</td>
      <td>0.606489</td>
      <td>0.134851</td>
      <td>0.734586</td>
      <td>0.127032</td>
      <td>0.811013</td>
      <td>0.105632</td>
      <td>0.767909</td>
      <td>0.116525</td>
      <td>0.838624</td>
      <td>0.094571</td>
      <td>0.780315</td>
      <td>0.119437</td>
      <td>0.853753</td>
      <td>0.093965</td>
      <td>0.961450</td>
      <td>0.920022</td>
      <td>0.960916</td>
      <td>0.946304</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1</td>
      <td>21</td>
      <td>0.780997</td>
      <td>0.123208</td>
      <td>0.853767</td>
      <td>0.094892</td>
      <td>0.544881</td>
      <td>0.172390</td>
      <td>0.609927</td>
      <td>0.137158</td>
      <td>0.733743</td>
      <td>0.128463</td>
      <td>0.811182</td>
      <td>0.106715</td>
      <td>0.768044</td>
      <td>0.116623</td>
      <td>0.838443</td>
      <td>0.096304</td>
      <td>0.778974</td>
      <td>0.122642</td>
      <td>0.854112</td>
      <td>0.094787</td>
      <td>0.965607</td>
      <td>0.936575</td>
      <td>0.968276</td>
      <td>0.955352</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1</td>
      <td>22</td>
      <td>0.781337</td>
      <td>0.122563</td>
      <td>0.855022</td>
      <td>0.094638</td>
      <td>0.545496</td>
      <td>0.172203</td>
      <td>0.612574</td>
      <td>0.138446</td>
      <td>0.734275</td>
      <td>0.128186</td>
      <td>0.810792</td>
      <td>0.106602</td>
      <td>0.768786</td>
      <td>0.116759</td>
      <td>0.839463</td>
      <td>0.095789</td>
      <td>0.780620</td>
      <td>0.120256</td>
      <td>0.855304</td>
      <td>0.094516</td>
      <td>0.972502</td>
      <td>0.948584</td>
      <td>0.975156</td>
      <td>0.964216</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1</td>
      <td>23</td>
      <td>0.782186</td>
      <td>0.122218</td>
      <td>0.855967</td>
      <td>0.094355</td>
      <td>0.546022</td>
      <td>0.172105</td>
      <td>0.613986</td>
      <td>0.138465</td>
      <td>0.734357</td>
      <td>0.128291</td>
      <td>0.810650</td>
      <td>0.106795</td>
      <td>0.769263</td>
      <td>0.116599</td>
      <td>0.839781</td>
      <td>0.094765</td>
      <td>0.781737</td>
      <td>0.120048</td>
      <td>0.856124</td>
      <td>0.094551</td>
      <td>0.984534</td>
      <td>0.947184</td>
      <td>0.981883</td>
      <td>0.973967</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>24</td>
      <td>0.780565</td>
      <td>0.121588</td>
      <td>0.856128</td>
      <td>0.094233</td>
      <td>0.546578</td>
      <td>0.171938</td>
      <td>0.615497</td>
      <td>0.139099</td>
      <td>0.733199</td>
      <td>0.127802</td>
      <td>0.809758</td>
      <td>0.106415</td>
      <td>0.767814</td>
      <td>0.116044</td>
      <td>0.839362</td>
      <td>0.094516</td>
      <td>0.780714</td>
      <td>0.119572</td>
      <td>0.856528</td>
      <td>0.094361</td>
      <td>0.989778</td>
      <td>0.972028</td>
      <td>0.987950</td>
      <td>0.982775</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1</td>
      <td>25</td>
      <td>0.781206</td>
      <td>0.121805</td>
      <td>0.856114</td>
      <td>0.093871</td>
      <td>0.547068</td>
      <td>0.171682</td>
      <td>0.617044</td>
      <td>0.139303</td>
      <td>0.733543</td>
      <td>0.127902</td>
      <td>0.810746</td>
      <td>0.106995</td>
      <td>0.768011</td>
      <td>0.116053</td>
      <td>0.839728</td>
      <td>0.094445</td>
      <td>0.781098</td>
      <td>0.121358</td>
      <td>0.856464</td>
      <td>0.093926</td>
      <td>0.994321</td>
      <td>0.969140</td>
      <td>0.993898</td>
      <td>0.991463</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2</td>
      <td>1</td>
      <td>0.591542</td>
      <td>0.146240</td>
      <td>0.630270</td>
      <td>0.107845</td>
      <td>0.537603</td>
      <td>0.172186</td>
      <td>0.518646</td>
      <td>0.148770</td>
      <td>0.696048</td>
      <td>0.131738</td>
      <td>0.680896</td>
      <td>0.134705</td>
      <td>0.654350</td>
      <td>0.135542</td>
      <td>0.676189</td>
      <td>0.124554</td>
      <td>0.683987</td>
      <td>0.139258</td>
      <td>0.692117</td>
      <td>0.129857</td>
      <td>0.738985</td>
      <td>0.545849</td>
      <td>0.668546</td>
      <td>0.513315</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2</td>
      <td>2</td>
      <td>0.681608</td>
      <td>0.130218</td>
      <td>0.700833</td>
      <td>0.103088</td>
      <td>0.538358</td>
      <td>0.172096</td>
      <td>0.520102</td>
      <td>0.148629</td>
      <td>0.732124</td>
      <td>0.125322</td>
      <td>0.701808</td>
      <td>0.132279</td>
      <td>0.722181</td>
      <td>0.121593</td>
      <td>0.724381</td>
      <td>0.114869</td>
      <td>0.734942</td>
      <td>0.122451</td>
      <td>0.726198</td>
      <td>0.125571</td>
      <td>0.814297</td>
      <td>0.618141</td>
      <td>0.713301</td>
      <td>0.592290</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2</td>
      <td>3</td>
      <td>0.735057</td>
      <td>0.119321</td>
      <td>0.719101</td>
      <td>0.102186</td>
      <td>0.539236</td>
      <td>0.171845</td>
      <td>0.521587</td>
      <td>0.147964</td>
      <td>0.744691</td>
      <td>0.128598</td>
      <td>0.705483</td>
      <td>0.131617</td>
      <td>0.745688</td>
      <td>0.122710</td>
      <td>0.739974</td>
      <td>0.110986</td>
      <td>0.753357</td>
      <td>0.122067</td>
      <td>0.740376</td>
      <td>0.120181</td>
      <td>0.843234</td>
      <td>0.661567</td>
      <td>0.750063</td>
      <td>0.651097</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2</td>
      <td>4</td>
      <td>0.762499</td>
      <td>0.112714</td>
      <td>0.744778</td>
      <td>0.092716</td>
      <td>0.540110</td>
      <td>0.171634</td>
      <td>0.522763</td>
      <td>0.147761</td>
      <td>0.752866</td>
      <td>0.124779</td>
      <td>0.711776</td>
      <td>0.128136</td>
      <td>0.764193</td>
      <td>0.116163</td>
      <td>0.754159</td>
      <td>0.108038</td>
      <td>0.770503</td>
      <td>0.116667</td>
      <td>0.752923</td>
      <td>0.115020</td>
      <td>0.862513</td>
      <td>0.688011</td>
      <td>0.778307</td>
      <td>0.697004</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2</td>
      <td>5</td>
      <td>0.771498</td>
      <td>0.112566</td>
      <td>0.754959</td>
      <td>0.098147</td>
      <td>0.541015</td>
      <td>0.171424</td>
      <td>0.524062</td>
      <td>0.147164</td>
      <td>0.754982</td>
      <td>0.124723</td>
      <td>0.715751</td>
      <td>0.128441</td>
      <td>0.768350</td>
      <td>0.113264</td>
      <td>0.759394</td>
      <td>0.110785</td>
      <td>0.775635</td>
      <td>0.112842</td>
      <td>0.761310</td>
      <td>0.113588</td>
      <td>0.882222</td>
      <td>0.716743</td>
      <td>0.799622</td>
      <td>0.733351</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2</td>
      <td>6</td>
      <td>0.780869</td>
      <td>0.111822</td>
      <td>0.759730</td>
      <td>0.101034</td>
      <td>0.541750</td>
      <td>0.171156</td>
      <td>0.525414</td>
      <td>0.146659</td>
      <td>0.757465</td>
      <td>0.123981</td>
      <td>0.718535</td>
      <td>0.128961</td>
      <td>0.775579</td>
      <td>0.112147</td>
      <td>0.764087</td>
      <td>0.110834</td>
      <td>0.781260</td>
      <td>0.110567</td>
      <td>0.765516</td>
      <td>0.116583</td>
      <td>0.895121</td>
      <td>0.741126</td>
      <td>0.819481</td>
      <td>0.763933</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2</td>
      <td>7</td>
      <td>0.788350</td>
      <td>0.109440</td>
      <td>0.770140</td>
      <td>0.104786</td>
      <td>0.542544</td>
      <td>0.170835</td>
      <td>0.526753</td>
      <td>0.146359</td>
      <td>0.759834</td>
      <td>0.123099</td>
      <td>0.720293</td>
      <td>0.127150</td>
      <td>0.781847</td>
      <td>0.110852</td>
      <td>0.772636</td>
      <td>0.112015</td>
      <td>0.787152</td>
      <td>0.109409</td>
      <td>0.774254</td>
      <td>0.116615</td>
      <td>0.907193</td>
      <td>0.760594</td>
      <td>0.833957</td>
      <td>0.788764</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2</td>
      <td>8</td>
      <td>0.793952</td>
      <td>0.108848</td>
      <td>0.779418</td>
      <td>0.102105</td>
      <td>0.543461</td>
      <td>0.170699</td>
      <td>0.527965</td>
      <td>0.145912</td>
      <td>0.761780</td>
      <td>0.120393</td>
      <td>0.724466</td>
      <td>0.123738</td>
      <td>0.786475</td>
      <td>0.108567</td>
      <td>0.778645</td>
      <td>0.108445</td>
      <td>0.790618</td>
      <td>0.107642</td>
      <td>0.781350</td>
      <td>0.112396</td>
      <td>0.912979</td>
      <td>0.778474</td>
      <td>0.850882</td>
      <td>0.811806</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2</td>
      <td>9</td>
      <td>0.797555</td>
      <td>0.106605</td>
      <td>0.782598</td>
      <td>0.104484</td>
      <td>0.544094</td>
      <td>0.170555</td>
      <td>0.529053</td>
      <td>0.145675</td>
      <td>0.762348</td>
      <td>0.120166</td>
      <td>0.723671</td>
      <td>0.125561</td>
      <td>0.789006</td>
      <td>0.107622</td>
      <td>0.779948</td>
      <td>0.112619</td>
      <td>0.793655</td>
      <td>0.105734</td>
      <td>0.780005</td>
      <td>0.117260</td>
      <td>0.921770</td>
      <td>0.796995</td>
      <td>0.862893</td>
      <td>0.831380</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2</td>
      <td>10</td>
      <td>0.801219</td>
      <td>0.106785</td>
      <td>0.785850</td>
      <td>0.106428</td>
      <td>0.544936</td>
      <td>0.170390</td>
      <td>0.530243</td>
      <td>0.145367</td>
      <td>0.763461</td>
      <td>0.119825</td>
      <td>0.724222</td>
      <td>0.127275</td>
      <td>0.790533</td>
      <td>0.106912</td>
      <td>0.780975</td>
      <td>0.110411</td>
      <td>0.795971</td>
      <td>0.105783</td>
      <td>0.783810</td>
      <td>0.114588</td>
      <td>0.931188</td>
      <td>0.811683</td>
      <td>0.875680</td>
      <td>0.849390</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2</td>
      <td>11</td>
      <td>0.801237</td>
      <td>0.105232</td>
      <td>0.787452</td>
      <td>0.103716</td>
      <td>0.545730</td>
      <td>0.170002</td>
      <td>0.531682</td>
      <td>0.144813</td>
      <td>0.763925</td>
      <td>0.118052</td>
      <td>0.722239</td>
      <td>0.126003</td>
      <td>0.791584</td>
      <td>0.105590</td>
      <td>0.783947</td>
      <td>0.107316</td>
      <td>0.796934</td>
      <td>0.104414</td>
      <td>0.785879</td>
      <td>0.112583</td>
      <td>0.933908</td>
      <td>0.835216</td>
      <td>0.890356</td>
      <td>0.865136</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2</td>
      <td>12</td>
      <td>0.804469</td>
      <td>0.102615</td>
      <td>0.787778</td>
      <td>0.106700</td>
      <td>0.546435</td>
      <td>0.169755</td>
      <td>0.532715</td>
      <td>0.144387</td>
      <td>0.766312</td>
      <td>0.116829</td>
      <td>0.722761</td>
      <td>0.126166</td>
      <td>0.794126</td>
      <td>0.104086</td>
      <td>0.783698</td>
      <td>0.108822</td>
      <td>0.799997</td>
      <td>0.102253</td>
      <td>0.786353</td>
      <td>0.114791</td>
      <td>0.939955</td>
      <td>0.843758</td>
      <td>0.896600</td>
      <td>0.878039</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2</td>
      <td>13</td>
      <td>0.806084</td>
      <td>0.102909</td>
      <td>0.789856</td>
      <td>0.107843</td>
      <td>0.547164</td>
      <td>0.169714</td>
      <td>0.533745</td>
      <td>0.144026</td>
      <td>0.765321</td>
      <td>0.117784</td>
      <td>0.723287</td>
      <td>0.126341</td>
      <td>0.795219</td>
      <td>0.104781</td>
      <td>0.784983</td>
      <td>0.109782</td>
      <td>0.801531</td>
      <td>0.103206</td>
      <td>0.789540</td>
      <td>0.113465</td>
      <td>0.945315</td>
      <td>0.855980</td>
      <td>0.905151</td>
      <td>0.889660</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2</td>
      <td>14</td>
      <td>0.807224</td>
      <td>0.102230</td>
      <td>0.791661</td>
      <td>0.109264</td>
      <td>0.548062</td>
      <td>0.169611</td>
      <td>0.535015</td>
      <td>0.143735</td>
      <td>0.765628</td>
      <td>0.117873</td>
      <td>0.724112</td>
      <td>0.126632</td>
      <td>0.795734</td>
      <td>0.104200</td>
      <td>0.786120</td>
      <td>0.109943</td>
      <td>0.803109</td>
      <td>0.102535</td>
      <td>0.787758</td>
      <td>0.117134</td>
      <td>0.951927</td>
      <td>0.869993</td>
      <td>0.916537</td>
      <td>0.901580</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2</td>
      <td>15</td>
      <td>0.808708</td>
      <td>0.102220</td>
      <td>0.793233</td>
      <td>0.111159</td>
      <td>0.548790</td>
      <td>0.169418</td>
      <td>0.535988</td>
      <td>0.143467</td>
      <td>0.765018</td>
      <td>0.118286</td>
      <td>0.723929</td>
      <td>0.127140</td>
      <td>0.796273</td>
      <td>0.104485</td>
      <td>0.786857</td>
      <td>0.111696</td>
      <td>0.804738</td>
      <td>0.102991</td>
      <td>0.789655</td>
      <td>0.115987</td>
      <td>0.956408</td>
      <td>0.883894</td>
      <td>0.925187</td>
      <td>0.914627</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2</td>
      <td>16</td>
      <td>0.809889</td>
      <td>0.101177</td>
      <td>0.793988</td>
      <td>0.112332</td>
      <td>0.549529</td>
      <td>0.169198</td>
      <td>0.537083</td>
      <td>0.143091</td>
      <td>0.765791</td>
      <td>0.117886</td>
      <td>0.723992</td>
      <td>0.127231</td>
      <td>0.797768</td>
      <td>0.103497</td>
      <td>0.787208</td>
      <td>0.111077</td>
      <td>0.805735</td>
      <td>0.102432</td>
      <td>0.791705</td>
      <td>0.114895</td>
      <td>0.959708</td>
      <td>0.892418</td>
      <td>0.931015</td>
      <td>0.923989</td>
    </tr>
    <tr>
      <th>41</th>
      <td>2</td>
      <td>17</td>
      <td>0.811246</td>
      <td>0.100706</td>
      <td>0.794395</td>
      <td>0.113527</td>
      <td>0.550230</td>
      <td>0.169020</td>
      <td>0.538064</td>
      <td>0.142792</td>
      <td>0.767099</td>
      <td>0.118084</td>
      <td>0.723815</td>
      <td>0.127625</td>
      <td>0.798633</td>
      <td>0.102214</td>
      <td>0.786233</td>
      <td>0.112849</td>
      <td>0.807145</td>
      <td>0.100918</td>
      <td>0.792656</td>
      <td>0.116857</td>
      <td>0.965609</td>
      <td>0.906850</td>
      <td>0.940784</td>
      <td>0.933062</td>
    </tr>
    <tr>
      <th>42</th>
      <td>2</td>
      <td>18</td>
      <td>0.812105</td>
      <td>0.099958</td>
      <td>0.794557</td>
      <td>0.113316</td>
      <td>0.550981</td>
      <td>0.168859</td>
      <td>0.539105</td>
      <td>0.142489</td>
      <td>0.767712</td>
      <td>0.117574</td>
      <td>0.724235</td>
      <td>0.127641</td>
      <td>0.799366</td>
      <td>0.101828</td>
      <td>0.786415</td>
      <td>0.113376</td>
      <td>0.808477</td>
      <td>0.100100</td>
      <td>0.791660</td>
      <td>0.116818</td>
      <td>0.969503</td>
      <td>0.916706</td>
      <td>0.946664</td>
      <td>0.940871</td>
    </tr>
    <tr>
      <th>43</th>
      <td>2</td>
      <td>19</td>
      <td>0.812089</td>
      <td>0.099966</td>
      <td>0.796554</td>
      <td>0.112850</td>
      <td>0.551809</td>
      <td>0.168691</td>
      <td>0.539993</td>
      <td>0.142296</td>
      <td>0.768231</td>
      <td>0.117825</td>
      <td>0.725868</td>
      <td>0.128507</td>
      <td>0.799878</td>
      <td>0.101963</td>
      <td>0.787874</td>
      <td>0.112361</td>
      <td>0.809282</td>
      <td>0.100378</td>
      <td>0.794694</td>
      <td>0.116455</td>
      <td>0.974171</td>
      <td>0.930711</td>
      <td>0.954290</td>
      <td>0.950795</td>
    </tr>
    <tr>
      <th>44</th>
      <td>2</td>
      <td>20</td>
      <td>0.813403</td>
      <td>0.099638</td>
      <td>0.797013</td>
      <td>0.113343</td>
      <td>0.552658</td>
      <td>0.168402</td>
      <td>0.541105</td>
      <td>0.141928</td>
      <td>0.769217</td>
      <td>0.117646</td>
      <td>0.725974</td>
      <td>0.128104</td>
      <td>0.800811</td>
      <td>0.102088</td>
      <td>0.787981</td>
      <td>0.113461</td>
      <td>0.810971</td>
      <td>0.099733</td>
      <td>0.795123</td>
      <td>0.116250</td>
      <td>0.977365</td>
      <td>0.940954</td>
      <td>0.962239</td>
      <td>0.958514</td>
    </tr>
    <tr>
      <th>45</th>
      <td>2</td>
      <td>21</td>
      <td>0.814293</td>
      <td>0.099241</td>
      <td>0.797514</td>
      <td>0.112407</td>
      <td>0.553472</td>
      <td>0.168216</td>
      <td>0.541971</td>
      <td>0.141694</td>
      <td>0.768856</td>
      <td>0.117873</td>
      <td>0.726450</td>
      <td>0.128143</td>
      <td>0.801634</td>
      <td>0.101645</td>
      <td>0.789065</td>
      <td>0.111828</td>
      <td>0.811952</td>
      <td>0.099384</td>
      <td>0.797578</td>
      <td>0.112865</td>
      <td>0.981691</td>
      <td>0.949502</td>
      <td>0.968963</td>
      <td>0.965618</td>
    </tr>
    <tr>
      <th>46</th>
      <td>2</td>
      <td>22</td>
      <td>0.814983</td>
      <td>0.099064</td>
      <td>0.799514</td>
      <td>0.112520</td>
      <td>0.554348</td>
      <td>0.168021</td>
      <td>0.543106</td>
      <td>0.141400</td>
      <td>0.769437</td>
      <td>0.118077</td>
      <td>0.727235</td>
      <td>0.128344</td>
      <td>0.802084</td>
      <td>0.101347</td>
      <td>0.791146</td>
      <td>0.111759</td>
      <td>0.812858</td>
      <td>0.099150</td>
      <td>0.798348</td>
      <td>0.114562</td>
      <td>0.985538</td>
      <td>0.960924</td>
      <td>0.975032</td>
      <td>0.972969</td>
    </tr>
    <tr>
      <th>47</th>
      <td>2</td>
      <td>23</td>
      <td>0.815880</td>
      <td>0.098741</td>
      <td>0.800143</td>
      <td>0.112297</td>
      <td>0.555142</td>
      <td>0.167987</td>
      <td>0.544080</td>
      <td>0.141130</td>
      <td>0.769947</td>
      <td>0.117631</td>
      <td>0.727866</td>
      <td>0.128803</td>
      <td>0.803069</td>
      <td>0.101068</td>
      <td>0.791322</td>
      <td>0.112221</td>
      <td>0.814406</td>
      <td>0.098601</td>
      <td>0.801270</td>
      <td>0.113455</td>
      <td>0.989627</td>
      <td>0.971770</td>
      <td>0.981876</td>
      <td>0.980550</td>
    </tr>
    <tr>
      <th>48</th>
      <td>2</td>
      <td>24</td>
      <td>0.815813</td>
      <td>0.098672</td>
      <td>0.799463</td>
      <td>0.113632</td>
      <td>0.555916</td>
      <td>0.167852</td>
      <td>0.544877</td>
      <td>0.141048</td>
      <td>0.769065</td>
      <td>0.118381</td>
      <td>0.728233</td>
      <td>0.129000</td>
      <td>0.802450</td>
      <td>0.101120</td>
      <td>0.790849</td>
      <td>0.113676</td>
      <td>0.814643</td>
      <td>0.098440</td>
      <td>0.801112</td>
      <td>0.114152</td>
      <td>0.993227</td>
      <td>0.981146</td>
      <td>0.987904</td>
      <td>0.986985</td>
    </tr>
    <tr>
      <th>49</th>
      <td>2</td>
      <td>25</td>
      <td>0.816141</td>
      <td>0.098563</td>
      <td>0.799089</td>
      <td>0.115177</td>
      <td>0.556683</td>
      <td>0.167495</td>
      <td>0.545762</td>
      <td>0.140699</td>
      <td>0.769321</td>
      <td>0.118930</td>
      <td>0.728320</td>
      <td>0.128460</td>
      <td>0.802770</td>
      <td>0.101260</td>
      <td>0.790093</td>
      <td>0.114195</td>
      <td>0.814874</td>
      <td>0.099474</td>
      <td>0.800905</td>
      <td>0.115212</td>
      <td>0.996576</td>
      <td>0.982339</td>
      <td>0.994081</td>
      <td>0.993571</td>
    </tr>
    <tr>
      <th>50</th>
      <td>3</td>
      <td>1</td>
      <td>0.668291</td>
      <td>0.141663</td>
      <td>0.683025</td>
      <td>0.116897</td>
      <td>0.543728</td>
      <td>0.177965</td>
      <td>0.562942</td>
      <td>0.147311</td>
      <td>0.729736</td>
      <td>0.126042</td>
      <td>0.749231</td>
      <td>0.119475</td>
      <td>0.708171</td>
      <td>0.135394</td>
      <td>0.728549</td>
      <td>0.115078</td>
      <td>0.718100</td>
      <td>0.133524</td>
      <td>0.753648</td>
      <td>0.118570</td>
      <td>0.705706</td>
      <td>0.533049</td>
      <td>0.654861</td>
      <td>0.504149</td>
    </tr>
    <tr>
      <th>51</th>
      <td>3</td>
      <td>2</td>
      <td>0.734835</td>
      <td>0.127364</td>
      <td>0.742815</td>
      <td>0.106122</td>
      <td>0.544658</td>
      <td>0.178081</td>
      <td>0.565125</td>
      <td>0.146970</td>
      <td>0.761911</td>
      <td>0.123006</td>
      <td>0.779269</td>
      <td>0.117599</td>
      <td>0.758292</td>
      <td>0.123570</td>
      <td>0.772711</td>
      <td>0.109653</td>
      <td>0.765159</td>
      <td>0.122646</td>
      <td>0.791191</td>
      <td>0.111976</td>
      <td>0.780808</td>
      <td>0.598153</td>
      <td>0.693672</td>
      <td>0.575050</td>
    </tr>
    <tr>
      <th>52</th>
      <td>3</td>
      <td>3</td>
      <td>0.772320</td>
      <td>0.121300</td>
      <td>0.758285</td>
      <td>0.105066</td>
      <td>0.545827</td>
      <td>0.177699</td>
      <td>0.567753</td>
      <td>0.145773</td>
      <td>0.775013</td>
      <td>0.125498</td>
      <td>0.790272</td>
      <td>0.117424</td>
      <td>0.781136</td>
      <td>0.119985</td>
      <td>0.790752</td>
      <td>0.110302</td>
      <td>0.782754</td>
      <td>0.123446</td>
      <td>0.805338</td>
      <td>0.112143</td>
      <td>0.816702</td>
      <td>0.637691</td>
      <td>0.729698</td>
      <td>0.630109</td>
    </tr>
    <tr>
      <th>53</th>
      <td>3</td>
      <td>4</td>
      <td>0.795407</td>
      <td>0.112135</td>
      <td>0.778415</td>
      <td>0.094739</td>
      <td>0.546874</td>
      <td>0.177404</td>
      <td>0.570293</td>
      <td>0.144793</td>
      <td>0.783511</td>
      <td>0.119202</td>
      <td>0.799598</td>
      <td>0.114705</td>
      <td>0.797506</td>
      <td>0.111028</td>
      <td>0.807288</td>
      <td>0.103773</td>
      <td>0.798898</td>
      <td>0.112550</td>
      <td>0.819340</td>
      <td>0.106176</td>
      <td>0.847794</td>
      <td>0.672250</td>
      <td>0.756213</td>
      <td>0.676347</td>
    </tr>
    <tr>
      <th>54</th>
      <td>3</td>
      <td>5</td>
      <td>0.805136</td>
      <td>0.111768</td>
      <td>0.791467</td>
      <td>0.094871</td>
      <td>0.547897</td>
      <td>0.177284</td>
      <td>0.572486</td>
      <td>0.144099</td>
      <td>0.785716</td>
      <td>0.118836</td>
      <td>0.800318</td>
      <td>0.114696</td>
      <td>0.801714</td>
      <td>0.110348</td>
      <td>0.813499</td>
      <td>0.102442</td>
      <td>0.805089</td>
      <td>0.111743</td>
      <td>0.823124</td>
      <td>0.103243</td>
      <td>0.868917</td>
      <td>0.701320</td>
      <td>0.779235</td>
      <td>0.713405</td>
    </tr>
    <tr>
      <th>55</th>
      <td>3</td>
      <td>6</td>
      <td>0.808663</td>
      <td>0.113221</td>
      <td>0.799950</td>
      <td>0.096352</td>
      <td>0.548903</td>
      <td>0.177027</td>
      <td>0.574774</td>
      <td>0.143096</td>
      <td>0.787778</td>
      <td>0.119678</td>
      <td>0.803257</td>
      <td>0.114810</td>
      <td>0.803839</td>
      <td>0.111178</td>
      <td>0.818852</td>
      <td>0.101299</td>
      <td>0.807666</td>
      <td>0.112003</td>
      <td>0.827916</td>
      <td>0.103611</td>
      <td>0.881503</td>
      <td>0.727311</td>
      <td>0.797132</td>
      <td>0.744126</td>
    </tr>
    <tr>
      <th>56</th>
      <td>3</td>
      <td>7</td>
      <td>0.814560</td>
      <td>0.110911</td>
      <td>0.813836</td>
      <td>0.097856</td>
      <td>0.549801</td>
      <td>0.176649</td>
      <td>0.577053</td>
      <td>0.142074</td>
      <td>0.789109</td>
      <td>0.118224</td>
      <td>0.807912</td>
      <td>0.112246</td>
      <td>0.808645</td>
      <td>0.109108</td>
      <td>0.825867</td>
      <td>0.100467</td>
      <td>0.812407</td>
      <td>0.109487</td>
      <td>0.833392</td>
      <td>0.101895</td>
      <td>0.893225</td>
      <td>0.746260</td>
      <td>0.821599</td>
      <td>0.769709</td>
    </tr>
    <tr>
      <th>57</th>
      <td>3</td>
      <td>8</td>
      <td>0.819244</td>
      <td>0.106301</td>
      <td>0.825801</td>
      <td>0.090810</td>
      <td>0.551046</td>
      <td>0.176562</td>
      <td>0.579511</td>
      <td>0.140809</td>
      <td>0.790212</td>
      <td>0.115766</td>
      <td>0.813191</td>
      <td>0.105418</td>
      <td>0.812675</td>
      <td>0.105444</td>
      <td>0.833400</td>
      <td>0.094199</td>
      <td>0.816127</td>
      <td>0.105589</td>
      <td>0.841200</td>
      <td>0.094227</td>
      <td>0.903259</td>
      <td>0.766288</td>
      <td>0.836742</td>
      <td>0.792233</td>
    </tr>
    <tr>
      <th>58</th>
      <td>3</td>
      <td>9</td>
      <td>0.821159</td>
      <td>0.104192</td>
      <td>0.833243</td>
      <td>0.090605</td>
      <td>0.551801</td>
      <td>0.176471</td>
      <td>0.581221</td>
      <td>0.140441</td>
      <td>0.791257</td>
      <td>0.117159</td>
      <td>0.814920</td>
      <td>0.107511</td>
      <td>0.815169</td>
      <td>0.103914</td>
      <td>0.837313</td>
      <td>0.094801</td>
      <td>0.819178</td>
      <td>0.103724</td>
      <td>0.843481</td>
      <td>0.095950</td>
      <td>0.911926</td>
      <td>0.784111</td>
      <td>0.852277</td>
      <td>0.813327</td>
    </tr>
    <tr>
      <th>59</th>
      <td>3</td>
      <td>10</td>
      <td>0.822575</td>
      <td>0.105066</td>
      <td>0.837450</td>
      <td>0.090690</td>
      <td>0.552816</td>
      <td>0.176367</td>
      <td>0.583622</td>
      <td>0.139686</td>
      <td>0.793015</td>
      <td>0.115750</td>
      <td>0.814813</td>
      <td>0.108304</td>
      <td>0.816177</td>
      <td>0.104120</td>
      <td>0.839493</td>
      <td>0.094690</td>
      <td>0.821101</td>
      <td>0.103835</td>
      <td>0.845583</td>
      <td>0.095361</td>
      <td>0.922874</td>
      <td>0.804368</td>
      <td>0.867344</td>
      <td>0.831862</td>
    </tr>
    <tr>
      <th>60</th>
      <td>3</td>
      <td>11</td>
      <td>0.826054</td>
      <td>0.103916</td>
      <td>0.840231</td>
      <td>0.089993</td>
      <td>0.553657</td>
      <td>0.175710</td>
      <td>0.586526</td>
      <td>0.138376</td>
      <td>0.793568</td>
      <td>0.113389</td>
      <td>0.815738</td>
      <td>0.105666</td>
      <td>0.817723</td>
      <td>0.103050</td>
      <td>0.841188</td>
      <td>0.092787</td>
      <td>0.822870</td>
      <td>0.103568</td>
      <td>0.848347</td>
      <td>0.093413</td>
      <td>0.930972</td>
      <td>0.822778</td>
      <td>0.878935</td>
      <td>0.850626</td>
    </tr>
    <tr>
      <th>61</th>
      <td>3</td>
      <td>12</td>
      <td>0.828725</td>
      <td>0.103022</td>
      <td>0.844371</td>
      <td>0.089284</td>
      <td>0.554505</td>
      <td>0.175458</td>
      <td>0.588299</td>
      <td>0.137414</td>
      <td>0.794349</td>
      <td>0.113927</td>
      <td>0.815856</td>
      <td>0.105562</td>
      <td>0.818968</td>
      <td>0.102478</td>
      <td>0.842661</td>
      <td>0.092962</td>
      <td>0.825311</td>
      <td>0.102505</td>
      <td>0.849442</td>
      <td>0.093772</td>
      <td>0.936388</td>
      <td>0.836308</td>
      <td>0.890483</td>
      <td>0.864773</td>
    </tr>
    <tr>
      <th>62</th>
      <td>3</td>
      <td>13</td>
      <td>0.829297</td>
      <td>0.102622</td>
      <td>0.848195</td>
      <td>0.089549</td>
      <td>0.555300</td>
      <td>0.175587</td>
      <td>0.589749</td>
      <td>0.137150</td>
      <td>0.794105</td>
      <td>0.114222</td>
      <td>0.816903</td>
      <td>0.105973</td>
      <td>0.819287</td>
      <td>0.102031</td>
      <td>0.846104</td>
      <td>0.092524</td>
      <td>0.825629</td>
      <td>0.102105</td>
      <td>0.851668</td>
      <td>0.092859</td>
      <td>0.939614</td>
      <td>0.848269</td>
      <td>0.900830</td>
      <td>0.878392</td>
    </tr>
    <tr>
      <th>63</th>
      <td>3</td>
      <td>14</td>
      <td>0.829769</td>
      <td>0.102256</td>
      <td>0.850039</td>
      <td>0.088652</td>
      <td>0.556560</td>
      <td>0.175646</td>
      <td>0.592463</td>
      <td>0.136475</td>
      <td>0.793518</td>
      <td>0.114166</td>
      <td>0.818050</td>
      <td>0.105220</td>
      <td>0.819066</td>
      <td>0.101973</td>
      <td>0.848442</td>
      <td>0.091472</td>
      <td>0.826454</td>
      <td>0.102040</td>
      <td>0.854015</td>
      <td>0.091998</td>
      <td>0.948872</td>
      <td>0.861514</td>
      <td>0.909741</td>
      <td>0.890004</td>
    </tr>
    <tr>
      <th>64</th>
      <td>3</td>
      <td>15</td>
      <td>0.830905</td>
      <td>0.102702</td>
      <td>0.852920</td>
      <td>0.088875</td>
      <td>0.557480</td>
      <td>0.175498</td>
      <td>0.595611</td>
      <td>0.136031</td>
      <td>0.793341</td>
      <td>0.115110</td>
      <td>0.819832</td>
      <td>0.104646</td>
      <td>0.819622</td>
      <td>0.102332</td>
      <td>0.851053</td>
      <td>0.091382</td>
      <td>0.827922</td>
      <td>0.102490</td>
      <td>0.856524</td>
      <td>0.092884</td>
      <td>0.953577</td>
      <td>0.875542</td>
      <td>0.919293</td>
      <td>0.903994</td>
    </tr>
    <tr>
      <th>65</th>
      <td>3</td>
      <td>16</td>
      <td>0.831241</td>
      <td>0.101252</td>
      <td>0.855349</td>
      <td>0.088399</td>
      <td>0.558397</td>
      <td>0.175056</td>
      <td>0.597697</td>
      <td>0.135309</td>
      <td>0.793504</td>
      <td>0.114012</td>
      <td>0.820475</td>
      <td>0.103977</td>
      <td>0.819382</td>
      <td>0.101396</td>
      <td>0.853041</td>
      <td>0.090726</td>
      <td>0.828375</td>
      <td>0.101331</td>
      <td>0.858812</td>
      <td>0.091207</td>
      <td>0.959861</td>
      <td>0.888062</td>
      <td>0.926262</td>
      <td>0.913804</td>
    </tr>
    <tr>
      <th>66</th>
      <td>3</td>
      <td>17</td>
      <td>0.832852</td>
      <td>0.100501</td>
      <td>0.857106</td>
      <td>0.087423</td>
      <td>0.558992</td>
      <td>0.175027</td>
      <td>0.599463</td>
      <td>0.135330</td>
      <td>0.795212</td>
      <td>0.114059</td>
      <td>0.820709</td>
      <td>0.103844</td>
      <td>0.821778</td>
      <td>0.100480</td>
      <td>0.853769</td>
      <td>0.089795</td>
      <td>0.830776</td>
      <td>0.100329</td>
      <td>0.860421</td>
      <td>0.090342</td>
      <td>0.964518</td>
      <td>0.903448</td>
      <td>0.936791</td>
      <td>0.924670</td>
    </tr>
    <tr>
      <th>67</th>
      <td>3</td>
      <td>18</td>
      <td>0.834447</td>
      <td>0.099031</td>
      <td>0.857889</td>
      <td>0.086501</td>
      <td>0.560175</td>
      <td>0.174739</td>
      <td>0.602015</td>
      <td>0.135100</td>
      <td>0.796225</td>
      <td>0.112942</td>
      <td>0.821422</td>
      <td>0.103334</td>
      <td>0.822355</td>
      <td>0.099237</td>
      <td>0.854627</td>
      <td>0.088840</td>
      <td>0.831918</td>
      <td>0.099173</td>
      <td>0.861035</td>
      <td>0.089398</td>
      <td>0.967534</td>
      <td>0.913010</td>
      <td>0.944539</td>
      <td>0.933869</td>
    </tr>
    <tr>
      <th>68</th>
      <td>3</td>
      <td>19</td>
      <td>0.834931</td>
      <td>0.098498</td>
      <td>0.859066</td>
      <td>0.086306</td>
      <td>0.561324</td>
      <td>0.174681</td>
      <td>0.605118</td>
      <td>0.135191</td>
      <td>0.797121</td>
      <td>0.112713</td>
      <td>0.822838</td>
      <td>0.103767</td>
      <td>0.823070</td>
      <td>0.098908</td>
      <td>0.855489</td>
      <td>0.088511</td>
      <td>0.833122</td>
      <td>0.098631</td>
      <td>0.862521</td>
      <td>0.088793</td>
      <td>0.974073</td>
      <td>0.926390</td>
      <td>0.952921</td>
      <td>0.945376</td>
    </tr>
    <tr>
      <th>69</th>
      <td>3</td>
      <td>20</td>
      <td>0.835562</td>
      <td>0.098537</td>
      <td>0.860320</td>
      <td>0.087729</td>
      <td>0.562268</td>
      <td>0.174517</td>
      <td>0.607916</td>
      <td>0.134459</td>
      <td>0.797340</td>
      <td>0.112791</td>
      <td>0.822790</td>
      <td>0.104385</td>
      <td>0.823487</td>
      <td>0.099468</td>
      <td>0.857172</td>
      <td>0.089738</td>
      <td>0.833939</td>
      <td>0.098474</td>
      <td>0.863581</td>
      <td>0.089822</td>
      <td>0.976390</td>
      <td>0.936843</td>
      <td>0.960220</td>
      <td>0.953894</td>
    </tr>
    <tr>
      <th>70</th>
      <td>3</td>
      <td>21</td>
      <td>0.836371</td>
      <td>0.098388</td>
      <td>0.861048</td>
      <td>0.088255</td>
      <td>0.563355</td>
      <td>0.174556</td>
      <td>0.610979</td>
      <td>0.135410</td>
      <td>0.797221</td>
      <td>0.113687</td>
      <td>0.822988</td>
      <td>0.104774</td>
      <td>0.823572</td>
      <td>0.099633</td>
      <td>0.857576</td>
      <td>0.089895</td>
      <td>0.835232</td>
      <td>0.098317</td>
      <td>0.864118</td>
      <td>0.090657</td>
      <td>0.982119</td>
      <td>0.948460</td>
      <td>0.967119</td>
      <td>0.961288</td>
    </tr>
    <tr>
      <th>71</th>
      <td>3</td>
      <td>22</td>
      <td>0.836381</td>
      <td>0.096907</td>
      <td>0.861901</td>
      <td>0.087720</td>
      <td>0.564397</td>
      <td>0.174430</td>
      <td>0.614066</td>
      <td>0.135267</td>
      <td>0.797447</td>
      <td>0.113248</td>
      <td>0.823421</td>
      <td>0.104498</td>
      <td>0.823727</td>
      <td>0.098531</td>
      <td>0.858527</td>
      <td>0.088987</td>
      <td>0.835353</td>
      <td>0.096786</td>
      <td>0.864825</td>
      <td>0.089239</td>
      <td>0.985126</td>
      <td>0.958050</td>
      <td>0.974176</td>
      <td>0.970071</td>
    </tr>
    <tr>
      <th>72</th>
      <td>3</td>
      <td>23</td>
      <td>0.836682</td>
      <td>0.096461</td>
      <td>0.862118</td>
      <td>0.087625</td>
      <td>0.565158</td>
      <td>0.174430</td>
      <td>0.616028</td>
      <td>0.135560</td>
      <td>0.797806</td>
      <td>0.113140</td>
      <td>0.822525</td>
      <td>0.104432</td>
      <td>0.824177</td>
      <td>0.098049</td>
      <td>0.859196</td>
      <td>0.088496</td>
      <td>0.835944</td>
      <td>0.096392</td>
      <td>0.864946</td>
      <td>0.089124</td>
      <td>0.988733</td>
      <td>0.969597</td>
      <td>0.980836</td>
      <td>0.977873</td>
    </tr>
    <tr>
      <th>73</th>
      <td>3</td>
      <td>24</td>
      <td>0.836138</td>
      <td>0.096393</td>
      <td>0.863461</td>
      <td>0.087355</td>
      <td>0.566109</td>
      <td>0.174214</td>
      <td>0.618088</td>
      <td>0.135822</td>
      <td>0.798013</td>
      <td>0.113313</td>
      <td>0.821959</td>
      <td>0.104469</td>
      <td>0.823855</td>
      <td>0.098195</td>
      <td>0.859821</td>
      <td>0.088392</td>
      <td>0.835750</td>
      <td>0.096526</td>
      <td>0.865060</td>
      <td>0.088818</td>
      <td>0.992799</td>
      <td>0.979753</td>
      <td>0.987583</td>
      <td>0.985773</td>
    </tr>
    <tr>
      <th>74</th>
      <td>3</td>
      <td>25</td>
      <td>0.837082</td>
      <td>0.097028</td>
      <td>0.864465</td>
      <td>0.087483</td>
      <td>0.566809</td>
      <td>0.173998</td>
      <td>0.620421</td>
      <td>0.135462</td>
      <td>0.797772</td>
      <td>0.113625</td>
      <td>0.822701</td>
      <td>0.104354</td>
      <td>0.823942</td>
      <td>0.098880</td>
      <td>0.861102</td>
      <td>0.087994</td>
      <td>0.837022</td>
      <td>0.097016</td>
      <td>0.865818</td>
      <td>0.088227</td>
      <td>0.996334</td>
      <td>0.990238</td>
      <td>0.993755</td>
      <td>0.992882</td>
    </tr>
  </tbody>
</table>
</div>



## Accuracy of the Five DA-based Classifiers vs the Number of Repetitions per Class in the Target Training Set $\hat{T}$

The following figure shows the classification accuracy, over the three databases
and the three feature sets, of the three adaptive and two non-adaptive
classifiers, which are listed below. For all classifiers, the accuracy
increases as the number of repetitions in the small training sets
(1-4 repetitions per class) increase as well. The QDA classifiers' accuracy
using the feature set 1 (Hahne set) is higher than using the other two feature
sets, even than all LDA ones. In this figure, we can also notice that most
DA classifiers' accuracy using our adaptation technique is higher than using
the other approaches.

Legend of the next figure: individual classifier (orange), multi-user classifier
(violet), Liu classifier (green),
Vidovic classifier (red), and our classifier (blue).


```python
VF1.graphACC(resultsNina5,resultsCote,resultsEPN)
```


![png](output_15_0.png)


## Friedman Rank Test and Holm Post- Hoc of the DA-based Classifiers

However, to determine whether the differences between these accuracies are
statistically significant, we rank the approaches using the Friedman rank
test and the Holm post-hoc test.
The following analysis indicates the approaches' rankings, where our
classifier using small training sets is statistically significant
best-ranked with at least 95\% of confidence (p-value $<$ 0.05) in both
LDA and QDA for all feature sets.


```python
VF1.AnalysisFriedman(place,windowSize)
```

    
    
    TYPE OF DA CLASSIFIER: LDA FEATURE SET: 1
    Number of classifiers:  5 
    Number of evaluations (10(people NinaPro5) x 4(shots) + 17(people Cote) x 4(shots) 30(people EPN) x 7(shots)):  228 
    
    Should we reject H0 (i.e. is there a difference in the means) at the 95.0 % confidence level? True 
    
    Individual LDA 1: 2.3
    Multi-User LDA 1: 5.0
    Liu LDA 1: 3.1
    Vidovic LDA 1: 2.7
    Our LDA 1: 2.0
    
     The best classifier is:  Our LDA 1
                                              p   sig
    Our LDA 1 vs Multi-User LDA 1  0.000000e+00  True
    Our LDA 1 vs Liu LDA 1         1.398881e-14  True
    Our LDA 1 vs Vidovic LDA 1     9.483763e-06  True
    Our LDA 1 vs Individual LDA 1  4.721475e-02  True
    
    
    TYPE OF DA CLASSIFIER: QDA FEATURE SET: 1
    Number of classifiers:  5 
    Number of evaluations (10(people NinaPro5) x 4(shots) + 17(people Cote) x 4(shots) 30(people EPN) x 7(shots)):  228 
    
    Should we reject H0 (i.e. is there a difference in the means) at the 95.0 % confidence level? True 
    
    Individual QDA 1: 3.0
    Multi-User QDA 1: 4.9
    Liu QDA 1: 2.7
    Vidovic QDA 1: 3.0
    Our QDA 1: 1.5
    
     The best classifier is:  Our QDA 1
                                              p   sig
    Our QDA 1 vs Individual QDA 1  0.000000e+00  True
    Our QDA 1 vs Multi-User QDA 1  0.000000e+00  True
    Our QDA 1 vs Vidovic QDA 1     0.000000e+00  True
    Our QDA 1 vs Liu QDA 1         4.440892e-16  True
    
    
    TYPE OF DA CLASSIFIER: LDA FEATURE SET: 2
    Number of classifiers:  5 
    Number of evaluations (10(people NinaPro5) x 4(shots) + 17(people Cote) x 4(shots) 30(people EPN) x 7(shots)):  228 
    
    Should we reject H0 (i.e. is there a difference in the means) at the 95.0 % confidence level? True 
    
    Individual LDA 2: 3.0
    Multi-User LDA 2: 4.8
    Liu LDA 2: 2.8
    Vidovic LDA 2: 2.4
    Our LDA 2: 1.9
    
     The best classifier is:  Our LDA 2
                                              p   sig
    Our LDA 2 vs Multi-User LDA 2  0.000000e+00  True
    Our LDA 2 vs Individual LDA 2  1.605382e-13  True
    Our LDA 2 vs Liu LDA 2         9.023134e-09  True
    Our LDA 2 vs Vidovic LDA 2     9.093957e-04  True
    
    
    TYPE OF DA CLASSIFIER: QDA FEATURE SET: 2
    Number of classifiers:  5 
    Number of evaluations (10(people NinaPro5) x 4(shots) + 17(people Cote) x 4(shots) 30(people EPN) x 7(shots)):  228 
    
    Should we reject H0 (i.e. is there a difference in the means) at the 95.0 % confidence level? True 
    
    Individual QDA 2: 3.1
    Multi-User QDA 2: 4.8
    Liu QDA 2: 2.9
    Vidovic QDA 2: 2.4
    Our QDA 2: 1.8
    
     The best classifier is:  Our QDA 2
                                              p   sig
    Our QDA 2 vs Individual QDA 2  0.000000e+00  True
    Our QDA 2 vs Multi-User QDA 2  0.000000e+00  True
    Our QDA 2 vs Liu QDA 2         1.687539e-14  True
    Our QDA 2 vs Vidovic QDA 2     2.603017e-05  True
    
    
    TYPE OF DA CLASSIFIER: LDA FEATURE SET: 3
    Number of classifiers:  5 
    Number of evaluations (10(people NinaPro5) x 4(shots) + 17(people Cote) x 4(shots) 30(people EPN) x 7(shots)):  228 
    
    Should we reject H0 (i.e. is there a difference in the means) at the 95.0 % confidence level? True 
    
    Individual LDA 3: 3.0
    Multi-User LDA 3: 4.9
    Liu LDA 3: 2.7
    Vidovic LDA 3: 2.5
    Our LDA 3: 2.0
    
     The best classifier is:  Our LDA 3
                                              p   sig
    Our LDA 3 vs Multi-User LDA 3  0.000000e+00  True
    Our LDA 3 vs Individual LDA 3  4.351253e-11  True
    Our LDA 3 vs Liu LDA 3         8.846653e-07  True
    Our LDA 3 vs Vidovic LDA 3     4.486680e-04  True
    
    
    TYPE OF DA CLASSIFIER: QDA FEATURE SET: 3
    Number of classifiers:  5 
    Number of evaluations (10(people NinaPro5) x 4(shots) + 17(people Cote) x 4(shots) 30(people EPN) x 7(shots)):  228 
    
    Should we reject H0 (i.e. is there a difference in the means) at the 95.0 % confidence level? True 
    
    Individual QDA 3: 3.3
    Multi-User QDA 3: 4.8
    Liu QDA 3: 2.5
    Vidovic QDA 3: 2.9
    Our QDA 3: 1.4
    
     The best classifier is:  Our QDA 3
                                              p   sig
    Our QDA 3 vs Individual QDA 3  0.000000e+00  True
    Our QDA 3 vs Multi-User QDA 3  0.000000e+00  True
    Our QDA 3 vs Vidovic QDA 3     0.000000e+00  True
    Our QDA 3 vs Liu QDA 3         2.056133e-13  True
    

## Large training sets

We also evaluate our DA classifier using large training sets
(1-25 repetitions per class) from the EPN database using the Hahne feature set,
as shown in the following figure. The multi-user classifier is removed from
this evaluation because its accuracy is the lowest in all databases using any
of the three feature sets (see the first figure).

Using small training sets, our classifier's accuracy is higher than the
individual classifier's one. However, when the number of repetitions per
class increase (large training sets), our classifier's accuracy converges
to the individual classifier's one.
This convergence is caused by the confidence coefficient $\kappa$, which reduces logarithmically as the
training sets increase. When this coefficient tends to be zero, the weights
$\hat{\omega}_c$ and $\hat{\lambda}_c$ for both LDA and QDA tends to be one,
which means our classifier tends to be equal to individual
classifier.
Contrarily, this figure shows the accuracy of the Liu and Vidovic
classifiers trained on large data sets is lower than the individual
classifier's one.

Legend of the next figure: individual classifier (orange), Liu classifier (green),
Vidovic classifier (red), and our classifier (blue).


```python
VF1.largeDatabase(resultsEPN,featureSet=1)
```


![png](output_20_0.png)


## Response Time and Training Time of Our Myoelectric Interface using Our Adaptation Technique

Moreover, we calculate the data-analysis time of the real-time  adaptive
myoelectric interface using our classifier, which is the sum of the
feature-extraction, pre-processing,
and classification times. The following analysis and figure show the data-analysis
time of our classifier, which is less than 5ms in the three databases for
both LDA and QDA. Therefore, our classifier is suitable for real-time
interfaces because the sum of the data-acquisition time (window of 295ms)
and data-analysis time is less than 300 ms in all cases. For both LDA and
QDA, this fugure also shows our classifier's training times,
which are less than 5 minutes in the three databases. All calculated times
were carried out on a desktop computer with an Intel® Core™ i5-8250U
processor and 8GB of RAM.

Nina Pro 5 database


```python
VF1.analysisTime(extractionTimeNina5, timeNina5)
```

    
    Our LDA Technique for feature set 1
    Feature set: 1
    Training Time [s]:  3.7 ± 0.2
    Extraction time [ms]:  0.75 ± 0.36
    Classification time [ms]:  1.08 ± 0.02
    Prprocessing time (min-max normalization) [µs]:  0.03 ± 0.01
    Analysis time (the sum of the extraction, classification, and preprocessing times) [ms]:  1.83 ± 0.36
    
    Our QDA Technique for feature set 1
    Feature set: 1
    Training Time [s]:  3.3 ± 0.1
    Extraction time [ms]:  0.75 ± 0.36
    Classification time [ms]:  1.69 ± 0.05
    Prprocessing time (min-max normalization) [µs]:  0.03 ± 0.01
    Analysis time (the sum of the extraction, classification, and preprocessing times) [ms]:  2.44 ± 0.36
    
    Our LDA Technique for feature set 2
    Feature set: 2
    Training Time [s]:  4.6 ± 1.1
    Extraction time [ms]:  1.0 ± 0.47
    Classification time [ms]:  1.6 ± 0.02
    Prprocessing time (min-max normalization) [µs]:  0.12 ± 0.03
    Analysis time (the sum of the extraction, classification, and preprocessing times) [ms]:  2.6 ± 0.47
    
    Our QDA Technique for feature set 2
    Feature set: 2
    Training Time [s]:  4.5 ± 0.1
    Extraction time [ms]:  1.0 ± 0.47
    Classification time [ms]:  2.28 ± 0.03
    Prprocessing time (min-max normalization) [µs]:  0.12 ± 0.03
    Analysis time (the sum of the extraction, classification, and preprocessing times) [ms]:  3.28 ± 0.47
    
    Our LDA Technique for feature set 3
    Feature set: 3
    Training Time [s]:  3.3 ± 0.1
    Extraction time [ms]:  2.38 ± 0.86
    Classification time [ms]:  1.58 ± 0.02
    Prprocessing time (min-max normalization) [µs]:  0.12 ± 0.02
    Analysis time (the sum of the extraction, classification, and preprocessing times) [ms]:  3.96 ± 0.86
    
    Our QDA Technique for feature set 3
    Feature set: 3
    Training Time [s]:  4.5 ± 0.1
    Extraction time [ms]:  2.38 ± 0.86
    Classification time [ms]:  2.27 ± 0.03
    Prprocessing time (min-max normalization) [µs]:  0.12 ± 0.02
    Analysis time (the sum of the extraction, classification, and preprocessing times) [ms]:  4.65 ± 0.86
    

Cote-Allard database


```python
VF1.analysisTime(extractionTimeCote, timeCote)
```

    
    Our LDA Technique for feature set 1
    Feature set: 1
    Training Time [s]:  0.5 ± 0.1
    Extraction time [ms]:  0.39 ± 0.32
    Classification time [ms]:  0.82 ± 0.45
    Prprocessing time (min-max normalization) [µs]:  0.11 ± 0.39
    Analysis time (the sum of the extraction, classification, and preprocessing times) [ms]:  1.21 ± 0.32
    
    Our QDA Technique for feature set 1
    Feature set: 1
    Training Time [s]:  0.8 ± 0.4
    Extraction time [ms]:  0.39 ± 0.32
    Classification time [ms]:  1.33 ± 0.71
    Prprocessing time (min-max normalization) [µs]:  0.11 ± 0.39
    Analysis time (the sum of the extraction, classification, and preprocessing times) [ms]:  1.72 ± 0.32
    
    Our LDA Technique for feature set 2
    Feature set: 2
    Training Time [s]:  0.7 ± 0.1
    Extraction time [ms]:  0.6 ± 0.41
    Classification time [ms]:  0.96 ± 0.43
    Prprocessing time (min-max normalization) [µs]:  0.2 ± 0.35
    Analysis time (the sum of the extraction, classification, and preprocessing times) [ms]:  1.56 ± 0.41
    
    Our QDA Technique for feature set 2
    Feature set: 2
    Training Time [s]:  0.9 ± 0.4
    Extraction time [ms]:  0.6 ± 0.41
    Classification time [ms]:  1.32 ± 0.62
    Prprocessing time (min-max normalization) [µs]:  0.2 ± 0.35
    Analysis time (the sum of the extraction, classification, and preprocessing times) [ms]:  1.92 ± 0.41
    
    Our LDA Technique for feature set 3
    Feature set: 3
    Training Time [s]:  0.8 ± 0.1
    Extraction time [ms]:  1.28 ± 0.6
    Classification time [ms]:  0.99 ± 0.65
    Prprocessing time (min-max normalization) [µs]:  0.18 ± 0.21
    Analysis time (the sum of the extraction, classification, and preprocessing times) [ms]:  2.27 ± 0.6
    
    Our QDA Technique for feature set 3
    Feature set: 3
    Training Time [s]:  0.8 ± 0.5
    Extraction time [ms]:  1.28 ± 0.6
    Classification time [ms]:  1.24 ± 0.58
    Prprocessing time (min-max normalization) [µs]:  0.18 ± 0.21
    Analysis time (the sum of the extraction, classification, and preprocessing times) [ms]:  2.52 ± 0.6
    

EPN database


```python
VF1.analysisTime(extractionTimeEPN, timeEPN)
```

    
    Our LDA Technique for feature set 1
    Feature set: 1
    Training Time [s]:  0.3 ± 0.1
    Extraction time [ms]:  0.24 ± 0.31
    Classification time [ms]:  0.62 ± 0.13
    Prprocessing time (min-max normalization) [µs]:  0.09 ± 0.3
    Analysis time (the sum of the extraction, classification, and preprocessing times) [ms]:  0.86 ± 0.31
    
    Our QDA Technique for feature set 1
    Feature set: 1
    Training Time [s]:  0.5 ± 0.1
    Extraction time [ms]:  0.24 ± 0.31
    Classification time [ms]:  0.9 ± 0.23
    Prprocessing time (min-max normalization) [µs]:  0.09 ± 0.3
    Analysis time (the sum of the extraction, classification, and preprocessing times) [ms]:  1.14 ± 0.31
    
    Our LDA Technique for feature set 2
    Feature set: 2
    Training Time [s]:  0.4 ± 0.1
    Extraction time [ms]:  0.44 ± 0.52
    Classification time [ms]:  0.83 ± 0.23
    Prprocessing time (min-max normalization) [µs]:  0.28 ± 1.03
    Analysis time (the sum of the extraction, classification, and preprocessing times) [ms]:  1.27 ± 0.52
    
    Our QDA Technique for feature set 2
    Feature set: 2
    Training Time [s]:  0.6 ± 0.2
    Extraction time [ms]:  0.44 ± 0.52
    Classification time [ms]:  1.08 ± 0.35
    Prprocessing time (min-max normalization) [µs]:  0.28 ± 1.03
    Analysis time (the sum of the extraction, classification, and preprocessing times) [ms]:  1.52 ± 0.52
    
    Our LDA Technique for feature set 3
    Feature set: 3
    Training Time [s]:  0.4 ± 0.1
    Extraction time [ms]:  0.89 ± 0.57
    Classification time [ms]:  0.81 ± 0.23
    Prprocessing time (min-max normalization) [µs]:  0.24 ± 0.52
    Analysis time (the sum of the extraction, classification, and preprocessing times) [ms]:  1.7 ± 0.57
    
    Our QDA Technique for feature set 3
    Feature set: 3
    Training Time [s]:  0.6 ± 0.2
    Extraction time [ms]:  0.89 ± 0.57
    Classification time [ms]:  1.08 ± 0.35
    Prprocessing time (min-max normalization) [µs]:  0.24 ± 0.52
    Analysis time (the sum of the extraction, classification, and preprocessing times) [ms]:  1.97 ± 0.57
    

Feature-extraction (blue), pre-processing (orange),
classification (green), and training (red) times of the
myoelectric interface that uses our technique.


```python
VF1.analysisTimeTotal(extractionTimeNina5, timeNina5, extractionTimeCote, timeCote, extractionTimeEPN,
                      timeEPN)
```

    Training Time [min] [3.86495206 4.07409078 0.6581855  0.852251   0.36578678 0.57611922]
    Training Time [min] STD [0.46510233 0.06966617 0.10198678 0.41811383 0.07265967 0.13334711] 
    
    Feature extraction Time [ms] [1.37704812 1.37704812 0.7568258  0.7568258  0.52298427 0.52298427]
    Feature extraction Time [ms] STD [0.56352801 0.56352801 0.44635791 0.44635791 0.46574528 0.46574528]
    Pre-processing Time [ms] [9.00000000e-05 9.00000000e-05 1.63333333e-04 1.63333333e-04
     2.03333333e-04 2.03333333e-04]
    Pre-processing Time [ms] STD [0.02       0.02       0.31666667 0.31666667 0.61666667 0.61666667]
    Classification Time [ms] [1.42       2.08       0.92333333 1.29666667 0.75333333 1.02      ]
    Classification Time [ms] STD [0.02       0.03666667 0.51       0.63666667 0.19666667 0.31      ]
    Data-Analysis Time [ms] [2.79713812 3.45713812 1.68032247 2.0536558  1.27652094 1.5431876 ]
    Data-Analysis Time [ms] STD [0.56352801 0.56352801 0.44635791 0.44635791 0.46574528 0.46574528] 
    
    


![png](output_29_1.png)

