# Experiment 3

In this experiment, we compare the proposed adaptive interface’s accuracy using our technique and an adaptive myoelectric interface using CNN and transfer learning, proposed by [Côté-Allard et al.](https://github.com/UlysseCoteAllard/MyoArmbandDataset). Côté-Allard interface assists the CNN training for a user taking advantage of a CNN trained using EMG data from the source users. The following table shows the accuracy of our adaptive interface and the Côté-Allard interface, and the difference between them. In this table, the differences were calculated as experiment 1 using the one-tail Wilcoxon signed-rank test. The only negative difference is not statistically significant and six differences are positive and statistically significant. Consequently, our interface’s accuracy is higher than or equal to the Côté-Allard interface’s accuracy.



## Import the library developed to visualize the results


```python
import Experiments.Experiment3.VisualizationFunctions as VF3
```

## Accuracy Comparison in percentage between Côté-Allard interface and Our QDA interface using the Feature Set 1 and the Three Database


```python
placeCoteResults="Experiments/Experiment3/"
placeOurTechniqueResults="Experiments/Experiment1/ResultsExp1/" 
VF3.AnalysisCote(placeOurTechniqueResults,placeCoteResults)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Our Interface's accuracy [%] over database: NinaPro5</th>
      <td>50.560000</td>
      <td>57.500000</td>
      <td>63.060000</td>
      <td>65.180000</td>
    </tr>
    <tr>
      <th>Cote Interface's accuracy [%] over database: NinaPro5</th>
      <td>44.760000</td>
      <td>55.200000</td>
      <td>59.870000</td>
      <td>62.520000</td>
    </tr>
    <tr>
      <th>Difference [%] over database: NinaPro5</th>
      <td>5.810000</td>
      <td>2.310000</td>
      <td>3.190000</td>
      <td>2.670000</td>
    </tr>
    <tr>
      <th>p-value over database: NinaPro5</th>
      <td>0.002531</td>
      <td>0.002531</td>
      <td>0.002531</td>
      <td>0.004672</td>
    </tr>
    <tr>
      <th>Our Interface's accuracy [%] over database: Cote</th>
      <td>96.530000</td>
      <td>99.200000</td>
      <td>99.410000</td>
      <td>99.090000</td>
    </tr>
    <tr>
      <th>Cote Interface's accuracy [%] over database: Cote</th>
      <td>96.130000</td>
      <td>99.000000</td>
      <td>99.000000</td>
      <td>99.430000</td>
    </tr>
    <tr>
      <th>Difference [%] over database: Cote</th>
      <td>0.400000</td>
      <td>0.200000</td>
      <td>0.410000</td>
      <td>-0.340000</td>
    </tr>
    <tr>
      <th>p-value over database: Cote</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Our Interface's accuracy [%] over database: EPN</th>
      <td>74.930000</td>
      <td>82.600000</td>
      <td>82.310000</td>
      <td>83.140000</td>
    </tr>
    <tr>
      <th>Cote Interface's accuracy [%] over database: EPN</th>
      <td>68.490000</td>
      <td>77.760000</td>
      <td>79.360000</td>
      <td>82.300000</td>
    </tr>
    <tr>
      <th>Difference [%] over database: EPN</th>
      <td>6.430000</td>
      <td>4.840000</td>
      <td>2.950000</td>
      <td>0.830000</td>
    </tr>
    <tr>
      <th>p-value over database: EPN</th>
      <td>0.000041</td>
      <td>0.003418</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


