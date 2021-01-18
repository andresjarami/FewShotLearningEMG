# Experiment 3
We compare the real-time myoelectric interface
(see section 3) using our technique (see section 2)
and the interface proposed by
[Côté-Allard et al.](https://github.com/UlysseCoteAllard/MyoArmbandDataset),
which uses CNN with transfer learning.

This CNN's scheme comprises two networks: source and target,
which share information through an element-wise summation in
a layer-by-layer fashion. Each network has three convolutional
and two fully connected layers.

Transfer learning occurs by freezing the source network's
weights (learned using source users' data) during the target
network's training on the target user's data.

The Côté-Allard approach uses continuous wavelet transform
to extract time-frequency information from windows of 260ms.

The following figure illustrates the classification accuracy
of the Côté-Allard approach and our classifier
(both LDA and QDA) using the three publicly databases,
the Hahne feature set, and two windows (295ms and 260ms).

*Import the library developed to visualize the results*


```python
import Experiments.Experiment3.VisualizationFunctions as VF3
```

Legend of the next figure: our QDA classifier 295ms (blue triangle markers),
our LDA classifier 295ms (blue circle markers),
our QDA classifier 260ms (orange triangle markers),
our LDA classifier 260ms (orange circle markers), and
Côté-Allard approach 260ms (orange x markers).


```python
placeOur260='Experiments/Experiment3/results/'
placeOur295='Experiments/Experiment1/results/'
placeCote='Experiments/Experiment3/'
featureSet=1
VF3.AnalysisCote(placeOur260,placeOur295, placeCote,featureSet)
```


![png](output_4_0.png)


The Côté-Allard approach's performance is lower than of our
QDA classifiers (260ms and 295ms) for the three databases.
Using the Friedman rank test and the Holm post-hoc test,
our QDA classifier using windows of 295ms are best-ranked
(ranking 1.9) than the Côté-Allard approach (ranking 2.8)
with 99\% of confidence.

Our approach's data-analysis time (less than 5ms),
as shown in experiment 1, is significantly shorter than
the Côté-Allard approach's one, which is 22.5$\pm$1.2ms
(using a NVIDIA V100 GPU). Therefore, our approach is suitable
for the frequent training of real-time myoelectric interfaces
because it can analyze more sEMG data per window (295ms),
unlike the Côté-Allard approach (260ms), improving
classification accuracy (see previous figure).


```python
VF3.AnalysisFriedman(placeOur260,placeOur295,placeCote,featureSet)
```

    
    
    ANALYSIS OF WINDOW 260ms
    Number of classifiers:  3 
    Number of evaluations (10(people NinaPro5) x 4(shots) + 17(people Cote) x 4(shots) 30(people EPN) x 7(shots)):  228 
    
    Should we reject H0 (i.e. is there a difference in the means) at the 95.0 % confidence level? True 
    
    vectOurLDA260: 2.6
    vectOurQDA260: 1.6
    vectCote: 1.8
    
     The best classifier is:  vectOurQDA260
                                           p   sig
    vectOurQDA260 vs vectOurLDA260  0.000000  True
    vectOurQDA260 vs vectCote       0.023134  True
    
    
    ANALYSIS OF WINDOW 295ms
    Number of classifiers:  3 
    Number of evaluations (10(people NinaPro5) x 4(shots) + 17(people Cote) x 4(shots) 30(people EPN) x 7(shots)):  228 
    
    Should we reject H0 (i.e. is there a difference in the means) at the 95.0 % confidence level? True 
    
    vectOurLDA295: 2.5
    vectOurQDA295: 1.5
    vectCote: 2.0
    
     The best classifier is:  vectOurQDA295
                                               p   sig
    vectOurQDA295 vs vectOurLDA295  0.000000e+00  True
    vectOurQDA295 vs vectCote       9.897734e-07  True
    
    
    ANALYSIS BOTH WINDOWS (260ms AND 295ms)
    Number of classifiers:  5 
    Number of evaluations (10(people NinaPro5) x 4(shots) + 17(people Cote) x 4(shots) 30(people EPN) x 7(shots)):  228 
    
    Should we reject H0 (i.e. is there a difference in the means) at the 95.0 % confidence level? True 
    
    vectOurLDA260: 4.4
    vectOurQDA260: 2.6
    vectOurLDA295: 3.4
    vectOurQDA295: 1.9
    vectCote: 2.8
    
     The best classifier is:  vectOurQDA295
                                               p   sig
    vectOurQDA295 vs vectOurLDA260  0.000000e+00  True
    vectOurQDA295 vs vectOurLDA295  0.000000e+00  True
    vectOurQDA295 vs vectCote       1.202935e-09  True
    vectOurQDA295 vs vectOurQDA260  1.190208e-06  True
    
