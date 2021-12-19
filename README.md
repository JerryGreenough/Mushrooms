# Mushrooms
<strong>Purpose:</strong> Predicting the Comestibility of Mushrooms

The purpose of this project is .

<p align="center">
    <img src="https://raw.githubusercontent.com/JerryGreenough/Mushrooms/master/images/fly_agaric.jpg" width="782" height="444">  
</p>

<p align="center">
    <strong><small>Fly Agaric Mushroom</small></strong>
</p>

## Informational Value

<p>Which of the dependent variables have high informational value when used to 
predict the outcome of the response/target variable? We will measure the information gain (with
respect to the target) as a means of assessing whether partitioning a dataset using the values of a
given independent variable is any more informative than partitioning the dataset based on the values
of another independent variable.</p>

<p>The entropy of a dataset with respect to a categorical target variable
<img src="https://render.githubusercontent.com/render/math?math=T">  
is given by:</p>

<p align="center">
<img align="center" src="https://render.githubusercontent.com/render/math?math=H_T=-\sum_{i=1}^{n_T}p_i%20log(p_i)">  
</p>

<p>where
<img align="center" src="https://render.githubusercontent.com/render/math?math=n_T"> 
is the cardinality (number of categories) of the variable
<img src="https://render.githubusercontent.com/render/math?math=T">
and
<img align="center" src="https://render.githubusercontent.com/render/math?math=p_i">  
is the relative frequency category
<img src="https://render.githubusercontent.com/render/math?math=i">
.</p>
<p>The split entropy, 
<img align="center" src="https://render.githubusercontent.com/render/math?math=S_T(C)">
incurred by assessing the entropy of the datset when partitioned based on values
of categorical variable
<img src="https://render.githubusercontent.com/render/math?math=C">
is given by:</p>

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=S_T(C)=\sum_{j=1}^{n_C}%20p_j%20H_T(C_j)">
</p>
    
<p>where
<img align="center" src="https://render.githubusercontent.com/render/math?math=n_C">
is the cardinality of the variable
<img src="https://render.githubusercontent.com/render/math?math=C">
,
<img align="center" src="https://render.githubusercontent.com/render/math?math=p_j">
is the relative frequency of category
<img align="center" src="https://render.githubusercontent.com/render/math?math=C_j"> 
within the dataset and 
<img align="center" src="https://render.githubusercontent.com/render/math?math=H_T(C_j)"> 
is the entropy of the data contained in 
<img align="center" src="https://render.githubusercontent.com/render/math?math=C_j"> 
with respect to the target variable
<img src="https://render.githubusercontent.com/render/math?math=T">
. 
In essence, the split entropy for a given feature is the sum of the weighted entropies for each set of observations that
is created by partitioning the observations based on the feature's values.</p>

<p>The information gain: 
<img align="center" src="https://render.githubusercontent.com/render/math?math=I_T(C)">    
with respect to the target variable that is afforded by variable 
<img src="https://render.githubusercontent.com/render/math?math=C">
is defined as follows: </p>
 
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=I_T(C)=H_T-S_T(C)">
</p>

<p>In order to find the most informative variable we have to determine
which variable
<img align="center" src="https://render.githubusercontent.com/render/math?math=C=C_{max}">     
maximizes the information gain
<img align="center" src="https://render.githubusercontent.com/render/math?math=I_T(C)">
, which is tantamount to finding the variable that minimizes the split entropy
<img align="center" src="https://render.githubusercontent.com/render/math?math=S_T(C)"> 
.</p>

```
import math
import numpy as np
import pandas as pd
```
