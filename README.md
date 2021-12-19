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


The information gain 
<img src="https://render.githubusercontent.com/render/math?math=I_T(C)">    
with respect to the target variable that is afforded by variable 
<img src="https://render.githubusercontent.com/render/math?math=C">
is defined as follows:
 
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=I_T(C)=H_T-S_T(C)">
</p>

<p>The objective of finding the most informative variable is tantamount 
to establishing which variable $C = C_{max}$ maximizes the information gain $I_T(C)$.</p>

```
import math
import numpy as np
import pandas as pd
```
