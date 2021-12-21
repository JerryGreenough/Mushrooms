# Mushrooms
<strong>Purpose:</strong> Predicting the features that are most informative in determining the edibility of mushrooms.

<p>The objective of this study is to explore the information value of features contained in the well known 'mushroom' dataset. 
The sample mushroom data can be sourced (together with a description of its content) from the following site:</p>

<a href = "https://www.kaggle.com/uciml/mushroom-classification">https://www.kaggle.com/uciml/mushroom-classification</a>

<p> This study demonstrates that only a small subset of the data's features (5 out of 20) is required 
to achieve a true positive rate (or recall) of 99.5% for predictions made by an ANN neural network that has been
trained using a portion of the data. In this instance, a true positive refers to an accurate predicition by a properly trained 
neural network of a mushroom being poisonous. </p>
    
<p>The data itself is contained in a .csv file:
    
```mushrooms.csv```

A description of the procedures and the mathematics that have been employed during this study are presented in the
sections that follow. The Phython source code that has been used to normalize and analyze the data
is contained in a Jupyter notebook: 
    
```mushrooms.ipynb``` 
    
The Python source code that has been used to both train as well as infer from the neural networks makes use of the
PyTorch library and is also contained in a Jupyter notebook:
    
```mushroom_predictor.ipynb``` 

</p>

<p align="center">
    <img src="https://raw.githubusercontent.com/JerryGreenough/Mushrooms/master/images/fly_agaric.jpg" width="782" height="444">  
</p>

<p align="center">
    <strong><small>Fly Agaric Mushroom</small></strong>
</p>

## Data Exploration

<p>The data consists of 8124 observations of mushrooms (of varying species), with each observation being a vector of
categorical values associated with the following features:-</p>

<ul>
<li>cap-shape, cap-surface, cap-color</li>
<li>veil-type, veil-color</li>
<li>gill-attachment, gill-spacing, gill-size, gill-color</li> 
<li>ring-number, ring-type</li>
<li>stalk-shape, stalk-root, stalk-surface-above-ring, stalk-surface-below-ring, stalk-color-above-ring, stalk-color-below-ring</li>
<li>spore-print-color</li>
<li>poulation</li>
<li>habitat</li>
<li>bruises</li>
<li>odor</li>
</ul>

<p>Each observation is also associated with a response variable 'class' which 
attains a value 'e' for edible and 'p' for poisonous. A portion of the dataset may be used to train
a machine learning model to use the feature values of a mushroom sample to predict whether the sample
is edible or poisonous. To this end, the study presented here seeks to demonstrate that certain of the
above listed features have more informational value than others - it will be seen that the edibility of a mushroom could
be characterized by just three of those features.</p>

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
<img <img align="center" src="https://render.githubusercontent.com/render/math?math=p_j">
is the relative frequency of category
<img align="center" src="https://render.githubusercontent.com/render/math?math=C_j"> 
within the dataset and 
<img align="center" src="https://render.githubusercontent.com/render/math?math=H_T(C_j)"> 
is the entropy of the data contained in 
<img align="center" src="https://render.githubusercontent.com/render/math?math=C_j"> 
with respect to the target variable
<img src="https://render.githubusercontent.com/render/math?math=T">. 
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

<p>The information gain (with respect to our response variable 'class') has been calculated for each of the features listed in the introduction and presented in
the following bar chart. The top three informative features appear to be 'gill-color', 'spore-print-color', and 'odor'.</p>
<p align="center">
    <img src="https://raw.githubusercontent.com/JerryGreenough/Mushrooms/master/images/information_gain.png" width="386" height="376">  
</p>


## Analyzing the Effect of Dropping Uninformative Features

<p>The normalized observation data will be used (as part of a separate study) to train a neural network in order 
to predict from observed feature values whether a mushroom is edible or poisonous. An experiment is performed
in order to determine how parsimonious the neural network's training data need be. 
To this end, a series of normalized datasets is constructed, with each dataset progressively making use of fewer of the least informative features. 
The datasets are saved in '.csv' fomat and are named using the following convention: </p>
    
```mushrooms_one_hot_encoded_partial_n.csv```
    
<p>in which the
<img src="https://render.githubusercontent.com/render/math?math=n">
denotes that the 
<img src="https://render.githubusercontent.com/render/math?math=n%2B1">
least informative features have been dropped from the original dataset. </p>

<p>A custom neural network is built and trained for each dataset. In order to assess how well the neural network has been trained, 
we can look at the loss function value 
In this instance, the problem that we seek to solve is binary classification, i.e. is the mushroom edible or not ? Accordingly,
the binary cross entropy loss function is used to assess whether the probability score produced by the network (a floating point value between 0.0 and 1.0) is close to the
value of the 'class' label in the hold out dataset (a categorical value which is either 0 or 1).</p>
    
<p>The following graph of test loss vs. number of dropped features suggests that convergence of the neural network's training algorithm
is only adversely affected after the first fifteen least informative features have been dropped from the dataset.</p>
<p align="center">
    <img src="https://raw.githubusercontent.com/JerryGreenough/Mushrooms/master/images/test_loss.png" width="432" height="288">  
</p>

<p>The recall score that is obtained by assessing the predictive performance 
of a trained neural network on hold-out (test) data is then used to assess whether reducing the number of uninformative features 
has a deleterious effect on the neural network's ability to predict whether a mushroom is posinonous or not. </p>

<p>For the purpose of this study, the recall score is given by
<img src="https://render.githubusercontent.com/render/math?math=R">
as follows: </p>

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=R={{TP}\over{{TP%20%2B%20FE}}}">
</p>

<p> where
<img src="https://render.githubusercontent.com/render/math?math=TP">
refers to the number of test observations that the neural network correctly classified as poisonous and
<img src="https://render.githubusercontent.com/render/math?math=FE">
refers to the number of test observations that the neural network incorrectly classified as edible. The rationale behind using the recall measure of 
model assessment is that we wish to predict as many
poisonous mushrooms as possible from those that are observed to be poisonous (namely
<img src="https://render.githubusercontent.com/render/math?math=TP%2BFE">
). </p>

<p>The following graph of recall vs. the number of dropped features demonstrates that there is little to be gained from including
the fifteen least informative features when attepting to predict mushroom edibility using a pre-trained neural network model.</p>
<p align="center">
    <img src="https://raw.githubusercontent.com/JerryGreenough/Mushrooms/master/images/recall.png" width="432" height="288">  
</p>






