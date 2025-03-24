# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A Logistical Regression model was used   
Model was trained on 1994 Census Data  
The goal was to predict whether a person makes over 50k a year based on other variables (age, sex, occupation, etc)  

## Intended Use
For educational purposes through Udacity  
To learn about ML deployment in pipelines and to predict the salary category of a person based on their features 

## Training Data
Census Data from 1994 was used. There are 14 Features and data is categorical and numerical

## Evaluation Data
20% of data was used to evaluate.

## Metrics
Evaluation of 3 metrics were tested. Precision: 0.7419 | Recall: 0.6384 | F1: 0.6863

## Ethical Considerations
Outliers were not considered.

## Caveats and Recommendations
This data set is old and may not reflect current earning potential.