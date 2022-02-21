# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data about individuals applying for bank loans. The task we set out to accomplish here is to develop a model that, based on the information provided about each individual, predicts whether they will subscribe to a service.

The best performing model was found to be a voting ensemble with 91.66% accuracy. However, many models were of comparable accuracy.

## Scikit-learn Pipeline
The Scikit-learn pipeline obtains the provided data from the provided URL. Following data download, a number of data cleaning steps are carried out. We remove NAs from the dataset, then perform feature engineering like one-hot encoding of categorical variables including target variable. 

Once the data has been prepared, we split the data into train and test sets. For the model built using Python SDK, we use logistic regression as the classification method and parameters such as "C" and "maximum number of iterations" are tuned using Hyperdrive.

#### Parameter sampling
Random parameter is used and searching is preferable to gridsearch also in that it is highly unlikely that the specified values in gridsearch are optimal, while there is a chance that the ones obtained randomly are closer to ideal values. I used a C values 0.001,0.01,0.1,1,10,20,50,100,200,500,1000 and maximum iteration values of 50,100,200,300.

#### Early stopping policy
BanditPolicy stopping policy is used because it allows one to select a cut-off at which models reporting metrics worse than the current best model are terminated. This method is better to select models with similar or better performance. This policy offers a little more flexibility than truncation and median stopping.

The best model accuracy was ~91.107%.

## AutoML
The autoML pipeline is very similar to the Scikit-learn pipeline as described. The data are retrieved from the provided URL using tabulardata format. We add compute cluster target to the autoML config as it allows us to run the autoML experiment using remote target.

The best model selected by autoML was a voting ensemble (~91.66% accurate). The Voting Ensemble model selected used an l1 regularization i.e penalty was placed the number of non-zero model coefficients. It also used a soft voting, where all models' class probabilities are averaged and the highest probablility selected to make a prediction.

## Pipeline comparison
The models from using SDK with hyperdrive had an accuracy of 91.107% and autoML model achieving 91.66% accuracy. The difference in accuracy could come down to slight variations in the cross-validation process. Both the algoritms are quite different in their methods, Logistic regression effectively makes use of a fitted logistic function with a threshold to carry out binary classification. The voting ensemble classifier uses numerous individual classifiers and, in this case, averages the class probabilities of each classifier to make a prediction. 

## Future work
In the future, better feature engineering and feature selection methods can be applied to the dataset. Also, many of the AutoML runs use a scaler prior to model training and evaluation. Running AutoML for much with using other metrics than accuracy for classification would likely find better models and running it for long time can also give better models though it wont be huge enough improvement. Furthermore, uisng hyperdrive with a ohter classification models might give us better results.
