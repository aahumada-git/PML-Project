# Human Activity Recognition : Qualitative Activity Recognition

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

## Goal

This project use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website [here](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset). 

> The goal of this project is to predict the manner in which they did the exercise.

## Loading Data


```r
library(utils)
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```


```r
base          <- getwd()
training.file <- paste(base, "pml-training.csv", sep="/")
testing.file  <- paste(base, "pml-testing.csv" , sep="/")

testing     <- read.csv(testing.file , header=TRUE, na.string="NA")
training    <- read.csv(training.file, header=TRUE, na.string="NA")
```

## Data Analisys

According to [document](http://groupware.les.inf.puc-rio.br/public/papers/2013.Velloso.QAR-WLE.pdf) were identified into data sets 3 groups:
 - Z : Measurement data related to user and times.
 - X : Sensor measures.
 - Y : Expected outcome (class)
 
Reviewing the sensor measures were detected many columns without data, they were eliminated reducing amount of X's variable from 151 to 52.

 

```r
Z   <- training[,1:7]   # Test info
X   <- training[,8:159] # Sensor measures
Y   <- training[,160]   # Expected outcome

NAs <- apply(X,2,function(x) { sum(is.na(x) | grepl("#DIV/0!|^[ ]*$",x))  })
idx <- which( NAs == 0 )

X <- X[,idx]            # Remove NAs columns
```

### Why you made the choices you did. 

In order to real variable reducction were applied pre-proccesing with PCA method, The result was that PCA needed 19 components to capture 90% of the variance. 

Then training set was transformed to PCA set.


```r
preProc <- preProcess(X, method=c("pca", "center", "scale"), thresh=0.9)
print( preProc )
```

```
## 
## Call:
## preProcess.default(x = X, method = c("pca", "center", "scale"), thresh
##  = 0.9)
## 
## Created from 19622 samples and 52 variables
## Pre-processing: principal component signal extraction, centered, scaled 
## 
## PCA needed 19 components to capture 90 percent of the variance
```

```r
pcaTrain <- data.frame(Y, predict(preProc, X) )
```


### How you built your model

Were used Random Forest method ("rf") due to the outcome type (discrete)


```r
ptm <- proc.time()
g <- train( Y ~ . , data=pcaTrain, method="rf")
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
print(g)
```

```
## Random Forest 
## 
## 19622 samples
##    19 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 19622, 19622, 19622, 19622, 19622, 19622, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     1         1      0.003        0.003   
##   10    1         1      0.003        0.003   
##   20    1         0.9    0.004        0.005   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

```r
cat(sprintf( "ChekPoint[%3.2f]:Training Time", proc.time()[3]-ptm[3] ))
```

```
## ChekPoint[3368.02]:Training Time
```

### What you think the expected out of sample error is

In order to validate the model were transformed testing data into PCA coordinated and predict its result:



```r
# Evaluate model in testing data
Xt    <- testing[,8:159]      # Sensor measures
Xt    <- Xt[,idx]             # Remove NAs columns
pcaXt <- predict(preProc, Xt) # Trasform to PCA

answer <- predict(g, pcaXt)
print(answer)
```

```
##  [1] B A A A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

### Result
Due to Accuracy reached by training was 97.3% were expected some error, and that was verifieded only with 3th test sample. 


