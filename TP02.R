##### TP02: Analysis #####
# Team 5: Mengting Ding, Dylan Koury, Olivia Siegal
# Due Tuesday, April 19

rm(list=ls())

# No scientific notation
options(scipen = 10000)

##### Libraries #####
# install.packages("MLmetrics")
library(MLmetrics)
# install.packages("unbalanced")
library(unbalanced) # https://rpubs.com/DeclanStockdale/799284
library(ggplot2)
library(caret)
library(dplyr)
library(MASS)
# install.packages("naniar")
library(naniar)
library(glmnet)
library(pls)
library(randomForest)
library(gbm)
# install.packages("xgboost")
library(xgboost)

##### Import data #####
# available on Porto Seguro Competition kaggle page
train = read.csv("train.csv",  row.names="id")

##### Normalized Gini Coefficient Function #####
# Code provided through kaggle to calculate normalized gini coefficient
normalizedGini <- function(aa, pp) {
  Gini <- function(a, p) {
    if (length(a) !=  length(p)) stop("Actual and Predicted need to be equal lengths!")
    temp.df <- data.frame(actual = a, pred = p, range=c(1:length(a)))
    temp.df <- temp.df[order(-temp.df$pred, temp.df$range),]
    population.delta <- 1 / length(a)
    total.losses <- sum(a)
    null.losses <- rep(population.delta, length(a)) 
    accum.losses <- temp.df$actual / total.losses
    gini.sum <- cumsum(accum.losses - null.losses)
    sum(gini.sum) / length(a)
  }
  Gini(aa,pp) / Gini(aa,aa)
}


##### Explore the training dataset #####
# Some of our models require that the y variable be a factor
train$target = as.factor(train$target)

head(train)
str(train) # all features are numeric or integer
summary(train)

##### Missing Values #####
# In this dataset, -1 represents missing data
# Need to change -1 values to NA

# Function to replace -1 with NA's
replaceWNAs = function(x){
  
  for (i in 1:ncol(x)){
    x[which(x[,i]== -1),i] = NA
  }
  
  return(x)
}

train = replaceWNAs(train)
summary(train$ps_car_11) # to confirm all instances of -1 were changed to NA

# Upon further exploration later in the analysis, we found that there were issues of multicollinearity with
# ps_ind_14, ps_ind_09_bin so removing them here.
train = train[,-c(10,15)]
head(train)

# Also need to remove NA's from dataset for complete analysis
# ps_reg_03, ps_car_03_cat, ps_car_05_cat, and ps_car_14 are comprised of mostly missing data, so drop these columns
train = train[,-c(20,23,25,35)]
summary(train)
# ps_car_07_cat contains 11489 NA's - removed
train = train[,-24]
summary(train)

# Change rest of the NA's to 0
replaceW0s = function(x){
  
  for (i in 1:ncol(x)){
    x[which(is.na(x[,i]) == TRUE),i] = 0
  }
  
  return(x)
}

train = replaceW0s(train)
summary(train)

##### Split train.csv into training & testing #####
set.seed(521)
divideData = createDataPartition(train$target,p = 0.7, list=FALSE)
train2 = train[divideData,]
validate = train[-divideData,]

# Determine whether or not we need to be worried about class imbalance in the training set?
summary(train2$target)
# Class imbalance? YES
#      0      1 
# 401463  15186 

# Plot frequency of each class
ggplot(data = train2, aes(fill = target)) +
  geom_bar(aes(x = target)) +
  ggtitle("Number of samples in each class", subtitle = "Training dataset") +
  xlab("") +
  ylab("Samples") +
  scale_y_continuous(expand = c(0,0)) +
  # scale_x_discrete(expance = c(0,0)) +
  theme(legend.position = "none",
        legend.title = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank())

##### Observations about the dataset & kaggle submissions #####

# BINARY CLASSIFICATION PROBLEM
# 0 - no claim; 1 - claim

# Note, data has been anonymized, so it makes it difficult to make informed decisions about which factors to keep in the model
# For prediction purposes, this shouldn't have a huge impact, but important to consider.

# MAJOR critique - many people didn't consider the class imbalance. 
# Dataset - anonymized so difficult to make informed decisions about which features to include/consider
# Did others center/scale the dataset?
# Competitors say KNN didn't work well
# multicollinearity considerations in the data?
# set seeds? for reproducability

# Models under consideration:
# logistic, lda, qda, knn, lasso, ridge, randomForest, boosted randomForest (gbm)

##### Balance the Dataset #####
# First need to separate into predictor and response variables
predictorVariables = train2[,-1]
responseVariable = train2[,1]

# For this function, the minority case must be 1 (true for this dataset)
# Undersampling the majority class
undersampledData = ubBalance(predictorVariables,responseVariable,type='ubUnder',verbose=TRUE)
underTrain = cbind(undersampledData$Y,undersampledData$X)
names(underTrain)[names(underTrain) == "undersampledData$Y"] <- "target"
head(underTrain)

# Plot the new fequencies
ggplot(data = underTrain, aes(fill = target)) +
  geom_bar(aes(x = target)) +
  ggtitle("Number of samples in each class", subtitle = "Training dataset") +
  xlab("") +
  ylab("Samples") +
  scale_y_continuous(expand = c(0,0)) +
  # scale_x_discrete(expance = c(0,0)) +
  theme(legend.position = "none",
        legend.title = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank())

summary(underTrain$target)
# 0     1 
# 15186 15186 

# Classes are balanced

##### Logistic regression - unbalanced #####
# Attempt to model the unbalanced dataset with the logistic regression
# Just to better understand how things change when the classes are balanced

# Preprocess the data
# This preprocessing step is specific to the unbalanced dataset
preprocessing2 <- train2 %>% preProcess(method = c("center", "scale"))
trainTransformed2 <- preprocessing2 %>% predict(train2)
testTransformed2 <- preprocessing2 %>% predict(validate)

Logistic = glm(target~.,data=trainTransformed2, family = 'binomial')
summary(Logistic)

# Save the predicted probabilities to a vector
LogisticProb = predict(Logistic,newdata=validate,type="response")

# Assign predictions to class based on probability
LogisticPred = ifelse(LogisticProb > 0.5,1,0) # Positive class is filing a claim (1)

# Produce Accuracy rates 
mean(LogisticPred==validate$target) # Accuracy rate of 0.96

# Confusion Matrix
table(LogisticPred,validate$target) # Only predicts 0 - lost of false positives

normalizedGini(as.numeric(validate$target),as.numeric(LogisticPred)) # 0.005

# Unbalanced dataset does not yield good models because the model assumes most of the
# assignments should be 0. Accuracy is high, but won't be good at predicting.

##### Logistic Regression #####
# Preprocess the balanced dataset
# Center & Scaled data will be used for Logistic, LDA, QDA, and KNN models
set.seed(521)
preprocessing <- underTrain %>% preProcess(method = c("center", "scale"))
trainTransformed <- preprocessing %>% predict(underTrain)
testTransformed <- preprocessing %>% predict(validate)

Logistic = glm(target~.,data=trainTransformed, family = 'binomial')
summary(Logistic)

# Logistic model assumption is that there is no multicollinearity
# When we ran this model the first time, we observed multicollinearity between two features
car::vif(Logistic)

# Used the correlation matrix to identify the collinear features & removed them (see earlier code)
cor(underTrain[,-1])
# ps_ind_12_bin and ps_ind_14 have a correlation value of 0.8962608
# ps_ind_09_bin included NA's in the summary of the Logistic model
# removed ps_ind_09_bin and ps_ind_12_bin

# Predict probabilities
LogisticProb = predict(Logistic,newdata=validate,type="response")

# Assign to class based on probability
LogisticPred = ifelse(LogisticProb > 0.5,0,1) # Positive class is filing a claim (1)

# Produce Accuracy rates 
mean(LogisticPred==validate$target) # Accuracy rate of 0.7898053

# Confusion Matrix
table(LogisticPred,validate$target) # Lots of false negatives

# Calculate the normalized gini coefficient
normalizedGini(as.numeric(validate$target),as.numeric(LogisticPred)) # -0.04246757

##### LDA #####
set.seed(521)
LDA = lda(target~.,data=trainTransformed)
summary(LDA)
plot(LDA)

LDApred <- LDA %>% predict(testTransformed)
names(LDApred)

# Accuracy
mean(LDApred$class==testTransformed$target) ## accuracy rate 0.6191652

# Confusion Matrix
confusionMatrix(LDApred$class,testTransformed$target)

# Calculate normalized gini coefficient
normalizedGini(as.numeric(validate$target),as.numeric(LDApred$class)) # 0.1629344

##### QDA #####
set.seed(521)
QDAmod = qda(target~.,data=underTrain,family="binomial")
summary(QDAmod)

QDApred = QDAmod %>% predict(testTransformed)
names(QDApred)

# Accuracy rate
mean(QDApred$class==testTransformed$target) # 0.3615811

# Confusion MAtrix
table(QDApred$class,testTransformed$target)

# Calculating Normalized Gini Coefficient
normalizedGini(as.numeric(testTransformed$target),as.numeric(QDApred$class)) # 0.0162174

##### KNN #####
# Need a smaller training set to run KNN
# Divide train into train3 (smaller training set) and validate2
set.seed(521)
divideData = createDataPartition(train$target,p = 0.2, list=FALSE)
train3 = train[divideData,]
validate2 = train[-divideData,]

# Need to balance train3
predictorVariables = train3[,-1]
responseVariable = train3[,1]

undersampledData = ubBalance(predictorVariables,responseVariable,type='ubUnder',verbose=TRUE)
underTrain2 = cbind(undersampledData$Y,undersampledData$X)
names(underTrain2)[names(underTrain2) == "undersampledData$Y"] <- "target"
head(underTrain2)

# Preprocess within the train function (center & scale)
KNNmod = train(target~., data=underTrain2,method="knn",preProcess = c("center","scale"))

KNNmod$bestTune #9
plot(KNNmod) #9 

KNNpred <- predict(KNNmod, newdata=validate)

confusionMatrix(KNNpred,validate$target)
# Accuracy rate = 0.5892

# Calculate normalized gini coefficient
normalizedGini(as.numeric(validate$target),as.numeric(KNNpred)) # 0.1065237

##### Lasso #####
# Need to organize the features into a matrix
# Be sure to use the balanced datasets
x.train = model.matrix(target~.,underTrain)[,-1]

# Isolate y into a variable
y.train = underTrain$target

# Repeat above prep for the test x's and y
x.test = model.matrix(target~., validate)[,-1]
y.test = validate$target

summary(y.train) # y.train is balanced
summary(y.test) # y.test is imbalanced, but that is okay because these data are not being used to train the model

set.seed(521)
# create a grid for possible lambdas
grid = 10^seq(-2,4,length=200)

# Lasso model without cross-validation
lasso = glmnet(x.train,y.train,alpha = 1,lambda=grid,family=binomial)

# Lasso model with cross-validation (to identify the best lambda)
cv.out.lasso = cv.glmnet(x.train,y.train,alpha=1,lambda=grid,family=binomial,nfolds=12)
bestLambda = cv.out.lasso$lambda.min
bestLambda # 0.01

# Make predictions using the best lambda value on the test set
lasso.pred = predict(lasso,s=bestLambda,newx=x.test)

target.hat = ifelse(lasso.pred>=0.5,1,0)

# Confusion matrix
table(y.test,target.hat)

# Accuracy rate
mean(y.test == target.hat) # 0.9188746

# Normalized gini coefficient
normalizedGini(as.numeric(y.test),as.numeric(target.hat)) # 0.06392911

##### Ridge #####
set.seed(521)
power.value = seq(from=10, to=-2,length=100)
grid = 10^power.value

# ridge = glmnet(x.train,y.train,alpha=0,lambda=grid,thres=1e-12)
cv.out.ridge = cv.glmnet(x.train,y.train,alpha=0,lambda=grid,family=binomial,thres = 1e-12,nfolds=12)

bestLambda = cv.out.ridge$lambda.min
bestLambda # 0.01747528

# Make predictions using best lambda on x.test
ridgePred = predict(cv.out.ridge,s=bestLambda,newx = x.test)
target.hat = ifelse(ridgePred>=0.5,1,0)

# Confusion matrix
table(y.test,target.hat)

# Accuracy rate
mean(y.test == target.hat) # 0.8878715

# Normalized gini coefficient
normalizedGini(as.numeric(y.test),as.numeric(target.hat)) # 0.09127149

##### randomForest #####
set.seed(521)

# May need to adjust memory to run the model
memory.limit(size=56000)

# start time
start = Sys.time()
modelrf <- randomForest(target~.,data=underTrain,ntree=200,importance=TRUE)
end = Sys.time()
end-start #run time 1.47 mins

# Make predictions
rfpred <- predict(modelrf, newdata=validate)

# Confusion matrix
table(rfpred,validate$target) 

# Accuracy rate
mean(rfpred==validate$target) #0.5941208

# Normalized gini coefficient
normalizedGini(as.numeric(validate$target),as.numeric(rfpred)) #0.16023

##### Bagging #####
set.seed(521)
start = Sys.time()
modelbag <- randomForest(target~.,data=underTrain,ntree=200,mtry=50,importance=TRUE)
end = Sys.time()
end-start #2 mins

# Made predictions
bagpred <- predict(modelbag,newdata=validate)

# Accuracy rate
mean(bagpred==validate$target) # 0.5934

# Confusion Matrix
table(bagpred,validate$target)

# Calculate normalized gini coefficient
normalizedGini(as.numeric(validate$target),as.numeric(bagpred)) #0.1575381

##### Boosting #####
set.seed(521)
start = Sys.time()
boost <- gbm(as.character(target)~.,underTrain, distribution="bernoulli"
             ,n.trees= 100,interaction.depth=3, shrinkage=0.02)
end = Sys.time()
end - start

# Make predictions
pred.gbm <- predict(boost,validate,type = "response")
yhat <- ifelse(pred.gbm>0.5,0,1)

# Accuracy rate
mean(yhat == validate$target) #0.405

# Confusion Matrix
table(yhat,validate$target)

# normalized gini coefficient
normalizedGini(as.numeric(validate$target),as.numeric(yhat)) #-0.1603158


##### XGBoost #####
#turn train2 features into matrix
Train.X <- data.matrix(scale(trainTransformed[-1]))
Train.Y <- as.numeric(trainTransformed$target)
Train.Y[Train.Y == "1"] <- "0"
Train.Y[Train.Y == "2"] <- "1"

# turn validate features into matrix
Test.X <- data.matrix(scale(testTransformed[-1]))
Test.Y <- as.numeric(testTransformed$target)
Test.Y[Test.Y == "1"] <- "0"
Test.Y[Test.Y == "2"] <- "1"

Train.X = xgb.DMatrix(data=Train.X, label=Train.Y)
Test.X = xgb.DMatrix(data=Test.X, label=Test.Y)

xgbmodel=xgboost( data = Train.X, #X features in matrix form
                  #label = as.numeric(Train.Y), # target is a vector
                  eta = .5, # learning rate
                  nthread = 1, # number of parallel threads
                  nrounds = 50, # number of rounds of predictions
                  objective = "binary:logistic",  # logistic
                  max.depth  = 2,  # number of splits
                  eval_metric = "error",
                  verbose = 1) # print training error

# Assess importance of features
(Importance <- xgb.importance(colnames(Train.X), model = xgbmodel))

# Make predictions
xgbpreds <- predict(xgbmodel, newdata=Test.X)
summary(xgbpreds)
predictions <- as.numeric(xgbpreds > 0.5)

# Accuracy rate
mean(predictions==Test.Y) # 0.5906319

# Confusion Matrix
table(predictions, Test.Y)

# Normalized Gini Coefficient
normalizedGini(as.numeric(Test.Y),predictions) # 0.1836209
