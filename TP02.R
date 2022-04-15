##### TP02: Analysis #####
# Due Tuesday, April 19

rm(list=ls())
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

##### Import data #####
train = read.csv("train.csv",  row.names="id")
test = read.csv("test.csv", row.names = "id")

##### Normalized Gini Coefficient Function #####
normalizedGini <- function(aa, pp) {
  Gini <- function(a, p) {
    if (length(a) !=  length(p)) stop("Actual and Predicted need to be equal lengths!")
    temp.df <- data.frame(actual = a, pred = p, range=c(1:length(a)))
    temp.df <- temp.df[order(-temp.df$pred, temp.df$range),]
    population.delta <- 1 / length(a)
    total.losses <- sum(a)
    null.losses <- rep(population.delta, length(a)) # Hopefully is similar to accumulatedPopulationPercentageSum
    accum.losses <- temp.df$actual / total.losses # Hopefully is similar to accumulatedLossPercentageSum
    gini.sum <- cumsum(accum.losses - null.losses) # Not sure if this is having the same effect or not
    sum(gini.sum) / length(a)
  }
  Gini(aa,pp) / Gini(aa,aa)
}

normalizedGini2 <- function(aa, pp) {
  gini <- function(aa, pp) {
    rank_pp <- data.table::frank(pp)
    return(cov(rank_pp / mean(rank_pp), aa / mean(aa)))
  }
  return(gini(aa, pp) / gini(aa, aa))}


##### Explore the training dataset #####
train$target = as.factor(train$target)
head(train)
str(train)
summary(train)
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
summary(train$ps_car_11)

# Upon further exploration later in the analysis, we found that there were issues of multicollinearity with
# ps_ind_14, ps_ind_09_bin so removing them here.
head(train)
train = train[,-c(10,15)]
head(train)

# Also need to remove NA's from dataset for complete analysis
summary(train)
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

str(test)
# Note, test.csv does not contain a target column
# This data set is only provided to predict the target values
# For training & testing purposes, we must split the train.csv dataset into a training and testing set

# Replace -1 with NA's in the test.csv dataset
test = replaceWNAs(test)
summary(test)

# Remove the same columns that were removed in train.csv
test = test[,-c(9,14,21,24,26,28,36)]
summary(test)

# Change remaining NA's to zero
test = replaceW0s(test)
summary(test)

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
# multicollinearity considerations in the data? - avoid QDA b/c we 
# set seeds? for reproducability

# Models under consideration:
# logistic, lda, qda, knn, lasso, ridge, randomForest, boosted randomForest (gbm)

##### Balance the Dataset #####
# First need to separate into predictor and response variables
predictorVariables = train2[,-1]
responseVariable = train2[,1]

# For this function, the minority case must be 1 (true for this dataset)
# Undersampling
undersampledData = ubBalance(predictorVariables,responseVariable,type='ubUnder',verbose=TRUE)
underTrain = cbind(undersampledData$Y,undersampledData$X)
names(underTrain)[names(underTrain) == "undersampledData$Y"] <- "target"
head(underTrain)

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

##### Logistic regression - unbalanced #####
preprocessing2 <- train2 %>% preProcess(method = c("center", "scale"))
trainTransformed2 <- preprocessing2 %>% predict(train2)
testTransformed2 <- preprocessing2 %>% predict(validate)

Logistic = glm(target~.,data=trainTransformed2, family = 'binomial')
summary(Logistic)

LogisticProb = predict(Logistic,newdata=validate,type="response")
LogisticPred = ifelse(LogisticProb > 0.5,1,0) # Positive class is filing a claim (1)

##Produce Accuracy rates 
mean(LogisticPred==validate$target) # Accuracy rate of 0.96
table(LogisticPred,validate$target) # Only predicts 0 - lost of false positives

# Not working
normalizedGini(as.numeric(validate$target),as.numeric(LogisticPred)) # 0.005
NormalizedGini(as.numeric(LogisticPred),as.numeric(validate$target)) # 0.005

# Unbalanced dataset does not yield good models because the model assumes most of the
# assignments should be 0. Accuracy is high, but won't be good at predicting.

##### Logistic Regression #####
preprocessing <- underTrain %>% preProcess(method = c("center", "scale"))
trainTransformed <- preprocessing %>% predict(underTrain)
testTransformed <- preprocessing %>% predict(validate)

Logistic = glm(target~.,data=trainTransformed, family = 'binomial')
summary(Logistic)

car::vif(Logistic) # Two or more of the predictor variables are correlated
cor(underTrain[,-1])
# ps_ind_12_bin and ps_ind_14 have a correlation value of 0.8962608
# ps_ind_09_bin included NA's in the summary of the Logistic model
# removed ps_ind_09_bin and ps_ind_12_bin

LogisticProb = predict(Logistic,newdata=validate,type="response")
LogisticPred = ifelse(LogisticProb > 0.5,0,1) # Positive class is filing a claim (1)

##Produce Accuracy rates 
mean(LogisticPred==validate$target) # Accuracy rate of 0.7898053
table(LogisticPred,validate$target) # Lots of false negatives

# Not working
normalizedGini(as.numeric(validate$target),as.numeric(LogisticPred)) # -0.1690838
NormalizedGini(as.numeric(LogisticPred),as.numeric(validate$target)) # -0.1690838

normalizedGini2(as.numeric(validate$target),as.numeric(LogisticPred))

##### LDA #####
LDA = lda(target~.,data=trainTransformed)
summary(LDA)
plot(LDA)

LDApred <- LDA %>% predict(testTransformed)
names(LDApred)

mean(LDApred$class==testTransformed$target) ## accuracy rate 0.6191652
confusionMatrix(LDApred$class,testTransformed$target)

# Not working
normalizedGini(as.numeric(validate$target),as.numeric(LDApred$class)) # 0.1629344
NormalizedGini(as.numeric(LDApred$class),as.numeric(validate$target)) # 0.1629344

##### QDA #####
QDAmod = qda(target~.-ps_ind_12_bin-ps_ind_09_bin,data=underTrain,family="binomial")
summary(QDAmod)


##### Lasso #####
x.train = model.matrix(target~.,underTrain)[,-1]
y.train = underTrain$target

x.test = model.matrix(target~., validate)[,-1]
y.test = validate$target

summary(y.train)
summary(y.test)

set.seed(521)
grid = 10^seq(-2,4,length=200)

lasso = glmnet(x.train,y.train,alpha = 1,lambda=grid,family=binomial)
cv.out.lasso = cv.glmnet(x.train,y.train,alpha=1,lambda=grid,family=binomial,nfolds=12)
bestLambda = cv.out.lasso$lambda.min
bestLambda # 0.01

lasso.pred = predict(lasso,s=bestLambda,newx=x.test)

target.hat = ifelse(lasso.pred>=0.5,1,0)
table(y.test,target.hat)
mean(y.test == target.hat) # 0.9188746

normalizedGini(as.numeric(y.test),as.numeric(target.hat)) # 0.06392911
NormalizedGini(as.numeric(target.hat),as.numeric(y.test)) # 0.06392911

##### Ridge #####
set.seed(521)
power.value = seq(from=10, to=-2,length=100)
grid = 10^power.value

ridge = glmnet(x.train,y.train,alpha=0,lambda=grid,thresh=1e-12)
cv.out.ridge = cv.glmnet(x.train,y.train,alpha=0,lambda=grid,family=binomial,thres = 1e-12,nfolds=12)

bestLambda = cv.out.ridge$lambda.min
bestLambda # 0.02146141

ridgePred = predict(cv.out.ridge,s=bestLambda,newx = x.test)

target.hat = ifelse(ridgePred>=0.5,1,0)
table(y.test,target.hat)
mean(y.test == target.hat) # 0.8891764

normalizedGini(as.numeric(y.test),as.numeric(target.hat)) # 0.09039913
NormalizedGini(as.numeric(target.hat),as.numeric(y.test)) # 0.09039913

##### randomForest #####
library(randomForest)
set.seed(521)
memory.limit(size=56000)
start = Sys.time()
modelrf <- randomForest(target~.,data=underTrain,ntree=200,importance=TRUE)
end = Sys.time()
end-start #run time 1min
rfpred <- predict(modelrf, newdata=validate)
table(rfpred,validate$target) 
mean(rfpred==validate$target) #0.5940704
#      0      1
#0 102643   2800
#1  69412   3708
normalizedGini(as.numeric(validate$target),as.numeric(rfpred)) #0.1691966
NormalizedGini(as.numeric(rfpred),as.numeric(validate$target)) #0.1691966

##### Bagging #####
library(randomForest)
set.seed(521)
start = Sys.time()
modelbag <- randomForest(target~.,data=underTrain,ntree=200,mtry=50,importance=TRUE)
end = Sys.time()
end-start #2.2min
bagpred <- predict(modelbag,newdata=validate)
mean(bagpred==validate$target) # 0.5934
table(bagpred,validate$target)
#      0      1
#0 102323   2865
#1  69732   3643

normalizedGini(as.numeric(validate$target),as.numeric(bagpred)) #0.1575381

##### Boosting #####
library(gbm)
set.seed(521)
start = Sys.time()
boost <- gbm(as.character(target)~.,underTrain, distribution="bernoulli"
             ,n.trees= 100,interaction.depth=3, shrinkage=0.02)
end = Sys.time()
end - start
pred.gbm <- predict(boost,validate,type = "response")
yhat <- ifelse(pred.gbm>0.5,0,1)
mean(yhat == validate$target) #0.405
table(yhat,validate$target) #
#     0      1
#0  69687   3703
#1 102368   2805
normalizedGini(as.numeric(validate$target),as.numeric(yhat)) #-0.1613004 


##### XGBoost #####
  #turn train2 features into matrix
Train.X <- data.matrix(scale(trainTransformed[-1]))
Train.Y <- as.numeric(trainTransformed$target)
Train.Y[Train.Y == "1"] <- "0"
Train.Y[Train.Y == "2"] <- "1"

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

(Importance <- xgb.importance(colnames(Train.X), model = xgbmodel))

xgbpreds <- predict(xgbmodel, newdata=Test.X)
summary(xgbpreds)

predictions <- as.numeric(xgbpreds > 0.5)
mean(predictions==Test.Y) ## accuracy rate 0.5906319
table(predictions, Test.Y)
normalizedGini(as.numeric(Test.Y),predictions) # 0.1836209
