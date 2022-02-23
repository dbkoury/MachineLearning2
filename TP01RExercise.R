

    #DEALING WITH CLASS IMBALANCE 
rm(list=ls())
setwd("C:/Users/dylan/MSBA/ML1/Assignments")

library(caret)
library(tidyverse)
library(mltools)
library(MASS)

#
data <- read.csv("Case4.csv", stringsAsFactors = TRUE)

data <- data[-c(3,17)]
attach(data)
set.seed(1)
divideData <- createDataPartition(data$intubated,p=.1,list=F)
train <- data[divideData,]
test <- data[-divideData,]

##Logistic Model
logisticmodel <- glm(intubated~.,data=train, family=binomial)
probs <- predict(logisticmodel, type="response", newdata=test)
pred <- ifelse(probs>0.5, "Yes", "No")
(logtable <- table(pred,test$intubated,dnn=c("Predicted","Observed")))
(logaccuracy <- mean(pred==test$intubated))
##Accuracy Rate = 0.8179191
(logerror <- 1-logaccuracy)
##Error Rate = 0.1820809

mean(data$intubated=="No") #0.818078
#In fact, we would actually have a slightly higher accuracy rate just predicting every observation as no than this model gives us

#True Positives: When we correctly identify an observation for not needing intubation
(logA <- logtable[1]) #46411
#True Negatives: When we correctly identify an observation for needing intubation
(logD <- logtable[4]) #1
#False Positives: When we wrongfully identify an observation as not needing intubation (but they do)
(logB <- logtable[3]) #10322
#False Negatives: When we wrongfully identify an observation for needing intubation (but they don't)
(logC <- logtable[2]) #10


imbal <- sum(ifelse(intubated=='No',1,0))/sum(ifelse(intubated=='Yes',1,0))

  #Lets balance it out
intub_yes <- data[data$intubated=='Yes',]
intub_no <- data[data$intubated=='No',]
set.seed(1)
index_no_balanced <- sample(1:dim(intub_no)[1], size = dim(intub_yes)[1])
data_balanced <- rbind.data.frame(intub_yes,intub_no[index_no_balanced,])

set.seed(1)
divideData <- createDataPartition(data_balanced$intubated,p=.1,list=F)
train <- data_balanced[divideData,]
test <- data_balanced[-divideData,]

##Logistic Model
logisticmodel <- glm(intubated~.,data=train, family=binomial)
probs <- predict(logisticmodel, type="response", newdata=test)
pred <- ifelse(probs>0.5, "Yes", "No")
(logtable <- table(pred,test$intubated,dnn=c("Predicted","Observed")))
(logaccuracy <- mean(pred==test$intubated))
##Accuracy Rate = 0.6217185
(logerror <- 1-logaccuracy)
##Error Rate = 0.3782815

#True Positives: When we correctly identify an observation for not needing intubation
(logA <- logtable[1]) #6543
#True Negatives: When we correctly identify an observation for needing intubation
(logD <- logtable[4]) #6293
#False Positives: When we wrongfully identify an observation as not needing intubation (but they do)
(logB <- logtable[3]) #4030
#False Negatives: When we wrongfully identify an observation for needing intubation (but they don't)
(logC <- logtable[2]) #3780


#Using RUSBoost
data$intubated <- ifelse(data$intubated=='Yes',1,0)

unbalanced_boost <- rus(intubated ~ ., data = data, size = 15 , alg = 'rf', ir = imbal, rf.ntree = 100)

unbalanced_boost

