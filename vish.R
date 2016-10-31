#################################### KNOCTOBER ####################################
###   	AUTHOR : ARUNKUMAR THANGAPANDIAN					###
###	EMAIL  : arunkumar.t@iiml.org						###
###   	DATE   : 30-OCT-2016							###
###################################################################################

######## LOAD ALL THE NECESSARY LIBRARIES #######

library(readr)
library(randomForest)
library(data.table)
library(bit64)
library(xgboost)
library(lubridate)
library(caret)
library(glmnet)
library(stringr)
library(dplyr)

set.seed(1)
setwd("Y:/AV/vishleshan")

###### READ DATA FILES #########

train <- fread("train.csv",header = T,sep = ',',stringsAsFactors=TRUE)
train <-as.data.frame(train)

test <- fread("test.csv",header = T,sep = ',',stringsAsFactors=TRUE)
test <-as.data.frame(test)

gc()

trainex <- train
trainex$Walc <- NULL
combined<-rbind(trainex,test)


################ FORMAT CHARACTER VARIABLES INTO FACTOR VARIABLES


combined$Sex <- as.numeric(factor(combined$Sex))
combined$Address <- as.numeric(factor(combined$Address))
combined$Famsize <- as.numeric(factor(combined$Famsize))
combined$Pstatus <- as.numeric(factor(combined$Pstatus))
combined$Mjob <- as.numeric(factor(combined$Mjob))
combined$Fjob <- as.numeric(factor(combined$Fjob))
combined$Guardian <- as.numeric(factor(combined$Guardian))
combined$Schoolsup <- as.numeric(factor(combined$Schoolsup))
combined$Famsup <- as.numeric(factor(combined$Famsup))
combined$Activities <- as.numeric(factor(combined$Activities))
combined$Nursery <- as.numeric(factor(combined$Nursery))
combined$Higher <- as.numeric(factor(combined$Higher))
combined$Internet <- as.numeric(factor(combined$Internet))
combined$Romantic <- as.numeric(factor(combined$Romantic))

################ FORMAT INTEGER VARIABLES INTO FACTOR VARIABLES

combined$Medu <- as.numeric(factor(combined$Medu))
combined$Fedu <- as.numeric(factor(combined$Fedu))
combined$Traveltime <- as.numeric(factor(combined$Traveltime))
combined$Studytime <- as.numeric(factor(combined$Studytime))
combined$Failures <- as.numeric(factor(combined$Failures))
combined$Famrel <- as.numeric(factor(combined$Famrel))
combined$Freetime <- as.numeric(factor(combined$Freetime))
combined$Goout <- as.numeric(factor(combined$Goout))
combined$Health <- as.numeric(factor(combined$Health))

################ FEATURE ENGINEERING 



#### REMOVE VARIABLES 

combined$ID <- NULL

################ PREPARE OUTCOME VARIABLE FOR XGBOOST

trainP<-head(combined,nrow(train))
testP<-tail(combined,nrow(test))

trainP$Walc <-train$Walc

feature.names <- c ("Sex","Age","Medu","Fedu","Mjob","Fjob","Address","Traveltime","Studytime","Famsup","Activities","Romantic","Famrel","Freetime","Goout","Health","Absences","Grade") 

################ XGB MODELLING #########################


cat(".......Training XGBOOST.........\n")

xgb_check <- xgboost(data        = data.matrix(trainP[,feature.names]),
	             label       = trainP$Walc,
        	     nrounds     =31,
		     max.depth   = 7,
	             objective   = "binary:logistic",
		     eval_metric = "logloss",
		     eval_metric = "error",
	             )

xgb_val <- data.frame(ID=test$ID)
xgb_val$Walc <- NA 
for (rows in split(1:nrow(testP), ceiling((1:nrow(testP))/1000))) {
    xgb_val[rows, "Walc"] <- predict(xgb_check, data.matrix(testP[rows,feature.names]))
}

################ ENSEMBLING #########################

ens_val <- xgb_val

ens_val$Walc <- as.numeric(ens_val$Walc > 0.5)

write.csv(ens_val, "sub_ens_1.csv", row.names=FALSE) 
