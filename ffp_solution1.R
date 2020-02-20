setwd("D:/ть/RProgramming/AdvancedSubjectsData")

rm(list = ls()) # remove all variables from global environment
cat("\014") # clear the screen

library(DMwR)
library(caret)
library(yardstick)
library(AUC)

f5.4 <- function(data, lev = NULL, model = NULL) 
{
  f5.4_val <- f_meas_vec(data$pred, data$obs, beta = 5.4)
  names(f5.4_val) <- "F5.4"
  f5.4_val
}

set.seed(5)

data.df <- read.csv("ffp_train_variable_reduce.csv")
data.df <- data.df[ -c(1) ]

# convert columns to correct classes
# data.df$GROUP <- as.factor(data.df$GROUP)
data.df$STATUS_PANTINUM <- as.factor(data.df$STATUS_PANTINUM)
data.df$STATUS_GOLD <- as.factor(data.df$STATUS_GOLD)
data.df$STATUS_SILVER <- as.factor(data.df$STATUS_SILVER)
data.df$CALL_FLAG <- as.factor(data.df$CALL_FLAG)
# data.df$CREDIT_PROBLEM <- as.factor(data.df$CREDIT_PROBLEM)
# data.df$RETURN_FLAG <- as.factor(data.df$RETURN_FLAG)
data.df$BENEFIT_FLAG <- as.factor(data.df$BENEFIT_FLAG)
data.df$BUYER_FLAG <- as.factor(data.df$BUYER_FLAG)

# data.df$AVG_FARE_L_Y1 <- data.df$FARE_L_Y1 / data.df$NUM_DEAL
# # data.df$AVG_FARE_L_Y2 <- data.df$FARE_L_Y2 / data.df$NUM_DEAL
# data.df$AVG_FARE_L_Y3 <- data.df$FARE_L_Y3 / data.df$NUM_DEAL
# data.df$AVG_FARE_L_Y4 <- data.df$FARE_L_Y4 / data.df$NUM_DEAL
# data.df$AVG_FARE_L_Y5 <- data.df$FARE_L_Y5 / data.df$NUM_DEAL

# fix imbalanced data
data.df <- SMOTE(BUYER_FLAG ~ ., data.df, perc.over = 600, perc.under = 100)

# partition the data
train.index <- sample(1:dim(data.df)[1], dim(data.df)*0.7)
train.df <- data.df[train.index, ]
valid.df <- data.df[-train.index, ]

ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, verboseIter = TRUE, summaryFunction = f5.4)
knnFit <- train(BUYER_FLAG ~ ., data = train.df, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20, metric = "F5.4", maximize = TRUE)

#Output of kNN fit
knnFit
plot(knnFit)
knnPredict <- predict(knnFit, newdata = valid.df)

confusionMatrix(knnPredict, valid.df$BUYER_FLAG, positive='1')

# f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 2)
f_meas_vec(knnPredict, valid.df$BUYER_FLAG, beta = 5.4)

r <- roc(knnPredict, as.factor(valid.df$BUYER_FLAG))
# plot(r)
auc(r)
