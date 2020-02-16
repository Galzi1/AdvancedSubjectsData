setwd("D:/ть/RProgramming/AdvancedSubjectsData")

rm(list = ls()) # remove all variables from global environment
cat("\014") # clear the screen

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

data.df$NUM_DEAL <- scale(data.df$NUM_DEAL)
data.df$LAST_DEAL <- scale(data.df$LAST_DEAL)
data.df$ADVANCE_PURCHASE <- scale(data.df$ADVANCE_PURCHASE)
data.df$FARE_L_Y1 <- scale(data.df$FARE_L_Y1)
data.df$FARE_L_Y3 <- scale(data.df$FARE_L_Y3)
data.df$FARE_L_Y4 <- scale(data.df$FARE_L_Y4)
data.df$FARE_L_Y5 <- scale(data.df$FARE_L_Y5)
data.df$POINTS_L_Y1 <- scale(data.df$POINTS_L_Y1)
data.df$POINTS_L_Y2 <- scale(data.df$POINTS_L_Y2)
data.df$POINTS_L_Y3 <- scale(data.df$POINTS_L_Y3)
data.df$POINTS_L_Y4 <- scale(data.df$POINTS_L_Y4)
data.df$POINTS_L_Y5 <- scale(data.df$POINTS_L_Y5)

# fix imbalanced data
library(DMwR)
data.df <- SMOTE(BUYER_FLAG ~ ., data.df, perc.over = 600, perc.under=100)


# partition the data
train.index <- sample(1:dim(data.df)[1], dim(data.df)*0.7)
train.df <- data.df[train.index, ]
valid.df <- data.df[-train.index, ]

# count BUYER_FLAG values per level of factor (0 or 1)
library(dplyr)
valid.df %>% 
  group_by(BUYER_FLAG) %>%
  summarise(no_rows = length(BUYER_FLAG))

# tc <- trainControl("cv",5)
# rpart.grid <- expand.grid(.cp=0.2)
# train.rpart <- train(BUYER_FLAG ~., data=train.df, method="rpart",trControl=tc)
# pred <- predict(train.rpart, valid.df)

# classification tree - rpart
library(rpart)
tr <- rpart(BUYER_FLAG ~ ., data = train.df, method = "class")
# plot(tr)
pred <- predict(tr, valid.df, type = "class")

library(caret)
confusionMatrix(pred, valid.df$BUYER_FLAG, positive='1')

library(yardstick)
# f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 2)
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 5.4)

library(AUC)
r <- roc(pred, as.factor(valid.df$BUYER_FLAG))
# plot(r)
auc(r)

# classification tree - ctree
library(party)
tr <- ctree(BUYER_FLAG ~ ., data = train.df)
# plot(tr)
pred <- predict(tr, valid.df)

library(caret)
confusionMatrix(pred, valid.df$BUYER_FLAG, positive='1')

library(yardstick)
# f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 2)
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 5.4)

library(AUC)
r <- roc(pred, as.factor(valid.df$BUYER_FLAG))
# plot(r)
auc(r)

# classification tree - C4.5 algorithm (J48 in RWeka)
library(RWeka)
library(partykit)
tr <- J48(BUYER_FLAG ~ ., data = train.df)
# plot(tr)
pred <- predict(tr, valid.df)

library(caret)
confusionMatrix(pred, valid.df$BUYER_FLAG, positive='1')

library(yardstick)
# f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 2)
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 5.4)

library(AUC)
r <- roc(pred, as.factor(valid.df$BUYER_FLAG))
# plot(r)
auc(r)

# # random forest - cforest
# rf <- cforest(BUYER_FLAG ~ ., data = train.df)
# 
# # plot(rf)
# pred <- predict(rf, valid.df)
# 
# library(caret)
# confusionMatrix(pred, valid.df$BUYER_FLAG, positive='1')
# 
# library(yardstick)
# # f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 2)
# f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 5.4)
# 
# library(AUC)
# r <- roc(pred, as.factor(valid.df$BUYER_FLAG))
# # plot(r)
# auc(r)
# 
# random forest - randomForest
library(randomForest)
rf <- randomForest(BUYER_FLAG ~ ., data = train.df, ntree = 1000)
pred <- predict(rf, valid.df)

library(caret)
confusionMatrix(pred, valid.df$BUYER_FLAG, positive='1')

library(yardstick)
# f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 2)
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 5.4)

library(AUC)
r <- roc(pred, as.factor(valid.df$BUYER_FLAG))
# plot(r)
auc(r)
# 
# # random forest - obliqueRF
# library(obliqueRF)
# rf <- obliqueRF(BUYER_FLAG ~ ., data = train.df)
# pred <- predict(rf, valid.df)
# 
# library(caret)
# confusionMatrix(pred, valid.df$BUYER_FLAG, positive='1')
# 
# library(yardstick)
# # f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 2)
# f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 5.4)
# 
# library(AUC)
# r <- roc(pred, as.factor(valid.df$BUYER_FLAG))
# plot(r)
# auc(r)

# k-nearest neighbour
knn_m <- knn(train = train.df, test = valid.df, cl = train.df$BUYER_FLAG, k=5)

library(caret)
confusionMatrix(knn_m, as.factor(valid.df$BUYER_FLAG), positive='1')

library(yardstick)
# f_meas_vec(knn_m, as.factor(valid.df$BUYER_FLAG), beta = 2)
f_meas_vec(knn_m, as.factor(valid.df$BUYER_FLAG), beta = 5.4)

library(AUC)
r <- roc(knn_m, as.factor(as.factor(valid.df$BUYER_FLAG)))
plot(r)
auc(r)

# logistic regression
lr <- glm(BUYER_FLAG ~ ., data = train.df, family = binomial())
pred <- predict(lr, valid.df, type = 'response')
pred_f <- as.factor(ifelse(pred > 0.5, 1, 0))

library(caret)
confusionMatrix(pred_f, as.factor(valid.df$BUYER_FLAG), positive='1')

library(yardstick)
# f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 2)
f_meas_vec(pred_f, valid.df$BUYER_FLAG, beta = 5.4)

library(AUC)
r <- roc(pred_f, as.factor(valid.df$BUYER_FLAG))
# plot(r)
auc(r)

# gradient boosting - gbm
library(gbm)
gb <- gbm(BUYER_FLAG ~ ., distribution = "bernoulli", data = train.df, n.trees = 1000, interaction.depth = 4, shrinkage = 0.01)
pred <- predict(gb, valid.df, n.trees = 100, type = "response")

library(caret)
confusionMatrix(pred, valid.df$BUYER_FLAG, positive='1')

library(yardstick)
# f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 2)
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 5.4)

library(AUC)
r <- roc(pred, as.factor(valid.df$BUYER_FLAG))
plot(r)
auc(r)

# gradient boosting - xgboost
xgb <-xgboost(data = train.df, label = train.df$BUYER_FLAG, max.depth = 2, eta = 1, nthread = 2, nround = 2, objective = "binary:logistic")
pred <- predict(xgb, valid.df)

library(caret)
confusionMatrix(pred, valid.df$BUYER_FLAG, positive='1')

library(yardstick)
# f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 2)
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 5.4)

library(AUC)
r <- roc(pred, as.factor(valid.df$BUYER_FLAG))
plot(r)
auc(r)

