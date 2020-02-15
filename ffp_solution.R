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

# fix imbalanced data
library(DMwR)
data.df <- SMOTE(BUYER_FLAG ~ ., data.df, perc.over = 600, perc.under=100)


# partition the data
train.index <- sample(1:dim(data.df)[1], dim(data.df)*0.6)
train.df <- data.df[train.index, ]
valid.df <- data.df[-train.index, ]

# count BUYER_FLAG values per level of factor (0 or 1)
library(dplyr)
valid.df %>% 
  group_by(BUYER_FLAG) %>%
  summarise(no_rows = length(BUYER_FLAG))

# classification tree - rpart
library(rpart)
tr <- rpart(BUYER_FLAG ~ ., data = train.df)
# plot(tr)
pred <- predict(tr, valid.df, type = "class")

library(caret)
confusionMatrix(pred, valid.df$BUYER_FLAG, positive='1')

library(yardstick)
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 2)
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
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 2)
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
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 2)
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 5.4)

library(AUC)
r <- roc(pred, as.factor(valid.df$BUYER_FLAG))
# plot(r)
auc(r)

# random forest - cforest
rf <- cforest(BUYER_FLAG ~ ., data = train.df)

# plot(rf)
pred <- predict(rf, valid.df)

library(caret)
confusionMatrix(pred, valid.df$BUYER_FLAG, positive='1')

library(yardstick)
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 2)
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 5.4)

library(AUC)
r <- roc(pred, as.factor(valid.df$BUYER_FLAG))
# plot(r)
auc(r)

# random forest - randomForest
library(randomForest)
rf <- randomForest(BUYER_FLAG ~ ., data = train.df, ntree = 100)
pred <- predict(rf, valid.df)

library(caret)
confusionMatrix(pred, valid.df$BUYER_FLAG, positive='1')

library(yardstick)
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 2)
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 5.4)

library(AUC)
r <- roc(pred, as.factor(valid.df$BUYER_FLAG))
# plot(r)
auc(r)

# random forest - obliqueRF
library(obliqueRF)
rf <- obliqueRF(BUYER_FLAG ~ ., data = train.df)
pred <- predict(rf, valid.df)

library(caret)
confusionMatrix(pred, valid.df$BUYER_FLAG, positive='1')

library(yardstick)
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 2)
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 5.4)

library(AUC)
r <- roc(pred, as.factor(valid.df$BUYER_FLAG))
plot(r)
auc(r)

# k-nearest neighbour
knn_m <- kNN(BUYER_FLAG ~ ., train.df, valid.df, norm=TRUE, k=2)

library(caret)
confusionMatrix(knn_m, as.factor(valid.df$BUYER_FLAG), positive='1')

library(yardstick)
f_meas_vec(knn_m, as.factor(valid.df$BUYER_FLAG), beta = 2)
f_meas_vec(knn_m, as.factor(valid.df$BUYER_FLAG), beta = 5.4)

library(AUC)
r <- roc(knn_m, as.factor(as.factor(valid.df$BUYER_FLAG)))
plot(r)
auc(r)

# logistic regression
lr <- glm(BUYER_FLAG ~ ., data = train.df, family = binomial())
pred <- predict(lr, valid.df, type = 'response')
pred_f <- as.factor(predict)

library(caret)
confusionMatrix(pred, as.factor(valid.df$BUYER_FLAG), positive='1')

library(yardstick)
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 2)
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 5.4)

library(AUC)
r <- roc(pred, as.factor(valid.df$BUYER_FLAG))
plot(r)
auc(r)

# gradient boosting - gbm
gb <-gbm(BUYER_FLAG ~ ., distribution = "bernoulli", data = train.df, n.trees = 1000, interaction.depth = 4, shrinkage = 0.01)
pred <- predict(gb, valid.df)

library(caret)
confusionMatrix(pred, valid.df$BUYER_FLAG, positive='1')

library(yardstick)
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 2)
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
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 2)
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 5.4)

library(AUC)
r <- roc(pred, as.factor(valid.df$BUYER_FLAG))
plot(r)
auc(r)

