setwd("D:/ть/RProgramming/AdvancedSubjectsData")

rm(list = ls()) # remove all variables from global environment
cat("\014") # clear the screen

library(DMwR)
library(caret)
library(dplyr)
library(yardstick)
library(rpart)
library(AUC)
library(party)
library(RWeka)
library(partykit)
library(randomForest)
library(tibble)
library(class)
library(gbm)

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

# preProcValues <- preProcess(x = trainX,method = c("center", "scale"))

# fix imbalanced data
data.df <- SMOTE(BUYER_FLAG ~ ., data.df, perc.over = 600, perc.under=100)


# partition the data
train.index <- sample(1:dim(data.df)[1], dim(data.df)*0.7)
train.df <- data.df[train.index, ]
valid.df <- data.df[-train.index, ]

# count BUYER_FLAG values per level of factor (0 or 1)

valid.df %>% 
  group_by(BUYER_FLAG) %>%
  summarise(no_rows = length(BUYER_FLAG))

f5.4 <- function(data, lev = NULL, model = NULL) {
  f5.4_val <- f_meas_vec(data$pred, data$obs, beta = 5.4)
  c(F5.4 = f5.4_val)
}

tc <- trainControl(method = "repeatedcv", number = 10, repeats = 3, summaryFunction = f5.4)
rpart.grid <- expand.grid(.cp=0.2)
train.rpart <- train(BUYER_FLAG ~., data=train.df, method="rpart",trControl=tc, metric="F5.4")
pred <- predict(train.rpart, valid.df)

# classification tree - rpart
tr <- rpart(BUYER_FLAG ~ ., data = train.df, method = "class")
# plot(tr)
pred <- predict(tr, valid.df, type = "class")

confusionMatrix(pred, valid.df$BUYER_FLAG, positive='1')

# f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 2)
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 5.4)

r <- roc(pred, as.factor(valid.df$BUYER_FLAG))
# plot(r)
auc(r)

# classification tree - ctree
tr <- ctree(BUYER_FLAG ~ ., data = train.df)
# plot(tr)
pred <- predict(tr, valid.df)

confusionMatrix(pred, valid.df$BUYER_FLAG, positive='1')

# f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 2)
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 5.4)

r <- roc(pred, as.factor(valid.df$BUYER_FLAG))
# plot(r)
auc(r)

# classification tree - C4.5 algorithm (J48 in RWeka)
tr <- J48(BUYER_FLAG ~ ., data = train.df)
# plot(tr)
pred <- predict(tr, valid.df)

confusionMatrix(pred, valid.df$BUYER_FLAG, positive='1')

# f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 2)
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 5.4)

r <- roc(pred, as.factor(valid.df$BUYER_FLAG))
# plot(r)
auc(r)

# random forest - randomForest
rf <- randomForest(BUYER_FLAG ~ ., data = train.df, ntree = 1000)
pred <- predict(rf, valid.df)

confusionMatrix(pred, valid.df$BUYER_FLAG, positive='1')

# f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 2)
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 5.4)

r <- roc(pred, as.factor(valid.df$BUYER_FLAG))
# plot(r)
auc(r)

k_values <- c(seq(from=5, to=10005, by=100))
num_k <- length(k_values)
error_df <- tibble(k=rep(0, num_k),
                   f5.4=rep(0, num_k),
                   auc=rep(0, num_k))

for(i in 1:num_k)
{
  
  # fix k for this loop iteration
  k_val <- k_values[i]
  
  # k-nearest neighbour
  knn_m <- knn(train = train.df, test = valid.df, cl = train.df$BUYER_FLAG, k=k_val)
  
  # f_meas_vec(knn_m, as.factor(valid.df$BUYER_FLAG), beta = 2)
  f_val <- f_meas_vec(knn_m, as.factor(valid.df$BUYER_FLAG), beta = 5.4)
  
  r <- roc(knn_m, as.factor(as.factor(valid.df$BUYER_FLAG)))
  # plot(r)
  auc_val <- auc(r)
  
  error_df[i, 'k'] <- k_val
  error_df[i, 'f5.4'] <- f_val
  error_df[i, 'auc'] <- auc_val
  print(paste('k:', k_val, 'f5.4:', f_val, 'auc:', auc_val, sep = " "))
}

print(error_df)

# library(class)
# # k-nearest neighbour
# knn_m <- knn(train = train.df, test = valid.df, cl = train.df$BUYER_FLAG, k=5)
# 
# library(caret)
# confusionMatrix(knn_m, as.factor(valid.df$BUYER_FLAG), positive='1')
# 
# library(yardstick)
# # f_meas_vec(knn_m, as.factor(valid.df$BUYER_FLAG), beta = 2)
# f_meas_vec(knn_m, as.factor(valid.df$BUYER_FLAG), beta = 5.4)
# 
# library(AUC)
# r <- roc(knn_m, as.factor(as.factor(valid.df$BUYER_FLAG)))
# # plot(r)
# auc(r)

# logistic regression
lr <- glm(BUYER_FLAG ~ ., data = train.df, family = binomial())
pred <- predict(lr, valid.df, type = 'response')
pred_f <- as.factor(ifelse(pred > 0.5, 1, 0))

confusionMatrix(pred_f, as.factor(valid.df$BUYER_FLAG), positive='1')

# f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 2)
f_meas_vec(pred_f, valid.df$BUYER_FLAG, beta = 5.4)

r <- roc(pred_f, as.factor(valid.df$BUYER_FLAG))
# plot(r)
auc(r)

# gradient boosting - gbm
gb <- gbm(BUYER_FLAG ~ ., distribution = "bernoulli", data = train.df, n.trees = 1000, interaction.depth = 4, shrinkage = 0.01)
pred <- predict(gb, valid.df, n.trees = 100, type = "response")

confusionMatrix(pred, valid.df$BUYER_FLAG, positive='1')

# f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 2)
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 5.4)

r <- roc(pred, as.factor(valid.df$BUYER_FLAG))
plot(r)
auc(r)

# gradient boosting - xgboost
xgb <-xgboost(data = train.df, label = train.df$BUYER_FLAG, max.depth = 2, eta = 1, nthread = 2, nround = 2, objective = "binary:logistic")
pred <- predict(xgb, valid.df)

confusionMatrix(pred, valid.df$BUYER_FLAG, positive='1')

# f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 2)
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 5.4)

r <- roc(pred, as.factor(valid.df$BUYER_FLAG))
plot(r)
auc(r)

