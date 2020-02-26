setwd("D:/ть/RProgramming/AdvancedSubjectsData")

rm(list = ls()) # remove all variables from global environment
cat("\014") # clear the screen

library(DMwR)
library(caret)
library(yardstick)
library(AUC)
library(parallel)
library(doParallel)
library(kernlab)
library(doSNOW)
library(klaR)
library(rpart)
library(dummies)
library(sets)

f5.4 <- function(data, lev = NULL, model = NULL) 
{
  f5.4_val <- F_meas(data$pred, data$obs, beta = 5.4, relevant = "1")
  names(f5.4_val) <- "F5.4"
  f5.4_val
}

get_outliers <- function(df, col, scale_factor)
{
  print(paste("median =", median(df[, col]), "; sd =", sd(df[, col])))
  outliers1 <- c(which(df[, col] > median(df[, col]) + scale_factor * sd(df[, col])))
  outliers2 <- c(which(df[, col] < median(df[, col]) - scale_factor * sd(df[, col])))
  # d <- df[df[, col] <= median(df[, col]) + 2 * sd(df[, col]),]
  # d <- d[d[, col] >= median(df[, col]) - 2 * sd(df[, col]),]
  # d
  union(outliers1, outliers2)
}

set.seed(5)

# cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
# registerDoParallel(cluster)

cl <- makeCluster(detectCores() - 1, outfile = "")
clusterEvalQ(cl, { library(yardstick) })
clusterEvalQ(cl, { library(AUC) })
clusterEvalQ(cl, { library(kernlab) })
clusterEvalQ(cl, { library(klaR) })
clusterEvalQ(cl, { library(rpart) })
clusterEvalQ(cl, { library(caret) })
registerDoSNOW(cl)

data.df <- read.csv("ffp_train.csv")
data.df <- data.df[ -c(1) ]

# convert columns to correct classes
data.df$GROUP <- as.factor(data.df$GROUP)
data.df$STATUS_PANTINUM <- as.factor(data.df$STATUS_PANTINUM)
data.df$STATUS_GOLD <- as.factor(data.df$STATUS_GOLD)
data.df$STATUS_SILVER <- as.factor(data.df$STATUS_SILVER)
data.df$CALL_FLAG <- as.factor(data.df$CALL_FLAG)
data.df$CREDIT_PROBLEM <- as.factor(data.df$CREDIT_PROBLEM)
data.df$RETURN_FLAG <- as.factor(data.df$RETURN_FLAG)
data.df$BENEFIT_FLAG <- as.factor(data.df$BENEFIT_FLAG)
data.df$BUYER_FLAG <- as.factor(data.df$BUYER_FLAG)

dummy_cols <- dummy(data.df$GROUP, sep = "_")
dummy_cols <- as.data.frame(dummy_cols)
dummy_cols <- sapply(dummy_cols, as.factor)
data.df <- cbind(data.df, dummy_cols)
data.df <- data.df[ -c(1) ]
data.df <- na.omit(data.df)

outliers <- c()

for (i in names(data.df))
{
  if (class(data.df[, i]) == "integer" || class(data.df[, i]) == "numeric")
  {
    outliers <- union(outliers, get_outliers(data.df, i, 3))
  }
}

data.df <- data.df[-outliers, ]

data.df$AVG_FARE_L_Y1 <- ifelse(data.df$NUM_DEAL == 0, 0, data.df$FARE_L_Y1 / data.df$NUM_DEAL)
data.df$AVG_FARE_L_Y2 <- ifelse(data.df$NUM_DEAL == 0, 0, data.df$FARE_L_Y2 / data.df$NUM_DEAL)
data.df$AVG_FARE_L_Y3 <- ifelse(data.df$NUM_DEAL == 0, 0, data.df$FARE_L_Y3 / data.df$NUM_DEAL)
data.df$AVG_FARE_L_Y4 <- ifelse(data.df$NUM_DEAL == 0, 0, data.df$FARE_L_Y4 / data.df$NUM_DEAL)
data.df$AVG_FARE_L_Y5 <- ifelse(data.df$NUM_DEAL == 0, 0, data.df$FARE_L_Y5 / data.df$NUM_DEAL)

outliers <- c()

for (i in c("AVG_FARE_L_Y1", "AVG_FARE_L_Y2", "AVG_FARE_L_Y3", "AVG_FARE_L_Y4", "AVG_FARE_L_Y5"))
{
  if (class(data.df[, i]) == "integer" || class(data.df[, i]) == "numeric")
  {
    outliers <- union(outliers, get_outliers(data.df, i, 3))
  }
}

data.df <- data.df[-outliers, ]

# HARD CODED-LY REMOVING STATUS_PANTINUM - all values are 0
data.df <- data.df[ -c(1) ]

# data.df <- read.csv("ffp_train_variable_reduce.csv")
# data.df <- data.df[ -c(1) ]
# 
# # convert columns to correct classes
# # data.df$GROUP <- as.factor(data.df$GROUP)
# data.df$STATUS_PANTINUM <- as.factor(data.df$STATUS_PANTINUM)
# data.df$STATUS_GOLD <- as.factor(data.df$STATUS_GOLD)
# data.df$STATUS_SILVER <- as.factor(data.df$STATUS_SILVER)
# data.df$CALL_FLAG <- as.factor(data.df$CALL_FLAG)
# # data.df$CREDIT_PROBLEM <- as.factor(data.df$CREDIT_PROBLEM)
# # data.df$RETURN_FLAG <- as.factor(data.df$RETURN_FLAG)
# data.df$BENEFIT_FLAG <- as.factor(data.df$BENEFIT_FLAG)
# data.df$BUYER_FLAG <- as.factor(data.df$BUYER_FLAG)
# 
# # data.df$AVG_FARE_L_Y1 <- data.df$FARE_L_Y1 / data.df$NUM_DEAL
# # # data.df$AVG_FARE_L_Y2 <- data.df$FARE_L_Y2 / data.df$NUM_DEAL
# # data.df$AVG_FARE_L_Y3 <- data.df$FARE_L_Y3 / data.df$NUM_DEAL
# # data.df$AVG_FARE_L_Y4 <- data.df$FARE_L_Y4 / data.df$NUM_DEAL
# # data.df$AVG_FARE_L_Y5 <- data.df$FARE_L_Y5 / data.df$NUM_DEAL
# 
# # fix imbalanced data
# data.df <- SMOTE(BUYER_FLAG ~ ., data.df, perc.over = 600, perc.under = 100)

# partition the data
train.index <- sample(1:dim(data.df)[1], dim(data.df)*0.7)
train.df <- data.df[train.index, ]
valid.df <- data.df[-train.index, ]

ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, verboseIter = TRUE, summaryFunction = f5.4, allowParallel = TRUE, sampling = "smote")

# knn
knn_grid <- expand.grid(k = 1:10)
knnFit <- train(BUYER_FLAG ~ ., data = train.df, method = "knn", trControl = ctrl, preProcess = c("center","scale", "pca"), tuneLength = 20, metric = "F5.4", maximize = TRUE)

knnPredict <- predict(knnFit, newdata = valid.df)

confusionMatrix(knnPredict, valid.df$BUYER_FLAG, positive = "1")


F_meas(knnPredict, valid.df$BUYER_FLAG, beta = 5.4, relevant = "1")

r <- roc(knnPredict, as.factor(valid.df$BUYER_FLAG))

auc(r)

# random forest
rf_grid <- expand.grid(mtry = c(2, 3, 4, 5, 10))
rf_ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, verboseIter = TRUE, summaryFunction = f5.4, sampling = "smote")

modellist <- list()
# for (ntree in c(500, 1000, 1500, 2000, 2500)) 
for (ntree in c(1000)) 
{
  print(paste("ntree = ", ntree))
  rfFit <- train(BUYER_FLAG ~ ., data = train.df, method = "rf", trControl = rf_ctrl, preProcess = c("center","scale", "pca"), tuneLength = 20, metric = "F5.4", maximize = TRUE, ntree = ntree, tuneGrid = rf_grid)
  key <- toString(ntree)
  modellist[[key]] <- rfFit
}

rfPredict <- predict(rfFit, newdata = valid.df)

confusionMatrix(rfPredict, valid.df$BUYER_FLAG, positive='1')


F_meas(rfPredict, valid.df$BUYER_FLAG, beta = 5.4, relevant = "1")

r <- roc(rfPredict, as.factor(valid.df$BUYER_FLAG))

auc(r)

# neural network
nnetCtrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, verboseIter = TRUE, summaryFunction = f5.4, allowParallel = TRUE, sampling = "smote")
nnet_grid <- expand.grid(size = c(2, 3, 4, 5, 10, 15), decay = c(0, 0.01, 0.05, 0.1))
nnetFit <- train(BUYER_FLAG ~ ., data = train.df, method = "nnet", trControl = nnetCtrl, preProcess = c("center","scale", "pca"), tuneLength = 20, metric = "F5.4", maximize = TRUE, trace = F, linout = 0, maxit = 1000, tuneGrid = nnet_grid)

nnetPredict <- predict(nnetFit, newdata = valid.df)

confusionMatrix(nnetPredict, valid.df$BUYER_FLAG, positive='1')


F_meas(nnetPredict, valid.df$BUYER_FLAG, beta = 5.4, relevant = "1")

r <- roc(nnetPredict, as.factor(valid.df$BUYER_FLAG))

auc(r)

# supported vector machines
svmFit <- train(BUYER_FLAG ~ ., data = train.df, method = "svmLinear", trControl = ctrl, preProcess = c("center","scale", "pca"), tuneLength = 20, metric = "F5.4", maximize = TRUE, maxit = 1000)

svmPredict <- predict(svmFit, newdata = valid.df)

confusionMatrix(svmPredict, valid.df$BUYER_FLAG, positive='1')


F_meas(svmPredict, valid.df$BUYER_FLAG, beta = 5.4, relevant = "1")

r <- roc(svmPredict, as.factor(valid.df$BUYER_FLAG))

auc(r)

# logistic regression
glmFit <- train(BUYER_FLAG ~ ., data = train.df, method = "glm", trControl = ctrl, preProcess = c("center","scale", "pca"), tuneLength = 20, metric = "F5.4", maximize = TRUE, maxit = 1000)

glmPredict <- predict(glmFit, newdata = valid.df)

confusionMatrix(glmPredict, valid.df$BUYER_FLAG, positive='1')


F_meas(glmPredict, valid.df$BUYER_FLAG, beta = 5.4, relevant = "1")

r <- roc(glmPredict, as.factor(valid.df$BUYER_FLAG))

auc(r)

# bayesian networks - naive bayes
nb_grid <- expand.grid(
  usekernel = c(TRUE, FALSE),
  fL = 0:5,
  adjust = seq(0, 5, by = 1)
)

nbFit <- train(BUYER_FLAG ~ ., data = train.df, method = "nb", trControl = ctrl, preProcess = c("center","scale", "pca"), tuneLength = 20, metric = "F5.4", maximize = TRUE, maxit = 1000, tuneGrid = nb_grid)

nbPredict <- predict(nbFit, newdata = valid.df)

confusionMatrix(nbPredict, valid.df$BUYER_FLAG, positive='1')


F_meas(nbPredict, valid.df$BUYER_FLAG, beta = 5.4, relevant = "1")

r <- roc(nbPredict, as.factor(valid.df$BUYER_FLAG))

auc(r)

# rpart
rpart_ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, verboseIter = TRUE, summaryFunction = f5.4, sampling = "smote")
rpart_grid <- expand.grid(cp = seq(0.001, 0.1, by = 0.001))
rpartFit <- train(BUYER_FLAG ~ ., data = train.df, method = "rpart", trControl = rpart_ctrl, preProcess = c("center","scale", "pca"), tuneLength = 20, metric = "F5.4", maximize = TRUE, tuneGrid = rpart_grid)

rpartPredict <- predict(rpartFit, newdata = valid.df)

confusionMatrix(rpartPredict, valid.df$BUYER_FLAG, positive='1')


F_meas(rpartPredict, valid.df$BUYER_FLAG, beta = 5.4, relevant = "1")

r <- roc(rpartPredict, as.factor(valid.df$BUYER_FLAG))

auc(r)

# After processing the data, we explicitly shut down the cluster
stopCluster(cluster)

