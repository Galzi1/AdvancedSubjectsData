# Gal Ziv,          ID: 205564198
# Asaf Ben Avraham, ID: 311573943

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
library(ada)
library(caTools)
library(party)
library(gbm)
library(plyr)
library(mlr)

f5.4 <- function(data, lev = NULL, model = NULL) 
{
  f5.4_val <- F_meas(data$pred, data$obs, beta = 5.4, relevant = "1")
  names(f5.4_val) <- "F5.4"
  f5.4_val
}

get_outliers <- function(df, col, scale_factor)
{
  outliers1 <- c(which(df[, col] > median(df[, col]) + scale_factor * sd(df[, col])))
  outliers2 <- c(which(df[, col] < median(df[, col]) - scale_factor * sd(df[, col])))
  union(outliers1, outliers2)
}

set.seed(5)

# creating clusters in order to run in parallel
cl <- makeCluster(detectCores() - 1, outfile = "")
clusterEvalQ(cl, { library(yardstick) })
clusterEvalQ(cl, { library(AUC) })
clusterEvalQ(cl, { library(kernlab) })
clusterEvalQ(cl, { library(klaR) })
clusterEvalQ(cl, { library(rpart) })
clusterEvalQ(cl, { library(caret) })
clusterEvalQ(cl, { library(ada) })
clusterEvalQ(cl, { library(caTools) })
clusterEvalQ(cl, { library(gbm) })
clusterEvalQ(cl, { library(plyr) })
clusterEvalQ(cl, { library(mlr) })
registerDoSNOW(cl)

# loading both files into dataframes and merging them
data.df <- read.csv("ffp_train.csv")
reviews.df <- read.csv("reviews_training_result.csv")

data.df <- merge(data.df, reviews.df, by = "ID", all = T)

# removing first column ("ID")
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

# remove columns with very low importance
data.df <- data.df[ , -which(names(data.df) %in% c("GROUP", "CREDIT_PROBLEM", "RETURN_FLAG"))]

data.df[["rating"]][is.na(data.df[["rating"]])] <- 2
data.df$rating <- as.factor(data.df$rating)
data.df <- na.omit(data.df)

# remove outliers in integer/numeric variables
outliers <- c()

for (i in names(data.df))
{
  if (class(data.df[, i]) == "integer" || class(data.df[, i]) == "numeric")
  {
    outliers <- union(outliers, get_outliers(data.df, i, 3))
  }
}

data.df <- data.df[-outliers, ]

# create new variable
data.df$AVG_FARE_L_Y1 <- ifelse(data.df$NUM_DEAL == 0, 0, data.df$FARE_L_Y1 / data.df$NUM_DEAL)
data.df$AVG_FARE_L_Y2 <- ifelse(data.df$NUM_DEAL == 0, 0, data.df$FARE_L_Y2 / data.df$NUM_DEAL)
data.df$AVG_FARE_L_Y3 <- ifelse(data.df$NUM_DEAL == 0, 0, data.df$FARE_L_Y3 / data.df$NUM_DEAL)
data.df$AVG_FARE_L_Y4 <- ifelse(data.df$NUM_DEAL == 0, 0, data.df$FARE_L_Y4 / data.df$NUM_DEAL)
data.df$AVG_FARE_L_Y5 <- ifelse(data.df$NUM_DEAL == 0, 0, data.df$FARE_L_Y5 / data.df$NUM_DEAL)

# remove outliers in newly created integer/numeric variables
outliers <- c()

for (i in c("AVG_FARE_L_Y1", "AVG_FARE_L_Y2", "AVG_FARE_L_Y3", "AVG_FARE_L_Y4", "AVG_FARE_L_Y5"))
{
  if (class(data.df[, i]) == "integer" || class(data.df[, i]) == "numeric")
  {
    outliers <- union(outliers, get_outliers(data.df, i, 3))
  }
}

data.df <- data.df[-outliers, ]

# binning of integer/numeric variables
for (i in names(data.df))
{
  if (class(data.df[, i]) == "integer" || class(data.df[, i]) == "numeric")
  {
    low_cut <- sort(data.df[, i])[0.30*length(data.df[, i])]
    med_cut <- sort(data.df[, i])[0.70*length(data.df[, i])]
    bins <- c(-Inf, low_cut, med_cut, Inf)
    names <- c("Low", "Medium", "High")
    data.df[, i] <- cut(data.df[, i], breaks = bins, labels = names)
    
    data.df <- createDummyFeatures(data.df, cols = i)
    data.df[, ncol(data.df)] <- as.factor(data.df[, ncol(data.df)])
    data.df[, ncol(data.df) - 1] <- as.factor(data.df[, ncol(data.df) - 1])
    data.df[, ncol(data.df) - 2] <- as.factor(data.df[, ncol(data.df) - 2])
  }
}

# partition the data
train.index <- sample(1:dim(data.df)[1], dim(data.df)*0.7)
train.df <- data.df[train.index, ]
valid.df <- data.df[-train.index, ]

ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, verboseIter = TRUE, summaryFunction = f5.4, allowParallel = TRUE, sampling = "smote")

# knn
knn_grid <- expand.grid(k = 1:10)
set.seed(5)
knnFit <- caret::train(BUYER_FLAG ~ ., data = train.df, method = "knn", trControl = ctrl, preProcess = c("zv", "center","scale", "pca"), tuneLength = 20, metric = "F5.4", maximize = TRUE)

knnPredict <- predict(knnFit, newdata = valid.df)

knnConfMat <- confusionMatrix(knnPredict, valid.df$BUYER_FLAG, positive='1')
knnConfMat
knnTP <- knnConfMat$table[2,2]
knnFP <- knnConfMat$table[2,1]
knnFN <- knnConfMat$table[1,2]
knnProfit <- knnTP * 54 - knnFP * 10 - knnFN * 54
knnProfit

knnF5.4 <- F_meas(knnPredict, valid.df$BUYER_FLAG, beta = 5.4, relevant = "1")
knnF5.4

knn_r <- roc(knnPredict, as.factor(valid.df$BUYER_FLAG))

knnAUC <- auc(knn_r)
knnAUC

# random forest
rf_grid <- expand.grid(mtry = c(2, 3, 4, 5, 10))
rf_ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, verboseIter = TRUE, summaryFunction = f5.4, sampling = "smote")

set.seed(5)
rfFit <- caret::train(BUYER_FLAG ~ ., data = train.df, method = "rf", trControl = rf_ctrl, preProcess = c("zv", "center","scale", "pca"), tuneLength = 20, metric = "F5.4", maximize = TRUE, ntree = 2000, tuneGrid = rf_grid)

rfPredict <- predict(rfFit, newdata = valid.df)

rfConfMat <- confusionMatrix(rfPredict, valid.df$BUYER_FLAG, positive='1')
rfConfMat
rfTP <- rfConfMat$table[2,2]
rfFP <- rfConfMat$table[2,1]
rfFN <- rfConfMat$table[1,2]
rfProfit <- rfTP * 54 - rfFP * 10 - rfFN * 54
rfProfit

rfF5.4 <- F_meas(rfPredict, valid.df$BUYER_FLAG, beta = 5.4, relevant = "1")
rfF5.4

rf_r <- roc(rfPredict, as.factor(valid.df$BUYER_FLAG))

rfAUC <- auc(rf_r)
rfAUC

# neural network
nnetCtrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, verboseIter = TRUE, summaryFunction = f5.4, allowParallel = TRUE, sampling = "smote")
nnet_grid <- expand.grid(size = c(2, 3, 4, 5, 10, 15), decay = c(0, 0.01, 0.05, 0.1))
set.seed(5)
nnetFit <- caret::train(BUYER_FLAG ~ ., data = train.df, method = "nnet", trControl = nnetCtrl, preProcess = c("zv", "center","scale", "pca"), tuneLength = 20, metric = "F5.4", maximize = TRUE, trace = F, linout = 0, maxit = 1000, tuneGrid = nnet_grid)

nnetPredict <- predict(nnetFit, newdata = valid.df)

nnetConfMat <- confusionMatrix(nnetPredict, valid.df$BUYER_FLAG, positive='1')
nnetConfMat
nnetTP <- nnetConfMat$table[2,2]
nnetFP <- nnetConfMat$table[2,1]
nnetFN <- nnetConfMat$table[1,2]
nnetProfit <- nnetTP * 54 - nnetFP * 10 - nnetFN * 54
nnetProfit


nnetF5.4 <- F_meas(nnetPredict, valid.df$BUYER_FLAG, beta = 5.4, relevant = "1")
nnetF5.4

nnet_r <- roc(nnetPredict, as.factor(valid.df$BUYER_FLAG))

nnetAUC <- auc(r)
nnetAUC

# supported vector machines with radial basis function kernel
svmr_grid <- expand.grid(C = c(-2, 0, 0.5, 2, 5, 10, 100), sigma = c(0.0001, 0.001, 0.01, 0.1, 1))
set.seed(5)
svmrFit <- caret::train(BUYER_FLAG ~ ., data = train.df, method = "svmRadial", trControl = ctrl, preProcess = c("zv", "center","scale", "pca"), tuneLength = 20, metric = "F5.4", maximize = TRUE, maxit = 1000, tuneGrid = svmr_grid)

svmrPredict <- predict(svmrFit, newdata = valid.df)

svmrConfMat <- confusionMatrix(svmrPredict, valid.df$BUYER_FLAG, positive='1')
svmrConfMat
svmrTP <- svmrConfMat$table[2,2]
svmrFP <- svmrConfMat$table[2,1]
svmrFN <- svmrConfMat$table[1,2]
svmrProfit <- svmrTP * 54 - svmrFP * 10 - svmrFN * 54
svmrProfit

svmrF5.4 <- F_meas(svmrPredict, valid.df$BUYER_FLAG, beta = 5.4, relevant = "1")
svmrF5.4

svmr_r <- roc(svmrPredict, as.factor(valid.df$BUYER_FLAG))

svmrAUC <- auc(svmr_r)
svmrAUC

# bayesian networks - naive bayes
nb_grid <- expand.grid(
  usekernel = c(TRUE, FALSE),
  fL = 0:5,
  adjust = seq(0, 5, by = 1)
)

set.seed(5)
nbFit <- caret::train(BUYER_FLAG ~ ., data = train.df, method = "nb", trControl = ctrl, preProcess = c("zv", "center","scale", "pca"), tuneLength = 20, metric = "F5.4", maximize = TRUE, maxit = 1000, tuneGrid = nb_grid)

nbPredict <- predict(nbFit, newdata = valid.df)

nbConfMat <- confusionMatrix(nbPredict, valid.df$BUYER_FLAG, positive='1')
nbConfMat
nbTP <- nbConfMat$table[2,2]
nbFP <- nbConfMat$table[2,1]
nbFN <- nbConfMat$table[1,2]
nbProfit <- nbTP * 54 - nbFP * 10 - nbFN * 54
nbProfit

nbF5.4 <- F_meas(nbPredict, valid.df$BUYER_FLAG, beta = 5.4, relevant = "1")
nbF5.4

nb_r <- roc(nbPredict, as.factor(valid.df$BUYER_FLAG))

nbAUC <- auc(nb_r)
nbAUC

# rpart
rpart_ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, verboseIter = TRUE, summaryFunction = f5.4, sampling = "smote")
rpart_grid <- expand.grid(cp = seq(0.01, 1, by = 0.01))
set.seed(5)
rpartFit <- caret::train(BUYER_FLAG ~ ., data = train.df, method = "rpart", trControl = rpart_ctrl, preProcess = c("zv", "center","scale", "pca"), tuneLength = 20, metric = "F5.4", maximize = TRUE, tuneGrid = rpart_grid)

rpartPredict <- predict(rpartFit, newdata = valid.df)

rpartConfMat <- confusionMatrix(rpartPredict, valid.df$BUYER_FLAG, positive='1')
rpartConfMat
rpartTP <- rpartConfMat$table[2,2]
rpartFP <- rpartConfMat$table[2,1]
rpartFN <- rpartConfMat$table[1,2]
rpartProfit <- rpartTP * 54 - rpartFP * 10 - rpartFN * 54
rpartProfit

rpartF5.4 <- F_meas(rpartPredict, valid.df$BUYER_FLAG, beta = 5.4, relevant = "1")
rpartF5.4

rpart_r <- roc(rpartPredict, as.factor(valid.df$BUYER_FLAG))

rpartAUC <- auc(r)
rpartAUC

# ctree
ctree_ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, verboseIter = TRUE, summaryFunction = f5.4, sampling = "smote")
ctree_grid <- expand.grid(mincriterion = seq(0.05, 0.95, by = 0.05))
set.seed(5)
ctreeFit <- caret::train(BUYER_FLAG ~ ., data = train.df, method = "ctree", trControl = ctree_ctrl, preProcess = c("zv", "center","scale", "pca"), tuneLength = 20, metric = "F5.4", maximize = TRUE, tuneGrid = ctree_grid)

ctreePredict <- predict(ctreeFit, newdata = valid.df)

ctreeConfMat <- confusionMatrix(ctreePredict, valid.df$BUYER_FLAG, positive='1')
ctreeConfMat
ctreeTP <- ctreeConfMat$table[2,2]
ctreeFP <- ctreeConfMat$table[2,1]
ctreeFN <- ctreeConfMat$table[1,2]
ctreeProfit <- ctreeTP * 54 - ctreeFP * 10 - ctreeFN * 54
ctreeProfit

ctreeF5.4 <- F_meas(ctreePredict, valid.df$BUYER_FLAG, beta = 5.4, relevant = "1")
ctreeF5.4

ctree_r <- roc(ctreePredict, as.factor(valid.df$BUYER_FLAG))

ctreeAUC <- auc(ctree_r)
ctreeAUC

# adaptive boosted classification trees (ada-boost)
ada_grid <- expand.grid(iter = c(100, 300, 500), maxdepth = c(2, 3, 4, 5), nu = c(0.01, 0.05, 0.1))
set.seed(5)
adaFit <- caret::train(BUYER_FLAG ~ ., data = train.df, method = "ada", trControl = ctrl, preProcess = c("zv", "center","scale", "pca"), tuneLength = 20, metric = "F5.4", maximize = TRUE, tuneGrid = ada_grid)

adaPredict <- predict(adaFit, newdata = valid.df)

adaConfMat <- confusionMatrix(adaPredict, valid.df$BUYER_FLAG, positive='1')
adaConfMat
adaTP <- adaConfMat$table[2,2]
adaFP <- adaConfMat$table[2,1]
adaFN <- adaConfMat$table[1,2]
adaProfit <- adaTP * 54 - adaFP * 10 - adaFN * 54
adaProfit

adaF5.4 <- F_meas(adaPredict, valid.df$BUYER_FLAG, beta = 5.4, relevant = "1")
adaF5.4

ada_r <- roc(adaPredict, as.factor(valid.df$BUYER_FLAG))

adaAUC <- auc(ada_r)
adaAUC

# logistic regression
set.seed(5)
glmFit <- caret::train(BUYER_FLAG ~ ., data = train.df, method = "glm", trControl = ctrl, preProcess = c("zv", "center","scale", "pca"), tuneLength = 20, metric = "F5.4", maximize = TRUE, family="binomial")

glmPredict <- predict(glmFit, newdata = valid.df)

glmConfMat <- confusionMatrix(glmPredict, valid.df$BUYER_FLAG, positive='1')
glmConfMat
glmTP <- glmConfMat$table[2,2]
glmFP <- glmConfMat$table[2,1]
glmFN <- glmConfMat$table[1,2]
glmProfit <- glmTP * 54 - glmFP * 10 - glmFN * 54
glmProfit

glmF5.4 <- F_meas(glmPredict, valid.df$BUYER_FLAG, beta = 5.4, relevant = "1")
glmF5.4

glm_r <- roc(glmPredict, as.factor(valid.df$BUYER_FLAG))

glmAUC <- auc(glm_r)
glmAUC

# stochastic gradient boosting
gbm_grid <- expand.grid(n.trees = c(50, 100, 500, 1000, 1500, 2000), interaction.depth = (1:5), shrinkage = c(0.005, 0.01, 0.05, 0.1, 0.5), n.minobsinnode = c(5, 10, 15))

set.seed(5)
gbmFit <- caret::train(BUYER_FLAG ~ ., data = train.df, method = "gbm", trControl = ctrl, preProcess = c("zv", "center","scale", "pca"), metric = "F5.4", maximize = TRUE, tuneGrid = gbm_grid, distribution = "bernoulli")

gbmPredict <- predict(gbmFit, newdata = valid.df)

gbmConfMat <- confusionMatrix(gbmPredict, valid.df$BUYER_FLAG, positive='1')
gbmConfMat
gbmTP <- gbmConfMat$table[2,2]
gbmFP <- gbmConfMat$table[2,1]
gbmFN <- gbmConfMat$table[1,2]
gbmProfit <- gbmTP * 54 - gbmFP * 10 - gbmFN * 54
gbmProfit

gbmF5.4 <- F_meas(gbmPredict, valid.df$BUYER_FLAG, beta = 5.4, relevant = "1")
gbmF5.4

gbm_r <- roc(gbmPredict, as.factor(valid.df$BUYER_FLAG))

gbmAUC <- auc(gbm_r)
gbmAUC

# After processing the data, we explicitly shut down the cluster
stopCluster(cluster)
