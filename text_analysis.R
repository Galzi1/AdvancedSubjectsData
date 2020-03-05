################################## text_training
setwd("E:\\proj\\")


text_training.df <- read.csv("text_training.csv")
text_training.df$rating = factor(text_training.df$rating)

library(randomForest)
library(AUC)
library(party)
library(caret)
set.seed(5)
rffit <- randomForest(rating ~ ., data=text_training.df[, -1], ntree=2000, keep.forest=FALSE, importance=TRUE)
importance(rffit) # relative importance of predictors (highest <-> most important)
a <- varImpPlot(rffit)

c <- a[a[, 1] > 0,]
d <- c[order(-c[,1]),]
write.csv(d[,0], "params_names.csv")
parames_names.df <- read.csv("params_names.csv")

params <- sapply(parames_names.df, as.character)

params <- rbind(params, 1)

params[length(params),] = "rating"

text_training_variable_reduce <- text_training.df[ , which(names(text_training.df) %in% params)]
write.csv(text_training_variable_reduce, "text_training_variable_reduce.csv")


###Apply model by training reduced variables data
text_training_variable_reduce.df <- read.csv("text_training_variable_reduce.csv")

reviews_training.df <- read.csv("reviews_training.csv")

reviews_rollout.df <- read.csv("reviews_rollout.csv")


library(clusterSim)
text_training_variable_reduce.df$rating = as.factor(text_training_variable_reduce.df$rating)

text_training_variable_reduce.df = text_training_variable_reduce.df[ ,-1]


train_control_4 <- trainControl(method = "cv", number = 4)

RF_model_4 <- train(rating ~ ., data = text_training_variable_reduce.df, trControl = train_control_4, method = "cforest")

reviews_training_ouput <- as.numeric(predict(RF_model_4, newdata = reviews_training.df)) - 1
write.csv(cbind(ID = reviews_training.df$ID, rating = reviews_training_ouput), "reviews_training_result.csv", row.names = FALSE)


reviews_rollout_rating <- as.numeric(predict(RF_model_4, newdata = reviews_rollout.df)) - 1
write.csv(cbind(ID = reviews_rollout.df$ID, rating = reviews_rollout_rating), "reviews_rolout_result.csv", row.names = FALSE)

