################################## text_training

text_training.df <- read.csv("text_training.csv")
text_training.df$rating = factor(text_training.df$rating)

library(randomForest)
library(AUC)
library(party)
set.seed(4543)
rffit <- randomForest(rating ~ ., data=text_training.df[, -1], ntree=2000, keep.forest=FALSE, importance=TRUE)
importance(rffit) # relative importance of predictors (highest <-> most important)
a <- varImpPlot(rffit)

c <- a[a[, 1] > 0,]
d <- c[order(-c[,1]),]
write.csv(d[,0], "params_names.csv")
parames_names.df <- read.csv("params_names.csv")

params <- sapply(parames_names.df, as.character)

temp <- text_training.df

params <- rbind(tmp, 1)
params[1525,] = "rating"


text_training_variable_reduce <- text_training.df[ , which(names(text_training.df) %in% params)]
write.csv(text_training_variable_reduce, "text_training_variable_reduce.csv")

text_training_variable_reduce.df <- read.csv("text_training_variable_reduce.csv")


# train.index <- sample(1:dim(text_training_variable_reduce.df)[1], dim(text_training_variable_reduce.df)*0.6)
# train.df <- text_training_variable_reduce.df[train.index, ]
# valid.df <- text_training_variable_reduce.df[-train.index, ]
# 
# 
# tr <- ctree(rating ~ ., data = train.df)
# pred <- predict(tr, newdata = valid.df)
# r <- roc(pred, as.factor(valid.df$rating))
# auc(r)
# 
# 
# #Create random forests
# rf <- cforest(rating ~ ., data = train.df)
# pred <- predict(rf, newdata = valid.df)
# r <- roc(pred, as.factor(valid.df$rating))
# auc(r)
# 
# library(nnet)
# nn <- nnet(rating ~ ., 
#            data = train.df[, -1], linear.output = F, size = 2, 
#            decay = 0.01, maxit = 200)
# pred <- predict(nn, valid.df)
# r <- roc(pred, as.factor(valid.df$rating))
# auc(r)



############### ffp_train
ffp_train.df <- read.csv("ffp_train.csv")
ffp_train.df$BUYER_FLAG = factor(ffp_train.df$BUYER_FLAG)
ffp_train.df

library(randomForest)
set.seed(4543)

rffit_b <- randomForest(BUYER_FLAG ~ ., data=ffp_train.df, ntree=2000, keep.forest=FALSE, importance=TRUE)
importance(rffit_b) # relative importance of predictors (highest <-> most important)
a_b <- varImpPlot(rffit_b)

c_b <- a_b[a_b[, 1] > 0,]
d_b <- c_b[order(-c_b[,1]),]

write.csv(d_b[,0], "params_b_names.csv") # In this stage we need to delete the first row from the csv(need to be fixed)
parames_b_n.df <- read.csv("params_b_names.csv")
parames_b_n.df

tmp_b <- sapply(parames_b_n.df, as.character)
tmp_b <- rbind(tmp_b, 1)
tmp_b[18,] = "BUYER_FLAG"

ffp_train_variable_reduce <- ffp_train.df[ , which(names(ffp_train.df) %in% tmp_b)]
write.csv(ffp_train_variable_reduce, "ffp_train_variable_reduce.csv")


ffp_train_variable_reduce.df <- read.csv("ffp_train_variable_reduce.csv")


library(clusterSim)
###quotient transformation (x/max)
ffp_train_variable_reduce.df = data.Normalization (ffp_train_variable_reduce.df,type="n8",normalization="column")
ffp_train_variable_reduce.df

library(ISLR)
smp_siz = floor(0.85*nrow(ffp_train_variable_reduce.df)) 

train_ind = sample(seq_len(nrow(ffp_train_variable_reduce.df)),size = smp_siz)  # Randomly identifies therows equal to sample size ( defined in previous instruction) from  all the rows of Smarket dataset and stores the row number in train_ind
train_ind
train =ffp_train_variable_reduce.df[train_ind,] #creates the training dataset with row numbers stored in train_ind
test=ffp_train_variable_reduce.df[-train_ind,]


train
test


#require(caTools)
#sample = sample.split(ffp_train_variable_reduce.df,SplitRatio = 0.99)
#train1 =subset(ffp_train_variable_reduce.df,sample ==TRUE) 
#test1=subset(ffp_train_variable_reduce.df, sample==FALSE)

#train1
#test1


# train.index <- sample(1:dim(ffp_train_variable_reduce.df)[1], dim(ffp_train_variable_reduce.df)*0.6)
# train <- ffp_train_variable_reduce.df[train.index, ]
# valid <- ffp_train_variable_reduce.df[-train.index, ]



tr <- ctree(BUYER_FLAG ~ ., data = train)
pred <- predict(tr, newdata = test)
r <- roc(pred, as.factor(test$BUYER_FLAG))
auc(r)



amdl_lr_v1 = glm(
  BUYER_FLAG ~ ., 
  family="binomial" , data=train)

mdl_lr_v1
summary(mdl_lr_v1)


y_tr = (test$BUYER_FLAG == 1) 
y_tr
#' 2. Find the predicted probabilities of each model for the testing set
#'    (use the table above):

#' 2.1. Logistic Regression model version 1   
y_hat_lr = predict(mdl_lr_v1, newdata=test, type="response")
y_hat_lr

# Choose a classification cutoff threshold, above it the probability values would be 
#'      considered as '1' (that is, predict that the passenger has survived)
#' In our case, we choose the cutoff value as an arbitrary 0.5  
threshold = 0.5
y_hat_lr = (y_hat_lr > threshold)


sum(y_hat_lr==TRUE)
sum(y_hat_lr==FALSE)


sum(y_tr==TRUE)
sum(y_tr==FALSE)


#' 2.4. Calculate the confusion matrix manually
tp_lr = sum(((y_hat_lr==TRUE) & (y_tr==TRUE)))   # true positive
fp_lr = sum(((y_hat_lr==TRUE) & (y_tr==FALSE)))  # false positive
fn_lr = sum(((y_hat_lr==FALSE) & (y_tr==TRUE)))  # false negative
tn_lr = sum(((y_hat_lr==FALSE) & (y_tr==FALSE))) # true negative

tp_lr
fp_lr
fn_lr
tn_lr



recall_lr = tp_lr/(tp_lr+fn_lr)
recall_lr


mdl_rf_v1 = randomForest(
  BUYER_FLAG ~ ., 
  data=train, ntree=2000)

y_hat_rf = predict(mdl_rf_v1, newdata=test, type="prob")##??
y_hat_rf = predict(mdl_rf_v1, newdata=test, type="response")#??


threshold = 0.5
y_hat_rf = (y_hat_rf > threshold)

tp_rf = sum(((y_hat_rf==TRUE) & (y_tr==TRUE)))   # true positive
fp_rf = sum(((y_hat_rf==TRUE) & (y_tr==FALSE)))  # false positive
fn_rf = sum(((y_hat_rf==FALSE) & (y_tr==TRUE)))  # false negative
tn_rf = sum(((y_hat_rf==FALSE) & (y_tr==FALSE))) # true negative

tp_rf
fp_rf
fn_rf
tn_rf


# sanity check
# sum(tp_lr,fp_lr,fn_lr,tn_lr) # should add up to the number of rows in `testing` 
# sum(tp_rf,fp_rf,fn_rf,tn_rf) # should add up to the number of rows in `testing` 
# nrow(testing)

#' 2.5. Calculate criteria
#' 
#' 2.5.1. Recall
recall_lr = tp_lr/(tp_lr+fn_lr)
recall_lr
recall_rf = tp_rf/(tp_rf+fn_rf)
recall_rf

#' 2.5.2. Precision
precision_lr = tp_lr/(tp_lr+fp_lr)
precision_lr
precision_rf = tp_rf/(tp_rf+fp_rf)
precision_rf


f1_lr=2*(precision_lr*recall_lr)/(precision_lr+recall_lr)
f1_lr

f1_rf=2*(precision_rf*recall_rf)/(precision_rf+recall_rf)
f1_rf

###############

