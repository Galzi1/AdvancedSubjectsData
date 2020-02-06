setwd("D:/ть/RProgramming/AdvancedSubjectsData")

set.seed(5)

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
plot(tr)
pred <- predict(tr, valid.df, type = "class")

library(caret)
confusionMatrix(pred, valid.df$BUYER_FLAG, positive='1')

library(yardstick)
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 2)
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 5.4)

library(AUC)
r <- roc(pred, as.factor(valid.df$BUYER_FLAG))
plot(r)
auc(r)

# classification tree - ctree
library(party)
tr <- ctree(BUYER_FLAG ~ ., data = train.df)
plot(tr)
pred <- predict(tr, valid.df)

library(caret)
confusionMatrix(pred, valid.df$BUYER_FLAG, positive='1')

library(yardstick)
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 2)
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 5.4)

library(AUC)
r <- roc(pred, as.factor(valid.df$BUYER_FLAG))
plot(r)
auc(r)

# classification tree - C4.5 algorithm (J48 in RWeka)
library(RWeka)
tr <- J48(BUYER_FLAG ~ ., data = train.df)
# make predictions
pred <- predict(tr, valid.df)

library(caret)
confusionMatrix(pred, valid.df$BUYER_FLAG, positive='1')

library(yardstick)
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 2)
f_meas_vec(pred, valid.df$BUYER_FLAG, beta = 5.4)

library(AUC)
r <- roc(pred, as.factor(valid.df$BUYER_FLAG))
plot(r)
auc(r)

# random forest
rf <- cforest(BUYER_FLAG ~ ., data = train.df)
plot(rf)
pred2 <- predict(rf, valid.df)

confusionMatrix(as.factor(ifelse(pred2>=0.5, 1, 0)), as.factor(valid.df$BUYER_FLAG))