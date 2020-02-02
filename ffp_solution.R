setwd("D:/ть/RProgramming/AdvancedSubjectsData")

traindata.df <- read.csv("ffp_train.csv")

# partition the data
set.seed(1)
train.index <- sample(1:dim(traindata.df)[1], dim(traindata.df)*0.6)
train.df <- traindata.df[train.index, ]
valid.df <- traindata.df[-train.index, ]

# classification tree
library(party)
tr <- ctree(BUYER_FLAG ~ ., data = train.df)
plot(tr)
pred <- predict(tr, valid.df)

confusionMatrix(as.factor(ifelse(pred>=0.5, 1, 0)), as.factor(valid.df$BUYER_FLAG))

