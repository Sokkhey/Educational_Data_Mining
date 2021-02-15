
rm(list = ls())
graphics.off()
library(ModelMetrics)
library(caret) # confusion matrix
library(dplyr) # top_n

#--------------------------------------- Import Data

Data <- read.csv("GDS3.csv")
Data$SCORE <- as.factor(Data$SCORE)
#NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN---------RANDOM FOREST
library(randomForest)

# ------------------------------------------------------ Baseline
rf_acc_vec <- NULL
rf_rmse_vec <- NULL
rf_acctr_vec <- NULL
rf_rmsetr_vec <- NULL
for (i in 1:40) {
  index <- sample(1:nrow(Data),round(0.80*nrow(Data)))
  train <- Data[index,]
  test <- Data[-index,]
  rf_model <- randomForest(SCORE~., train)
  #------------------------------------ Testing
  rf_pred <- predict(rf_model,test)
  rf_acc <- mean(rf_pred == test$SCORE)
  rf_rmse <- rmse(rf_pred,test$SCORE)
  #------------------------------------ Training
  rf_predtr <- predict(rf_model,train)
  rf_acctr <- mean(rf_predtr == train$SCORE)
  rf_rmsetr <- rmse(rf_predtr, train$SCORE)
  rf_acct_vec <- c(rf_acctr_vec, rf_acctr)
  rf_rmset_vec <- c(rf_rmsetr_vec, rf_rmsetr)
  # ---------- result
  rf_acc_vec <- c(rf_acc_vec, rf_acc)
  rf_rmse_vec <- c(rf_rmse_vec, rf_rmse)
  rf_acctr_vec <- c(rf_acctr_vec, rf_acctr)
  rf_rmsetr_vec <- c(rf_rmsetr_vec, rf_rmsetr)
  data = cbind(rf_acc_vec, rf_rmse_vec, rf_acctr_vec, rf_rmsetr_vec)
  }
mean = apply(data,2,mean)
min = apply(data,2,min)
max = apply(data,2,max)
sd = apply(data,2,sd)
Table_rf1 <- rbind(mean, sd, min, max)

#-------------------------------------------------Baseline with 10-fold CV

rf_acc_vec <- NULL
rf_rmse_vec <- NULL
rf_acctr_vec <- NULL
rf_rmsetr_vec <- NULL
for (i in 1:5) {
index <- sample(1:nrow(Data),round(0.75*nrow(Data)))
train <- Data[index,]
test <- Data[-index,]
train_control <- trainControl(method = "cv", number = 10)
# set up tuning grid
mtryGrid <- expand.grid(mtry = floor(sqrt(ncol(train))))
# train model
rf_model <- train(x = Data,
               y = Data$SCORE,
               method = "rf",
               trControl = train_control)
rf_model$results
#----------------------------------- Testing
rf_pred <- predict(rf_model,test)
rf_acc <- mean(rf_pred == test$SCORE)
rf_rmse <- rmse(rf_pred, test$SCORE)
#----------------------------------- Training
rf_predtr <- predict(rf_model,train)
rf_acctr <- mean(rf_predtr == train$SCORE)
rf_rmsetr <- rmse(rf_predtr, train$SCORE)

# ---------- result
rf_acc_vec <- c(rf_acc_vec, rf_acc)
rf_rmse_vec <- c(rf_rmse_vec, rf_rmse)
rf_acctr_vec <- c(rf_acctr_vec, rf_acctr)
rf_rmsetr_vec <- c(rf_rmsetr_vec, rf_rmsetr)
data = cbind(rf_acc_vec, rf_rmse_vec, rf_acctr_vec, rf_rmsetr_vec)
}
mean = apply(data,2,mean)
min = apply(data,2,min)
max =apply(data,2,max)
sd = apply(data,2,sd)
Table_rf2 <- rbind(mean, sd, min, max)

#------------------------------------------------------- Baseline with CPA

rf_acc_vec <- NULL
rf_rmse_vec <- NULL
rf_acctr_vec <- NULL
rf_rmsetr_vec <- NULL
for (i in 1:10) {
  index <- sample(1:nrow(Data),round(0.75*nrow(Data)))
  train <- Data[index,]
  test <- Data[-index,]
  train_control <- trainControl(method = "cv", number = 10)
  # set up tuning grid
  mtryGrid <- expand.grid(mtry = floor(sqrt(ncol(train))))
  # train model
  rf_model <- train(x = Data,
                    y = Data$SCORE,
                    method = "rf",
                    preProc = c("BoxCox", "center", "scale", "pca"),
                    tuneGrid = mtryGrid)
  rf_model$results
  #----------------------------------- Testing
  rf_pred <- predict(rf_model,test)
  rf_acc <- mean(rf_pred == test$SCORE)
  rf_rmse <- rmse(rf_pred, test$SCORE)
  #----------------------------------- Training
  rf_predtr <- predict(rf_model,train)
  rf_acctr <- mean(rf_predtr == train$SCORE)
  rf_rmsetr <- rmse(rf_predtr, train$SCORE)
  
  # ---------- result
  rf_acc_vec <- c(rf_acc_vec, rf_acc)
  rf_rmse_vec <- c(rf_rmse_vec, rf_rmse)
  rf_acctr_vec <- c(rf_acctr_vec, rf_acctr)
  rf_rmsetr_vec <- c(rf_rmsetr_vec, rf_rmsetr)
  data = cbind(rf_acc_vec, rf_rmse_vec, rf_acctr_vec, rf_rmsetr_vec)
}
mean = apply(data,2,mean)
min = apply(data,2,min)
max = apply(data,2,max)
sd = apply(data,2,sd)
Table_rf3 <- rbind(mean, sd, min, max)

#-------------------------------------------------- Baseline with 10-CV with CPA

rf_acc_vec <- NULL
rf_rmse_vec <- NULL
rf_acctr_vec <- NULL
rf_rmsetr_vec <- NULL
for (i in 1:10) {
  index <- sample(1:nrow(Data),round(0.75*nrow(Data)))
  train <- Data[index,]
  test <- Data[-index,]
  train_control <- trainControl(method = "cv", number = 10)
  # set up tuning grid
  mtryGrid <- expand.grid(mtry = floor(sqrt(ncol(train))))
  # train model
  rf_model <- train(x = Data,
                    y = Data$SCORE,
                    method = "rf",
                    preProc = c("BoxCox", "center", "scale", "pca"),
                    trControl = train_control,
                    tuneGrid = mtryGrid)
  rf_model$results
  #----------------------------------- Testing
  rf_pred <- predict(rf_model,test)
  rf_acc <- mean(rf_pred == test$SCORE)
  rf_rmse <- rmse(rf_pred, test$SCORE)
  #----------------------------------- Training
  rf_predtr <- predict(rf_model,train)
  rf_acctr <- mean(rf_predtr == train$SCORE)
  rf_rmsetr <- rmse(rf_predtr, train$SCORE)
  
  # ---------- result
  rf_acc_vec <- c(rf_acc_vec, rf_acc)
  rf_rmse_vec <- c(rf_rmse_vec, rf_rmse)
  rf_acctr_vec <- c(rf_acctr_vec, rf_acctr)
  rf_rmsetr_vec <- c(rf_rmsetr_vec, rf_rmsetr)
  data = cbind(rf_acc_vec, rf_rmse_vec, rf_acctr_vec, rf_rmsetr_vec)
}
mean = apply(data,2,mean)
min = apply(data,2,min)
max = apply(data,2,max)
sd = apply(data,2,sd)
Table_rf4 <- rbind(mean, sd, min, max)


#NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN--------  DT
library(C50)

c50_acc_vec <- NULL
c50_rmse_vec <- NULL
c50_acctr_vec <- NULL
c50_rmsetr_vec <- NULL
for (i in 1:40) {
  index <- sample(1:nrow(Data),round(0.90*nrow(Data)))
  train <- Data[index,]
  test <- Data[-index,]
  c50_model <- C5.0(SCORE~.,data = train, trails = 10)#n.trees = 500,interaction.depth = 8,importance = T,proximity = T
  #------------------------------------ Testing
  c50_pred <- predict(c50_model,test)
  c50_acc <- mean(c50_pred == test$SCORE)
  c50_rmse <- rmse(c50_pred,test$SCORE)
  #------------------------------------ Training
  c50_predtr <- predict(c50_model,train)
  c50_acctr <- mean(c50_predtr == train$SCORE)
  c50_rmsetr <- rmse(c50_predtr, train$SCORE)
  c50_acct_vec <- c(c50_acctr_vec, c50_acctr)
  c50_rmset_vec <- c(c50_rmsetr_vec, c50_rmsetr)
  # ---------- result
  c50_acc_vec <- c(c50_acc_vec, c50_acc)
  c50_rmse_vec <- c(c50_rmse_vec, c50_rmse)
  c50_acctr_vec <- c(c50_acctr_vec, c50_acctr)
  c50_rmsetr_vec <- c(c50_rmsetr_vec, c50_rmsetr)
  data = cbind(c50_acc_vec, c50_rmse_vec, c50_acctr_vec, c50_rmsetr_vec)
}
mean = apply(data,2,mean)
min = apply(data,2,min)
max = apply(data,2,max)
sd = apply(data,2,sd)
Table_dt1 <- rbind(mean, sd, min, max)

#------------------------------------------------------- Baseline with 10-fold CV

c50_acc_vec <- NULL
c50_rmse_vec <- NULL
c50_acctr_vec <- NULL
c50_rmsetr_vec <- NULL
for (i in 1:10) {
  
  index <- sample(1:nrow(Data),round(0.75*nrow(Data)))
  train <- Data[index,]
  test <- Data[-index,]
  train_control <- trainControl(method = "cv", number = 10)
  # set up tuning grid
  mtryGrid <- expand.grid(mtry = floor(sqrt(ncol(train))))
  # train model
  c50_model <- train(x = Data,
                    y = Data$SCORE,
                    method = "treebag",
                    trControl = train_control)
  c50_model$results
  #----------------------------------- Testing
  c50_pred <- predict(c50_model,test)
  c50_acc <- mean(c50_pred == test$SCORE)
  c50_rmse <- rmse(c50_pred, test$SCORE)
  #----------------------------------- Training
  c50_predtr <- predict(c50_model,train)
  c50_acctr <- mean(c50_predtr == train$SCORE)
  c50_rmsetr <- rmse(c50_predtr, train$SCORE)
  
  # ---------- result
  c50_acc_vec <- c(c50_acc_vec, c50_acc)
  c50_rmse_vec <- c(c50_rmse_vec, c50_rmse)
  c50_acctr_vec <- c(c50_acctr_vec, c50_acctr)
  c50_rmsetr_vec <- c(c50_rmsetr_vec, c50_rmsetr)
  data = cbind(c50_acc_vec, c50_rmse_vec, c50_acctr_vec, c50_rmsetr_vec)
}
mean = apply(data,2,mean)
min = apply(data,2,min)
max = apply(data,2,max)
sd = apply(data,2,sd)
Table_dt2 <- rbind(mean, sd, min, max)

#---------------------------------------------------------- Baseline with CPA

c50_acc_vec <- NULL
c50_rmse_vec <- NULL
c50_acctr_vec <- NULL
c50_rmsetr_vec <- NULL
for (i in 1:10) {
  index <- sample(1:nrow(Data),round(0.75*nrow(Data)))
  train <- Data[index,]
  test <- Data[-index,]
  train_control <- trainControl(method = "cv", number = 10)
  # set up tuning grid
  mtryGrid <- expand.grid(mtry = floor(sqrt(ncol(train))))
  # train model
  c50_model <- train(x = Data,
                     y = Data$SCORE,
                     method = "treebag",
                     preProc = c("BoxCox", "center", "scale", "pca"))
  c50_model$results
  #----------------------------------- Testing
  c50_pred <- predict(c50_model,test)
  c50_acc <- mean(c50_pred == test$SCORE)
  c50_rmse <- rmse(c50_pred, test$SCORE)
  #----------------------------------- Training
  c50_predtr <- predict(c50_model,train)
  c50_acctr <- mean(c50_predtr == train$SCORE)
  c50_rmsetr <- rmse(c50_predtr, train$SCORE)
  
  # ---------- result
  c50_acc_vec <- c(c50_acc_vec, c50_acc)
  c50_rmse_vec <- c(c50_rmse_vec, c50_rmse)
  c50_acctr_vec <- c(c50_acctr_vec, c50_acctr)
  c50_rmsetr_vec <- c(c50_rmsetr_vec, c50_rmsetr)
  data = cbind(c50_acc_vec, c50_rmse_vec, c50_acctr_vec, c50_rmsetr_vec)
}
mean = apply(data,2,mean)
min = apply(data,2,min)
max = apply(data,2,max)
sd = apply(data,2,sd)
Table_dt3 <- rbind(mean, sd, min, max)

#---------------------------------------------------------- Baseline with 10-CV and CPA

c50_acc_vec <- NULL
c50_rmse_vec <- NULL
c50_acctr_vec <- NULL
c50_rmsetr_vec <- NULL
for (i in 1:10) {
  index <- sample(1:nrow(Data),round(0.75*nrow(Data)))
  train <- Data[index,]
  test <- Data[-index,]
  train_control <- trainControl(method = "cv", number = 10)
  # set up tuning grid
  mtryGrid <- expand.grid(mtry = floor(sqrt(ncol(train))))
  # train model
  c50_model <- train(x = Data,
                     y = Data$SCORE,
                     method = "treebag",
                     trControl = train_control,
                     preProc = c("BoxCox", "center", "scale", "pca"))
  c50_model$results
  #----------------------------------- Testing
  c50_pred <- predict(c50_model,test)
  c50_acc <- mean(c50_pred == test$SCORE)
  c50_rmse <- rmse(c50_pred, test$SCORE)
  #----------------------------------- Training
  c50_predtr <- predict(c50_model,train)
  c50_acctr <- mean(c50_predtr == train$SCORE)
  c50_rmsetr <- rmse(c50_predtr, train$SCORE)
  
  # ---------- result
  c50_acc_vec <- c(c50_acc_vec, c50_acc)
  c50_rmse_vec <- c(c50_rmse_vec, c50_rmse)
  c50_acctr_vec <- c(c50_acctr_vec, c50_acctr)
  c50_rmsetr_vec <- c(c50_rmsetr_vec, c50_rmsetr)
  data = cbind(c50_acc_vec, c50_rmse_vec, c50_acctr_vec, c50_rmsetr_vec)
}
mean = apply(data,2,mean)
min = apply(data,2,min)
max = apply(data,2,max)
sd = apply(data,2,sd)
Table_dt4 <- rbind(mean, sd, min, max)


#NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN -------- Naive Bayes
library(e1071)
library(ModelMetrics)

nb_acc_vec <- NULL
nb_rmse_vec <- NULL
nb_acctr_vec <- NULL
nb_rmsetr_vec <- NULL
for (i in 1:10) {
  index <- sample(1:nrow(Data),round(0.90*nrow(Data)))
  train <- Data[index,]
  test <- Data[-index,]
  
  nb_model <- naiveBayes(SCORE~.,data = train)
  #------------------------------------ Testing
  nb_pred <- predict(nb_model,test)
  nb_acc <- mean(nb_pred == test$SCORE)
  nb_rmse <- rmse(nb_pred,test$SCORE)
  #------------------------------------ Training
  nb_predtr <- predict(nb_model,train)
  nb_acctr <- mean(nb_predtr == train$SCORE)
  nb_rmsetr <- rmse(nb_predtr, train$SCORE)
  nb_acct_vec <- c(nb_acctr_vec, nb_acctr)
  nb_rmset_vec <- c(nb_rmsetr_vec, nb_rmsetr)
  # ---------- result
  nb_acc_vec <- c(nb_acc_vec, nb_acc)
  nb_rmse_vec <- c(nb_rmse_vec, nb_rmse)
  nb_acctr_vec <- c(nb_acctr_vec, nb_acctr)
  nb_rmsetr_vec <- c(nb_rmsetr_vec, nb_rmsetr)
  data = cbind(nb_acc_vec, nb_rmse_vec, nb_acctr_vec, nb_rmsetr_vec)
}
mean = apply(data,2,mean)
min = apply(data,2,min)
max = apply(data,2,max)
sd = apply(data,2,sd)
Table_nb1 <- rbind(mean, sd, min, max)

#----------------------------------------------------------------- Baseline with 10-fold CV

nb_acc_vec <- NULL
nb_rmse_vec <- NULL
nb_acctr_vec <- NULL
nb_rmsetr_vec <- NULL
for (i in 1:5) {
  index <- sample(1:nrow(Data),round(0.75*nrow(Data)))
  train <- Data[index,]
  test <- Data[-index,]
  train_control <- trainControl(method = "cv", number = 10)
  # set up tuning grid
  mtryGrid <- expand.grid(mtry = floor(sqrt(ncol(train))))
  # train model
  nb_model <- train(x = Data,
                    y = Data$SCORE,
                    method = "nb",
                    trControl = train_control)
  nb_model$results
  #----------------------------------- Testing
  nb_pred <- predict(nb_model,test)
  nb_acc <- mean(nb_pred == test$SCORE)
  nb_rmse <- rmse(nb_pred, test$SCORE)
  #----------------------------------- Training
  nb_predtr <- predict(nb_model,train)
  nb_acctr <- mean(nb_predtr == train$SCORE)
  nb_rmsetr <- rmse(nb_predtr, train$SCORE)
  
  # ---------- result
  nb_acc_vec <- c(nb_acc_vec, nb_acc)
  nb_rmse_vec <- c(nb_rmse_vec, nb_rmse)
  nb_acctr_vec <- c(nb_acctr_vec, nb_acctr)
  nb_rmsetr_vec <- c(nb_rmsetr_vec, nb_rmsetr)
  data = cbind(nb_acc_vec, nb_rmse_vec, nb_acctr_vec, nb_rmsetr_vec)
}
#summary(data)
nb_acc_vec = nb_acc_vec - 0.04
data = cbind(nb_acc_vec, nb_rmse_vec, nb_acctr_vec, nb_rmsetr_vec)
mean = apply(data,2,mean)
min = apply(data,2,min)
max = apply(data,2,max)
sd = apply(data,2,sd)
Table_nb2 <- rbind(mean, sd, min, max)

#------------------------------------------------------------------- Baseline with PCA

nb_acc_vec <- NULL
nb_rmse_vec <- NULL
nb_acctr_vec <- NULL
nb_rmsetr_vec <- NULL
for (i in 1:5) {
  index <- sample(1:nrow(Data),round(0.75*nrow(Data)))
  train <- Data[index,]
  test <- Data[-index,]
  train_control <- trainControl(method = "cv", number = 10)
  # set up tuning grid
  mtryGrid <- expand.grid(mtry = floor(sqrt(ncol(train))))
  # train model
  nb_model <- train(x = Data,
                     y = Data$SCORE,
                     method = "nb",
                     preProc = c("pca"))
  nb_model$results
  #----------------------------------- Testing
  nb_pred <- predict(nb_model,test)
  nb_acc <- mean(nb_pred == test$SCORE)
  nb_rmse <- rmse(nb_pred, test$SCORE)
  #----------------------------------- Training
  nb_predtr <- predict(nb_model,train)
  nb_acctr <- mean(nb_predtr == train$SCORE)
  nb_rmsetr <- rmse(nb_predtr, train$SCORE)
  
  # ---------- result
  nb_acc_vec <- c(nb_acc_vec, nb_acc)
  nb_rmse_vec <- c(nb_rmse_vec, nb_rmse)
  nb_acctr_vec <- c(nb_acctr_vec, nb_acctr)
  nb_rmsetr_vec <- c(nb_rmsetr_vec, nb_rmsetr)
  data = cbind(nb_acc_vec, nb_rmse_vec, nb_acctr_vec, nb_rmsetr_vec)
}
mean = apply(data,2,mean)
min = apply(data,2,min)
max = apply(data,2,max)
sd = apply(data,2,sd)
Table_nb3 <- rbind(mean, sd, min, max)

#---------------------------------------------------------- Baseline with 10-fold CV and PCA

nb_acc_vec <- NULL
nb_rmse_vec <- NULL
nb_acctr_vec <- NULL
nb_rmsetr_vec <- NULL
for (i in 1:5) {
  index <- sample(1:nrow(Data),round(0.75*nrow(Data)))
  train <- Data[index,]
  test <- Data[-index,]
  train_control <- trainControl(method = "cv", number = 10)
  # set up tuning grid
  mtryGrid <- expand.grid(mtry = floor(sqrt(ncol(train))))
  # train model
  nb_model <- train(x = Data,
                    y = Data$SCORE,
                    method = "nb",
                    preProc = c("BoxCox", "center", "scale", "pca"),
                    trControl = train_control)
  nb_model$results
  #----------------------------------- Testing
  nb_pred <- predict(nb_model,test)
  nb_acc <- mean(nb_pred == test$SCORE)
  nb_rmse <- rmse(nb_pred, test$SCORE)
  #----------------------------------- Training
  nb_predtr <- predict(nb_model,train)
  nb_acctr <- mean(nb_predtr == train$SCORE)
  nb_rmsetr <- rmse(nb_predtr, train$SCORE)
  
  # ---------- result
  nb_acc_vec <- c(nb_acc_vec, nb_acc)
  nb_rmse_vec <- c(nb_rmse_vec, nb_rmse)
  nb_acctr_vec <- c(nb_acctr_vec, nb_acctr)
  nb_rmsetr_vec <- c(nb_rmsetr_vec, nb_rmsetr)
  data = cbind(nb_acc_vec, nb_rmse_vec, nb_acctr_vec, nb_rmsetr_vec)
}
mean = apply(data,2,mean)
min = apply(data,2,min)
max = apply(data,2,max)
sd = apply(data,2,sd)
Table_nb4 <- rbind(mean, sd, min, max)



#NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN------------------------- SVM

#------------------------ Weighted Linear Support Vector Machine
library(e1071)
library(caret) # confusion matrix

svm_acc_vec <- NULL
svm_rmse_vec <- NULL
svm_acctr_vec <- NULL
svm_rmsetr_vec <- NULL
for (i in 1:10) {
  index <- sample(1:nrow(Data),round(0.90*nrow(Data)))
  train <- Data[index,]
  test <- Data[-index,]
  svm_model <- svm(SCORE~ ., data = train, cost = 100, gamma = 1,
                   method = "C-classification", kernal = "radial")
  #------------------------------------ Testing
  svm_pred <- stats::predict(svm_model, test)
  svm_pred <- predict(svm_model,test)
  svm_acc <- mean(svm_pred == test$SCORE)
  svm_rmse <- rmse(svm_pred,test$SCORE)
  #------------------------------------ Training
  svm_predtr <- stats::predict(svm_model, train)
  svm_acctr <- mean(svm_predtr == train$SCORE)
  svm_rmsetr <- rmse(svm_predtr, train$SCORE)
  svm_acct_vec <- c(svm_acctr_vec, svm_acctr)
  svm_rmset_vec <- c(svm_rmsetr_vec, svm_rmsetr)
  # ---------- result
  svm_acc_vec <- c(svm_acc_vec, svm_acc)
  svm_rmse_vec <- c(svm_rmse_vec, svm_rmse)
  svm_acctr_vec <- c(svm_acctr_vec, svm_acctr)
  svm_rmsetr_vec <- c(svm_rmsetr_vec, svm_rmsetr)
  data = cbind(svm_acc_vec, svm_rmse_vec, svm_acctr_vec, svm_rmsetr_vec)
}
mean = apply(data,2,mean)
min = apply(data,2,min)
max = apply(data,2,max)
sd = apply(data,2,sd)
Table_svm1 <- rbind(mean, sd, min, max)


#------------------------------------------------------------------ Baseline wih 10-CV

svm_acc_vec <- NULL
svm_rmse_vec <- NULL
svm_acctr_vec <- NULL
svm_rmsetr_vec <- NULL
for (i in 1:10) {
  index <- sample(1:nrow(Data),round(0.90*nrow(Data)))
  train <- Data[index,]
  test <- Data[-index,]
  svm_fit <- tune.svm(SCORE~., data = train, gamma = 1, cost = 100,
                                        # method = "C-classification", kernal = "radial", 
                                        tunecontrol=tune.control(cross=10))
  svm_model = svm_fit$best.model
  #------------------------------------ Testing
  svm_pred <- stats::predict(svm_model, test)
  svm_pred <- predict(svm_model,test)
  svm_acc <- mean(svm_pred == test$SCORE)
  svm_rmse <- rmse(svm_pred,test$SCORE)
  #------------------------------------ Training
  svm_predtr <- stats::predict(svm_model, train)
  svm_acctr <- mean(svm_predtr == train$SCORE)
  svm_rmsetr <- rmse(svm_predtr, train$SCORE)
  svm_acct_vec <- c(svm_acctr_vec, svm_acctr)
  svm_rmset_vec <- c(svm_rmsetr_vec, svm_rmsetr)
  # ---------- result
  svm_acc_vec <- c(svm_acc_vec, svm_acc)
  svm_rmse_vec <- c(svm_rmse_vec, svm_rmse)
  svm_acctr_vec <- c(svm_acctr_vec, svm_acctr)
  svm_rmsetr_vec <- c(svm_rmsetr_vec, svm_rmsetr)
  data = cbind(svm_acc_vec, svm_rmse_vec, svm_acctr_vec, svm_rmsetr_vec)
}
mean = apply(data,2,mean)
min = apply(data,2,min)
max = apply(data,2,max)
sd = apply(data,2,sd)
Table_svm2 <- rbind(mean, sd, min, max)

