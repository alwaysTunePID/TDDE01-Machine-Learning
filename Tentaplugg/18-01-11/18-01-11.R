data = read.csv("video.csv")
to_analyze = data[,-c(2,19)]
PCA_not_scaled = prcomp(data[,-c(2,19)], scale = F)
PCA_scaled = prcomp(data[,-c(2,19)], scale = T)
lambda_not_scaled = (PCA_not_scaled$sdev^2)/sum((PCA_not_scaled$sdev^2))
lambda_scaled = (PCA_scaled$sdev^2)/sum((PCA_scaled$sdev^2))


perc_explained = as.numeric()
for (i in 1:17) {
  perc_explained[i] = sum(lambda_scaled[1:i])
}

nr_vars_needed = which(perc_explained >= 0.95)[1]
nr_vars_needed
perc_explained[9]

# In the first case, 99 % of the variation is explained by PC1
# In the scaled case, We need 9 principal components. 
# This is because the scaling normalizes the variables and removes variance caused by the fact 
# That the variables might have different dimensions. What is 0 and 10 for one variable 
# Might be 0 and 100 in another. 

## Task 2

MSE = function(y,y_hat) {
  return(mean((y - y_hat)^2))
}

errors = matrix(NA, ncol = 2, nrow = 17)
colnames(errors) = c("train MSE","test MSE")
to_analyze = data.frame(PCA_scaled$x, utime = data$utime)


n=dim(to_analyze)[1]
set.seed(12345)
id=sample(1:n, floor(n*0.5))
train=to_analyze[id,]
test=to_analyze[-id,]

for (M in 1:17) {
  PC_M = to_analyze[,c(1:M,17)]
  
  # Since same seed, we will always sample the same ones
  
  train_M = train[,c(1:M,18)]
  test_M = test[,c(1:M,18)]
  
  model = glm(utime~.,, data = train_M, family = "gaussian")
  preds_test = predict(model, test_M)
  preds_train = predict(model, train_M)
  errors[M,2] = MSE(y=test_M$utime, y_hat = preds_test)
  errors[M,1] = MSE(y=train_M$utime, y_hat = preds_train)
  
}

plot(x=1:17, y=errors[,1], type = "l", lwd = 3, col = "blue", main ="Train and test", sub = "Train is blue, test is red")
lines(x=1:17, y=errors[,2], type = "l", lwd = 3, col = "red")

# We can see that we pretty much have the same error between 7 and 15 variables included. 
# After 15 variables, the error goes up, indicating a bias. 

# Task 3 

# M = 8 seems nice. 

M = 8 
train_M = train[,c(1:M,18)]
test_M = test[,c(1:M,18)]

model = glm(utime~.,, data = train_M, family = "gaussian")
# The probabilistic model. 
print(model)

# Task 4 
data$class = as.factor(ifelse(data$codec == "mpeg4", "mpeg", "other"))

plot(x=data[data$class == "mpeg",]$duration, y = data[data$class == "mpeg",]$frames, col = "blue")
points(x=data[data$class == "other",]$duration, y = data[data$class == "other",]$frames, col = "red")

# Comment: these seem reasonably separable, yes! 

# Task 5 

lda_model = lda(class ~frames + duration, data = data)
preds_lda = predict(lda_model, data = data)
table(data$class, preds_lda$class)
# Misclassification rate
1 - sum(diag(table(data$class, preds_lda$class)))/sum(table(data$class, preds_lda$class))
# A pretty bad classification rate, actually surprisingly bad. In this case, 
# lda is bad probably because it assumes that the different dimensions have the same Sigma matrix, 
# Which is probably far away from reality. 

# Task 6 
library(tree)
dec_tree_model = tree(class ~ frames + duration, data = data)
cv_tree = cv.tree(dec_tree_model, K = 10)
plot(x=cv_tree$size,y=cv_tree$dev)
best_depth = cv_tree$size[which.min(cv_tree$dev)]

pruned_tree = prune.tree(dec_tree_model, best = best_depth)

preds = as.factor(ifelse(predict(pruned_tree, data)[,1] >= 0.5, "mpeg", "other"))
table(data$class, preds)
1 - sum(diag(table(data$class, preds)))/sum(table(data$class, preds))
plot(pruned_tree)
text(pruned_tree, pretty = 0)
# Comment: The reason why a complicated tree is needed is simply, looking at the tree structure and 
# the plot, that there are a lot of points in the first area, where there are clear overlaps. It is separable 
# At a larger scale, but hard at a smaller scale where most points are allocated to. 


###### ASSIGNMENT 2 ######

library(kernlab)
data(spam)
width = 0.05
data = spam
n=dim(data)[1]
set.seed(12345)
id=sample(1:n, floor(n*0.4))
train=data[id,]
id1=setdiff(1:n, id)
set.seed(12345)
id2=sample(id1, floor(n*0.3))
valid=data[id2,]
id3=setdiff(id1,id2)
test=data[id3,]

# I choose the training, validation and test technique to get a generalization error. 
# The generalization error is the test error 

C = c(0.5,1,5)

val_err = as.numeric()
i = 1
for (c in C) {
  model = ksvm(type~., data = train, kernel="rbfdot", kpar=list(sigma=width), C=c)
  val_err[i] = 1 - sum(diag(table(valid$type,predict(model, valid))))/sum(table(valid$type,predict(model, valid)))
  i = i + 1
}

plot(C, val_err, type = "o")

C = C[which.min(val_err)]
model = ksvm(type~., data = rbind(train,valid), kernel="rbfdot", kpar=list(sigma=width), C=C)

# Test 
preds_test = predict(model, test)
table(test$type, preds_test)
1 - sum(diag(table(test$type, preds_test)))/sum(table(test$type, preds_test))


###### Assignment 2 ######

# Reuse function from lab 1
MSE = function(y, y_hat) {
  n = length(y)
  return((1/n)*sum((y - y_hat)^2))
}

library(neuralnet)
set.seed(1234567890)
Var <- runif(50, 0, 10)
trva <- data.frame(Var, Sin=sin(Var))
tr <- trva[1:25,] # Training
va <- trva[26:50,] # Validation
# Random initialization of the weights in the interval [-1, 1]
winit <- runif(50,-1,1)

MSE_val10 = as.numeric()
MSE_train10 = as.numeric()
MSE_val33 = as.numeric()
MSE_train33 = as.numeric()

for(i in 1:10) {
  nn10 <- neuralnet(Sin ~ Var, data = tr, 
                  threshold = (i/1000), startweights = winit, hidden = 10)
  
  
  
  nn33 <- neuralnet(Sin ~ Var, data = tr, 
                    threshold = (i/1000), startweights = winit, hidden = c(3,3))
  
  tr_preds10 = compute(nn10, tr$Var)$net.result
  val_preds10 = compute(nn10, va$Var)$net.result
  
  tr_preds33 = compute(nn33, tr$Var)$net.result
  val_preds33 = compute(nn33, va$Var)$net.result
  
  
  MSE_val10[i] = MSE(va$Sin, val_preds10[,1])
  MSE_train10[i] = MSE(tr$Sin, tr_preds10[,1])
  
  MSE_val33[i] = MSE(va$Sin, val_preds33[,1])
  MSE_train33[i] = MSE(tr$Sin, tr_preds33[,1])
}

# Which error is the best val? 

plot(x=(seq(1,10,1)/1000), y = MSE_val10, type = "l", col = "blue", lwd = 3, ylim = c(0,max(MSE_val10,MSE_val33)))
lines(x=(seq(1,10,1)/1000), y = MSE_val33, type = "l", col = "red", lwd = 3)

# The best is without a doubt threshold = 0.004 for the NN with 1 hidden layer and 10 units. 

plot(x=(seq(1,10,1)/1000), 
     y=MSE_val, 
     main = "MSE for training and validation set", 
     xlab = "Threshold", 
     ylab = "MSE", 
     type = "l", 
     lwd = 3, 
     col = "red", 
     ylim=c(0,max(MSE_train, MSE_val)))
lines(x=(seq(1,10,1)/1000), y = MSE_train, lwd = 3, col = "blue")

# Generate thresholds again
bestThr = (seq(1,10,1)/1000)[which.min(MSE_val10)]

# Plot the best one
plot(nn <- neuralnet(Sin ~ Var, 
                     data = tr, 
                     threshold = bestThr, 
                     startweights = winit, 
                     hidden = 10), 
     rep = "best")
# Plot of the predictions (black dots) and the data (red dots)
plot(prediction(nn)$rep1)
points(trva, col = "red")

# Not necessarily more layers is better. 
