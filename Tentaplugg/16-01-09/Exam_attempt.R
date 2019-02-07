##### Assignment 1 #####

# Task 1

data = read.csv("crx.csv", header=T)

### Setup
set.seed(12345)
data.crx = read.csv("crx.csv")
data.crx$Class = as.factor(data.crx$Class)
n = dim(data.crx)[1]
ids = sample(1:n, n*0.8)
train = data.crx[ids,]
test = data.crx[-ids,]



library(tree)

tree_model = tree(Class~., data = train)
plot(tree_model)
text(tree_model, pretty = 0)

train_new = train[-c(2),]
tree_model_2 = tree(Class~., data=train_new)
plot(tree_model_2)
text(tree_model_2, pretty = 0)


# Task 2
set.seed(12345)
cv_tree = cv.tree(tree_model)
plot(x=cv_tree$size, y = cv_tree$dev, type = "l")
cv_tree$size
plot(cv_tree$dev)
depth = cv_tree$size[which.min(cv_tree$dev)]
print(paste("So best size is ",depth))
finalTree=prune.tree(tree_model, best=depth)
plot(finalTree)
text(finalTree, pretty = 0)
summary(finalTree)



# It chose only A9. 
test_preds = predict(finalTree, test)

test_preds = as.factor(apply(as.matrix(test_preds), 1,function(row){return(which.max(row) - 1)}))

conf_m = table(Actual = test$Class, test_preds)
1 - sum(diag(conf_m))/sum(conf_m)
# Task 3

x_train = model.matrix( ~ .-1, train[,-16])

lassoModel = glmnet(x_train, train$Class, alpha = 1, family="binomial")
plot(lassoModel, xvar="lambda", label=TRUE)

lasso_cv = cv.glmnet(x_train, train$Class)
lambda = seq(0,2, length = 1000)
lassoCVModel = cv.glmnet(x_train, train$Class, lambda = lambda, alpha = 1, family="binomial")
lassoCVModel$lambda.min
plot(lassoCVModel)
coef(lassoCVModel, s = "lambda.min")

E = function(Y, p_hat) {
  return(sum(Y*log(p_hat) + (1 - Y)*log(1 - p_hat)))
}

test_preds = predict(finalTree, test)
test_preds_classif = as.factor(apply(as.matrix(test_preds), 1,function(row){return(which.max(row) - 1)}))
E_tree = E(Y = (as.numeric(test$Class) - 1), p_hat = test_preds[,2])

x_test = model.matrix( ~ .-1, test[,-16])
lasso_preds = predict(lassoCVModel, newx=x_test, type = "response")
E_lasso = E(Y = (as.numeric(test$Class) - 1), p_hat = lasso_preds)

# I am not sure as to why this criterion is better sometimes. It is probably something with netroyp

# This criterion is sometimes more reasonable to use than missclassification rate since
# it includes information about how certain the predictions are, where confident eg. p=0.99 -> R=-0.01005034
# correct predictions are punished less than "lucky" correct predictions eg. p=0.51 -> R = -0.6733446
# Both make the correct prediction given that class is 1 but the more confident prediction contributes
# less to the error. 
# Suitible when you want to know which model is the most confident in its decision making

###### ASSIGNMENT 2 ######

#install.packages("mboost")
library(mboost)
bf <- read.csv2("bodyfatregression.csv") 
set.seed(1234567890)
m <- blackboost(Bodyfat_percent ~ Waist_cm+Weight_kg, data=bf) 
mstop(m)
cvf <- cv(model.weights(m), type="kfold")
cvm <- cvrisk(m, folds=cvf, grid=1:100)
plot(cvm)
mstop(cvm)
#SUPPORT VECTOR MACHINES

# We can see that the cm does not improve after 32 boosting iterations, but rather increases. 
# Thus, it is starting to overfit after 32 iterations. 

data(spam)

C = c(1,5)
lin_width = c(0.01,0.05)

c_coeffs = as.numeric()
lin_width = as.numeric()
error = as.numeric()

i = 1

set.seed(12345)
for (c in C) {
  svm_linear = ksvm(type~., data=spam, C=c, kernel="vanilladot", cross = 2)
  
  # Insert model
  
  
  
  i = i + 1
  for (width in lin_width) {
    
    svm_rbf = ksvm(type~., data = spam,C=c, kernel="rbf", cross=2,kpar=(sigma=(1/width)))
    
    
    i = i + 1
  }
  
  
  
}

cross_validation = function(data, model) {
  
}





####### ASSIGNMENT 2B - NN #######

set.seed(1234567890)
Var <- runif(50, 0, 10)
trva <- data.frame(Var, Sin=sin(Var)) 
tr <- trva[1:25,] # Training
va <- trva[26:50,] # Validation

w_j = runif(10, -1,1)

b_j = runif(10,-1,1)
w_k = runif(10,-1,1)
b_k = runif(10,-1,1)

learning_rate = 1/(nrow(tr)^2)

n_iter = 5000
error = rep(0, n_iter)
error_va = rep(0,n_iter)

# Implementing stochastic gradient descent

for (i in 1:n_iter) {
  
  # Looping through the training set to evaluate the current error. Calculating the current prediction
  for (n in 1:nrow(tr)) {
    z_j = tanh(w_j* tr[n,]$Var + b_j)
    y_k = sum(w_k * z_j) + b_k
    error[i] = error[i] + (y_k - tr[n,]$Sin)^2
  }
  
  for (n in 1:nrow(va)) {
    z_j = tanh(w_j* va[n,]$Var + b_j)
    y_k = sum(w_k * z_j) + b_k
    
    error_va[i] = error_va[i] + (y_k - va[n,]$Sin)^2
  }
  
  cat("i: ", i, ", error: ", error[i]/2, ", error_va: ", error_va[i]/2, "\n")
  flush.console()
  
  # Now train the network on all data points! 
  
  for (n in 1:nrow(tr)) {
    # First do forward propagation to calculate the current derivates and partial derivates 
    z_j = tanh(w_j* tr[n,]$Var + b_j)
    y_k = sum(w_k * z_j) + b_k
    
    # Now do backpropagation to change everything
    d_k = y_k - tr[n,]$Sin 
    d_j = (1 - z_j^2)*w_k*d_k
    partial_w_k = d_k*z_j
    partial_b_k = d_k
    partial_w_j = d_j * tr[n,]$Var
    partial_b_j = d_j
    w_k = w_k - learning_rate*partial_w_k
    b_k = b_k - learning_rate*partial_b_k
    w_j = w_j - learning_rate*partial_w_j
    b_j = b_j - learning_rate*partial_b_j
    
  }
  
}


w_j
b_j
w_k
b_k

plot(error/2, ylim=c(0, 5))
points(error_va/2, col = "red")


# prediction on training data

pred <- matrix(nrow=nrow(tr), ncol=2)

for(n in 1:nrow(tr)) {
  
  z_j <- tanh(w_j * tr[n,]$Var + b_j)
  y_k <- sum(w_k * z_j) + b_k
  pred[n,] <- c(tr[n,]$Var, y_k[1])
  
}

plot(pred)
points(tr, col = "red")

# prediction on validation data

pred <- matrix(nrow=nrow(tr), ncol=2)

for(n in 1:nrow(va)) {
  
  z_j <- tanh(w_j * va[n,]$Var + b_j)
  y_k <- sum(w_k * z_j) + b_k
  pred[n,] <- c(va[n,]$Var, y_k[1])
  
}

plot(pred)
points(va, col = "red")

