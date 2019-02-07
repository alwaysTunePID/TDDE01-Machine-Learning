###### ASSIGNMENT 1 ######

library(readxl)


# Task 1


data = read_excel("spambase.xlsx")
data$Spam = as.factor(data$Spam)
n=dim(data)[1]
set.seed(12345)
id=sample(1:n, floor(n*0.5))
train=data[id,]
test=data[-id,]
# Task 2

log_reg_model = glm(Spam~., data=train, family="binomial")

preds_test = predict(log_reg_model, test, type = "response")

preds_test = as.factor(ifelse(preds_test > 0.5, 1, 0))

conf_m = table(Actual = test$Spam, Predicted = preds_test)
table(Actual = test$Spam, Predicted = preds_test)

1 - sum(diag(conf_m))/sum(conf_m)

# Analysis: The results are not perfect. 

# Task 3 

preds_test = predict(log_reg_model, test, type = "response")

preds_test = as.factor(ifelse(preds_test > 0.9, 1, 0))

conf_m = table(Actual = test$Spam, Predicted = preds_test)
table(Actual = test$Spam, Predicted = preds_test)

1 - sum(diag(conf_m))/sum(conf_m)

# Analysis: When choosing the limit 0.9, we get a significantly higher classification rate. 
# This means that the model is not very robust and has many probabilites between 0.5 and 0.9. 

# Task 4

library("kknn")

kknn_model = kknn(Spam~., train = train, test= train, k = 30)



conf_m = table(train$Spam,kknn_model$fitted.values)
conf_m
"Train set misclassification rate"
1 - sum(diag(conf_m))/sum(conf_m)


# Checking test set
kknn_model = kknn(Spam~., train = train, test= test, k = 30)



conf_m = table(test$Spam,kknn_model$fitted.values)
conf_m
"Test set misclassification rate"
1 - sum(diag(conf_m))/sum(conf_m)
"High misclassification rate"

# Task 5 

kknn_model = kknn(Spam~., train = train, test= train, k = 1)

conf_m = table(train$Spam,kknn_model$fitted.values)
conf_m
"Train set misclassification rate"
1 - sum(diag(conf_m))/sum(conf_m)


# Evaluting accuracy on test set
kknn_model = kknn(Spam~., train = train, test= test, k = 1)

conf_m = table(test$Spam,kknn_model$fitted.values)
conf_m

1 - sum(diag(conf_m))/sum(conf_m)

# When we use 1, we overfit. This is because it picks its own point, as this is the closest one. Thus, we receive an extreme accruacy if the points is already seen. 


##### ASSIGNMENT 2 #####
# Task 1
data = read_excel("machines.xlsx")

# Task 2

# The distribution type is exponential. 

log_likelihood = function(theta, x) {
  n = length(x)
  return(sum(log(theta) - theta*x))
}



thetas = seq(0.001, max(data$Length), length.out = 1000)

scale = 1/(thetas[2] - thetas[1])

log_lik = apply(as.matrix(thetas), 1, log_likelihood, data$Length)
plot(x=thetas, y = log_lik, type = "l")


plot(x=thetas, y = (exp(log_lik)/sum(exp(log_lik))), type = "l")
max(thetas[which.max(log_lik)])

# Task 3 

log_lik_new = apply(as.matrix(thetas), 1, log_likelihood, data$Length[1:6])


plot(x=thetas, y = log_lik, type = "l", ylim=c(min(log_lik,log_lik_new), max(log_lik,log_lik_new)))
lines(x=thetas, y = log_lik_new, type = "l", col = "blue")
plot(x=thetas, y = (exp(log_lik)/sum(exp(log_lik))), type = "l")
lines(x=thetas, y = (exp(log_lik_new)/sum(exp(log_lik_new))), lwd = 2, col = "blue")
# We obtain a more biased estimate, which is not as accurate. 

# Task 4

l_theta = function(theta, x) {
  
  log_lik = log_likelihood(theta,x)
  
  return(log_lik + dexp(theta, rate = 10,log = T))
  
}

l = apply(as.matrix(thetas), 1, l_theta, data$Length)
plot(x=thetas, y = log_lik, ylim=c(min(l,log_lik,log_lik_new), max(l,log_lik,log_lik_new)), type = "l")
lines(x=thetas, y = log_lik_new, col = "blue", lwd = 2)
lines(x=thetas, y = l, col = "red", lwd = 2)

plot(x=thetas, y = (exp(log_lik)/sum(exp(log_lik))), type = "l", ylim=c(min((exp(log_lik)/sum(exp(log_lik))),(exp(log_lik_new)/sum(exp(log_lik_new))),(exp(l)/sum(exp(l)))),max((exp(log_lik)/sum(exp(log_lik))),(exp(log_lik_new)/sum(exp(log_lik_new))),(exp(l)/sum(exp(l))))))
lines(x=thetas, y = (exp(log_lik_new)/sum(exp(log_lik_new))), lwd = 2, col = "blue")
lines(x=thetas, y = (exp(l)/sum(exp(l))))

optimal_theta = thetas[which.max(l)]

# Task 5

simulations = rexp(50, rate = optimal_theta)
par(mfrow=c(2,1))
hist(data$Length)
hist(simulations)
par(mfrow=c(1,1))
# We see that the distribution is quite similar. 


###### Assignment 3 ######

# Here, we should produce and R function that performs feature selection (best subset selection) in linear regression by using k-fold cross-validation


split_into_folds = function(X, Y, K = 5, seed = 12345) {
  smp_size = floor(nrow(X)/K)
  
  indices_chosen = as.numeric()
  folds = list()
  
  set.seed(seed)
  for (k in 1:(K-1)) {
    indices = sample(x=seq(1,nrow(X),1)[!(seq(1,nrow(X),1) %in% indices_chosen)], size=smp_size, replace = FALSE)
    x = X[indices,]
    y = Y[indices]
    print(y)
    fold = as.matrix(data.frame(x,y))
    folds[[k]] = fold
    indices_chosen = append(indices_chosen, indices)
  }
  
  # last fold
  indices = seq(1,nrow(X),1)[!seq(1,nrow(X),1) %in% indices_chosen]
  x = X[indices,]
  y = Y[indices]
  fold = as.matrix(data.frame(x,y))
  folds[[K]] = fold
  return(folds)
}

CVtrainAndValidationSet <- function(splits, splitIndex = -1) {
  if (splitIndex == -1 || splitIndex > length(splits)) {
    testIndex = floor(runif(1, min = 1, max = length(splits)))
    test = splits[[testIndex]]
    train = splits
    train[[testIndex]] = NULL
  } else {
    test = splits[[splitIndex]]
    train = splits
    train[[splitIndex]] = NULL
  }
  trainSet = train[[1]]
  for (i in 2:length(train)) {
    trainSet = rbind(trainSet, train[[i]])
  }
  
  return(list(trainSet,test))
}

feature_selection = function(X, y, Nfolds = 5) {
  X_new = X
  # Store all possible combinations of 2^5 combinations = 32 different possible combinations
  
  combinations = matrix(NA, nrow=(2^5), ncol = 6)
  
  i = 1
  for (i1 in c(0,1)) {
    for (i2 in c(0,1)) {
      for (i3 in c(0,1)) {
        for (i4 in c(0,1)) {
          for (i5 in c(0,1)) {
            combinations[i,] = c(i1,i2,i3,i4,i5,1)
            i = i + 1
          }
        }
      }
    }
  }
  # Permute 
  
  k = Nfolds
  set.seed(12345)
  
  folds = split_into_folds(X=X,Y=y,K=5)
  cv_err = matrix(NA, nrow=(2^5), ncol = 1)
  # Loop through combinations 
  for (i in 1:(2^5)) {
    cv_err[i] = CV(folds, combination = combinations[i,])
  }
  
  nr_feats = apply(combinations, 1, function(comb) {
    return(sum(comb) - 1)
  })
  
  plot(x=nr_feats[2:32], y = cv_err[2:32], ylab="MSE",
       xlab = "Number of features included", 
       main = "MSE error plotted against number of features")
  
  smallest_error = as.numeric()
  for (j in 1:5) {
    smallest_error[j] = min(cv_err[which(nr_feats == j),])
  }
  lines(x=1:5, y=smallest_error, type = 'o', col = "blue", lwd = 2)
  points(x=nr_feats[which.min(cv_err)], 
         y = cv_err[which.min(cv_err),], 
         pch = "o", cex= 3,
          col = "red")
  optimal_subset = c(colnames(X),"Fertility")[combinations[which.min(cv_err),] == 1]
  
  return(optimal_subset)
}

MSE = function(y, y_hat) {
  n = length(y)
  return(mean((y - y_hat)^2))
}

get_formula = function(response, feats) {
  f = paste(c(response, paste(best_subset, collapse = ' + ')), collapse = " ~ ")
}

CV = function(folds, combination, k = 5) {
  errors = as.numeric()
  for (j in 1:k) {
    # Pick v
    train_test = CVtrainAndValidationSet(splits= folds,splitIndex = j)
    train = train_test[[1]]
    test = train_test[[2]]
    
    
    # Remove the columns not included
    X_new_train = as.matrix(train[,combination == 1])
    y_train = X_new_train[,ncol(X_new_train)]
    X_new_test = as.matrix(test[,combination == 1])
    y_test = X_new_test[,ncol(X_new_test)]
    
    X_new_train = matrix(c(rep(1,nrow(X_new_train)),(X_new_train[,-ncol(X_new_train)])), byrow = FALSE, nrow = nrow(X_new_train))
    X_new_test = matrix(c(rep(1,nrow(X_new_test)),(X_new_test[,-ncol(X_new_test)])), byrow = FALSE, nrow = nrow(X_new_test))
    
    
    if (dim(X_new_train)[2] >= 2) {
      # Fit model
      betas = solve(t(X_new_train)%*%X_new_train)%*%t(X_new_train)%*%as.matrix(y_train)
      y_hat = X_new_test%*%betas
      errors[j] = MSE(y=y_test, y_hat=y_hat)
    }
    
  }
  # Return the error
  return(mean(errors))
}
set.seed(12345)
data(swiss)
swiss_data = swiss
y = swiss_data$Fertility
X = as.matrix(swiss_data[,-c(1)])
best_subset = feature_selection(X,y)
print(paste("best formula is:", get_formula("Fertility", best_subset[-5])))



###### Assignment 4 #####
library(readxl)
data = read_excel("tecator.xlsx")

plot(x=data$Moisture, y = data$Protein)

# It looks like it might be described by a linear model, yes. 


MSE = function(y, y_hat) {
  n = length(y)
  return((1/n)*sum((y - y_hat)^2))
}

polynomial_model = function(X_train, y_train, X_test, y_test, degree) {
  
  x_train = matrix(NA, nrow=nrow(as.matrix(X_train)), ncol = degree)
  for (i in 1:degree) {
    x_train[,i] = X_train^i
  }
  
  x_test = matrix(NA, nrow = nrow(as.matrix(X_test)), ncol = degree)
  for (i in 1:degree) {
    x_test[,i] = X_test^i
  }
  y = y_train
  train = data.frame(x_train,y)
  y = y_test
  if (degree == 1) {
    x_train = x_test
    test = data.frame(x_train, y)
  } else {
    test = data.frame(x_test, y)
  }
  
  
  
  lm_model = lm(y~., data = train)
  preds_train = predict(lm_model, train)
  preds_test = predict(lm_model, test)
  
  
  err_train = MSE(y=y_train, y_hat = preds_train)
  err_test = MSE(y=y_test, y_hat=preds_test)
  return(c(err_train, err_test))
}


  
  
  
n=dim(data)[1]
set.seed(12345)
id=sample(1:n, floor(n*0.5))
train=data[id,]
test=data[-id,]

X_train = train$Protein
y_train = train$Moisture
X_test = test$Protein
y_test = test$Moisture

errors = matrix(NA, nrow=6, ncol = 2)
colnames(errors) = c("training MSE", "test MSE")
for (i in 1:6) {
  errors[i,] = polynomial_model(X_train = X_train,y_train = y_train, X_test = X_test, y_test = y_test, degree=i)
}

plot(x=seq(1,nrow(errors),1),y=errors[,1], type = "l", lwd = 3, col = "blue", ylim=c(30,max(errors)))
lines(x=seq(1,6,1), y=errors[,2], col="red", lwd=3)
which.min(errors[,2])

# Task 4

library(MASS)
to_remove = c("Sample","Moisture", "Protein")
data_task_4 = data[,!(colnames(data) %in% to_remove)]

lm_model_task_4 = lm(Fat~., data = data_task_4)

reduction = stepAIC(lm_model_task_4)

summary(reduction)

# Task 5 - fit a ridge regression model with the same predictor and response variables. 

library(glmnet)

X_ridge = scale(data_task_4[,-101])
y_ridge = scale(data_task_4[,101])
ridge_reg_model = glmnet(X_ridge, y_ridge, alpha = 0, family = "gaussian")
plot(ridge_reg_model, xvar="lambda", label=T)


# Task 6 - LASSO

X_lasso = X_ridge
y_lasso = y_ridge

lasso_model = glmnet(X_ridge, y_ridge, alpha = 1, family = "gaussian")
plot(lasso_model, xvar="lambda", label=T)


# Task 7 - cross-validation

cv_lasso = cv.glmnet(X_lasso, y_lasso, lambda = seq(0,5,length=1000), alpha = 1)
cv_lasso$lambda.min
plot(cv_lasso)
coef(cv_lasso, s = "lambda.min")
