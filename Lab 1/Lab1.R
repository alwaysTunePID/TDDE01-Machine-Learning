setwd("~/Desktop/Plugg_Lkpg/HT18/TDDE01/Lab 1")



##############################################
################ ASSIGNMENT 1 ################
##############################################


library(readxl)
data = read_excel("spambase.xlsx")
#data$Spam = factor(data$Spam)
# Ensure this is correct when online again
log_reg = function(data, fitModel, pLimit = 0.5) {
  # Specify response in predict,
  fits = predict(fitModel, data)
  
  probabilities = fits
  
  classifications = apply(as.matrix(probabilities), 1, function(row) {
    if (row > pLimit) {
      return(1)
    } else {
      return(0)
    }
  })
  return(classifications)
}

classificationRate = function(confMatrix) {
  return(sum(diag(confMatrix))/sum(confMatrix))
}

# Task 1
n=dim(data)[1]
set.seed(12345)
id=sample(1:n, floor(n*0.5))
train=data[id,]
test=data[-id,] 


#Task 2
# Performing a fit
# TODO: HOW DO I KNOW I AM USING LOGISTIC REGRESSION. 
# THIS IS PROBABLY WRONG FIT. I HAVE TO CHECK LECTURES. 
#fit = glm(Spam ~ ., data = train)
# I think this is correct. 
fit = glm(Spam ~ ., data = train, family = "binomial")
#summary(fit)
fits = predict(fit, train)
classificationsTrain = log_reg(train, fit)
classificationsTest = log_reg(test, fit)

table(train$Spam, classificationsTrain)
classificationRate(table(train$Spam, classificationsTrain))
table(test$Spam, classificationsTest)
classificationRate(table(test$Spam, classificationsTest))

## Task 3

classificationsTrain = log_reg(train, fit, pLimit = 0.9)
classificationsTest = log_reg(test, fit, pLimit = 0.9)

table(train$Spam, classificationsTrain)
classificationRate(table(train$Spam, classificationsTrain))
table(test$Spam, classificationsTest)
classificationRate(table(test$Spam, classificationsTest))

# Task 4 
#install.packages("kknn")
library("kknn")

kkn.fit = kknn(Spam ~., train = train, test = test, k = 30)

table(test$Spam, kkn.fit$fitted.values)
classificationRate(table(test$Spam, kkn.fit$fitted.values))


# Task 5

kkn.fit = kknn(Spam ~., train, test, k = 1)
table(test$Spam, kkn.fit$fitted.values)
classificationRate(table(test$Spam, kkn.fit$fitted.values))


##############################################
################ ASSIGNMENT 2 ################
##############################################

# Task 1
library(readxl)
data = read_excel("machines.xlsx")

# Task 2

logLikelihood = function(theta, data) {
  return(sum(log(theta) - theta*data))
}

normalize = function(scale, probs) {
  return(scale*probs/sum(probs))
}

thetaGrid = seq(0,max(data), length = 1000)
scale = 1/(thetaGrid[2] - thetaGrid[1])
logProbs = apply(as.matrix(thetaGrid), 1, logLikelihood, data)

probsNormalizedTask2 = normalize(scale, exp(logProbs))

plot(x = thetaGrid, y = probsNormalizedTask2, type = "l", lwd = 3, xlab = expression(theta), main = "Task 2 and 3 - maximum likelihood", ylab = "Probabilities")
maxTheta = thetaGrid[which.max(probsNormalizedTask2)]
print(paste("Max value of theta is ", maxTheta))

# Task 3 
cutData = data$Length[1:6]
logProbsTask3 = apply(as.matrix(thetaGrid), 1, logLikelihood, cutData)
probsNormalizedTask3 = normalize(scale, exp(logProbsTask3))

lines(x= thetaGrid, y = probsNormalizedTask3, type = "l", lwd = 3, col = "green")


# Task 4


l_theta = function(theta, data, lambda = 10) {
  
  
  log_dpois = log(lambda*exp(-lambda*theta))
  return(logLikelihood(theta, data) + log_dpois)
}

#thetaGrid = seq(0,4,1)
logProbsTask4 = apply(as.matrix(thetaGrid), 1, l_theta, data)

probsNormalizedTask4 = normalize(scale, exp(logProbsTask4))
plot(x = thetaGrid, y = probsNormalizedTask4, type = "l", col = "red", lwd = 3, xlab = expression(theta), main = "Task 2 and 3 - maximum likelihood", ylab = "Probabilities")
lines(x=thetaGrid, y = probsNormalizedTask2, lwd = 3)
lines(x= thetaGrid, y = probsNormalizedTask3, type = "l", lwd = 3, col = "green")

# TASK 5 

# here we use maxTheta

sims = rexp(n=50, rate=maxTheta)

par(mfrow=c(2,1))
hist(sims, main = "Simulations", col = "blue", xlim = c(0,5), ylim = c(0,20), xlab = "Simulations")
hist(data$Length, main = "Actual values", col = "green", xlim = c(0,5), ylim = c(0,20), xlab = "Actual length values")
par(mfrow=c(1,1))

##############################################
################ ASSIGNMENT 3 ################
##############################################

# Should not be solved by TDDE01 students





##############################################
################ ASSIGNMENT 4 ################
##############################################


# Task 1
library(readxl)
data = read_excel("tecator.xlsx")

plot(x=data$Moisture, y = data$Protein)


# Task 2 

prob_model = function(train, test, i) {
  n_train = length(train$Moisture)
  x_train = train$Moisture
  X_train = matrix(c(rep(1,n_train), x_train, x_train^2, x_train^3,x_train^4, x_train^5,x_train^6), nrow = n_train, ncol = 7)
  
  
  n_test = length(test$Moisture)
  x_test = test$Moisture
  X_test = matrix(c(rep(1,n_test), x_test, x_test^2, x_test^3,x_test^4, x_test^5,x_test^6), nrow = n_test, ncol = 7)
  
  if (i != 6) {
    X_train = X_train[,-((i+2):7)]
    X_test = X_test[,-((i+2):7)]
  }
  
  
  y_train = train$Protein
  y_test = test$Protein
  
  return(list(y_train,X_train, y_test, X_test))
  #return misclassification rate and confusion matrix 
  
  
  
}

MSE = function(y, y_hat) {
  n = length(y)
  return((1/n)*sum((y - y_hat)^2))
}

convert_to_poly = function(x, i) {
  n = length(x)
  x_matrix = matrix(c(rep(1,n), x, x^2, x^3, x^4, x^5, x^6), nrow = n, ncol = 7)
  
  if (i != 6) {
    x_matrix = x_matrix[,-((i+2):7)]
  }
    
  
  return(x_matrix)
}
# Task 3 
# Copying from Assignment 1
n=dim(data)[1]
set.seed(1)
id=sample(1:n, floor(n*0.5))
train=data[id,]
test=data[-id,] 


MSE_vals = matrix(NA, nrow = 6, ncol = 2)
rownames(MSE_vals) = seq(1,6,1)
colnames(MSE_vals) = c("Training MSE", "Test MSE")


x_seq = seq(min(data$Moisture), max(data$Moisture), length = 1000)

for (i in 1:6) {
  Model = prob_model(train, test, i)
  y = Model[[1]]
  X = Model[[2]]
  lm.fit = lm(y ~ 0 + X)
  trainPredict = predict(lm.fit, train)
  testPredict = predict(lm.fit, test)
  
  betas = lm.fit$coefficients
  print(dim(as.matrix(betas)))
  
  x_matrix = convert_to_poly(x_seq, i)
  
  print(dim(x_matrix))
  
  lines(x=x_seq, y = t(as.matrix(betas))%*%t(x_matrix))
  MSE_vals[i, 1] = MSE(train$Protein ,trainPredict)
  MSE_vals[i, 2] = MSE(test$Protein, testPredict)
}

#plot(x=seq(1,6,1), y = MSE_vals[,2], col = "green", type = "l", ylim = c(min(c(MSE_vals[,1],MSE_vals[,2])), max(c(MSE_vals[,1],MSE_vals[,2]))), xlab = "Model", ylab = "MSE")

#lines(x=seq(1,6,1), y = MSE_vals[,1], col = "red")
plot(x=seq(1,6,1), y = MSE_vals[,1], col = "green", type = "l", xlab = "Model", ylab = "MSE", main = "Training MSE")
plot(x=seq(1,6,1), y = MSE_vals[,2], col = "green", type = "l", xlab = "Model", ylab = "MSE", main = "Test MSE")


#Task 4 - use entire data set.

# Perform variable selection of a linear model in which fat is a response and Channel1-Channel100 are predictors. 
# Comment on how many variables were selected. 

library(MASS)
dataToUse = data
#Remove protein and
dataToUse$Protein = NULL
dataToUse$Moisture = NULL
dataToUse$Sample = NULL
lm.fit = lm(Fat ~ ., data = dataToUse)

step = stepAIC(lm.fit, direction = "both")

nrVarsChosen = length(step$coefficients)

# Task 5 - ridge regression
#install.packages("glmnet")
library(glmnet)
# ridgeRegModel = lm.ridge(formula = Fat ~ Channel1 + Channel2 + Channel4 + Channel5 + Channel7 + 
#                          Channel8 + Channel11 + Channel12 + Channel13 + Channel14 + 
#                          Channel15 + Channel17 + Channel19 + Channel20 + Channel22 + 
#                          Channel24 + Channel25 + Channel26 + Channel28 + Channel29 + 
#                          Channel30 + Channel32 + Channel34 + Channel36 + Channel37 + 
#                          Channel39 + Channel40 + Channel41 + Channel42 + Channel45 + 
#                          Channel46 + Channel47 + Channel48 + Channel50 + Channel51 + 
#                          Channel52 + Channel54 + Channel55 + Channel56 + Channel59 + 
#                          Channel60 + Channel61 + Channel63 + Channel64 + Channel65 + 
#                          Channel67 + Channel68 + Channel69 + Channel71 + Channel73 + 
#                          Channel74 + Channel78 + Channel79 + Channel80 + Channel81 + 
#                          Channel84 + Channel85 + Channel87 + Channel88 + Channel92 + 
#                          Channel94 + Channel98 + Channel99, data = dataToUse)

variablesSelected = names(step$coefficients)
variablesSelected = variablesSelected[-c(1)]
dataSteps = dataToUse[,variablesSelected]
dataSteps = scale(dataSteps)
response = scale(dataToUse$Fat)
ridgeRegModel = glmnet(as.matrix(dataSteps), response, alpha = 0, family="gaussian")
plot(ridgeRegModel, xvar="lambda", label=TRUE)


# Task 6 - Lasso regression
#Here, alpha = 1 instead
lassoModel = glmnet(as.matrix(dataSteps), response, alpha = 1, family="gaussian")
plot(lassoModel, xvar="lambda", label=TRUE)

# Task 7 - use cross-validation to find the optimal lasso model

lassoCVModel = cv.glmnet(as.matrix(dataSteps), response, alpha = 1, family="gaussian")
lassoCVModel$lambda.min
plot(lassoCVModel)
coef(lassoCVModel, s = "lambda.min")


# Task 8 - compare results from steps 4 and 7
# Compare how many variables and the residuals I guess? 
par(mfrow=c(1,2))
plot(lassoCVModel)
plot(ridgeRegModel, xvar="lambda", label=TRUE)
par(mfrow=c(1,1))

##############################################
################ SPECIAL ASSIGNMENT 1 ################
##############################################

library(readxl)
data = read_excel("spambase.xlsx")


# Check this one https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761

knearest = function(data, K, newData) {
  #data is the training data, 
  #newData returns the predicted class probabilities for newData by using K-nearest neighbour approach. 
  responsesX = as.matrix(data[,ncol(data)])
  responsesY = as.matrix(data[,ncol(newData)])
  
  X = as.matrix(data[,-c(ncol(data))])
  Y = as.matrix(newData[,-c(ncol(newData))])
  
  # Here is where it went wrong before
  Xhat = X/sqrt(rowSums(X^2))
  Yhat = Y/sqrt(rowSums(Y^2))
  
  
  C_matrix = Xhat%*%t(Yhat)

  D_matrix = 1 - C_matrix
  #For the data test set, take the distance to the training data points, take the k nearest neighbours and classify them as the average of the training set's labels of the k points
  
  # return the indices of the k nearest neighbours for all different new data points, generating a matrix with the k nearest neighbou
  indicesKnearest = apply(D_matrix, 2, function(col, k){
    names(col) = seq(1, length(col), 1)
    
    col = sort(col)
    
    return(strtoi(names(col[1:k])))
  }, K)

  classifications = apply(indicesKnearest, 2, function(row) {
    # Classify according to the mean of the training data
    avg = mean(responsesX[row,])
    return(round(avg))
  })
  
  
  return(classifications)
}

# Use same seed and partition as before. 
n=dim(data)[1]
set.seed(12345)
id=sample(1:n, floor(n*0.5))
train=data[id,]
test=data[-id,]

labelsTest = knearest(train, 30, test)
labelsTrain = knearest(train, 30, train)
table(train$Spam, labelsTrain)
table(test$Spam, labelsTest)

print("misclassification rate train")
1 - sum(diag(table(train$Spam, labelsTrain)))/sum(table(train$Spam, labelsTrain))
print("misclassification rate test")
1 - sum(diag(table(test$Spam, labelsTest)))/sum(table(test$Spam, labelsTest))

##############################################
################ SPECIAL ASSIGNMENT 2 ################
##############################################

# K nearest neighbour density estimation

attach(mtcars)

k = 6

volume_N_sphere = function(nDim, radius) {
  n = nDim
  R = radius
  return(((pi^(n/2))/gamma(n/2 + 1))*R^n)
}



kNearestNeighDensEstimation = function(data, k = 6) {
  #data is the training data, 
  #newData returns the predicted class probabilities for newData by using K-nearest neighbour approach. 
  responsesX = as.matrix(data[,ncol(data)])
  
  N = nrow(data)
  nDim = ncol(data)
  #X = as.matrix(data[,-c(ncol(data))])
  
  X = as.matrix(data) # We do not assume responses to exist
  
  
  # Here is where it went wrong before
  Xhat = X/sqrt(rowSums(X^2))
  
  
  
  C_matrix = Xhat%*%t(Xhat)
  
  D_matrix = 1 - C_matrix
  #For the data test set, take the distance to the training data points, take the k nearest neighbours and classify them as the average of the training set's labels of the k points
  
  # return the indices of the k+1 nearest neighbours for all different new data points, generating a matrix with the k nearest neighbours
  # Since we include the point itself, we return k+1 nearest. 
  radiuses = apply(D_matrix, 2, function(col, k){
    names(col) = seq(1, length(col), 1)
    
    col = sort(col)
    
    return(strtoi(names(col[k])))
  }, k+1)
  
  #Density estimation
  densities = apply(as.matrix(radiuses), 1, function(R, nDim) {
    return(k/(volume_N_sphere(nDim,R)))
  }, nDim)
  return(densities)
}

densities = kNearestNeighDensEstimation(cars, 6)



hist(cars$speed, breaks = 15, freq = FALSE, ylim=c(0,max(densities) + 0.1))
lines(x=cars$speed, y = densities)

##############################################
################ SPECIAL ASSIGNMENT 2 ################
##############################################