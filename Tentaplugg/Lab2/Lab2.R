###### ASSIGNMENT 1 ######

## Task 1

data = read.csv("australian-crabs.csv")

plot(data[data$sex == "Male",]$CL, data[data$sex == "Male",]$RW, col = "blue", xlab = "CL", ylab = "RW", main = "Crab's sex, separated by CL and RW")
points(data[data$sex == "Female",]$CL, data[data$sex == "Female",]$RW, col = "pink")

# Comment: Yes, it does look linearly separable

## Task 2

library(MASS)
lda_model = lda(sex ~ CL + RW, data = data, prior = c(sum(data$sex == "Female")/nrow(data), sum(data$sex == "Male")/nrow(data)))

preds = predict(lda_model, data)

# Confusion matrix
conf_m = table(Actual = data$sex, Predictions = preds$class)
conf_m

# Misclassification rate
1 - sum(diag(conf_m))/sum(conf_m)

# Comment: Quality of fit is nearly impeccable in this case. 

## Task 3

lda_model = lda(sex ~ CL + RW, data = data, prior = c(0.1,0.9))
preds = predict(lda_model, data)

# Confusion matrix
conf_m = table(Actual = data$sex, Predictions = preds$class)
conf_m

# Misclassification rate
1 - sum(diag(conf_m))/sum(conf_m)

# Comment: Now we steer towards males. We get a less accurate results, but steer our predictions towards males

## Task 4 

log_reg_model = glm(sex ~ CL + RW, data = data, family = "binomial")

preds = predict(log_reg_model, data, type = "response")

preds = as.factor(ifelse(preds >= 0.5, "Male", "Female"))
conf_m = table(Actual = data$sex, Predictions = preds)
conf_m
# Misclassification rate
1 - sum(diag(conf_m))/sum(conf_m)

#log_reg_model$coefficients
beta_0 = log_reg_model$coefficients[1]
beta_CL = log_reg_model$coefficients[2]
beta_RW = log_reg_model$coefficients[3]
c(1,50)
RWs = -(beta_0 + beta_CL*c(1,50))/beta_RW
lines(x=c(1,50),y=RWs, col = "green", lwd = 3)


###### ASSIGNMENT 2 ######

## Task 1

library(readxl)
data = read_excel("creditscoring.xls")
data$good_bad = factor(data$good_bad)
n=dim(data)[1]
set.seed(12345)
id=sample(1:n, floor(n*0.5))
train=data[id,]
id1=setdiff(1:n, id)
set.seed(12345)
id2=sample(id1, floor(n*0.25))
valid=data[id2,]
id3=setdiff(id1,id2)
test=data[id3,]

## Task 2

library(tree)

deviance_tree = tree(good_bad ~., data = train, split = c("deviance"))

gini_tree = tree(good_bad~., data = train, split=c("gini"))

classif_func = function(probs) {
  if (which.max(probs) == 2) {
    return("good")
  } else {
    return("bad")
  }
}

preds_train_dev = as.factor(apply(predict(deviance_tree, train),1,classif_func))
preds_test_dev = as.factor(apply(predict(deviance_tree, test), 1, classif_func))

conf_m_train_dev = table(train$good_bad, preds_train_dev)
conf_m_test_dev = table(test$good_bad, preds_test_dev)

# train misclassification rate dev
1 - sum(diag(conf_m_train_dev))/sum(conf_m_train_dev)
# test misclassification rate dev
1 - sum(diag(conf_m_test_dev))/sum(conf_m_test_dev)
preds_train_gini = as.factor(apply(predict(gini_tree, train),1,classif_func))
preds_test_gini = as.factor(apply(predict(gini_tree, test),1,classif_func))
conf_m_train_gini = table(train$good_bad, preds_train_gini)
conf_m_test_gini = table(test$good_bad, preds_test_gini)

1 - sum(diag(conf_m_train_gini))/sum(conf_m_train_gini)
1 - sum(diag(conf_m_test_gini))/sum(conf_m_test_gini)

# Comment: Gini yields significantly worse results. Use deviance! 

plot(deviance_tree)
text(deviance_tree, pretty = 0)
print(deviance_tree)

## Task 3

tree = deviance_tree
trainScore = rep(0,15)
validationScore = rep(0,15)
for (i in 2:15) {
  prunedTree=prune.tree(tree,best=i)
  pred=predict(prunedTree, newdata=valid,
               type="tree")
  trainScore[i]=deviance(prunedTree)
  validationScore[i]=deviance(pred)
}

plot(x=seq(2,15,1), y = trainScore[2:15],type="b", ylim = c(min(trainScore,validationScore),max(trainScore,validationScore)), col = "blue")
lines(x=seq(2,15,1), y = validationScore[2:15],type = "b", col = "red")

# Best depth is 

best_depth = seq(2,15,1)[which.min(validationScore[2:15])]
best_depth

final_tree = prune.tree(tree, best=best_depth)
plot(final_tree)
text(final_tree, pretty = 0)

test_classifs = as.factor(apply(predict(final_tree, test),1,classif_func))
conf_m = table(test$good_bad, test_classifs)
1 - sum(diag(conf_m))/sum(conf_m)

# Comment: Quite a good misclassification rate, but far from optimal. 

# Task 4 
library(e1071)
nb_model = naiveBayes(good_bad ~., data=train)

test_preds = predict(nb_model, test)
table(test$good_bad, test_preds)
1 - sum(diag(table(test$good_bad, test_preds)))/sum(table(test$good_bad, test_preds))
# Not extremely good either. Should probably be optimized. 

# Task 5 - plot ROC etc

Pi = as.matrix(seq(0,1,by=0.05))

test_preds_nb = predict(nb_model, test, type = "raw")
test_preds_tree = predict(final_tree, test)

TPR = function(y_hat, y) {
  conf_m = table(Actual = y, Predicted = y_hat)
  # True positive rate - true positives divided by all actually being positive
  if (dim(conf_m)[2] == 1 && "good" %in% y_hat) {
    return(1) # If we only have good, all are classified as positive correctly (although many are predicted incorrectly) and this is thus 1
  } else if (dim(conf_m)[2] == 1 && "bad" %in% y_hat) {
    return(0)
  } else {
    TP = conf_m[2,2] # True positives
    N_plus = sum(conf_m[2,])
    return(TP/N_plus)
  }
  
}

FPR = function(y_hat,y) {
  conf_m = table(Actual = y, Predicted = y_hat)
  # False positive rate - false positives divided by all actually being negative
  if (dim(conf_m)[2] == 1 && "good" %in% y_hat) {
    return(1) # If we only have "good", it means all are classified as positive, and the false positives is thus as many as the number of false ones. Thus, the FP/N_minus = 1
  } else if (dim(conf_m)[2] == 1 && "bad" %in% y_hat) {
    return(0) # If we only have bad, we do not have a single false positive as there are no positives. Hence, 0
  } else {
    FP = conf_m[1,2]
    N_minus = sum(conf_m[1,])
    return(FP/N_minus)
  }
  
}

# Now calculate the classifications

classifs_nb = data.frame(apply(Pi,1,function(pi) {
  return(factor(ifelse(test_preds_nb[,2] > pi, "good", "bad")))
}))

classifs_tree = data.frame(apply(Pi,1,function(pi) {
  return(factor(ifelse(test_preds_tree[,2] > pi, "good", "bad")))
}))

# Now get the FPR and TPR

TPR_nb = apply(classifs_nb, 2, TPR,test$good_bad)
FPR_nb = apply(classifs_nb, 2, FPR, test$good_bad)

TPR_tree = apply(classifs_tree,2, TPR, test$good_bad)
FPR_tree = apply(classifs_tree,2, FPR, test$good_bad)

plot(x=FPR_nb,y=TPR_nb, type = "l", col = "blue", lwd = 3, ylim=c(0,1), xlim = c(0,1))
lines(x=TPR_tree,y=TPR_tree, type = "l", col = "red", lwd = 3)

# Task 6 - use loss matrix

nb_model = naiveBayes(good_bad~., data = train)

test_preds_w_loss = as.factor(apply(predict(nb_model, test, type = "raw"),1,function(probs) {
  losses = c(1 - probs[1], 10*(1 - probs[2]))
  if(which.min(losses) == 1) {
    return("bad")
  } else {
    return("good")
  }
}))

conf_m = table(Actual = test$good_bad, Predicted = test_preds_w_loss)


###### ASSIGNMENT 3 ######

## Task 1

data = read.csv("State.csv", header = T, sep = ";", dec = ",")

data = data[order(data$MET),]

plot(x=data$MET, y = data$EX)
data$MET
# Here, a polynomial model looks a appropriate?


## Task 3

library(tree)
set.seed(12345)
tree_model = tree(EX~MET,data=data, control = tree.control(nobs = nrow(data),minsize = 8))
cv_tree = cv.tree(tree_model)
best_size = cv_tree$size[which.min(cv_tree$dev)]

plot(x=cv_tree$size, y= cv_tree$dev, type = "l")

final_tree = prune.tree(tree_model, best=best_size)

preds = predict(final_tree, data)



plot(x=data$MET, y = data$EX, col = "blue", pch = "x")
lines(x=data$MET, y = preds)

hist(residuals(final_tree), breaks = 20)

# Many of the residuals seem to be around -50, and it is mainly negative.

## Task 3

# Compute and plot the 95 % confidence bands using non-parametric bootstrap for the tree regression model. 

# Do non-parametric bootstrap

set.seed(12345)
B = 1000

all_preds = matrix(NA, nrow=nrow(data),ncol=B)

for (i in 1:B) {
  # Sample indices to pick out
  samples = sample(seq(1,nrow(data),1),size = nrow(data), replace=TRUE)
  data_sampled = data[samples,]
  tree_model = tree(EX~MET,
                    data=data_sampled, 
                    control = tree.control(nobs = nrow(data),
                                           minsize = 8)
                    )
  
  final_tree = prune.tree(tree_model, best=best_size)
  
  all_preds[,i] = predict(final_tree, data)
  
}

confidence_bands = apply(all_preds,1,function(row) {
  return(quantile(row, probs = c(0.025,0.5,0.975)))
})
plot(x=data$MET, y = data$EX, col = "blue", pch = "x")
lines(x=data$MET, y = preds, lwd=3)
lines(x=data$MET, y = confidence_bands[1,], lwd=3, col = "red")
lines(x=data$MET, y = confidence_bands[2,], lwd=3, col = "green")
lines(x=data$MET, y = confidence_bands[3,], lwd=3, col = "red")


# Use parametric bootstrap

library(mvtnorm)

# Parametric bootstrap: Fit model to D, use this model all the time.

# Step 1: Fit model to D
tree_model = tree(EX~MET,
                  data=data, 
                  control = tree.control(nobs = nrow(data),
                                         minsize = 8)
)

random_gen = function(data, tree_model) {
  
  y_hat = predict(tree_model, data)
  y = data$EX
  residuals = y - y_hat
  sigma = sd(residuals)
  data$EX = rnorm(nrow(data), mean = y_hat, sd = sigma)
  return(data)
}

ci_tree = function(data) {
  tree_model = tree(EX~MET,
                    data=data, 
                    control = tree.control(nobs = nrow(data),
                                           minsize = 8)
  )
  
  final_tree = prune.tree(tree_model, best=best_size)
  preds = predict(final_tree, data)
  
  return(preds)
}

pred_band_tree = function(data) {
  tree_model = tree(EX~MET,
                    data=data, 
                    control = tree.control(nobs = nrow(data),
                                           minsize = 8)
  )
  
  final_tree = prune.tree(tree_model, best=best_size)
  preds = predict(final_tree,data)
  y = orig_data$EX
  residuals = y - predict(final_orig_tree, orig_data)
  preds = rnorm(nrow(data), mean = preds, sd = sd(residuals))
  return(preds)
}

final_orig_tree = prune.tree(tree_model, best=best_size)

#sigma2 = sd(summary(final_tree)$residuals)^2
orig_data = data

# Do
ci_estimates = boot(orig_data, statistic = ci_tree, mle=final_orig_tree, R=1000,sim="parametric", ran.gen = random_gen)

ci_res = envelope(ci_estimates)

pred_bands_estimates = boot(orig_data, mle = final_orig_tree, statistic = pred_band_tree, R = 1000, sim = "parametric", ran.gen = random_gen)

pred_res = envelope(pred_bands_estimates)

plot(x=data$MET, y = data$EX, col = "blue", pch = "x")
lines(x=data$MET, y = pred_res$point[1,], lwd = 3, col = "red")
lines(x=data$MET, y = pred_res$point[2,], lwd = 3, col = "red")
lines(x=data$MET, y = ci_res$point[1,], lwd = 3, col = "blue")
lines(x=data$MET, y = ci_res$point[2,], lwd = 3, col = "blue")

## Task 5
tree_model = tree(EX~MET,data=data, control = tree.control(nobs = nrow(data),minsize = 8))
final_tree = prune.tree(tree_model, best=best_size)

preds = predict(final_tree, data)
hist(residuals(final_tree), breaks = 20)
# Comment: a chi-square model would be able to handle this nicely. 


####### ASSIGNMENT 4 #######

data = read.csv("NIRspectra.csv", sep = ";", dec = ",")

result = prcomp(data[,-ncol(data)])

lambda=result$sdev^2
explanationFactor = lambda/sum(lambda)*100

names(explanationFactor) = seq(1,length(lambda),1)
plot(explanationFactor, type = "b", xlab = "Variable", ylab = "% explanatory")
explanationFactor = sort(explanationFactor, decreasing = T)

sumP = 0
i = 0
while (sumP < 99) {
  i = i + 1
  sumP = sumP + explanationFactor[i]
  
  
}

nrVarsToInvolve = i


plot(result$x[,1], result$x[,2], xlab = "PC1", ylab = "PC2")

## Task 2

# Task 2: Make trace plots

U = result$rotation
plot(U[,1], main="Traceplot, PC1", xlab = "Actual variables")
plot(U[,2],main="Traceplot, PC2", xlab = "Actual variables")


# Task 3: Perform independent component analysis

library(fastICA)
#fICA = fastICA(as.matrix(featSpace),)
#X = as.matrix()

# a. Compute W' = K * W
set.seed(12345)
fICAResult = fastICA(as.matrix(data[,-ncol(data)]), n.comp = nrVarsToInvolve)
W_prime = fICAResult$K%*%fICAResult$W

plot(W_prime[,1], xlab = "Variable", ylab = "W'")
plot(W_prime[,2], xlab = "Variable", ylab = "W'")

plot(x=fICAResult$S[,1],fICAResult$S[,2])
