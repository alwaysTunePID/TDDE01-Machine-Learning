
#### Assignment 1 ####

### Task 1 ###

data = read.csv("video.csv")

result = prcomp(data[,-c(2,19)])
lambda=result$sdev^2
explanationFactor = lambda/sum(lambda)*100

names(explanationFactor) = seq(1,length(lambda),1)
plot(explanationFactor, type = "b", xlab = "Variable", ylab = "% explanatory")
explanationFactor = sort(explanationFactor, decreasing = T)

# Only 1 variable is needed when scaling is not done. 
# This is because. 

# Task 2

library(pls)

task2_data = data[,-c(2)]

n=dim(data)[1]
set.seed(12345)
id=sample(1:n, floor(n*0.5))
train=data[id,]
test=data[-id,] 

MSE = function(y, y_hat) {
  n = length(y)
  #return((1/n)*sum((y - y_hat)^2))
  return(mean((y - y_hat)^2))
}

MSE_train = as.numeric()
MSE_test = as.numeric()
M = (ncol(task2_data) - 1)
for (m in 1:M) {
  pcr_model = mvr(utime ~., scale = TRUE, data = train, validation = "none")
  train_preds = predict(pcr_model, train, ncomp = m)
  test_preds = predict(pcr_model, test, ncomp = m)
  MSE_train[m] = MSE(train$utime, train_preds)
  MSE_test[m] = MSE(test$utime, test_preds)
}

plot(x=seq(1,M,1), y = MSE_train, col = "green", type = "l", lwd = 3, xlab = "M", ylab = "Mean squared error", main = "Red = test, green = train")
lines(x=seq(1,M,1), y = MSE_test, col = "red", type = "l", lwd = 3)

# Task 3

m = 8
pcr_model = mvr(utime ~., ncomp = m, scale = TRUE, data = train, validation = "none")

summary(pcr_model)

pcr_model$coefficients
Yloadings(pcr_model)
"
coeff_matrix = result$rotation

# Reduce to m components

coeff_matrix = coeff_matrix[,1:m]

coefficients = rowSums(coeff_matrix)

print('The model can be described as: ')

names(coefficients) = colnames(task2_data[,1:17])
coefficients
"


# Task 4

data$class = factor(apply(as.matrix(data$codec), 1, function(val) {
  if (val == "mpeg4") {
    return("mpeg")
  } else {
    return("other")
  }
}))


mpegs = data[data$class == "mpeg",]
others = data[data$class == "other",]
plot(mpegs$duration, mpegs$frames, col = "red", xlim = c(min(data$duration), max(data$duration)), ylim = c(min(data$frames),max(data$frames)), main = "Blue = others, red = mpeg")
points(others$duration, others$frames, col = "blue")
print("Yes, it does look linearly separable")

# Task 5

library(MASS)

task_5_data = data[, c(1,10,20)]
task_5_data$duration = scale(task_5_data$duration)
task_5_data$frames = scale(task_5_data$frames)

plot(task_5_data[task_5_data$class == "mpeg",]$duration, task_5_data[task_5_data$class == "mpeg",]$frames, col = "red")
points(task_5_data[task_5_data$class == "other",]$duration, task_5_data[task_5_data$class == "other",]$frames, col = "blue")

lda_model = lda(class ~., data = task_5_data)
summary(lda_model)
preds = predict(lda_model, task_5_data)
conf_m = table(Actual = task_5_data$class, Predicted = preds$class)
print(conf_m)
print(1 - sum(diag(conf_m))/sum(conf_m))


print("The classification is far from perfect. Why is this? Because the cov matrix is far from diagonal")
cor(task_5_data$duration, task_5_data$frames)

# Task 6

task_6_data = data[, c(1,10,20)]

library(tree)
tree_model = tree(class ~.,task_6_data)
cv = cv.tree(tree_model)

optimal_size = cv$size[which.min(cv$dev)]
finalTree=prune.tree(tree_model, best=optimal_size)

plot(finalTree)
text(finalTree, pretty = 0)
# Training error
preds = factor(apply(predict(finalTree, task_6_data), 1, function(row) {
  best = which.max(row)
  if (best == 1) {
    return("mpeg")
  } else {
    return("other")
  }
}))
conf_m = table(Actual = task_6_data$class, Predictions = preds)
print(conf_m)
print("There are many leaves since there are many smaller special cases and a more dense area in one place. ")
print("The optimal decision boundaries are not parallell to the axes")


########### ASSIGNMENT 2 ############

data = read.csv("spambase.csv", header = T, sep = ";", dec = ",")
data$Spam = factor(data$Spam)
library(kernlab)

C = c(0.5,1,5)

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

valid_classif_rate = as.numeric()

for (c in C) {
  model = ksvm(Spam ~., data = train,C = c)
  val_preds = predict(model, valid, type = "response")
  conf_m = table(Actual = valid$Spam, Predictions = val_preds)
  print(conf_m)
  valid_classif_rate = sum(diag(conf_m))/sum(conf_m)
}

C = C[which.max(valid_classif_rate)]

chosen_model = ksvm(Spam ~., data = train, C = C)

# Task 2 

# Estimate generalization error 

test_preds = predict(chosen_model, test, type = "response")
conf_m = table(Actual = test$Spam, Predictions = test_preds)
conf_m
sum(diag(conf_m))/sum(conf_m)
# Task 4

print("The parameter C is the regularizer, which basically instructs how much we punish the function for overfitting. With a low value, it will tend to overfit to data points, but with a higher value, it will have a harder time to 'adapt' to new points. ")



####### Task NN ######
set.seed(12345)
x = runif(50, 0,10)
y = sin(x)
data = data.frame(x,y)
plot(x,y)

n=dim(data)[1]
set.seed(12345)
id=sample(1:n, floor(n*0.5))
train=data[id,]
val=data[-id,] 

thresholds = seq(1,10,1)/1000
set.seed(12345)
winit = runif(100, -1,1)

MSE_validation_one_layer = as.numeric()
MSE_validation_two_layers = as.numeric()
MSE_train_one_layer = as.numeric()
MSE_train_two_layers = as.numeric()

library(neuralnet)
MSE = function(y, y_hat) {
  n = length(y)
  return((1/n)*sum((y - y_hat)^2))
}

i = 1
for (thr in thresholds) {
  nn_one_layer <- neuralnet(y ~ x, data = train, 
                    threshold = thr, startweights = winit, hidden = 10)
  nn_two_layers = neuralnet(y ~ x, data = train, 
                    threshold = thr, startweights = winit, hidden = c(3,3))
  # Use validation set here to get error
  yValidation_one_layer = compute(nn_one_layer, val$x)$net.result
  yValidation_two_layers = compute(nn_two_layers, val$x)$net.result
  
  MSE_validation_one_layer[i] = MSE(y = val$y, y_hat = yValidation_one_layer[,1])
  MSE_validation_two_layers[i] = MSE(y = val$y, y_hat = yValidation_two_layers[,1])
  
  yTrain_one_layer = compute(nn_one_layer, train$x)$net.result
  yTrain_two_layers = compute(nn_two_layers, train$x)$net.result
  
  MSE_train_one_layer[i] = MSE(y = train$y, y_hat = yTrain_one_layer[,1])
  MSE_train_two_layers[i] = MSE(y = train$y, y_hat = yTrain_two_layers[,1])
  
  i = i + 1
}

#plot(nn_two_layers)

plot(x=seq(1,10,1), y = MSE_validation_one_layer, type = "l", lwd = 3, col = "red", main = "MSE for the two models", sub = "Thick lines = validation. Thinner lines = train. Red = one layer, blue = two layers", ylim = c(0,max(MSE_validation_one_layer, MSE_validation_two_layers, MSE_train_one_layer, MSE_train_two_layers)))
lines(x= seq(1,10,1), y = MSE_train_one_layer, col = "red", lwd = 2)
lines(x = seq(1,10,1), y = MSE_validation_two_layers, col = "blue", lwd = 3)
lines(x = seq(1,10,1), y = MSE_train_two_layers, col = "blue", lwd = 2)



