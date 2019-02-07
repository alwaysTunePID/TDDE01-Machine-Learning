##### ASSIGNMENT 1 #####

data = read.csv("australian-crabs.csv")

## Task 1

plot(data[data$species == "Blue",]$CW, data[data$species == "Blue",]$BD, col = "blue", main = "CW-BD, colored by species", xlab = "CW", ylab ="BD")
points(data[data$species == "Blue",]$CW, data[data$species == "Orange",]$BD, col = "orange")

# Comment: They do certainly look fairly linearly separable. 

## Task 2 

library(e1071)

nb_model = naiveBayes(species ~ CW+BD, data = data)

preds = predict(nb_model, data)
table(data$species, preds)
1 - sum(diag(table(data$species, preds)))/sum(table(data$species, preds))

# Comment: Since these two are actually very dependent (linearly dependent) and NB assumes conditional independence, it is not appropriate to use here. 

#cov(data[,-c(1,2,3)])
# As we see they have a high covariance
cor(data[,c(7,8)])
cov(data[,c(7,8)])

## Task 3 

log_reg_model = glm(species ~ CW+BD, data = data, family = "binomial")
preds_log_reg = predict(log_reg_model, data, type = "response")
preds_log_reg = as.factor(ifelse(preds_log_reg >= 0.5, "Orange", "Blue"))

table(data$species, preds_log_reg)
1 - sum(diag(table(data$species, preds_log_reg)))/sum(table(data$species, preds_log_reg))
# Much better, since this is linearly separable. 

# Equation of the decision boundary is -(beta_CW*CW + beta_0)/beta_BD = BD
beta_0 = log_reg_model$coefficients[1]
beta_CW = log_reg_model$coefficients[2]
beta_BD = log_reg_model$coefficients[3]

lines(x=data$CW, y = (-(beta_CW*data$CW + beta_0)/beta_BD))

# Comment: The quality of fit is almost impeccable due to the linear separableness

# Task 4

PCA = prcomp(data[,c(7,8)], scale = TRUE)

# the sdev squared will give us the variation. 
# Normalizing the variance vector will give us how much variation is explained by each principal component
lambda = PCA$sdev^2
how_much_explained = lambda/sum(lambda)
how_much_explained
# So we explain 98 percent of the variation with the first principal component. 

# "Present the equations expressing principal component coordinates thorugh original coordinates"
#Coordinates of any data point $x=(x_1,..,x_p)$ in the new coordinate system will be given by
#$$z = (z_1,...,z_n), z_i = x^Tu_i$$
#  In matrix form, we see it as $Z = X\cdot U$
# Is it 
z_1 = as.matrix(data[1,c(7,8)])%*%PCA$rotation
# or is it
z_1 = as.matrix(data[1,c(7,8)])%*%t(PCA$rotation)
# ? 

# Answer: It is the first one!
z_1 = as.matrix(data[1,c(7,8)])%*%PCA$rotation

# Task 5

data_to_use = data.frame(PCA$x, data$species)
nb_model_PCA = naiveBayes(data.species~., data = data_to_use)
preds = predict(nb_model_PCA, data_to_use)

table(data_to_use$data.species, preds)
1 - sum(diag(table(data_to_use$data.species, preds)))/sum(table(data_to_use$data.species, preds))
# Comment: The new is nearly impeccable, since we remove all the variance and thus the covariance of the features. 

###### ASSIGNMENT 2 ######

data = read.csv("bank.csv", sep = ";", dec = ",")
plot(data$Time,data$Visitors, xlim = c(9,13), ylim = c(min(data$Visitors), 250))
model = glm(Visitors ~ Time, data = data, family=poisson(link = "log"))
model$coefficients
lines(x=seq(0,13,by=0.5), y = exp((model$coefficients[1] + model$coefficients[2]*seq(0,13, by = 0.5))))
lines(x=data$Time, y = exp(predict(model, data)), col = "blue", lwd = 3)
# The probabilistic expression for the fitted model is (paste into latex) $$y_i = t\cdot exp(x_i^T \beta)$$

library(boot)

# Task 2: Now, compute a prediction band. 

# Statistical function to return predictions. 
statistic_func = function(..data) {
  model = glm(Visitors ~ Time, data = ..data, family=poisson(link = "log"))
  preds = exp(predict(model, data1213)) # predict on the globally first sampled data.
  preds = rpois(nrow(..data), preds) # Sample out of a poisson distribution out of this. 
  return(preds)
}

# A random generator
rand_gen = function(.data, model) {
  
  preds = exp(predict(model,.data))
  # Return the new sampled ones out of the new model. 
  .data$Visitors = rpois(nrow(.data), preds)
  return(.data)
}

B = 1000
set.seed(12345)


data1213 = data.frame(Time = seq(9,13,by=0.05), Visitors = as.numeric(21))
set.seed(12345)
# Sample the data first once through random gen once to start with that data. 
data1213 = rand_gen(data1213, model)
set.seed(12345)
# Send that sampled data in!
pred_band = boot(data1213, 
                 mle = model, 
                 statistic = statistic_func, 
                 R=1000, 
                 sim = "parametric", 
                 ran.gen = rand_gen)

e1 = envelope(pred_band)

lines(x=seq(9,13,by=0.05),y= e1$point[1,], col = "green", lwd = 3)
lines(x=seq(9,13,by=0.05),y= e1$point[2,], col = "green", lwd = 3)


###### ASSIGNMENT 3 #######
library(kernlab)
data(spam)

C = c(1,10,100)

width = 0.05

errors = as.numeric()
i = 1
set.seed(1234567890)
for (c in C) {
  model = ksvm(type~., data = spam,C=c, kernel = "rbfdot", kpar=list(sigma=width), cross=2)
  errors[i] = cross(model)
  print(paste("C = ",c, ", Error:",errors[i]))
}
# We see that the best error is for C = 10. 
results = data.frame(C = C, Errors = errors)
plot(results)





##### Neural net #####

MSE = function(y,y_hat) {
  return(mean((y - y_hat)^2))
}

library(neuralnet)
set.seed(1234567890)
Var <- runif(50, 0, 10)
tr <- data.frame(Var, Sin=sin(Var)) 
tr1 <- tr[1:25,] # Fold 1
tr2 <- tr[26:50,] # Fold 2
thr = 1/1000
winit = runif(50,-1,1)

nn1 = neuralnet(Sin~Var, data=tr1, hidden=10, threshold = thr, startweights = winit)
nn2 = neuralnet(Sin~Var, data = tr2, hidden=10, threshold = thr, startweights = winit)

MSE1 = MSE(tr2$Sin,compute(nn1, tr2$Var)$net.result)
MSE2 = MSE(tr1$Sin,compute(nn2, tr1$Var)$net.result)
paste("Error is", mean(MSE1,MSE2))
