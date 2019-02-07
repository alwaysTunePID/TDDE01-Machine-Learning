set.seed(1234567890)
library(geosphere)
stations <- read.csv("stations.csv", 
                     stringsAsFactors=FALSE, 
                     fileEncoding="latin1")
temps <- read.csv("temps50k.csv")
st <- merge(stations,temps,by="station_number")

h_distance <- 100000
  h_date <- 40
  h_time <- 2
a <- 58.4274 # The point to predict (up to the students)
b <- 14.826
date <- "2013-11-04" # The date to predict (up to the students)
times <- c("04:00:00", "06:00:00", "08:00:00", "10:00:00",
           "12:00:00", "14:00:00", "16:00:00", "18:00:00", 
           "20:00:00", "22:00:00", "00:00:00")
temp <- vector(length=length(times))
# My own code




convert_to_mins = function(time) {
  
  return(as.numeric(substr(time,1,2))*60 + as.numeric(substr(time,4,5)))
}

st$time = convert_to_mins(st$time)
st$day_num = as.numeric(factor(substr(as.character(temps$date), 
                               start = 6, 
                               stop = 10)))
# point1 is the longitude and latitude
# If distance_calc = True, the distance is sent
dist_kernel = function(point1, point2, distance = NULL) {
  if (is.null(distance)) {
    
    distance = distHaversine(p1=point1, p2 = point2)
  } 
  
  return(exp(-(distance/h_distance)^2))
}

time_kernel = function(time1, time2) {
  timeDiff = abs((time1 - time2)/60)
  
  
  if (length(timeDiff) > 1) {
    timeDiff = apply(as.matrix(timeDiff), 1, function(row) {
      if (row > 12) {
        return(12 - row %% 12)
      } else {
        return(row)
      }
    })
  }
  
  return(exp(-(timeDiff^2)/(h_time^2)))
}

date_kernel = function(date1, date2) {
  if (length(date2 > 1)) {
    daysBetween = apply(as.matrix(date2), 1, function(d2) {
      min(366 - abs(date1 - d2), abs(date1 - d2))  
    })
  } else {
    daysBetween = min(366 - abs(date1 - date2), abs(date1 - date2))  
  }
  return(exp(-(daysBetween^2)/(h_date^2)))
}


# Kernels created. Now start the prediciton

# Show sensibleness
# Show date
plot(x=seq(1,366,1), y=date_kernel(date1 = 45, 
                                   date2 = seq(1,366,1)), 
     type="l", 
     lwd=3, 
     col = "blue", 
     main = "Sensibleness for date kernel", 
     sub ="Sensibleness for day 45 of year")
# Show sensibleness for time kernel for 06:00:00
plot(x=seq(1,24*60,1), 
     y = time_kernel(time1 = 60, time2 = seq(1,24*60,1)), 
     type="l", lwd=3, col = "blue", 
     main ="Sensibleness for time kernel", 
     sub = "Sensibleness for 01:00:00")
# Show sensibleness for distance
plot(x=seq(-200000,200000,5),
     y=dist_kernel(0,0,distance = seq(-200000,200000,5)), 
     type = "l", lwd=3, col ="blue", 
     main="Sensibleness for distance kernel")


# Now, time to predict the temperature for point a,b

# pred_point is the longitude and latitude as c(latitude, longitude)
# data is st sent. Is assumed to have time converted to minutes and day_num, which is the day of the year.
sum_kernels = function(data, pred_point, times, date) {
  times = convert_to_mins(times)
  longitude = pred_point[2]
  latitude = pred_point[1]
  # Convert date to its number on the year
  date = which(
    substr(
      as.character(date), 
      start=6, 
      stop=10) == levels(
        factor(
          substr(as.character(data$date), 
                      start = 6, stop = 10)
          )
        )
    )
  
  # Now sum up the kernels!!!!
  
  k_date = date_kernel(date,data$day_num)
  k_distance = dist_kernel(c(longitude,latitude),
                           matrix(c(data$longitude,data$latitude), ncol=2, byrow=F))
  
  k_time = apply(as.matrix(times),1,time_kernel,data$time)
  
  # Sum up the kernels
  kern = apply(k_time, 2, function(time_k, k_date, k_distance) {
    return(time_k + k_date + k_distance)
  },k_date,k_distance)
  
  kern_t = apply(kern,2,function(col) {
    return(col*data$air_temperature)
  })
  
  # Now calculate the temperature for each time period
  n_time_periods = (dim(kern_t)[2])
  
  temps = as.numeric()
  
  for (i in 1:n_time_periods) {
    temps[i] = sum(kern_t[,i])/sum(kern[,i])
  }
  print(temps)
  return(temps)
}

prod_kernels = function(data, pred_point, times, date) {
  times = convert_to_mins(times)
  longitude = pred_point[2]
  latitude = pred_point[1]
  # Convert date to its number on the year
  date = which(substr(
    as.character(date), 
    start=6, stop=10) == 
      levels(factor(substr(as.character(data$date), 
                           start = 6, stop = 10)))
    )
  
  # Now sum up the kernels!!!!
  
  k_date = date_kernel(date,data$day_num)
  k_distance = dist_kernel(c(longitude,latitude),
                           matrix(c(data$longitude,data$latitude), 
                                  ncol=2, 
                                  byrow=F))
  
  k_time = apply(as.matrix(times),1,time_kernel,data$time)
  
  # Sum up the kernels
  kern = apply(k_time, 2, function(time_k, k_date, k_distance) {
    return(time_k*k_date*k_distance)
  },k_date,k_distance)
  
  kern_t = apply(kern,2,function(col) {
    return(col*data$air_temperature)
  })
  
  # Now calculate the temperature for each time period
  n_time_periods = (dim(kern_t)[2])
  
  temps = as.numeric()
  
  for (i in 1:n_time_periods) {
    temps[i] = sum(kern_t[,i])/sum(kern[,i])
  }
  print(temps)
  return(temps)
}

temps = sum_kernels(data = st, pred_point = c(a,b), times = times, date = date)
temps_prod = prod_kernels(data = st, pred_point = c(a,b), times = times, date = date)
plot(temps, type="o", ylim = c(min(temps_prod, temps), max(temps_prod, temps)), col = "red")
lines(temps_prod, type = "o", col = "blue")

##### Assignment 2 #####

# Task 1

library(kernlab)

data(spam)
data = spam
gauss_width = 0.05
C = c(0.5,1,5)

n=dim(data)[1]
set.seed(12345)
id=sample(1:n, floor(n*0.7))
train=data[id,]
val=data[-id,] 


for (c in C) {
  svm_model = ksvm(type~., data = train, C=c, kernel="rbfdot", kpar=list(sigma=(1/gauss_width)))
  pred_vals = predict(svm_model, val)
  print("Misclassification rate")
  print((1 - sum(diag(table(Actual = val$type, Predicted = pred_vals)))/sum(table(Actual = val$type, Predicted = pred_vals))))
}

# Best model is with C = 5

# Comment: C has the purpose of regularizing and avoiding overfitting by reducing the coefficients in the vector w

##### Assignment 3 #####

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




MSE_val = as.numeric()
MSE_train = as.numeric()
for(i in 1:10) {
    nn <- neuralnet(Sin ~ Var, data = tr, 
                  threshold = (i/1000), startweights = winit, hidden = 10)
    #nn <- neuralnet(Sin ~ Var, data=tr,hidden=10,threshold = (i/1000), startweights = winit)# Your code here)
      # Your code here
    tr_preds = compute(nn, tr$Var)$net.result
    val_preds = compute(nn, va$Var)$net.result
    MSE_val[i] = MSE(va$Sin, val_preds[,1])
    MSE_train[i] = MSE(tr$Sin, tr_preds[,1])
    
    
}

# Which error is the best? 
plot(x=(seq(1,10,1)/1000), y=MSE_val, main = "MSE for training and validation set", xlab = "Threshold", ylab = "MSE", type = "l", lwd = 3, col = "red", ylim=c(0,max(MSE_train, MSE_val)))
lines(x=(seq(1,10,1)/1000), y = MSE_train, lwd = 3, col = "blue")

#plot(nn <- neuralnet(Sin ~ Var,)# Your code here))
  # Plot of the predictions (black dots) and the data (red dots)
plot(prediction(nn)$rep1)
points(trva, col = "red")









### Toss a coin log-likelihood


tosses = c(0,1,1,1,1,0,1,1,1,1)

logLikCoinToss = function(theta, data) {
  return(sum(data*log(theta) + (1 - data)*log(1 - theta)))
}

thetas = seq(0,1,by=0.01)

logLiks = apply(as.matrix(thetas),1,logLikCoinToss, tosses)
plot(x=thetas, y=exp(logLiks), type = "l")

logPosterior = function(theta, data) {
  n1 = 5
  n2 = 5
  
  return(logLikCoinToss(theta,data) + dbeta(theta, n1,n2, log = TRUE))
}

logPosteriors = apply(as.matrix(thetas),1,logPosterior, tosses)
lines(x=thetas, y = exp(logPosteriors), col = "green")
