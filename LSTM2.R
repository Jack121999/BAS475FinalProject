library(keras)
library(tensorflow)
library(fpp3)
use_condaenv("apple_tensorflow", required = T)

data = read.csv("credit.csv")

data = as.data.frame(t(rev(as.data.frame(t(data)))))

data$m1 = 0
data$m2 = 0
data$m3 = 0
data$m4 = 0
data$m5 = 0
data$m6 = 0
data$m7 = 0
data$m8 = 0
data$m9 = 0
data$m10 = 0
data$m11 = 0
data$m12 = 0

for (i in 1:nrow(data)) {
  data[i, (i%%12)+2] = 1 
}

holdout = tail(data, 13)
data = head(data, 492-12)

model <- keras_model_sequential()

model %>%
  layer_lstm(units = 256, input_shape = c(24, 13), return_sequences = TRUE) %>% 
  layer_dropout(0.25) %>%
  layer_lstm(units = 512,return_sequences = FALSE) %>%
  layer_dropout(0.25) %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dropout(0.25) %>%
  layer_dense(units = 1, activation = "relu")

model %>% compile(optimizer = optimizer_adam(learning_rate = 1e-4), loss = "mse")

train_x = array(numeric(),c(480-24,24,13))
train_y = array(numeric(),c(480-24,1,1))
for(i in 1:(480-24)) {
  print(i)
  train_x[i, 1:24, 1:13] = data.matrix(data[i:(i+23),])
  train_y[i, 1, 1] = data$credit_in_millions[(i+23+1)]
}

model %>% fit(as.array(train_x), as.array(train_y), epochs = 50, validation_split=.2)

out = numeric(12)
x = array_reshape(train_x[480-24,,], c(1, 24, 13))
for (i in 1:12) {
  pred = predict(model, x)[1, 1]
  x[1, 1:23,] = x[1, 2:24,]
  x[1, 24, 1] = pred + .17
  x[1, 24, 2:13] = data.matrix(holdout[i, 2:13])
  out[i] = pred
}

rmse <- function(y_actual, y_pred) {
  sqrt(mean((y_actual - y_pred)^2))
}

out = out + .1

plot(holdout$credit_in_millions,type="l",col="red")
lines(out,col="green")

rmse(holdout$credit_in_millions[1:12], out)


#-------------------CREATE FINAL MODEL-------------------#


data = read.csv("credit.csv")

data = as.data.frame(t(rev(as.data.frame(t(data)))))

data$m1 = 0
data$m2 = 0
data$m3 = 0
data$m4 = 0
data$m5 = 0
data$m6 = 0
data$m7 = 0
data$m8 = 0
data$m9 = 0
data$m10 = 0
data$m11 = 0
data$m12 = 0

for (i in 1:nrow(data)) {
  data[i, (i%%12)+2] = 1 
}

model <- keras_model_sequential()

model %>%
  layer_lstm(units = 256, input_shape = c(24, 13), return_sequences = TRUE) %>% 
  layer_dropout(0.25) %>%
  layer_lstm(units = 512,return_sequences = FALSE) %>%
  layer_dropout(0.25) %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dropout(0.25) %>%
  layer_dense(units = 1, activation = "relu")

model %>% compile(optimizer = optimizer_adam(learning_rate = 1e-4), loss = "mse")

train_x = array(numeric(),c(492-24,24,13))
train_y = array(numeric(),c(492-24,1,1))
for(i in 1:(492-24)) {
  print(i)
  train_x[i, 1:24, 1:13] = data.matrix(data[i:(i+23),])
  train_y[i, 1, 1] = data$credit_in_millions[(i+23+1)]
}

model %>% fit(as.array(train_x), as.array(train_y), epochs = 50, validation_split=.2)

out = numeric(12)
x = array_reshape(train_x[492-24,,], c(1, 24, 13))
months = x[1, 1:12, 2:13]
for (i in 1:12) {
  pred = predict(model, x)[1, 1]
  x[1, 1:23,] = x[1, 2:24,]
  x[1, 24, 1] = pred + .2
  x[1, 24, 2:13] = data.matrix(months[i,])
  out[i] = pred
}

out = out + .1

plot(1:492, data$credit_in_millions, type="l",col="red")
lines(493:504, out, col="green")

outDF = data.frame(out)
write.csv(outDF, "predictions.csv")
