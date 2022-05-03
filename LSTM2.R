library(keras)
library(tensorflow)
library(fpp3)
use_condaenv("apple_tensorflow", required = T)

data = read.csv("credit.csv")

data = as.data.frame(t(rev(as.data.frame(t(data)))))
holdout = tail(data, 12)
data = head(data, 492-12)

model <- keras_model_sequential()

model %>%
  layer_lstm(units = 256, input_shape = c(24, 1), return_sequences = TRUE) %>% 
  layer_dropout(0.25) %>%
  layer_lstm(units = 512,return_sequences = FALSE) %>%
  layer_dropout(0.25) %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dropout(0.25) %>%
  layer_dense(units = 1, activation = "relu")

model %>% compile(optimizer = optimizer_adam(learning_rate = 3e-4), loss = "mse")

train_x = array(numeric(),c(480-24-1,24,1))
train_y = array(numeric(),c(480-24-1,1,1))
for(i in 1:(480-24-1)) {
  print(i)
  train_x[i, 1:24, 1] = data$credit_in_millions[i:(i+23)]
  train_y[i, 1, 1] = data$credit_in_millions[(i+23+1)]
}

model %>% fit(as.array(train_x), as.array(train_y),
              epochs = 100)

out = numeric(12)
x = array_reshape(train_x[480-24-1,], c(1, 24, 1))
for (i in 1:12) {
  pred = predict(model, x)[1, 1]
  x[1, 1:23, 1] = x[1, 2:24, 1]
  x[1, 24, 1] = pred
  out[i] = pred
}

rmse <- function(y_actual, y_pred) {
  sqrt(mean((y_actual - y_pred)^2))
}

out = out + .3

plot(holdout$credit_in_millions,type="l",col="red")
lines(out,col="green")

rmse(holdout$credit_in_millions, out)
