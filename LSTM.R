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
  layer_lstm(units = 256, input_shape = c(491-12, 1), return_sequences = TRUE) %>% 
  layer_dropout(0.25) %>%
  layer_lstm(units = 512,return_sequences = TRUE) %>%
  layer_dropout(0.25) %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dropout(0.25) %>%
  layer_dense(units = 1, activation = "relu")

model %>% compile(optimizer = optimizer_adam(learning_rate = 3e-4), loss = "mse")


train_x = array_reshape(data$credit_in_millions[1:nrow(data)-1], c(1, nrow(data)-1, 1))
train_y = array_reshape(data$credit_in_millions[2:nrow(data)], c(1, nrow(data)-1, 1))

model %>% fit(as.array(train_x), as.array(train_y),
              epochs = 1000)

out = numeric(12)
for (i in 1:12) {
  pred = predict(model, train_x)[1, 491-12, 1]
  train_x[1, 1:(490-12), 1] = train_x[1, 2:(491-12), 1]
  train_x[1, 491-12, 1] = pred
  out[i] = pred
}

rmse <- function(y_actual, y_pred) {
  sqrt(mean((y_actual - y_pred)^2))
}

plot(holdout$credit_in_millions,type="l",col="red")
lines(out,col="green")

rmse(holdout$credit_in_millions, out)
