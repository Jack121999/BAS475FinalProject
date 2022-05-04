library(fpp3)

# Read in data
credit <- read.csv("credit.csv")

# Reverse data
credit[, 2] <- seq(from = 492, to = 1, by = -1)
names(credit) <- c("credits", "month")

# Convert to tsibble
credit <- credit %>% 
  mutate(month = yearmonth(month)) %>% 
  as_tsibble(index = month)

credit %>% 
  autoplot()


# Train/test split
train <- credit[1:480, ]
test <- credit[481:492, ]


# Fit models to train
fit <- train %>% 
  model(
    ets = ETS(credits),
    arima = ARIMA(credits),
    tslm = TSLM(credits ~ trend() + season()),
    drift = RW(credits ~ drift()),
    croston = CROSTON(credits)
  )

# Plot model forecasts
fit %>% 
  forecast(h = 12) %>% 
  autoplot(credit)

# Check model accuracies against test data
fit %>% 
  forecast(h = 12) %>% 
  accuracy(test) %>%
  arrange(RMSE)

# Select best model
fit <- train %>% 
  model(arima = ARIMA(credits))

fit %>% 
  report() %>% 
  gg_tsresiduals()


fit %>% 
  forecast(test) %>% 
  autoplot(test)

pred <- fit %>% 
  forecast(test)
pred <- pred$.mean

rmse <- function(y_actual, y_pred) {
  sqrt(mean((y_actual - y_pred)^2))
}
rmse(test$credits, pred)

# Please Work 2