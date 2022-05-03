Credit <- read.csv("credit.csv")

Credit <- Credit %>%
  mutate(Time = rev(row_number()))%>%
  as_tsibble(index = Time)

gg_tsdisplay(Credit, plot_type = "partial")

# -----------------------------------------------------------------------------

Training <- Credit[1:(nrow(Credit)-12),]
Holdout <- Credit[481:492,]

# -----------------------------------------------------------------------------

lambda <- Training %>%
  features(ï..credit_in_millions, features=guerrero)%>%
  pull(lambda_guerrero)

Training <- Training %>%
  mutate(bc_training=box_cox(ï..credit_in_millions, lambda))

Training <- Training %>%
  mutate(bcd_training=difference(bc_training))

Training %>%
  autoplot(bcd_training)

# -----------------------------------------------------------------------------

fit <- Training %>%
  model(
    arima = ARIMA(bcd_training)
  )

fit %>%
  glance()

bc_pred <- fit %>%
  forecast(Holdout)

pred <- inv_box_cox(bc_pred$.mean,lambda)

# -----------------------------------------------------------------------------

mape <- function(y_pred,y_actual){
  mean(abs((y_pred-y_actual)/y_actual))
}

rmse <- function(y_pred,y_actual){
  sqrt(mean((y_pred-y_actual)^2))
}

mape(pred, Holdout$ï..credit_in_millions)
rmse(pred, Holdout$ï..credit_in_millions)
