## R
require(forecast)

phi         <- 0.7
time_steps  <- 24 
N           <- 1000
sigma_error <- 1

sd_series   <- sigma_error^2 / (1 - phi^2)
starts      <- rnorm(N, sd = sqrt(sd_series))
estimates   <- numeric(N)
res         <- numeric(time_steps)

for (i in 1:N) {
  errs = rnorm(time_steps, sd = sigma_error)
  res[1] <- starts[i] + errs[1]
  
  for (t in 2:time_steps) {
    res[t] <- phi * res[t-1] + errs[t]
  }
  
  estimates[i] <- arima(res, c(1, 0, 0))$coef[1]
}

hist(estimates,
     main = "Oszacowanie Phi dla procesu AR(1) \n przy założeniu, że szereg jest procesem AR(1)",
     xlab = "Oszacowania",
     ylab = "Liczba wystąpień",
     breaks = 50)

##R
summary(estimates)
