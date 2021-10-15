## R
## Załóżmy, że proces jest naprawdę procesem AR(2);
## ponieważ jego symulacja jest bardziej skomplikowana, skorzystamy z arima.sim

require(forecast)
time_steps  <- 24 
N           <- 1000
sigma_error <- 1


phi_1 <- 0.7
phi_2 <- -0.2
estimates <- numeric(N)
for (i in 1:N) {
  res <- arima.sim(list(order = c(2,0,0),
                        ar = c(phi_1, phi_2)),
                        n = time_steps)

  estimates[i] <- arima(res, c(1, 0, 0))$coef[1]
}

hist(estimates,
     main = "Oszacowanie Phi dla procesu AR(1) \n przy założeniu, że szereg jest procesem AR(2)",
     xlab = "Oszacowania",
     ylab = "Liczba wystąpień",
     breaks = 50)

##R
summary(estimates)
