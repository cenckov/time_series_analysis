## R
require(forecast)
set.seed(1017)
## Rząd metody został specjalnie zakomentowany (taki kod nie zadziała)
y = arima.sim(n = 1000, list(ar = c(0.8,-0.4), ma = c(-0.7)))
                               
## R
ar1.ma1.model = Arima(y, order = c(1, 0, 1))
par(mfrow = c(2,1))
acf(ar1.ma1.model$residuals)
pacf(ar1.ma1.model$residuals)

## R
ar2.ma1.model = Arima(y, order = c(2, 0, 1))
plot(y, type = 'l')
lines(ar2.ma1.model$fitted, col = 2)
plot(y, ar2.ma1.model$fitted)
par(mfrow = c(2,1))
acf(ar2.ma1.model$residuals)
pacf(ar2.ma1.model$residuals)
## R
ar2.ma2.model = Arima(y, order = c(2, 0, 2))
plot(y, type = 'l')
lines(ar2.ma2.model$fitted, col = 2)
plot(y, ar2.ma2.model$fitted)
par(mfrow = c(2,1))
acf(ar2.ma2.model$residuals)
pacf(ar2.ma2.model$residuals)

ar2.d1.ma2.model = Arima(y, order = c(2, 1, 2))
plot(y, type = 'l')
lines(ar2.d1.ma2.model$fitted, col = 2)
plot(y, ar2.d1.ma2.model$fitted)
par(mfrow = c(2,1))
acf(ar2.d1.ma2.model$residuals)
pacf(ar2.d1.ma2.model$residuals)

## R
cor(y, ar1.ma1.model$fitted)
#[1] 0.3018926
cor(y, ar2.ma1.model$fitted) 
#[1] 0.4683598
cor(y, ar2.ma2.model$fitted)
#[1] 0.4684905
cor(y, ar2.d1.ma2.model$fitted)
#[1] 0.4688166


## R
## Oryginalne wspołczynniki
y = arima.sim(n = 1000, list(ar = c(0.8, -0.4), ma = c(-0.7)))
ar2.ma1.model$coef
