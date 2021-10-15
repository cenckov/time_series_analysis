## R korzysta z radianów do określania częstotliwości 

x = 1:100

## Szereg bez szumów
y = sin(x * pi / 4) 
plot(y, type = 'b')
acf(y)
pacf(y)

y = sin(x * pi / 10)
plot(y, type = 'b')
acf(y)
pacf(y)

y = sin(x * pi / 4) + sin(x * pi / 10)
plot(y, type = 'b')
acf(y)
pacf(y)

par(mfrow = c(3, 3))
## Szereg odrobinę zaszumiony
noise1 = rnorm(100, sd = 0.05)
noise2 = rnorm(100, sd = 0.05)
par(mfrow = c(3, 3))
y = sin(x * pi / 4) + noise1
plot(y, type = 'b')
acf(y)
pacf(y)

y = sin(x * pi / 10) + noise2
plot(y, type = 'b')
acf(y)
pacf(y)

y = sin(x * pi / 4) + sin(x * pi / 10) + noise1 + noise2
plot(y, type = 'b')
acf(y)
pacf(y)

## Bardzo zaszumiony szereg
noise1 = rnorm(100, sd = 0.3)
noise2 = rnorm(100, sd = 0.3)
par(mfrow = c(3, 3))
y = sin(x * pi / 4) + noise1
plot(y, type = 'b')
acf(y)
pacf(y)

y = sin(x * pi / 10) + noise2
plot(y, type = 'b')
acf(y)
pacf(y)

y = sin(x * pi / 4) + sin(x * pi / 10) + noise1 + noise2
plot(y, type = 'b')
acf(y)
pacf(y)


