require(data.table)
## R
x <- 1:100
y <- sin(x * pi /3)
plot(y, type = "b")
acf(y)

## R
cor(y, shift(y, 1), use = "pairwise.complete.obs")
cor(y, shift(y, 2), use = "pairwise.complete.obs") 

## R
y <- sin(x * pi /3)
plot(y[1:30], type = "b") 
pacf(y)
