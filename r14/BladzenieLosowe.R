## R
## Do stworzenia wykresów wykorzystaj kolejno każde z trzech poniższych ustawień ziarna
set.seed(1)
##set.seed(100)
##set.seed(30)

N <- 10000
x <- cumsum(sample(c(-1, 1), N, TRUE))
plot(x)

## R
require(data.table)
cor(x, shift(x), use = "complete.obs")

##R
cor (diff (x), shift (diff (x)), use = "complete.obs") 

