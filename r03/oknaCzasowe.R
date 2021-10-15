## R
## Do obliczenia średniej ruchomej można wykorzystać dostępną w R funkcję filter 
x <- rnorm(n = 100, mean = 0, sd = 10) + 1:100
mn <- function(n) rep(1/n, n)
plot(x, type = 'l', lwd = 1)
lines(filter(x, mn( 5)), col = 2, lwd = 3, lty = 2)
lines(filter(x, mn(50)), col = 3, lwd = 3, lty = 3)


## R
## rollapply() pozwala dostosować działanie do potrzeb użytkownika
require(zoo)
f1 <- rollapply(zoo(x), 20, function(w) min(w),align = "left", partial = TRUE)
f2 <- rollapply(zoo(x), 20, function(w) min(w), align = "right", partial = TRUE)
plot (x,           lwd = 1,         type = 'l')
lines(f1, col = 2, lwd = 3, lty = 2)
lines(f2, col = 3, lwd = 3, lty = 3)


## R
# Rozszerzające się okna
plot(x, type = 'l', lwd = 1)
lines(cummax(x),             col = 2, lwd = 3, lty = 2) # maksimum
lines(cumsum(x)/1:length(x), col = 3, lwd = 3, lty = 3) # średnia

## R
plot(x, type = 'l', lwd = 1)
lines(rollapply(zoo(x), seq_along(x), function(w) max(w),
                partial = TRUE, align = "right"),
      col = 2, lwd = 3, lty = 2)
lines(rollapply(zoo(x), seq_along(x), function(w) mean(w),
                partial = TRUE, align = "right"),
      col = 3, lwd = 3, lty = 3)

