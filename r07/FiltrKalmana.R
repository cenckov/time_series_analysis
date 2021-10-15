## R
## Zajmujemy się lotem rakiety przez 100 pierwszych jednostek czasu
ts.length <- 100
## Ruch napędzany jest przez przyspieszenie
a <- rep(0.5, ts.length)
## Zarówno pozycja jak i prędkość wynoszą początkowo zero
x <- rep(0, ts.length)
v <- rep(0, ts.length)
for (ts in 2:ts.length) {
  x[ts] <- v[ts - 1] * 2 + x[ts - 1] + 1/2 * a[ts-1] ^ 2
  x[ts] <- x[ts] + rnorm(1, sd = 20) ## składnik losowy
  v[ts] <- v[ts - 1] + 2 * a[ts-1]
}

## R
par(mfrow = c(3, 1))
plot(x,            main = "Położenie",     type = 'l')
plot(v,            main = "Prędkość",     type = 'l')
plot(a, main = "Przyspieszenie", type = 'l')

## R
z <- x + rnorm(ts.length, sd = 300)
plot(x, ylim = range(c(x, z)))
lines(z)

## R
kalman.motion <- function(z, Q, R, A, H) {
  dimState = dim(Q)[1] 
  xhatminus <- array(rep(0, ts.length * dimState), c(ts.length, dimState))
  xhat <- array(rep(0, ts.length * dimState), c(ts.length, dimState))
  Pminus <- array(rep(0, ts.length * dimState * dimState), c(ts.length, dimState, dimState))
  P <- array(rep(0, ts.length * dimState * dimState), c(ts.length, dimState, dimState))
  
  K <- array(rep(0, ts.length * dimState), c(ts.length, dimState)) # Wzmocnienie Kalmana
  
  # Założenie początkowe, zaczynamy od zera dla każdego z parametrów
  xhat[1, ] <- rep(0, dimState)
  P[1, , ] <- diag(dimState)
  
  for (k in 2:ts.length) {
    # Zmiana czasu
    xhatminus[k, ] <- A %*% matrix(xhat[k-1, ])
    Pminus[k, , ] <- A %*% P[k-1, , ] %*% t(A) + Q
    K[k, ] <- Pminus[k, , ] %*% H %*% solve( t(H) %*% Pminus[k, , ] %*% H + R )
    xhat[k, ] <- xhatminus[k, ] + K[k, ] %*% (z[k]- t(H) %*% xhatminus[k, ])
    P[k, , ] <- (diag(dimState)-K[k,] %*% t(H)) %*% Pminus[k, , ]
    }
  ## zwracamy zarówno prognozę, jak i wygładzone wartości 
  return(list(xhat = xhat, xhatminus = xhatminus))
}

## R
## Parametry szumów
R <- 10^2
## Wariancja pomiaru musi zostać ustalona
## Biorąc pod uwagę fizyczne ograniczenia aparatury pomiarowej ustalamy
## jej wartość na taka samą jak ta dodana do składnika x podczas jego generowania

## Wariancja procesu jest najczęściej uznawana za hiperparametr,
## którego wartość dopasowuje się tak aby zwiększyć wydajność modelu
Q <- 10
## Parametry dynamiczne
A <- matrix(1) ## x_t = A * x_t-1 (w jaki sposób poprzednie wartości x wpływają na obecną)
H <- matrix(1) ## y_t = H * x_t (przekształcenie stanu w pomiar)

## Wywołanie metody implementującej filtr Kalmana
xhat <- kalman.motion(z, diag(1) * Q, R, A, H)[[1]]


