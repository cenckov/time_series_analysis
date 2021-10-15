## R
## Ponieważ będziemy korzystać z liczb losowych musimy ustalić ziarno
## Jeżeli pozostawisz je niezmienione, powinieneś otrzymać dokładnie takie same rezultaty
set.seed(123)

## Ustawiamy parametry każdego z czterech stanów na rynkach finansowych
bull_mu <- 0.1
bull_sd <- 0.1

neutral_mu <- 0.02
neutral_sd <- 0.08

bear_mu <- -0.03
bear_sd <- 0.2

panic_mu <- -0.1
panic_sd <- 0.3


## Aby ułatwić sobie indeksowanie, umieszczamy  parametry w wektorach
mus <- c(bull_mu, neutral_mu, bear_mu, panic_mu)
sds <- c(bull_sd, neutral_sd, bear_sd, panic_sd)

## Ustalamy wartości kilku stałych opisujących szereg
NUM.PERIODS <- 10
SMALLEST.PERIOD <- 20
LONGEST.PERIOD <- 40

## Tworzymy wektor zawierający, otrzymane losowo, długości trwania (w dniach) każdego ze stanów na rynku
days <- sample(SMALLEST.PERIOD:LONGEST.PERIOD, NUM.PERIODS,replace = TRUE)

## Dla każdego okresu z wektora days tworzymy szereg czasowy opisujący dany stan
## Otrzymane w ten sposób szeregi łączymy w wektor returns

returns <- numeric()
true.mean <- numeric()
for (d in days) {
  idx = sample(1:4, 1, prob = c(0.2, 0.6, 0.18, 0.02))
  returns <- c(returns, rnorm(d, mean = mus[idx], sd = sds[idx]))
  true.mean <- c(true.mean, rep(mus[idx], d))
}

## R
table(true.mean)

## R
require(depmixS4)
hmm.model <- depmix(returns ~ 1, family = gaussian(),
                    nstates = 4, data=data.frame(returns=returns))
model.fit <- fit(hmm.model)
post_probs <- posterior(model.fit)

## R
plot(returns, type = 'l', lwd = 3, col = 1, 
     yaxt = "n", xaxt = "n", xlab = "", ylab = "",
     ylim = c(-0.6, 0.6))
lapply(0:(length(returns) - 1), function (i) {
  ## Dodanie prostokątów stanowiących tło, których kolor wskazuje na konkretny stan
  rect(i,-0.6,(i + 1),0.6,
       col = rgb(0.0,0.0,0.0,alpha=(0.2 * post_probs$state[i + 1])), border = NA)
  
})

attr(model.fit, "response")
