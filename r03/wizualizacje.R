## R
require(timevis)
require(data.table)
donations <- fread("donations.csv")
d <- donations[, .(min(timestamp), max(timestamp)), user]
names(d) <- c("content", "start", "end")
d <- d[start != end]
timevis(d[sample(1:nrow(d), 20)])

## R - polski opis

colors <- c("green", "red", "pink", "blue", "yellow","lightsalmon", "black", "gray",
                        "cyan",  "lightblue",   "maroon", "purple")
matplot(matrix(AirPassengers, nrow = 12, ncol = 12),
          type = 'l', col = colors, lty = 1, lwd = 2.5,
          xaxt = "n", ylab = "Ilość Pasażerow")
legend("topleft", legend = 1949:1960, lty = 1, lwd = 2.5,
         col = colors)
axis(1, at = 1:12, labels = c("STY", "LUT", "MAR", "KWI",
                              "MAJ", "CZE", "LIP", "SIE",
                              "WRZ", "PAZ", "LIS", "GRU"))

## R — opis oryginalny

colors <- c("green", "red", "pink", "blue", "yellow","lightsalmon", "black", "gray",
            "cyan",  "lightblue",   "maroon", "purple")
matplot(matrix(AirPassengers, nrow = 12, ncol = 12),
        type = 'l', col = colors, lty = 1, lwd = 2.5,
        xaxt = "n", ylab = "Passenger Count")
legend("topleft", legend = 1949:1960, lty = 1, lwd = 2.5,
       col = colors)
axis(1, at = 1:12, labels = c("Jan", "Feb", "Mar", "Apr",
                              "May", "Jun", "Jul", "Aug",
                              "Sep", "Oct", "Nov", "Dec"))

## R
require(forecast)
seasonplot(AirPassengers)


## R
months <- c("Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
matplot(t(matrix(AirPassengers, nrow = 12, ncol = 12)),
            type = 'l', col = colors, lty = 1, lwd = 2.5)
legend("left", legend = months,
                        col = colors, lty = 1, lwd = 2.5)


## R
monthplot(AirPassengers)

##R
hist2d <- function(data, nbins.y, xlabels) {
  ## Zapewniamy równomierne odstępy pomiędzy danymi
  ## w ybins, uwzględniamy również minimum i maksimum
  ymin = min(data)
  ymax = max(data) * 1.0001
  ## Łatwy sposób na porawdzenie sobie z problemem przynależności wartości granicznych
  
  ybins = seq(from = ymin, to = ymax, length.out = nbins.y + 1 )
  
  ## Tworzenie macierzy zer o odpowiednim rozmiarze
  hist.matrix = matrix(0, nrow = nbins.y, ncol = ncol(data))
  
  ## Umieszczamy dane w macierzy zgodnie z zasadą że każdy wiersz zawiera informację o jednym punkcie danych
  for (i in 1:nrow(data)) {
    ts = findInterval(data[i, ], ybins)
    for (j in 1:ncol(data)) {
     hist.matrix[ts[j], j] = hist.matrix[ts[j], j] + 1
    }
  }
  hist.matrix
}
     

## R
h = hist2d(t(matrix(AirPassengers, nrow = 12, ncol = 12)), 5, months)
image(1:ncol(h), 1:nrow(h), t(h), col = heat.colors(5),
      axes = FALSE, xlab = "Time", ylab = "Passenger Count")


## R
require(data.table)
words <- fread("50words_TEST.csv")
w1 <- words[V1 == 1]
h = hist2d(w1, 25, 1:ncol(w1))
colors <- gray.colors(20, start = 1, end = .5)
par(mfrow = c(1, 2))
image(1:ncol(h), 1:nrow(h), t(h),
      col = colors, axes = FALSE, xlab = "Time", ylab = "Projection Value")
image(1:ncol(h), 1:nrow(h), t(log(h)),
      col = colors, axes = FALSE, xlab = "Time", ylab = "Projection Value")

## R
require(hexbin)
w1 <- words[V1 == 1]
## Zmiana danych na pary wartości xy wymagane przez większość implementacji histogramów 2d
names(w1) <- c("type", 1:270)
w1 <- melt(w1, id.vars = "type")
w1 <- w1[, -1]
names(w1) <- c("Time point", "Value")

plot(hexbin(w1))

## R
require(plotly)
require(data.table)
months = 1:12
ap = data.table(matrix(AirPassengers, nrow = 12, ncol = 12))
names(ap) = as.character(1949:1960)
ap[, month := months]
ap = melt(ap, id.vars = 'month')
names(ap) = c("month", "year", "count")

p <- plot_ly(ap, x = ~month, y = ~year, z = ~count,
             color = ~as.factor(month)) %>%
  add_markers() %>%
  layout(scene = list(xaxis = list(title = 'Month'),
                      yaxis = list(title = 'Year'),
                      zaxis = list(title = 'PassengerCount')))

## R

file.location <- 'https://raw.githubusercontent.com/plotly/datasets/master/_3d-line-plot.csv'
data <- read.csv(file.location)
p <- plot_ly(data, x = ~x1, y = ~y1, z = ~z1,
             type = 'scatter3d', mode = 'lines',
             line = list(color = '#1f77b4', width = 1))

