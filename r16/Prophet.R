## R
library(pageviews)
df_wiki = article_pageviews(project = "en.wikipedia",
                              article = "Facebook",
                              start = as.Date('2015-11-01'),
                              end = as.Date("2018-11-02"),
                              user_type = c("user"),
                              platform = c("mobile-web"))
colnames(df_wiki)

require(prophet)
## Tworzymy podzbiór zawierający pożądane przez nas dane 
## i nadajemy im nazwy zgodne z naszymi oczekiwaniami
df = df_wiki[, c("date", "views")]
colnames(df) = c("ds", "y")

## Ponieważ w danych pojawiają się ekstremalne różnice, 
## zmniejszamy dzienną zmienność, stosując transformację 
## logarytmiczną
df$y = log(df$y)

## Tworzymy ramkę danych zawierającą przyszłe daty,
## w których chcemy dokonać prognozy
## Prognozujemy na 365 dni do przodu
m = prophet(df)
future <- make_future_dataframe(m, periods = 365)
tail(future)

## Tworzymy prognozę w interesujących nas datach
forecast <- predict(m, future)
tail(forecast[c('ds', 'yhat', 'yhat_lower', 'yhat_upper')])

## Teraz gdy dysponujemy już prognozą interesujących nas wartości 
## tworzymy wykres, aby ocenić jakość prognozy
plot(df$ds, df$y, col = 1, type = 'l', xlim = range(forecast$ds),
     main = "Rzeczywiste i prognozowane liczby odwiedzin artykułu o Facebooku dostępnego w anglojęzycznej Wikipedii")
points(forecast$ds, forecast$yhat, type = 'l', col = 2)

## R
prophet_plot_components(m, forecast)

