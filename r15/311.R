## R
require(data.table)
df = fread("311.csv", skip = 0, nrows = 10) 
colnames(df)

setnames(df, gsub(" ", "", colnames(df)))

## Aby móc pracować z datami należy ustawić odpowiedni czas lokalny w parametrach R
Sys.setlocale("LC_TIME", "en_US.UTF-8")
## R

df$CreatedDate = df[, CreatedDate := as.POSIXct(CreatedDate, format = "%m/%d/%Y %I:%M:%S %p")]
df$ClosedDate = df[, ClosedDate := as.POSIXct(ClosedDate, format = "%m/%d/%Y %I:%M:%S %p")]

## R
summary(as.numeric(df$ClosedDate - df$CreatedDate, units = "days"))

##R
range(df$CreatedDate)

## R
## Wczytujemy tylko interesujące nas kolumny
df = fread("311.csv", select = c("Created Date", "Closed Date"))
  
## Zmiana nazw kolumn na zalecany  w data.table zapis bez spacji
setnames(df, gsub(" ", "", colnames(df)))

## Usuwamy wiersze, w których w miejscu dat są puste pola
df = df[nchar(CreatedDate) > 1 & nchar(ClosedDate) > 1]

## Konwersja daty do formatu POSIXct

fmt.str = "%m/%d/%Y %I:%M:%S %p"
df$CreatedDate = df[, CreatedDate := as.POSIXct(CreatedDate, format = fmt.str)]
df$ClosedDate = df[, ClosedDate := as.POSIXct(ClosedDate, format = fmt.str)]
## Ustalenie porządku wg kolumny CreatedDate
setorder(df, CreatedDate)

## Obliczenie liczby dni pomiędzy otwarciem a zamknięciem zgłoszenia
## na infolinii 311
df[, LagTime := as.numeric(difftime(ClosedDate, CreatedDate, 
                           units = "days"))]

## R
summary(df$LagTime)
nrow(df[LagTime < 0]) / nrow(df)
nrow(df[LagTime > 1000]) / nrow(df)
df = df[LagTime < 1000]
df = df[LagTime > 0]
df.new = df[seq(1, nrow(df), 2), ]
write.csv(df.new[order(ClosedDate)], "abridged.df.csv")

  