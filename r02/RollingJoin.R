## R
require(data.table)
## Posiadamy skromny rejestr informacji o przekazanych darowiznach
donations <- data.table(amt = c(99, 100, 5, 15, 11, 1200), dt = as.Date(c("2019-2-27", "2019-3-2", "2019-6-13", "2019-8-1", "2019-8-31", "2019-9-15")))

## Oraz informacje na temat przeprowadzonych kampanii marketingowych,
publicity <- data.table(identifier = c("q4q42", "4299hj", "bbg2"),dt = as.Date(c("2019-1-1", "2019-4-1", "2019-7-1")))

## Na każdej tablicy ustalamy, która wartość ma być kluczem głównym.
setkey(publicity, "dt")
setkey(donations, "dt")

## Chcemy połączyć informacje o każdej dotacji z ostatnią kampanią reklamową, która ją poprzedziła. Możemy to zrobić, ustawiając parametr roll = TRUE

publicity[donations, roll = TRUE]
