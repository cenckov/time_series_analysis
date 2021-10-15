## R
library(AnomalyDetection)
data(raw_data)
head(raw_data)

## R
## Wykrywanie większego procentu anomalii w obu kierunkach
general_anoms = AnomalyDetectionTs(raw_data, max_anoms=0.05,
                                   direction='both')

## Wykrywanie mniejszego procentu anomalii tylko w dodatnim kierunku
high_anoms = AnomalyDetectionTs(raw_data, max_anoms=0.01,
                                direction='pos')
