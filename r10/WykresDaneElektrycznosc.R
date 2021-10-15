## R
require(data.table)
elec = fread("electricity.txt")
elec

ncol(elec)
nrow(elec)

elec[125:148, plot(V4,    type = 'l', col = 1, ylim = c(0, 1000), ylab='Zużycie energii', xlab='')]
elec[125:148, lines(V14,  type = 'l', col = 2)]
elec[125:148, lines(V114, type = 'l', col = 3)]

elec[1:168, plot(V4,    type = 'l', col = 1, ylim = c(0, 1000), ylab='Zużycie energii', xlab='')]
elec[1:168, lines(V14,  type = 'l', col = 2)]
elec[1:168, lines(V114, type = 'l', col = 3)]


elec.diff = fread("electricity.diff.txt")
elec.diff[1:168, plot( V4, type = 'l', col = 1, ylim = c(-350, 350),ylab='Zużycie energii', xlab='')]
elec.diff[1:168, lines(V14, type = 'l', col = 2)]
elec.diff[1:168, lines(V114, type = 'l', col = 3)]
