setwd("/home/levicivita/HonorsThesisTZ/ThesisCode/gen_lightcurves/gp_smoothed")
library("Bagidis");
library("rjson");
json_file <- "SN2002fk_gpsmoothed.json";
json_data <- fromJSON(paste(readLines(json_file), collapse=""));

#mag = json_data$r$mag
#mag = mag[seq(1, length(mag), 2)]
modelmag = json_data$r$modelmag
modelmag = modelmag[seq(1,length(modelmag), 5)]
#modelmag_shift = shift(modelmag, 20)

#modelmag = c(0,0,0,0,0,0,0,0,1,2,3,4,5,6,6,5,4,3,0,0,0,0)
#modelmag_shift = c(0,0,0,0,0,0,0,0,0,0,1,2,3,4,5,6,6,5,4,3,0,0)
#modelmag_diff  = c(0,0,0,0,0,0,0,0,0,0,0,3,4,5,6,6,5,4,3,2,1,0)

print(modelmag)
#mag_decomp <- Bagidis::BUUHWE(mag)
modelmag_decomp <- Bagidis::BUUHWE(modelmag)
#modelmag_decomp_shift <- Bagidis::BUUHWE(modelmag_shift)
#modelmag_decomp_diff <- Bagidis::BUUHWE(modelmag_diff)
#Bagidis::BUUHWE.plot(mag_decomp, Color = FALSE, row.max=20, border=FALSE)
Bagidis::BUUHWE.plot(modelmag_decomp, Color = FALSE)
#Bagidis::BUUHWE.plot(modelmag_decomp_shift, Color = FALSE)
#Bagidis::BUUHWE.plot(modelmag_decomp_diff, Color = FALSE)
modelmag_decomp$detail[0:10]
class(modelmag_decomp$basis)
#jpeg(filename = "Bagidis_decomp.jpg",width=850,height=550,units="px")

dev.off()

#Take the first 10 coefficients and recompute the series values
detail = modelmag_decomp$detail
basis = modelmag_decomp$basis

detail[10:length(detail)] <- 0
basis[,10:dim(basis)[2]] <- 0

#detail <- detail[2:length(detail)]
#basis <- basis[,2:dim(basis)[2]]

rebuilt_modelmag <- t(as.matrix(detail)) %*% t(basis)
#plot(1:length(rebuilt_modelmag), rebuilt_modelmag)

modelmag_rebuilt <- Bagidis::BUUHWE(modelmag)
modelmag_rebuilt$series <- rebuilt_modelmag
modelmag_rebuilt$detail <- detail[1:10]
modelmag_rebuilt$basis <- basis[,1:10]
Bagidis::BUUHWE.plot(modelmag_rebuilt, Color = FALSE)

