require("Bagidis")

# Fetch command line arguments
my_args <- commandArgs(trailingOnly = TRUE)

#reorder and convert to numeric
magnitude = as.numeric(my_args[1])
num_coeffs = as.numeric(my_args[2])

Bagidis.lcurve = Bagidis::BUUHWE(magnitude)

#Subset by the number of coefficients
Bagidis.out.lcurve = Bagidis.lcurve[1:num_coeffs]

cat(Bagidis.out.lcurve)