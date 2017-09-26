from __future__ import division
import numpy as np
from mpmath import *

# --------- Haar wavelet approximation of a function
# algorithm from : http://fourier.eng.hmc.edu/e161/lectures/wavelets/node5.html
# implementation only handle [0,1] for the moment: scaling and wavelet fcts need to be periodice

phi = lambda x : (0 <= x < 1) #scaling fct
psi = lambda x : (0 <= x < .5) - (.5 <= x < 1) #wavelet fct
phi_j_k = lambda x, j, k : 2**(j/2) * phi(2**j * x - k)
psi_j_k = lambda x, j, k : 2**(j/2) * psi(2**j * x - k)

def haar(f, interval, level):
    c0 = quadgl(  lambda t : f(t) * phi_j_k(t, 0, 0), interval  )

    coef = []
    for j in xrange(0, level):
        for k in xrange(0, 2**j):
                djk = quadgl(  lambda t: f(t) * psi_j_k(t, j, k), interval  )
                coef.append( (j, k, djk) )

    return c0, coef

def haarval(haar_coef, x):
    c0, coef = haar_coef
    s = c0 * phi_j_k(x, 0, 0)
    for j, k ,djk in coef:
            s += djk * psi_j_k(x, j, k)
    return s

# --------- to plot an Haar wave
interval = [0, 1]
plot([lambda x : phi_j_k(x,1,1)],interval)

# ---------- main
# below is code to compate : Haar vs Fourier vs Chebyshev

nb_coeff = 4
interval = [0, 1] # haar only handle [0,1] for the moment: scaling and wavelet fcts need to be periodice

fct = lambda x : (x-0.5)**2

haar_coef = haar(fct, interval, nb_coeff+1)
haar_series_apx = lambda x : haarval(haar_coef, x)

fourier_coef = haar(fct, interval, nb_coeff+1)
fourier_series_apx = lambda x: haarval(fourier_coef, x)
chebyshev_coef = haar(fct, interval, nb_coeff+1)
chebyshev_series_apx = lambda x : haarval(chebyshev_coef, x)


#print 'fourier %d chebyshev %d haar %d' % ( len(fourier_coef[0]) + len(fourier_coef[1]),len(chebyshev_coef), 1 + len(haar_coef[1]))
#print 'error:'
#print 'fourier', quadgl(  lambda x : abs( fct(x) - fourier_series_apx(x) ), interval  )
#print 'chebyshev', quadgl(  lambda x : abs( fct(x) - chebyshev_series_apx(x) ), interval  )
#print 'haar', quadgl(  lambda x : abs( fct(x) - haar_series_apx(x) ), interval  )

plot([fct, chebyshev_series_apx, haar_series_apx], interval) 

