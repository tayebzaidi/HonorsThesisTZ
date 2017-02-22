function bestrest, obspb, z, lameff=lameff, nolookup=nolookup, lookupfile=lookupfile
;
; bestrest.pro -- determines the best match rest-frame passbands
;
; Created 2005-Nov-12
; Version 2007-Feb-12
;               
; INPUTS
;     obspb: observer frame passband names [string; array or scalar]
;         z: redshift [scalar]
;
; OPTIONAL INPUTS
;     nolookup:  if set, don't use the lookup table, just go with
;                the default algorithm based on effective wavelength
;     lookupfile: alternate lookup table filename [string]    
;                default is aux/passbands/obsrest.lookup
;
; OPTIONAL OUTPUTS
;     lameff: effective wavelengths for the input passbands
;
; RETURN VALUE
;  array of best rest-frame passband names ('U','B','V','R', or 'I')
;


nolookup = keyword_set(nolookup)
if n_elements(lookupfile) eq 0 then $
  lookupfile = getenv('MLCS2K2_BASEDIR') + '/aux/passbands/obsrest.lookup'

; right now we just match the effective wavelength as best we can

n = n_elements(obspb)
out = obspb

lameff = dblarr(n)

z1 = 1.0d + z

if not nolookup then begin
    obstr = ''
    restr = ''
    readit,lookupfile,restr,obstr,zmin,zmax
endif

for i=0,n-1 do begin
    pb = getpb(obspb[i])
    
; saurabh's old convention for the effective wavelength: 
;    message, 'Using sj convention for effective wavelength', /info
;    efflam = synflux(pb.wavelength,pb.wavelength,pb)

; we're using rick's convention -- see pbefflam.pro
;    message, 'Using RK convention for effective wavelength', /info
    efflam = pbefflam(pb)

    reff = efflam/z1

    usedefault = 1
    if not nolookup then begin

        qq = where((obstr eq obspb[i]) and (zmin le z) and (zmax gt z),nqq)

        if nqq gt 1 then $
          message,'multiple matches in lookup table? ' + $
                  obspb[i] + ' ' + string(z)
    
        if nqq eq 1 then begin
            out[i] = restr[qq[0]]
            usedefault = 0
        endif
    endif

    if usedefault then begin

; Default v005 passband boundaries        
;         if      (reff ge 3200d and reff lt 3900d) then out[i] = 'U' $
;         else if (reff ge 3900d and reff lt 4850d) then out[i] = 'B' $
;         else if (reff ge 4850d and reff lt 5800d) then out[i] = 'V' $
;         else if (reff ge 5800d and reff lt 7300d) then out[i] = 'R' $
;         else if (reff ge 7300d and reff lt 9500d) then out[i] = 'I' $
;         else                                           out[i] = 'none'

; Rick Kessler's boundaries, which ensure unique obs -> rest mappings for SDSS
;    V/R boundary -> 5850  (was 5800)
;    R/I boundary -> 7050  (was 7300)
;        message, 'USING RK rest-frame conventions', /info
        if      (reff ge 3200d and reff lt 3900d) then out[i] = 'U' $
        else if (reff ge 3900d and reff lt 4850d) then out[i] = 'B' $
        else if (reff ge 4850d and reff lt 5850d) then out[i] = 'V' $
        else if (reff ge 5850d and reff lt 7050d) then out[i] = 'R' $
        else if (reff ge 7050d and reff lt 9500d) then out[i] = 'I' $
        else                                           out[i] = 'none'

    endif 

    lameff[i] = efflam

endfor

defaultskcgx, rfcolset, gxlaw, gxrv, samerest

if strlowcase(samerest) eq 'no' then begin
; if samerest is not allowed, we check here that the
; rest-frame passbands are unique (no two observer-frame
; passbands map to the same rest-frame passband)

    if (n_elements(uniq(out,sort(out))) lt n) then begin
        message, 'rest-frame passbands are NOT unique!: '+$
          strjoin(obspb,',')+' --> '+strjoin(out,',')
    endif

endif

return,out

end
