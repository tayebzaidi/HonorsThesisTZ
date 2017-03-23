import logging
import json
import sys
import os

def remapBands(lightcurve, z=0):
    """
    Remaps the bands of a given lightcurve file to the "grizy" of the DES survey
    Stages:
        Assess which band the given band most clearly falls under
        E.g. map g_Catalina_Schmidt --> g

        Given a redshift z, adjust the band to be in the restframe
        If no redshift is given, skip this stage (or in this case, divide by 1)

    This code is based off of Gautham Narayan's "bestrest.pro" function

    Passband choice order (descending):
        CfA
        CSP
        Kait3
        Swift
        Vega
        Pairitel
        SDSS
        Landolt
        Standard

    Passbands to ignore
        Catalina Schmidt
        (No survey listed)

    Inputs:
        lightcurve: A lightcurve dictionary that uses the standard setup, defined elsewhere
        z:          A redshift, optional input (default zero)

    Output:
        lightcurve_mapped: Lightcurve dictionary that has been remapped
    """
    #Initialize the lightcurve object to be returned
    lightcurve_mapped = {}

    repeat_passbands = {'u':0, 'g':0, 'r':0, 'i':0, 'z':0, 'Y':0}
    passband_options = {'u':[], 'g':[], 'r':[], 'i':[], 'z':[], 'Y':[]}
    surveys_ranked = ['cfa', 'csp', 'kait', 'swift', 'vega', 'pairitel',
                      'sdss', 'landolt', 'standard']

    redshift = 1.0 + z

    for passband in lightcurve.keys():

        passband_lambda = get_passband_lambda(passband)

        if passband_lambda is None:
            continue

        eff_lambda = passband_lambda/redshift
        passband_mapped = get_passband(eff_lambda)

        if passband_mapped is None:
            continue

        #if passband_mapped != passband:
        #    print("Passband {} changed to {}".format(passband, passband_mapped))

        #Add to passband to see if there is repetition later
        repeat_passbands[passband_mapped] += 1
        #Store the options for the future passband
        passband_options[passband_mapped] += [passband]
    #print(passband_options)


    for passband_mapped in passband_options:
        if repeat_passbands[passband_mapped] == 0:
            continue

        found_mapping = False
        for survey in surveys_ranked:
            for original_passband in passband_options[passband_mapped]:

                if survey in original_passband.lower():
                    found_mapping = True
                    lightcurve_mapped[passband_mapped] = lightcurve[original_passband]
                    lightcurve_mapped[passband_mapped]['mapping'] = '{} to {}'.format(original_passband, passband_mapped)
                    #print("Survey {} chose, Passband {} changed to {}".format(survey, original_passband, passband_mapped))
                    if found_mapping:
                        break
            if found_mapping:
                break


    return lightcurve_mapped


def get_passband_lambda(passband):
    """
    Returns the wavelength of a given filter

    Filter table (values in nanometers):

    Dark Energy Survey filter wavelengths:
        #u --'lb': 350  --   ub: 100
        g -- 'lb': 472 --    ub: 152
        #VR -'lb': 630 --    ub: 260
        r -- 'lb': 641.5 --  ub: 148
        i -- 'lb': 783.5 --  ub: 147
        z -- 'lb': 926 --    ub: 152
        Y -- 'lb': 1009.5 -- ub: 113

    Standard filter wavelengths:
        U -- 'lb': 365 --    ub: 66
        B -- 'lb': 445 --    ub: 94
        V -- 'lb': 551 --    ub: 88
        R -- 'lb': 658 --    ub: 138
        I -- 'lb': 806 --    ub: 149

    Inputs:
        passband: A string containing the passband using the "*_ + survey" format

    Outputs:
        lambda:'lb' wavelength of a given passband
    """

    des_passbands = {
        'u': {'lb': 350, 'ub': 100},
        'g': {'lb': 472, 'ub': 152},
        'VR': {'lb': 630, 'ub': 260},
        'r': {'lb': 641.5, 'ub': 148},
        'i': {'lb': 783.5, 'ub': 147},
        'z': {'lb': 926, 'ub': 152},
        'Y': {'lb': 1009.5, 'ub': 113}
    }

    standard_passbands = {
        'U': {'lb': 365, 'ub': 66},
        'B': {'lb': 445, 'ub': 94},
        'V': {'lb': 551, 'ub': 88},
        'R': {'lb': 658, 'ub': 138},
        'I': {'lb': 806, 'ub': 149}
    }
    #I can add other dicionary lookups as necessary

    #Take the first letter as the passband
    pb = passband[0]

    #Check to see if the passband is in the either of the lookup dictionaries
    if pb in des_passbands:
        #For now, only return the'lb' and ignore ub
        return des_passbands[pb]['lb']
    elif pb in standard_passbands:
        return standard_passbands[pb]['lb']
    else:
        #If the passband isn't in the lookup dictionaries, return error message
        logging.info("Could not find the passband")
        return None

def get_passband(pb_lambda):
    """
    Returns the filter corresponding to a given wavelength
    Only using filters from the Dark Energy survey "ugrizY"

    Filter table (values in nanometers):

    Dark Energy Survey filter wavelengths:
        #u --'lb': 350  --   ub: 100
        g -- 'lb': 472 --    ub: 152
        r -- 'lb': 641.5 --  ub: 148
        i -- 'lb': 783.5 --  ub: 147
        z -- 'lb': 926 --    ub: 152
        Y -- 'lb': 1009.5 -- ub: 113

    Clearly demarcate where the boundaries lie between filters
    Boundaries (values in nanometers):
        u --> 250    -- 385
        g --> 385    -- 558.75
        r --> 558.75 -- 713
        i --> 713    -- 852.25
        z --> 852.25 -- 987.25
        Y --> 987.25 -- 1122.5

    Inputs:
        lambda:'lb' wavelength of a given passband

    Outputs:
        passband: A string containing the passband using the  format
    """

    #Demarcate the upper and lower bounds of the filter cutoffs
    des_passbands = {
        'u': {'lb': 250, 'ub': 385},
        'g': {'lb': 385, 'ub': 558.75},
        'r': {'lb': 558.75, 'ub': 713},
        'i': {'lb': 713, 'ub': 852.25},
        'z': {'lb': 852.25, 'ub': 987.25},
        'Y': {'lb': 987.25, 'ub': 1122.5}
    }

    pb_lambda = float(pb_lambda)

    #First check if the wavelength is within the u-Y range
    lower_bound = des_passbands['u']['lb'] - des_passbands['u']['lb']
    upper_bound = des_passbands['Y']['lb'] + des_passbands['Y']['lb']

    if pb_lambda > lower_bound and pb_lambda <= upper_bound:
        #Find the closest filter to the pb_lambda
        #Set min_diff to an arbitrarily large value
        for filt in des_passbands:
            #Compare to lower_bound and upper_bound
            if pb_lambda > des_passbands[filt]['lb'] and pb_lambda <= des_passbands[filt]['ub']:
                min_passband = filt
        #Return the passband with the minimum total distance
        return min_passband
    else:
        logging.info("Effective lambda is out of range of current filters")
        return None


def main():
    print("No programmed output from main()")
    pass

if __name__ == "__main__":
    sys.exit(main())