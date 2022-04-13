
from astropy import units as u
from astropy.coordinates import Angle

import numpy as np

'''
def decra_to_hms(dra):
    """Convert Right Ascension in decimal degrees to hours, minutes, seconds.

    :param dra: float
        Right Ascension in decimal degrees
    :return: integer, integer, float
        Right Ascension in hours, minutes, seconds
    """

    if isinstance(dra, float) or isinstance(dra, int):

        ra_minutes, ra_seconds = divmod(dra / 15 * 3600, 60)
        ra_hours, ra_minutes = divmod(ra_minutes, 60)

    elif isinstance(dra, np.ndarray) or isinstance(dra, list):

        dra = np.array(dra / 15 * 3600)
        ra_minutes, ra_seconds = np.divmod(dra, 60)
        ra_hours, ra_minutes = np.divmod(ra_minutes, 60)

    else:
        raise TypeError("Input type {} not understood (float, int, "
                        "np.ndarray)".format(type(dra)))

    return ra_hours, ra_minutes, ra_seconds


def decdecl_to_dms(ddecl):
    """Convert Declination in decimal degrees to degrees, minutes, seconds

    :param ddecl: float
        Declination in decimal degrees
    :return: integer, integer, float
        Declination in degrees, minutes, seconds
    """

    if isinstance(ddecl, float) or isinstance(ddecl, int):
        is_negative = ddecl < 0
        ddecl = abs(ddecl)
        decl_minutes, decl_seconds = divmod(ddecl * 3600, 60)
        decl_degrees, decl_minutes = divmod(decl_minutes, 60)
        decl_degrees[is_negative] = - decl_degrees[is_negative]

    elif isinstance(ddecl, np.ndarray) or isinstance(ddecl, list):
        ddecl = np.array(ddecl)
        is_negative = ddecl < 0
        ddecl = np.abs(ddecl)
        decl_minutes, decl_seconds = np.divmod(ddecl * 3600, 60)
        decl_degrees, decl_minutes = np.divmod(decl_minutes, 60)
        decl_degrees[is_negative] = - decl_degrees[is_negative]

    else:
        raise TypeError("Input type {} not understood (float, int, "
                        "np.ndarray)".format(type(ddecl)))

    return decl_degrees, decl_minutes, decl_seconds
'''

def coord_to_name(dra, dd, epoch='J'):
    """Return an object name based on its Right Ascension and Declination.

    :param dra: float
        Right Ascension of the target in decimal degrees
    :param dd: float
        Declination of the target in decimal degrees
    :param epoch: string
        Epoch string (default: J), can also be substituted for survey
        abbreviation.
    :return: string
        String based on the targets coordings [epoch][RA in HMS][Dec in DMS]
    """

    #ra_hours, ra_minutes, ra_seconds = decra_to_hms(dra)
    #decl_degrees, decl_minutes, decl_seconds = decdecl_to_dms(dd)

    coord_name_list = []

    if type(dra) is np.float64:
        hms_ra = Angle(dra, u.degree).to_string(u.hour, sep='', precision=2, pad=True)
        if dd >= 0:
            sing = '+'
        else:
            sing = '-'
        deg_dec = Angle(abs(dd), u.degree).to_string(u.deg, sep='', precision=2, pad=True)

        coord_name_list.append('{:}{:}{:}{:}'.format(epoch,
                                                     hms_ra,
                                                     sing,
                                                     deg_dec))
    else:
        for idx in range(len(dra)):

            hms_ra = Angle(dra, u.degree).to_string(u.hour,sep='', precision=2,pad=True)
            if dd[idx]>=0:
                sing='+'
            else:
                sing='-'
            deg_dec = Angle(abs(dd), u.degree).to_string(u.deg,sep='', precision=2,pad=True)

            coord_name_list.append('{:}{:}{:}{:}'.format(epoch,
                                                 hms_ra[idx],
                                                 sing,
                                                 deg_dec[idx]))

    return coord_name_list