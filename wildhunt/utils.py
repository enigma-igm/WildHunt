
from astropy import units as u
from astropy.coordinates import Angle

import numpy as np

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
        Names based on the targets coordinates [epoch][RA in HMS][Dec in DMS]
    """

    coord_name_list = []

    if type(dra) is np.float64:
        hms_ra = Angle(dra, u.degree).to_string(u.hour, sep='', precision=2, pad=True)
        if dd >= 0:
            sign = '+'
        else:
            sign = '-'
        deg_dec = Angle(abs(dd), u.degree).to_string(u.deg, sep='', precision=2, pad=True)

        coord_name_list.append('{:}{:}{:}{:}'.format(epoch,
                                                     hms_ra,
                                                     sign,
                                                     deg_dec))
    else:
        for idx in range(len(dra)):

            hms_ra = Angle(dra[idx], u.degree).to_string(u.hour,sep='', precision=2,pad=True)
            if dd[idx]>=0:
                sign='+'
            else:
                sign='-'
            deg_dec = Angle(abs(dd[idx]), u.degree).to_string(u.deg,sep='', precision=2,pad=True)

            coord_name_list.append('{:}{:}{:}{:}'.format(epoch,
                                                 hms_ra,
                                                 sign,
                                                 deg_dec))

    return coord_name_list


# FROM https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"
