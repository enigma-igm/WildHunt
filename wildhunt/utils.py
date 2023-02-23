
from astropy import units as u
from astropy.coordinates import Angle

import numpy as np


def coord_to_name(dra, ddec, epoch='J'):
    """Return an object name based on its Right Ascension and Declination.

    :param dra: Right Ascension of the target in decimal degrees
    :type dra: float or numpy.ndarray
    :param ddec: Declination of the target in decimal degrees
    :type ddec: float or numpy.ndarray
    :param epoch:  Epoch string (default: J), can also be substituted for survey
        abbreviation.
    :type: string
    :return:  Names based on the targets coordinates
        [epoch][RA in HMS][Dec in DMS]
    :rtype: string
    """

    coord_name_list = []

    if type(dra) is np.float64:
        hms_ra = Angle(dra, u.degree).to_string(u.hour, sep='', precision=2, pad=True)
        if ddec >= 0:
            sign = '+'
        else:
            sign = '-'
        deg_dec = Angle(abs(ddec), u.degree).to_string(u.deg, sep='', precision=2, pad=True)

        coord_name_list.append('{:}{:}{:}{:}'.format(epoch,
                                                     hms_ra,
                                                     sign,
                                                     deg_dec))
    else:
        for idx in range(len(dra)):

            hms_ra = Angle(dra[idx], u.degree).to_string(u.hour,sep='', precision=2,pad=True)
            if ddec[idx]>=0:
                sign='+'
            else:
                sign='-'
            deg_dec = Angle(abs(ddec[idx]), u.degree).to_string(u.deg, sep='', precision=2, pad=True)

            coord_name_list.append('{:}{:}{:}{:}'.format(epoch,
                                                 hms_ra,
                                                 sign,
                                                 deg_dec))

    return coord_name_list

def degree_to_hms(dra, ddec, epoch='J'):
    """Convert the Right Ascension and Declination from degrees to hours, min, sec.

    :param dra: Right Ascension of the target in decimal degrees
    :type dra: float
    :param ddec: Declination of the target in decimal degrees
    :type ddec: float
    :param epoch:  Epoch string (default: J), can also be substituted for survey
        abbreviation.
    :type: string
    :return:  Right Ascension and Declination on the targets coordinates
        (RA in HMS and Dec in DMS)
    :rtype: string
    """

    if type(dra) is np.float64:
        hms_ra = Angle(dra, u.degree).to_string(u.hour, sep=':', precision=2, pad=True)
        if ddec >= 0:
            sign = '+'
        else:
            sign = '-'
        deg_dec = sign + Angle(abs(ddec), u.degree).to_string(u.deg, sep=':', precision=2, pad=True)

    else:
        hms_ra = []
        deg_dec = []

        for idx in range(len(dra)):

            hms_ra.append(Angle(dra[idx], u.degree).to_string(u.hour,sep=':', precision=2,pad=True))
            if ddec[idx]>=0:
                sign='+'
            else:
                sign='-'
            deg_dec.append(sign + Angle(abs(ddec[idx]), u.degree).to_string(u.deg, sep=':', precision=2, pad=True))

    return hms_ra, deg_dec


def convert_dmsdec2decdeg(dec_dms,delimiter=':'):

    if delimiter is None:
        dec_degrees = float(dec_dms[0:3])
        dec_minutes = float(dec_dms[3:5])
        dec_seconds = float(dec_dms[5:10])
    if delimiter is not None:
        dec_degrees = float(dec_dms.split(delimiter)[0])
        dec_minutes = float(dec_dms.split(delimiter)[1])
        dec_seconds = float(dec_dms.split(delimiter)[2])

    # print(dec_dms[0])

    if dec_dms[0] == '-':
        is_positive = False
    else:
        is_positive = True

    dec = abs(dec_degrees) + dec_minutes/60. + dec_seconds/3600.

    if is_positive is False:
        dec = -dec

    return dec


def convert_hmsra2decdeg(ra_hms, delimiter=':'):

    if delimiter is None:
        ra_hours = float(ra_hms[0:2])
        ra_minutes = float(ra_hms[2:4])
        ra_seconds = float(ra_hms[4:10])
    if delimiter ==':':
        ra_hours = float(ra_hms[0:2])
        ra_minutes = float(ra_hms[3:5])
        ra_seconds = float(ra_hms[6:12])

    return (ra_hours + ra_minutes/60. + ra_seconds/3600.) * 15.




# FROM https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"
