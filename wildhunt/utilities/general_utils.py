import base64

import numpy as np
from astropy import stats
from astropy import units as u
from astropy.coordinates import Angle

# =========================================================================== #


def nan_mad_std(data, axis=None, func=None):
    """

    Wrapper for astropy.stats.mad_std which ignores nans, so as to
    prevent bugs when using sigma_clipped_stats with the axis keyword
    and stdfunc=astropy.stats.mad_std

    Args:
        data (array-like):
            Data array or object that can be converted to an array.
        axis (int, sequence of int, None, optional):
            Axis along which the robust standard deviations are
            computed.  The default (`None`) is to compute the robust
            standard deviation of the flattened array.

    Returns:
        float, `numpy.ndarray`: The robust standard deviation of the
        input data.  If ``axis`` is `None` then a scalar will be
        returned, otherwise a `~numpy.ndarray` will be returned.
    """
    return stats.mad_std(data, axis=axis, func=func, ignore_nan=True)


# =========================================================================== #


def coord_to_name(dra, ddec, epoch="J"):
    """Return an object name based on its Right Ascension and Declination.

    :param dra: Right Ascension of the target in decimal degrees
    :type dra: float or numpy.ndarray
    :param ddec: Declination of the target in decimal degrees
    :type ddec: float or numpy.ndarray
    :param epoch:  Epoch string (default: J), can also be substituted for
     survey abbreviation.
    :type: string
    :return:  Names based on the targets coordinates
        [epoch][RA in HMS][Dec in DMS]
    :rtype: string
    """

    coord_name_list = []

    if type(dra) is np.float64:
        hms_ra = Angle(dra, u.degree).to_string(u.hour, sep="", precision=2, pad=True)
        if ddec >= 0:
            sign = "+"
        else:
            sign = "-"
        deg_dec = Angle(abs(ddec), u.degree).to_string(
            u.deg, sep="", precision=2, pad=True
        )

        coord_name_list.append("{:}{:}{:}{:}".format(epoch, hms_ra, sign, deg_dec))
    else:
        for idx in range(len(dra)):
            hms_ra = Angle(dra[idx], u.degree).to_string(
                u.hour, sep="", precision=2, pad=True
            )
            if ddec[idx] >= 0:
                sign = "+"
            else:
                sign = "-"
            deg_dec = Angle(abs(ddec[idx]), u.degree).to_string(
                u.deg, sep="", precision=2, pad=True
            )

            coord_name_list.append("{:}{:}{:}{:}".format(epoch, hms_ra, sign, deg_dec))

    return coord_name_list


# =========================================================================== #


def degree_to_hms(dra, ddec):
    """Convert the Right Ascension and Declination from degrees to hours, min,
     sec.

    :param dra: Right Ascension of the target in decimal degrees
    :type dra: float or numpy.ndarray
    :param ddec: Declination of the target in decimal degrees
    :type ddec: float or numpy.ndarray
    :return:  Right Ascension and Declination on the targets coordinates
        (RA in HMS and Dec in DMS)
    :rtype: string
    """

    if type(dra) in [np.float64, float, int]:
        ra_hms = Angle(dra, u.degree).to_string(u.hour, sep=":", precision=2, pad=True)
        if ddec >= 0:
            sign = "+"
        else:
            sign = "-"
        dec_dms = sign + Angle(abs(ddec), u.degree).to_string(
            u.deg, sep=":", precision=2, pad=True
        )

    elif type(dra) is np.ndarray:
        ra_hms_list = []
        dec_dms_list = []

        for idx in range(len(dra)):
            ra_hms_idx = Angle(dra[idx], u.degree).to_string(
                u.hour, sep=":", precision=2, pad=True
            )

            ra_hms_list.append(ra_hms_idx)

            if ddec[idx] >= 0:
                sign = "+"
            else:
                sign = "-"
            dec_dms_idx = sign + Angle(abs(ddec[idx]), u.degree).to_string(
                u.deg, sep=":", precision=2, pad=True
            )
            dec_dms_list.append(dec_dms_idx)

        ra_hms = np.array(ra_hms_list)
        dec_dms = np.array(dec_dms_list)

    else:
        raise ValueError("dra must be a float or numpy.ndarray")

    return ra_hms, dec_dms


# =========================================================================== #


def convert_dmsdec2decdeg(dec_dms, delimiter=":"):
    """Convert a Declination coordinate string of the form DD:MM:SS.SS to
    decimal degrees.

    :param dec_dms: Declination coordinate string of the form DD:MM:SS.SS
    :type dec_dms: string
    :param delimiter: Delimiter between the degrees, minutes, and seconds.
    :type delimiter: string
    :return: Declination in decimal degrees
    :rtype: float
    """

    if delimiter is None or delimiter == "":
        dec_degrees = float(dec_dms[0:3])
        dec_minutes = float(dec_dms[3:5])
        dec_seconds = float(dec_dms[5:10])
    elif delimiter is not None and delimiter != "":
        dec_degrees = float(dec_dms.split(delimiter)[0])
        dec_minutes = float(dec_dms.split(delimiter)[1])
        dec_seconds = float(dec_dms.split(delimiter)[2])
    else:
        raise ValueError("Delimiter must be a string or None")

    if dec_dms[0] == "-":
        is_positive = False
    else:
        is_positive = True

    dec = abs(dec_degrees) + dec_minutes / 60.0 + dec_seconds / 3600.0

    if is_positive is False:
        dec = -dec

    return dec


# =========================================================================== #


def convert_hmsra2decdeg(ra_hms, delimiter=":"):
    """Convert a Right Ascension coordinate string of the form HH:MM:SS.SS to
    decimal degrees.

    :param ra_hms: Right Ascension coordinate string of the form HH:MM:SS.SS
    :type ra_hms: string
    :param delimiter: Delimiter between the hours, minutes, and seconds.
    :type delimiter: string
    :return: Right Ascension in decimal degrees
    :rtype: float
    """

    if delimiter is None or delimiter == "":
        ra_hours = float(ra_hms[0:2])
        ra_minutes = float(ra_hms[2:4])
        ra_seconds = float(ra_hms[4:10])
    elif delimiter is not None and delimiter != "":
        ra_hours = float(ra_hms[0:2])
        ra_minutes = float(ra_hms[3:5])
        ra_seconds = float(ra_hms[6:12])
    else:
        raise ValueError("Delimiter must be a string or None")

    return (ra_hours + ra_minutes / 60.0 + ra_seconds / 3600.0) * 15.0


# =========================================================================== #


def sizeof_fmt(num, suffix="B"):
    """Get human-readable file size.

    Adopted from https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size

    :param num: Files size in bytes.
    :type num: int
    :param suffix: Suffix of the files sizes (default: B)
    :type suffix: str
    :return: File size in human-readable format.
    :rtype: str
    """
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


# =========================================================================== #


def b64e(s):
    """Encode a string to Base64 format.

    This function takes a string, encodes it into bytes, and then
    encodes those bytes into a Base64 string.

    :param s: The string to encode in Base64.
    :type s: str
    :return: The Base64 encoded string.
    :rtype: str
    """
    return base64.b64encode(s.encode()).decode()


# =========================================================================== #


def b64d(s):
    """Decode a Base64 encoded string.

    This function takes a Base64 encoded string, decodes it into bytes,
    and then converts those bytes back into a regular string.

    :param s: The Base64 encoded string to decode.
    :type s: str
    :return: The decoded string.
    :rtype: str
    """
    return base64.b64decode(s).decode()


# =========================================================================== #
