

from ..imaging import Survey


class panstarrs(Survey):

    pass


def get_ps1_filenames(ra, dec, bands='g'):
    """

    :param ra:
    :param dec:
    :param bands:
    :return:
    """
    url_base = 'http://ps1images.stsci.edu/cgi-bin/ps1filenames.py?'
    ps1_url = url_base + 'ra={}&dec={}&filters={}'.format(ra, dec, bands)

    table = Table.read(ps1_url, format='ascii')

    # Sort filters from red to blue
    flist = ["yzirg".find(x) for x in table['filter']]
    table = table[np.argsort(flist)]

    filenames = table['filename']

    if len(filenames) > 0:
        return filenames
    else:
        print("No PS1 image is available for this position.")
        return None



def get_ps1_image_url(ra, dec, bands='g'):
    """

    :param ra:
    :param dec:
    :param bands:
    :return:
    """
    filenames = get_ps1_filenames(ra, dec, bands)

    if filenames is not None:

        url_list = []

        for filename in filenames:
            url_list.append('http://ps1images.stsci.edu{}'.format(filename))

        return url_list
    else:
        return None


def get_ps1_image_cutout_url(ra, dec, fov, bands='g', verbosity=0):
    """

    :param ra:
    :param dec:
    :param fov:
    :param bands:
    :param verbosity:
    :return:
    """


    # Convert field of view in arcsecond to pixel size (1 pixel = 0.25 arcseconds)
    size = fov * 4

    filenames = get_ps1_filenames(ra, dec, bands)

    if filenames is not None:

        url = ("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
               "ra={ra}&dec={dec}&size={size}&format=fits").format(**locals())

        urlbase = url + "&red="
        url_list = []
        for filename in filenames:
            url_list.append(urlbase + filename)

        return url_list
    else:
        return None