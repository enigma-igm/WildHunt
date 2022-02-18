"""

Main module for downloading and manipulating image data.

"""

import numpy as np

from astropy.table import Table

# from wildhunt import utils
import utils

# from wildhunt.surveys.panstarrs import Panstarrs

surveys = ['vhsdr6', 'vhsdr6', 'decals']
bands = ['J', 'K', 'z']
fovs = [30, 30, 30]

survey_dict = [{'survey': 'vhsdr6', 'bands': ['J', 'K'], 'fov':30},
               {'survey': 'decals', 'bands': ['r, z'], 'fov':30}
              ]


class ImagingSurvey(object):
    """
    Survey class for downloading imaging data
    """

    def __init__(self, bands, fov, name):
        """

        :param bands:
        :param fov:
        :param name:
        """


        ImagingSurvey.name = name
        ImagingSurvey.bands = bands
        ImagingSurvey.fov = fov

    def download_images(self, ra, dec, image_folder_path, n_jobs):
        """

        :param ra:
        :param dec:
        :param image_folder_path:
        :param n_jobs:
        :return:
        """

        raise NotImplementedError


class Panstarrs(ImagingSurvey):

    def __init__(self, bands, fov):
        """

        :param bands:
        :param fov:
        :param name:
        """

        super(Panstarrs, self).__init__(bands, fov, 'ps1')

    def download_images(self, ra, dec, image_folder_path, n_jobs):
        """

        :param ra:
        :param dec:
        :param image_folder_path:
        :param n_jobs:
        :return:
        """

        pass

    def get_ps1_filenames(self, ra, dec, bands='g'):
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

    def get_ps1_image_url(self, ra, dec, bands='g'):
        """

        :param ra:
        :param dec:
        :param bands:
        :return:
        """
        filenames = self.get_ps1_filenames(ra, dec, bands)

        if filenames is not None:

            url_list = []

            for filename in filenames:
                url_list.append('http://ps1images.stsci.edu{}'.format(filename))

            return url_list
        else:
            return None

    def get_ps1_image_cutout_url(self, ra, dec, fov, bands='g', verbosity=0):
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

        filenames = self.get_ps1_filenames(ra, dec, bands)

        if filenames is not None:

            url = ("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
                   "ra={ra}&dec={dec}&size={size}&format=fits").format(
                **locals())

            urlbase = url + "&red="
            url_list = []
            for filename in filenames:
                url_list.append(urlbase + filename)

            return url_list
        else:
            return None



def retrieve_survey(survey_name, bands, fov):

    survey = None

    if survey_name == 'ps1':

        survey = Panstarrs(bands, fov)


    if survey == None:
        print('ERROR')

    return survey


def get_images(ra, dec, image_folder_path, survey_dict, n_jobs=1, verbosity=0):
    """

    :param ra:
    :param dec:
    :param image_folder_path:
    :param survey_dict:
    :param n_jobs:
    :param verbosity:
    :return:
    """

    obj_names = utils.coord_to_name(ra, dec, epoch="J")

    for dict in survey_dict:

        survey = retrieve_survey(dict['survey'], dict['bands'], dict['fov'])

        survey.download_images(ra, dec, image_folder_path, n_jobs,
                               verbosity=verbosity)


