#!/usr/bin/env python
"""

Main module for downloading and manipulating image data.

"""

import glob
from astropy.io import fits


from wildhunt.surveys import panstarrs, vsa_wsa, legacysurvey


def retrieve_survey(survey_name, bands, fov):

    survey = None

    if survey_name == 'PS1':

        survey = panstarrs.Panstarrs(bands, fov)

    if survey_name[:3] in ['VHS', 'VVV', 'VMC', 'VIK', 'VID', 'UKI', 'UHS']:

        survey = vsa_wsa.VsaWsa(bands, fov, survey_name)

    if survey_name[:4] == 'DELS':

        survey = legacysurvey.LegacySurvey(bands, fov, survey_name)


    if survey == None:
        print('ERROR')

    return survey


def get_images(ra, dec, image_folder_path, survey_dict, n_jobs=1, verbosity=1):
    """

    :param ra:
    :param dec:
    :param image_folder_path:
    :param survey_dict:
    :param n_jobs:
    :param verbosity:
    :return:
    """


    for dict in survey_dict:

        survey = retrieve_survey(dict['survey'], dict['bands'], dict['fov'])

        survey.download_images(ra, dec, image_folder_path, n_jobs)



def open_image(filename, ra, dec, fov, image_folder_path, verbosity=0):

    """Opens an image defined by the filename with a fov of at least the
    specified size (in arcseonds).

    :param filename:
    :param ra:
    :param dec:
    :param fov:
    :param image_folder_path:
    :param verbosity:
    :return:
    """

    filenames_available = glob.glob(filename)
    file_found = False
    open_file_fov = None
    file_path = None
    if len(filenames_available) > 0:
        for filename in filenames_available:

            try:
                file_fov = int(filename.split("_")[3].split(".")[0][3:])
            except:
                file_fov = 9999999

            if fov <= file_fov:
                data, hdr = fits.getdata(filename, header=True)
                file_found = True
                file_path =filename
                open_file_fov = file_fov

    if file_found:
        if verbosity > 0:
            print("Opened {} with a fov of {} "
                  "arcseconds".format(file_path, open_file_fov))

        return data, hdr, file_path

    else:
        if verbosity > 0:
            print("File {} in folder {} not found. Target with RA {}"
                  " and Decl {}".format(filename, image_folder_path,
                                        ra, dec))
        return None, None, None
