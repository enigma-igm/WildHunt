#!/usr/bin/env python
"""

Main module for downloading and manipulating image data.

"""



from wildhunt.surveys import panstarrs


def retrieve_survey(survey_name, bands, fov):

    survey = None

    if survey_name == 'ps1':

        survey = panstarrs.Panstarrs(bands, fov)


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


