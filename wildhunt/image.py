#!/usr/bin/env python
"""

Main module for downloading and manipulating image data.

"""

import glob
from astropy.io import fits



def mp_get_forced_photometry(ra, dec, survey_dict):
    # Get aperture photometry for one source but all bands/surveys


    # return photometry for each source but all filters/surveys (a row in a
    # ra/dec table
    pass



class image(object):

    def __init__(self):
        self.data = None  # image data
        self.header = None
        pass

    def open(self, filename):
        hdul = fits.open(filename)
        self.header = hdul[0].header
        self.data = hdul[1].data

        pass


    def get_aperture_photometry(self, ra, dec, survey, band):
        # This function calculates aperture photometry on the image

        # Possible return a catalog entry
        pass


        result_dict = {'ap_flux_2': None}

        return result_dict




# def get_images(ra, dec, image_folder_path, survey_dict, n_jobs=1, verbosity=1):
#     """
#
#     :param ra:
#     :param dec:
#     :param image_folder_path:
#     :param survey_dict:
#     :param n_jobs:
#     :param verbosity:
#     :return:
#     """
#
#
#     for dict in survey_dict:
#
#         survey = retrieve_survey(dict['survey'], dict['bands'], dict['fov'])
#
#         survey.download_images(ra, dec, image_folder_path, n_jobs)



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
