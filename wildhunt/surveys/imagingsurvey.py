#!/usr/bin/env python

class ImagingSurvey(object):
    """
    Survey class for downloading imaging data
    """

    def __init__(self, bands, fov, name, verbosity=1):
        """

        :param bands:
        :param fov:
        :param name:
        """

        self.name = name
        self.bands = bands
        self.fov = fov
        self.verbosity=verbosity

    def download_images(self, ra, dec, image_folder_path, n_jobs):
        """

        :param ra:
        :param dec:
        :param image_folder_path:
        :param n_jobs:
        :return:
        """

        raise NotImplementedError


    def download_image_from_url(self, url, image_folder_path):


        raise NotImplementedError