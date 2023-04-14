#!/usr/bin/env python

import os
import requests
import pandas as pd
import multiprocessing as mp
from urllib.error import HTTPError
from http.client import IncompleteRead

from wildhunt import utils
from wildhunt import pypmsgs

msgs = pypmsgs.Messages()


class ImagingSurvey(object):
    """
    Survey class for handling imaging data from online survey services.

    This is a general implementation of the survey class introducing
    functions to download images from online archives and provide parameters
    for aperture photometry on the image data.

    """

    def __init__(self, bands, fov, name, verbosity=1):
        """ Initialize the survey class.

        :param bands: List of survey filter bands.
        :type bands: list
        :param fov: Field of view of requested imaging in arcseconds.
        :type fov: int
        :param name: Name of the survey.
        :type name: str
        """

        self.name = name
        self.bands = bands
        self.fov = fov
        self.verbosity = verbosity

        # Initialize the survey specific variables
        # These will be set in the survey specific setup
        self.image_folder_path = None
        self.n_jobs = None

        self.source_table = None
        self.download_table = pd.DataFrame(columns=['image_name', 'url'])

        # Initialize the aperture photometry parameters with None
        self.ab_corr = None  # AB correction
        self.exp = None  # Exposure time
        self.back = None  # Background flag (True/False)
        self.zpt = None  # Zero point
        self.nanomag_corr = None  # Nanomaggies correction

    def download_images(self, ra, dec, image_folder_path, n_jobs):
        """ Download images from the online image server.

        This function is overriden in the survey specific class.

        :param ra: Right ascension of the sources in decimal degrees.
        :type ra: numpy.ndarray
        :param dec: Declination of the sources in decimal degrees.
        :type dec: numpy.ndarray
        :param image_folder_path: Path to the folder where the images are
         stored.
        :type image_folder_path: str
        :param n_jobs: Number of parallel jobs to use for downloading.
        :type n_jobs: int
        :return: None
        """

        raise NotImplementedError

    def survey_setup(self, ra, dec, image_folder_path, epoch='J', n_jobs=1):
        """ Set up the survey for downloading images.

        :param ra: List of right ascensions in decimal degrees.
        :type ra: numpy.ndarray
        :param dec: List of declinations in decimal degrees.
        :type dec: numpy.ndarray
        :param image_folder_path: Path to the folder where the images are
         stored.
        :type image_folder_path: str
        :param epoch: Epoch string for the coordinates.
        :type epoch: str
        :param n_jobs: Number of parallel jobs to use for downloading.
        :type n_jobs: int
        :return:
        """

        # Check if image_folder_exists, if not create
        if not os.path.exists(image_folder_path):
            os.makedirs(image_folder_path)

        self.image_folder_path = image_folder_path

        # Create the download table with the obj_name
        obj_names = utils.coord_to_name(ra, dec, epoch=epoch)
        self.source_table = pd.DataFrame({'obj_name': obj_names,
                                          'ra': ra,
                                          'dec': dec})

        # Check for pre-exisiting images and remove sources if all
        # images in the specified bands already exist
        self.pre_check_for_existing_images()

        # # Check n_jobs value
        cpu_cores = mp.cpu_count()
        if n_jobs > cpu_cores-1:
            self.n_jobs = cpu_cores-1
            msgs.info('Number of jobs too large for CPU')
            msgs.info('Setting number of jobs to {}'.format(self.n_jobs))
        elif n_jobs >= 1:
            self.n_jobs = n_jobs
        else:
            self.n_jobs = 1
            msgs.info('Number of jobs not understood')
            msgs.info('Setting number of jobs to {}'.format(self.n_jobs))

    def pre_check_for_existing_images(self):
        """ Check whether pre-existing images are complete for sources in
        the main table.

        :return: None
        """

        rows_to_drop = []

        for idx in self.source_table.index:
            band_exists = True
            obj_name = self.source_table.loc[idx, 'obj_name']

            for band in self.bands:

                image_name = obj_name + "_" + self.name + "_" + \
                             band + "_fov" + '{:d}'.format(self.fov)

                file_path = self.image_folder_path + '/' + image_name + '.fits'

                file_exists = os.path.isfile(file_path)

                # Checks whether one band does not exist
                if band_exists:
                    band_exists = file_exists

            if band_exists:
                msgs.info('All images for source {} already exist. Source '
                          'is removed from source table.'.format(obj_name))
                rows_to_drop.append(idx)

        # Drop rows of existing sources from source table
        self.source_table.drop(rows_to_drop, inplace=True)

    def check_for_existing_images_before_download(self):
        """ Check for pre-existing whether images requested in the download
        table already exist in the image folder.

        :return:
        """

        rows_to_drop = []

        for idx in self.download_table.index:

            image_name = self.download_table.loc[idx, 'image_name']

            file_path = self.image_folder_path + '/' + image_name + '.fits'

            file_exists = os.path.isfile(file_path)

            if file_exists:
                msgs.info('Image {} already exists and is removed from '
                          'download.'.format(image_name))

                rows_to_drop.append(idx)

        # Drop rows of existing sources from source table
        self.download_table.drop(rows_to_drop, inplace=True)

    def download_image_from_url(self, url, image_name):
        """ Download the image with name image_name from the given url.

        :param url: URL of the image to download.
        :type url: str
        :param image_name: Name of the image to download.
        :type image_name: str
        :return:
        """

        # Try except clause for downloading the image
        try:

            r = requests.get(url)
            if 'unwise.me' in url:
                open(self.image_folder_path + '/' + image_name + '.tar.gz',
                     "wb").write(r.content)
            else:
                open(self.image_folder_path + '/' + image_name + '.fits',
                     "wb").write(r.content)

            if self.verbosity > 0:
                msgs.info("Download of {} to {} completed".format(
                    image_name, self.image_folder_path))

        except (IncompleteRead, HTTPError, AttributeError, ValueError) as err:
            msgs.warn("Download error encountered: {}".format(err))
            if self.verbosity > 0:
                msgs.warn("Download of {} unsuccessful".format(image_name))

    def mp_download_image_from_url(self):
        """ Execute image download in parallel.

        :return: None
        """

        mp_args = list(zip(self.download_table.loc[:, 'url'].values,
                           self.download_table.loc[:, 'image_name'].values))

        with mp.Pool(processes=self.n_jobs) as pool:
            pool.starmap(self.download_image_from_url, mp_args)

    def force_photometry_params(self, header, band, filepath=None):
        """ Abstract function to set the force photometry parameters."""

        raise NotImplementedError




