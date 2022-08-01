#!/usr/bin/env python

import os

import itertools

import requests

import multiprocessing as mp

from urllib.request import urlopen  # python3
from urllib.error import HTTPError
from http.client import IncompleteRead

import numpy as np
import pandas as pd
from IPython import embed

from wildhunt import utils

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
        self.verbosity = verbosity

        # Set to none and overwritten by survey setup
        self.image_folder_path = None
        self.n_jobs = None

        self.source_table = None
        self.download_table = pd.DataFrame(columns=['image_name', 'url'])

    def download_images(self, ra, dec, image_folder_path, n_jobs):
        """

        :param ra:
        :param dec:
        :param image_folder_path:
        :param n_jobs:
        :return:
        """

        raise NotImplementedError

    def survey_setup(self, ra, dec, image_folder_path, epoch='J', n_jobs=1):

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
            print('[INFO] Number of jobs too large for CPU')
            print('[INFO] Setting number of jobs to {}'.format(self.n_jobs))
        elif n_jobs >= 1:
            self.n_jobs = n_jobs
        else:
            self.n_jobs = 1
            print('[INFO] Number of jobs not understood')
            print('[INFO] Setting number of jobs to {}'.format(self.n_jobs))


    def pre_check_for_existing_images(self):

        rows_to_drop = []

        for idx in self.source_table.index:
            band_exists = True

            for band in self.bands:
                obj_name = self.source_table.loc[idx, 'obj_name']

                image_name = obj_name + "_" + self.name + "_" + \
                         band + "_fov" + '{:d}'.format(self.fov)

                file_path = self.image_folder_path + '/' + image_name + '.fits'

                file_exists = os.path.isfile(file_path)

                # Checks whether one band does not exist
                if band_exists:
                    band_exists = file_exists

            if band_exists:
                print('[INFO] All images for source {} already exist. Source '
                      'is removed from source table.'.format(obj_name))
                rows_to_drop.append(idx)

        # Drop rows of existing sources from source table
        self.source_table.drop(rows_to_drop, inplace=True)


    def check_for_existing_images_before_download(self):

        rows_to_drop = []

        for idx in self.download_table.index:

            image_name = self.download_table.loc[idx, 'image_name']

            file_path = self.image_folder_path + '/' + image_name + '.fits'

            file_exists = os.path.isfile(file_path)

            if file_exists:
                print('[INFO] Image {} already exists and is removed from '
                      'download.'.format(image_name))

                rows_to_drop.append(idx)

        # Drop rows of existing sources from source table
        self.download_table.drop(rows_to_drop, inplace=True)

    def download_image_from_url(self, url, image_name):

        # Try except clause for downloading the image
        try:

            r = requests.get(url)
            open(self.image_folder_path + '/' + image_name + '.fits', "wb").write(r.content)

            if self.verbosity > 0:
                print("[INFO] Download of {} to {} completed".format(image_name, self.image_folder_path))

            #datafile = urlopen(url)

            #check_ok = datafile.msg == 'OK'

            #if check_ok:

                #file = datafile.read()

                #output = open(self.image_folder_path + '/' + image_name +
                #              '.fits', 'wb')
                #output.write(file)
                #output.close()
                #if self.verbosity > 0:
                #    print("[INFO] Download of {} to {} completed".format(
                #        image_name, self.image_folder_path))

        except (IncompleteRead, HTTPError, AttributeError, ValueError) as err:
            print(err)
            if self.verbosity > 0:
                print("[ERROR] Download of {} unsuccessful".format(image_name))

    def mp_download_image_from_url(self):
        """

        :param image_folder_path:
        :param n_jobs:
        :return:
        """

        mp_args = list(zip(self.download_table.loc[:, 'url'].values,
                           self.download_table.loc[:, 'image_name'].values))

        with mp.Pool(processes=self.n_jobs) as pool:
            pool.starmap(self.download_image_from_url, mp_args)
