#!/usr/bin/env python

import os
import requests
import pandas as pd
import numpy as np
from io import StringIO
from astropy.table import Table

from wildhunt import utils, pypmsgs
from wildhunt.surveys import imagingsurvey

from IPython import embed

msgs = pypmsgs.Messages()


class Panstarrs(imagingsurvey.ImagingSurvey):
    """ Panstarrs class deriving from the ImagingSurvey class to handle
    image downloads and aperture photometry for the Pan-STARRS1 survey.

    """

    def __init__(self, bands, fov, name='PS1', verbosity=1):
        """ Initialize the LegacySurvey class.

        :param bands: List of survey filter bands.
        :type bands: list
        :param fov: Field of view of requested imaging in arcseconds.
        :type fov: int
        :param name: Name of the survey.
        :type name: str
        :return: None
        """

        self.batch_size = 10000

        self.ra = None
        self.dec = None
        self.nbatch = 1

        super(Panstarrs, self).__init__(bands, fov, name, verbosity)

    def download_images(self, ra, dec, image_folder_path, n_jobs=1):
        """ Download images from the online image server for Pan-STARRS1.

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

        self.survey_setup(ra, dec, image_folder_path, epoch='J', n_jobs=n_jobs)

        if self.source_table.shape[0] > 0:

            self.batch_setup()

            for i in range(self.nbatch):
                self.retrieve_image_url_list(imagetypes="stack",
                                             batch_number=i)

                self.check_for_existing_images_before_download()

                if self.n_jobs > 1:
                    self.mp_download_image_from_url()

                else:
                    for idx in self.download_table.index:
                        image_name = self.download_table.loc[idx, 'image_name']
                        url = self.download_table.loc[idx, 'url']
                        self.download_image_from_url(url, image_name)

            for i in range(self.nbatch):
                os.remove(str(i) + '_PS1_download_urls.csv')

        else:
            msgs.info('All images already exist.')
            msgs.info('Download canceled.')


    def batch_setup(self,):

        self.ra = self.source_table.loc[:, 'ra'].values
        self.dec = self.source_table.loc[:, 'dec'].values

        # Compute the number of batches to retrieve the urls
        if np.size(self.ra) > self.batch_size:
            self.nbatch = int(np.ceil(np.size(self.ra) / self.batch_size))

    def retrieve_image_url_list(self, batch_number=0, imagetypes="stack"):
        """ Retrieve the list of image URLs from the online image server.

        :param batch_number: Number of the batch to retrieve the urls for.
        :type batch_number: int
        :param imagetypes: Type of the image data to retrieve. Defaults to
         stacked images ("stack").
        :return: None
        """

        # Convert field of view in arcsecond to pixel size
        # (1 pixel = 0.25 arcseconds)
        img_size = self.fov * 4

        bands = ''.join(self.bands)

        # Retrieve bulk file table
        url_ps1filename = 'http://ps1images.stsci.edu/cgi-bin/ps1filenames.py'

        ra_batch = self.ra[batch_number * self.batch_size:
                           batch_number * self.batch_size + self.batch_size]
        dec_batch = self.dec[batch_number * self.batch_size:
                             batch_number * self.batch_size + self.batch_size]

        # Put the positions in an in-memory file object
        cbuf = StringIO()
        cbuf.write('\n'.join(
                    ["{} {}".format(ra_idx, dec_idx) for (ra_idx, dec_idx)
                     in zip(ra_batch, dec_batch)]))
        cbuf.seek(0)

        # Use requests.post to pass in positions as a file
        r = requests.post(url_ps1filename,
                          data=dict(filters=bands, type=imagetypes),
                          files=dict(file=cbuf))
        r.raise_for_status()
        # Convert retrieved file table to pandas DataFrame
        df = Table.read(r.text, format="ascii").to_pandas()

        # Group table by filter and do not sort!
        groupby = df.groupby(by='filter', sort=False)

        for group_key in groupby.groups.keys():

            group_df = groupby.get_group(group_key)
            band = group_key

            for idx in group_df.index:

                filename = group_df.loc[idx, 'filename']

                obj_name = utils.coord_to_name(group_df.loc[idx, 'ra'],
                                               group_df.loc[idx, 'dec'])[0]

                # Create image name
                image_name = obj_name + "_" + self.name + "_" + \
                             band + "_fov" + '{:d}'.format(self.fov)

                url = ("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
                       "ra={}&dec={}&size={}&format=fits").format(
                    group_df.loc[idx, 'ra'],
                    group_df.loc[idx, 'dec'],
                    img_size)
                urlbase = url + "&red="

                new_entry = pd.DataFrame(data={'image_name': image_name,
                                               'url': urlbase + filename},
                                         index=[0])
                self.download_table = pd.concat([self.download_table,
                                                 new_entry],
                                                ignore_index=True)

        self.download_table.to_csv('{}_PS1_download_urls.csv'.format(
            str(batch_number)))

    def force_photometry_params(self, header, band, filepath=None):
        """Set parameters to calculate aperture photometry for the Pan-STARRS1
        survey imaging.

        :param header: Image header
        :type header: astropy.io.fits.header.Header
        :param band: The filter band of the image
        :type band: str
        :param filepath: File path to the image
        :type filepath: str

        :return: None
        """

        zpt = {"g": 25.0, "r": 25.0, "i": 25.0, "z": 25.0, "y": 25.0,}

        self.exp = header['EXPTIME']
        self.back = True
        self.zpt = zpt[band]
        self.ab_corr = 0.
        self.nanomag_corr = np.power(10, 0.4*(22.5-self.zpt))
