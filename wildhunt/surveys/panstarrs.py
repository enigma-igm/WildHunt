#!/usr/bin/env python

import os
import requests
import numpy as np
from io import StringIO
from astropy.table import Table
from astropy.io import fits

from wildhunt.surveys import imagingsurvey


from IPython import embed


class Panstarrs(imagingsurvey.ImagingSurvey):

    def __init__(self, bands, fov, verbosity=1):
        """

        :param bands:
        :param fov:
        :param name:
        """

        self.batch_size = 10000

        self.ra = None
        self.dec = None
        self.nbatch = 1

        super(Panstarrs, self).__init__(bands, fov, 'PS1', verbosity)

    def download_images(self, ra, dec, image_folder_path, n_jobs=1):
        """

        :param ra:
        :param dec:
        :param image_folder_path:
        :param n_jobs:
        :return:
        """

        self.survey_setup(ra, dec, image_folder_path, epoch='J', n_jobs=n_jobs)

        if self.source_table.shape[0] > 0 :

            self.batch_setup()

            for i in range(self.nbatch):
                self.retrieve_image_url_list(imagetypes="stack", batch_number = i)

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

            print('All images already exist.')
            print('Downloading aborted.')

    def batch_setup(self,):

        self.ra = self.source_table.loc[:, 'ra'].values
        self.dec = self.source_table.loc[:, 'dec'].values

        # Compute the number of batches to retrieve the urls
        if np.size(self.ra) > self.batch_size:
            self.nbatch = int(np.ceil(np.size(self.ra) / self.batch_size))

    def retrieve_image_url_list(self, batch_number = 0, imagetypes="stack"):

        # Convert field of view in arcsecond to pixel size (1 pixel = 0.25 arcseconds)
        self.size = self.fov * 4

        bands = ''.join(self.bands)

        # Retrieve bulk file table
        url_ps1filename = 'http://ps1images.stsci.edu/cgi-bin/ps1filenames.py'

        ra_batch = self.ra[batch_number * self.batch_size : batch_number * self.batch_size + self.batch_size]
        dec_batch = self.dec[batch_number * self.batch_size : batch_number * self.batch_size + self.batch_size]
        # Put the positions in an in-memory file object
        cbuf = StringIO()
        cbuf.write(
                '\n'.join(
                    ["{} {}".format(ra_idx, dec_idx) for (ra_idx, dec_idx) in zip(
                        ra_batch, dec_batch)]))
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

        for idx in range(len(ra_batch)):

            obj_name = self.source_table.iloc[batch_number * self.batch_size + idx]['obj_name']

            for jdx, (key, group) in enumerate(groupby):
                    band = group.loc[jdx, 'filter']
                    filename = df.loc[jdx+idx*len(self.bands), 'filename']

                    # Create image name
                    image_name = obj_name + "_" + self.name + "_" + \
                                    band + "_fov" + '{:d}'.format(self.fov)

                    url = ("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
                                   "ra={}&dec={}&size={}&format=fits").format(ra_batch[idx],
                                                                              dec_batch[idx],
                                                                              self.size)
                    urlbase = url + "&red="

                    self.download_table = self.download_table.append(
                                {'image_name': image_name,
                                 'url': urlbase + filename},
                                ignore_index=True)
        self.download_table.to_csv('{}_PS1_download_urls.csv'.format(str(batch_number)))

    def force_photometry_params(self, header, band, filepath=None):
        '''Set the parameters that are used in the aperture_photometry to perform forced photometry based on the
        :param heade: header of the image
        :param band: image band
        :param filepath: file path to the image

        Returns:
            self
        '''

        zpt = {"g": 25.0, "r": 25.0, "i": 25.0, "z": 25.0, "y": 25.0,}


        self.exp = header['EXPTIME']
        self.back = 'back'
        self.zpt = zpt[band]
        self.ABcorr = 0.
        self.nanomag_corr = np.power(10, 0.4*(22.5-self.zpt))

        return self