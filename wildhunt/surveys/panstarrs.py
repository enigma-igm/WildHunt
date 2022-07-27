#!/usr/bin/env python

import time
import os
import requests
import numpy as np
from io import StringIO
from astropy.table import Table
from astropy.io import fits

from wildhunt.surveys import imagingsurvey


from IPython import embed


class Panstarrs(imagingsurvey.ImagingSurvey):

    def __init__(self, bands, fov, batch_size=10000, verbosity=1):
        """

        :param bands:
        :param fov:
        :param name:
        :param batch_size:
        """

        super(Panstarrs, self).__init__(bands, fov, 'PS1', batch_size, verbosity)

    def download_images(self, ra, dec, image_folder_path, n_jobs=1):
        """

        :param ra:
        :param dec:
        :param image_folder_path:
        :param n_jobs:
        :return:
        """

        self.survey_setup(ra, dec, image_folder_path, epoch='J', n_jobs=n_jobs)

        if len(ra) == len(dec) and len(ra) > 0:

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

        else:

            print('All images already exist.')
            print('Downloading aborted.')

    def batch_setup(self,):

        self.ra = self.source_table.loc[:, 'ra'].values
        self.dec = self.source_table.loc[:, 'dec'].values

        # Compute the number of batches to retrieve the urls
        if np.size(self.ra) > self.batch_size:
            self.nbatch = int(np.ceil(np.size(self.ra) / self.batch_size))
        else:
            self.nbatch = 1

    def retrieve_image_url_list(self, batch_number = 0, imagetypes="stack"):

        # Convert field of view in arcsecond to pixel size (1 pixel = 0.25 arcseconds)
        self.size = self.fov * 4

        bands = ''.join(self.bands)

        # Retrieve bulk file table
        url_ps1filename = 'http://ps1images.stsci.edu/cgi-bin/ps1filenames.py?'

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

    def data_setup(self, obj_name, band, image_folder_path):
        '''
            Set the parameters that are used in the aperture_photometry to perform forced photometry based on the
            PS1 survey
            Args:
            Args:
                obj_name:
                band:
                image_folder_path:
            '''

        zpt = {"g": 25.0, "r": 25.0, "i": 25.0, "z": 25.0, "y": 25.0,}

        # Read in the data and header
        image_name = obj_name + "_" + self.name + "_" + band + "_fov" + '{:d}'.format(self.fov)
        fitsname = os.path.join(image_folder_path, image_name + '.fits')

        par = fits.open(fitsname)
        self.data = par[0].data.copy()
        self.hdr = par[0].header
        self.exp = par[0].header['EXPTIME']
        self.extCorr = 0.0
        self.back = 'back'
        self.zpt = zpt[band]

        del par[0].data

        return self



    # def retrieve_image_url_list(self):
    #
    #     # Convert field of view in arcsecond to pixel size (1 pixel = 0.25 arcseconds)
    #     size = self.fov * 4
    #
    #     for idx in self.source_table.index:
    #
    #         ra = self.source_table.loc[idx, 'ra']
    #         dec = self.source_table.loc[idx, 'dec']
    #         obj_name = self.source_table.loc[idx, 'obj_name']
    #
    #         bands = ''.join(self.bands)
    #
    #         try:
    #             filetable = self.get_ps1_filetable(ra, dec, bands=bands)
    #
    #         except :
    #             fname = '{}_download_table_temp.csv'.format(self.name)
    #             self.download_table.to_csv(fname)
    #             raise Exception('Download of url table failed. Wrote survey '
    #                             'download table to file: {}'.format(fname))
    #
    #         for row in filetable:
    #             band = row['filter']
    #             filename = row['filename']
    #
    #             # Create image name
    #             image_name = obj_name + "_" + self.name + "_" + \
    #                          band + "_fov" + '{:d}'.format(self.fov)
    #
    #             url = ("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
    #                    "ra={}&dec={}&size={}&format=fits").format(ra,
    #                                                               dec,
    #                                                               size)
    #
    #             urlbase = url + "&red="
    #
    #             self.download_table = self.download_table.append(
    #                 {'image_name': image_name,
    #                  'url': urlbase + filename},
    #                 ignore_index=True)
    #
    #
    # def get_ps1_filetable(self, ra, dec, bands='g'):
    #     """
    #
    #     :param ra:
    #     :param dec:
    #     :param bands:
    #     :return:
    #     """
    #     url_base = 'http://ps1images.stsci.edu/cgi-bin/ps1filenames.py?'
    #     ps1_url = url_base + 'ra={}&dec={}&filters={}'.format(ra, dec, bands)
    #
    #     table = Table.read(ps1_url, format='ascii')
    #
    #     # Sort filters from red to blue
    #     flist = ["yzirg".find(x) for x in table['filter']]
    #     table = table[np.argsort(flist)]
    #
    #     return table
