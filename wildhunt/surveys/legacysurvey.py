#!/usr/bin/env python

import os
import numpy as np
from astropy.table import Table
from astropy.io import fits

from wildhunt.surveys import imagingsurvey

class LegacySurvey(imagingsurvey.ImagingSurvey):

    def __init__(self, bands, fov, name, verbosity=1):
        """

        :param bands:
        :param fov:
        :param name:
        """

        super(LegacySurvey, self).__init__(bands, fov, name, verbosity)

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

            self.retrieve_image_url_list()

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

    def retrieve_image_url_list(self):

        for idx in self.source_table.index:

            ra = self.source_table.loc[idx, 'ra']
            dec = self.source_table.loc[idx, 'dec']
            obj_name = self.source_table.loc[idx, 'obj_name']
            release = 'ls-dr' + self.name[6:]
            bands = ''.join(self.bands)

            for band in bands:

                # Convert field of view in arcsecond to pixel size (1 pixel = 0.262 arcseconds
                # for g,r,z and 2.75 arcseconds for W1, W2)
                if band in ['1','2']:
                    pixelscale = 2.75
                else:
                    pixelscale = 0.262
                size = self.fov * 1 / pixelscale

                # Create image name
                image_name = obj_name + "_" + self.name + "_" + \
                             band + "_fov" + '{:d}'.format(self.fov)

                url = ("http://legacysurvey.org/viewer/fits-cutout/?ra={}&dec={}&layer={}"
                       "&pixscale={}&bands={}&size={}").format(ra,dec,release,str(pixelscale),band,str(int(size)))


                self.download_table = self.download_table.append(
                    {'image_name': image_name,'url': url},
                    ignore_index=True)

    def data_setup(self, obj_name, band, image_folder_path):
        '''
            Set the parameters that are used in the aperture_photometry to perform forced photometry based on the
            LS survey
            Args:
                obj_name:
                band:
                image_folder_path:
            '''

        zpt = {"g": 22.5, "r": 22.5, "z": 22.5, "1": 22.5, "2": 22.5}
        aboffset={"1": 2.699, "2": 3.339}

        # Read in the data and header
        image_name = obj_name + "_" + self.name + "_" + band + "_fov" + '{:d}'.format(self.fov)
        fitsname = os.path.join(image_folder_path, image_name + '.fits')

        par = fits.open(fitsname)
        if band in ['1','2']:
            self.data = par[0].data.copy() * 10 ** (-aboffset[band] / 2.5)
        else:
            self.data = par[0].data.copy()

        self.hdr = par[0].header
        self.exp = 1
        self.extCorr = 0.0
        self.back = 'no_back'
        self.zpt = zpt[band]

        del par[0].data

        return self