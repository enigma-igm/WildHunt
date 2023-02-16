#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
from astropy.table import Table
import tarfile

from wildhunt.surveys import imagingsurvey
from wildhunt import pypmsgs

from IPython import embed

msgs = pypmsgs.Messages()


class WISE(imagingsurvey.ImagingSurvey):

    def __init__(self, bands, fov, name, verbosity=1):
        """

        :param bands:
        :param fov:
        :param name:
        """

        super(WISE, self).__init__(bands, fov, name, verbosity)

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

            self.extract_images()

        else:

            print('All images already exist.')
            print('Downloading aborted.')

    def retrieve_image_url_list(self):

        for idx in self.source_table.index:

            ra = self.source_table.loc[idx, 'ra']
            dec = self.source_table.loc[idx, 'dec']
            obj_name = self.source_table.loc[idx, 'obj_name']
            bands = ''.join(self.bands)

            for band in bands:

                # Convert field of view in arcsecond to pixel size (1 pixel =  2.75 arcseconds for WISE)

                if 'all' in self.name:
                    release = 'allwise'
                else:
                    release = self.name.split("-",1)[1]

                pixelscale = 2.75
                size = self.fov * 1 / pixelscale

                # Create image name
                image_name = obj_name + "_" + self.name + "_" + \
                             band + "_fov" + '{:d}'.format(self.fov)

                url = ("http://unwise.me/cutout_fits?version={}&ra={}&dec={}"
                       "&size={}&bands={}&file_img_m=on").format(release,ra,dec,str(int(size)),band)

                # Implementing concat instead of review append
                new_entry = pd.DataFrame(data={'image_name': image_name,
                                               'url': url},
                                         index=[0])
                self.download_table = pd.concat([self.download_table,
                                                new_entry],
                                                ignore_index=True)

    def extract_images(self,):

        for image_name in self.download_table['image_name']:
            tar_file = tarfile.open(self.image_folder_path + '/' + image_name + '.tar.gz')
            file_name = tar_file.firstmember.name
            untar = tar_file.extractfile(file_name)
            untar = untar.read()

            output = open(self.image_folder_path + '/' + image_name + '.fits', 'wb')
            output.write(untar)
            output.close()

            os.remove(self.image_folder_path + '/' + image_name + '.tar.gz')