#!/usr/bin/env python

import numpy as np
from astropy.table import Table

from wildhunt.surveys import imagingsurvey


from IPython import embed


class Panstarrs(imagingsurvey.ImagingSurvey):

    def __init__(self, bands, fov, verbosity=1):
        """

        :param bands:
        :param fov:
        :param name:
        """

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

        # Convert field of view in arcsecond to pixel size (1 pixel = 0.25 arcseconds)
        size = self.fov * 4

        for idx in self.source_table.index:

            ra = self.source_table.loc[idx, 'ra']
            dec = self.source_table.loc[idx, 'dec']
            obj_name = self.source_table.loc[idx, 'obj_name']

            bands = ''.join(self.bands)

            try:
                filetable = self.get_ps1_filetable(ra, dec, bands=bands)

            except :
                fname = '{}_download_table_temp.csv'.format(self.name)
                self.download_table.to_csv(fname)
                raise Exception('Download of url table failed. Wrote survey '
                                'download table to file: {}'.format(fname))

            for row in filetable:
                band = row['filter']
                filename = row['filename']

                # Create image name
                image_name = obj_name + "_" + self.name + "_" + \
                             band + "_fov" + '{:d}'.format(self.fov)

                url = ("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
                       "ra={}&dec={}&size={}&format=fits").format(ra,
                                                                  dec,
                                                                  size)

                urlbase = url + "&red="

                self.download_table = self.download_table.append(
                    {'image_name': image_name,
                     'url': urlbase + filename},
                    ignore_index=True)


    def get_ps1_filetable(self, ra, dec, bands='g'):
        """

        :param ra:
        :param dec:
        :param bands:
        :return:
        """
        url_base = 'http://ps1images.stsci.edu/cgi-bin/ps1filenames.py?'
        ps1_url = url_base + 'ra={}&dec={}&filters={}'.format(ra, dec, bands)

        table = Table.read(ps1_url, format='ascii')

        # Sort filters from red to blue
        flist = ["yzirg".find(x) for x in table['filter']]
        table = table[np.argsort(flist)]

        return table
