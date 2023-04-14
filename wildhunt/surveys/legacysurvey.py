#!/usr/bin/env python

import pandas as pd


from wildhunt import pypmsgs
from wildhunt.surveys import imagingsurvey

msgs = pypmsgs.Messages()


class LegacySurvey(imagingsurvey.ImagingSurvey):
    """ Legacy Survey class deriving from the ImagingSurvey class to handle
    image downloads and aperture photometry for the Legacy Survey.

    """

    def __init__(self, bands, fov, name, verbosity=1):
        """ Initialize the LegacySurvey class.

        :param bands: List of survey filter bands.
        :type bands: list
        :param fov: Field of view of requested imaging in arcseconds.
        :type fov: int
        :param name: Name of the survey.
        :type name: str
        """

        super(LegacySurvey, self).__init__(bands, fov, name, verbosity)

    def download_images(self, ra, dec, image_folder_path, n_jobs=1):
        """ Download images from the online image server for Legacy Survey.

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

            msgs.info('All images already exist.')
            msgs.info('Download canceled.')

    def retrieve_image_url_list(self):
        """ Retrieve the list of image URLs from the online image server.

        :return: None
        """

        for idx in self.source_table.index:

            ra = self.source_table.loc[idx, 'ra']
            dec = self.source_table.loc[idx, 'dec']
            obj_name = self.source_table.loc[idx, 'obj_name']
            bands = ''.join(self.bands)

            for band in bands:

                # Convert field of view in arcsecond to pixel size
                # (1 pixel = 0.262 arcseconds for g,r,z and 2.75 arcseconds
                # for W1, W2)

                if band in ['1', '2']:
                    release = 'unwise-neo6'
                    pixelscale = 2.75
                else:
                    release = 'ls-dr' + self.name[6:]
                    pixelscale = 0.262
                size = self.fov * 1 / pixelscale

                # Create image name
                image_name = obj_name + "_" + self.name + "_" + \
                             band + "_fov" + '{:d}'.format(self.fov)

                url = (
                    'http://legacysurvey.org/viewer/fits-cutout/?ra={}&dec='
                    '{}&layer={} &pixscale={}&bands={}&size={}').format(
                    ra, dec, release, str(pixelscale), band, str(int(size)))

                # Implementing concat instead of deprecated append
                new_entry = pd.DataFrame(data={'image_name': image_name,
                                               'url': url},
                                         index=[0])
                self.download_table = pd.concat([self.download_table,
                                                new_entry],
                                                ignore_index=True)

    def force_photometry_params(self, header, band, filepath=None):
        """Set parameters to calculate aperture photometry for the Legacy
        Survey imaging.

        :param header: Image header
        :type header: astropy.io.fits.header.Header
        :param band: The filter band of the image
        :type band: str
        :param filepath: File path to the image
        :type filepath: str

        :return: None
        """

        zpt = {"g": 22.5, "r": 22.5, "z": 22.5, "1": 22.5, "2": 22.5}

        self.exp = 1.
        self.back = False
        self.zpt = zpt[band]
        self.nanomag_corr = 1.

        if band == '1':
            self.ab_corr = 2.699
        elif band == '2':
            self.ab_corr = 3.339
        else:
            self.ab_corr = 0.
