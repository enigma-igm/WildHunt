#!/usr/bin/env python

import pandas as pd


from wildhunt import pypmsgs
from wildhunt import utils
from wildhunt.surveys import imagingsurvey

msgs = pypmsgs.Messages()


class LoTSSDR2(imagingsurvey.ImagingSurvey):
    """ LoTSSDR2 class deriving from the ImagingSurvey class to handle
    image downloads and aperture photometry for the LOw-Frequency ARray
    (LOFAR) Two-metre Sky Survey (LoTSS).

    It is designed to download cutouts from the LoTSS DR2 mosaics at 150 MHz.

    The LoTSSDR2 survey provides two resolution options, which are encoded in
    the available bands as follows:
    - 150MHz (default; 6 arcsec resolution)
    - 150MHz_lowres (20 arcsec resolution)

    """

    def __init__(self, bands, fov, name, verbosity=1):
        """ Initialize the LoTSSDR2 class.

        :param bands: List of survey filter bands.
        :type bands: list
        :param fov: Field of view of requested imaging in arcseconds.
        :type fov: int
        :param name: Name of the survey.
        :type name: str
        """

        # Check if the provided filter bands are valid for LoTSSDR2
        valid_bands = ['150MHz', '150MHz_lowres']
        valid_band_counter = 0
        for band in bands:
            if band not in valid_bands:
                msgs.info('Provided band {} is not valid for LoTSSDR2.'.format(band))
                msgs.info('Valid bands for LoTSSDR2 are: {}'.format(valid_bands))
            else:
                valid_band_counter += 1
        if valid_band_counter == 0:
            msgs.info('No valid bands provided for LoTSSDR2.')
            msgs.info('Defaulting to 150MHz band.')
            bands = ['150MHz']

        super(LoTSSDR2, self).__init__(bands, fov, name, verbosity)

    def download_images(self, ra, dec, image_folder_path, n_jobs=1):
        """ Download images from the online image server for LoTSSDR2.

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
            ra_hms, dec_dms = utils.degree_to_hms(ra, dec)

            size = self.fov / 60.0  # Convert arcseconds to arcminutes

            for band in self.bands:

                # Create image name
                image_name = obj_name + "_" + self.name + "_" + \
                             band + "_fov" + '{:d}'.format(self.fov)

                base_url = 'https://lofar-surveys.org/'

                if band == '150MHz_lowres':
                    page = 'dr2-low-cutout.fits'
                elif band == '150MHz':
                    page = 'dr2-cutout.fits'
                else:
                    msgs.error('Invalid band for LoTSSDR2.')



                url = base_url + page + '?pos=' + ra_hms + '%20' + dec_dms + \
                        '&size=' + str(size)

                # Implementing concat instead of deprecated append
                new_entry = pd.DataFrame(data={'image_name': image_name,
                                               'url': url},
                                         index=[0])
                self.download_table = pd.concat([self.download_table,
                                                new_entry],
                                                ignore_index=True)