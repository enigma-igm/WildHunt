#!/usr/bin/env python

import os
import time
import numpy as np
import pandas as pd
from astropy.io import fits

from wildhunt import pypmsgs
from wildhunt.surveys import imagingsurvey

msgs = pypmsgs.Messages()

# List of programmeID:
# VHS - 110, VVV - 120, VMC - 130, VIKING - 140, VIDEO - 150,
# UHS - 107, UKIDSS: LAS - 101,
# GPS - 102, GCS - 103, DXS - 104, UDS - 105


class VsaWsa(imagingsurvey.ImagingSurvey):
    """ VsaWsa class deriving from the ImagingSurvey class to handle
    image downloads and aperture photometry for the 1 VSA and WSA image
    archives.

    WSA: http://wsa.roe.ac.uk/

    VSA: http://vsa.roe.ac.uk/

    """

    def __init__(self, bands, fov, name, verbosity=1):
        """ Initialize the VsaWsa class.

        :param bands: List of survey filter bands.
        :type bands: list
        :param fov: Field of view of requested imaging in arcseconds.
        :type fov: int
        :param name: Name of the survey.
        :type name: str
        :param verbosity: Verbosity level.
        :type verbosity: int
        :return: None
        """

        self.database = name

        if name[0] == 'U':
            archive = 'WSA'
            if name[0:3] == 'UKI':
                if name[-3::] == 'LAS':
                    program_id = '101'
                elif name[-3::] == 'GPS':
                    program_id = '102'
                elif name[-3::] == 'GCS':
                    program_id = '103'
                elif name[-3::] == 'DXS':
                    program_id = '104'
                elif name[-3::] == 'UDS':
                    program_id = '105'
                else:
                    msgs.error('Unknown UKIDSS survey program.')
                    raise ValueError
                self.database = name[:-3]
            elif name[0:3] == 'UHS':
                program_id = '107'
            else:
                msgs.error('Unknown WSA survey program.')
                raise ValueError
        else:
            archive = 'VSA'
            if name[0:3] == 'VHS':
                program_id = '110'
            elif name[0:3] == 'VVV':
                program_id = '120'
            elif name[0:3] == 'VMC':
                program_id = '130'
            elif name[0:3] == 'VIK':
                program_id = '140'
            elif name[0:3] == 'VID':
                program_id = '150'
            else:
                msgs.error('Unknown VSA survey program.')
                raise ValueError
            self.database = name

        self.archive = archive
        self.program_id = program_id
        self.batch_size = 10000

        super(VsaWsa, self).__init__(bands, fov, name, verbosity)

    def download_images(self, ra, dec, image_folder_path, n_jobs=1):
        """ Download images from the online services for VSA or WSA.

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

            self.retrieve_image_url_list()

            print(self.download_table)

            self.check_for_existing_images_before_download()

            if n_jobs > 1:
                self.mp_download_image_from_url()
            else:
                for idx in self.download_table.index:
                    image_name = self.download_table.loc[idx, 'image_name']
                    url = self.download_table.loc[idx, 'url']
                    self.download_image_from_url(url, image_name)

            os.system('rm out_*')
            os.system('rm upload_*')

        else:
            msgs.info('All images already exist.')
            msgs.info('Download canceled.')

    def retrieve_image_url_list(self):
        """ Retrieve the list of image URLs from the online image server.

        :return: None
        """

        ra = self.source_table.loc[:, 'ra'].values
        dec = self.source_table.loc[:, 'dec'].values
        obj_names = self.source_table.loc[:,'obj_name'].values

        # Survey parameters
        survey_param = {'archive': self.archive, 'database': self.database,
                        'programmeID': self.program_id,
                        'bands': self.bands,
                        'idPresent': 'noID', 'userX': self.fov/60, 'email': '',
                        'email1': '', 'crossHair': 'n',
                        'mode': 'wget'}

        boundary = "--FILEUPLOAD"  # separator

        if np.size(ra) > self.batch_size:
            nbatch = int(np.ceil(np.size(ra) / self.batch_size))
        else:
            nbatch = 1

        loop = 0

        for i in range(nbatch):

            start_id = self.batch_size * loop
            loop += 1
            upload_file = "upload_" + str(loop) + ".txt"
            output_file = "out_" + str(loop) + ".txt"
            up = open(upload_file, 'w+')

            for param in survey_param:
                if param == 'bands':
                    for band in (survey_param[param]):
                        print(boundary, file=up)
                        print('Content-Disposition: form-data; name=\"band\"\n',
                              file=up)
                        print(band + ' ', file=up)
                else:
                    print(boundary, file=up)
                    print(
                        'Content-Disposition: form-data; name=\"{}\"\n'.format(
                            param), file=up)
                    print(str(survey_param[param]) + ' ', file=up)
            print(boundary, file=up)
            print('Content-Disposition: form-data; name=\"startID\"\n',
                  file=up)
            print(str(start_id) + ' ', file=up)
            print(boundary, file=up)
            print(
                'Content-Disposition: form-data; name=\"fileName\"; filename='
                '\"{}\" Content-Type: text/plain\n'.format(
                    upload_file), file=up)

            if i != (nbatch - 1):
                n = self.batch_size
            else:
                n = len(ra) - start_id
            for jdx in range(n):
                if dec[start_id + jdx] >= 0:
                    print('{:>15},{:>15}'.format(
                        '+' + str(ra[start_id + jdx]),
                        '+' + str(dec[start_id + jdx])),
                        file=up)
                else:
                    print('{:>15},{:>15}'.format(
                        '+' + str(ra[start_id + jdx]),
                        str(dec[start_id + jdx])),
                        file=up)

            print('\n', file=up)
            print(boundary + '--', file=up)

            up.close()

            if self.archive == 'WSA':

                os.system(
                    "wget --keep-session-cookies --header=\"Content-Type: "
                    "multipart/form-data;  boundary=FILEUPLOAD\""
                    " --post-file {} http://wsa.roe.ac.uk:8080/wsa/"
                    "tmpMultiGetImage -O {}".format(
                        upload_file,
                        output_file))

            elif self.archive == 'VSA':
                os.system(
                    "wget --keep-session-cookies --header=\"Content-Type: "
                    "multipart/form-data;  boundary=FILEUPLOAD\""
                    " --post-file {} http://horus.roe.ac.uk:8080/vdfs"
                    "/MultiGetImage -O {}".format(
                        upload_file,
                        output_file))
            else:
                raise NotImplementedError('Specified archive {} not '
                                          'understood. Can be "VSA" or '
                                          '"WSA".'.format(self.archive))

            out = open(output_file, 'r')

            lines = out.readlines()

            j = 0
            n_bands = 1
            for line in lines:
                record = '--http'

                if record in line:

                    obj_name = obj_names[start_id + j]
                    if n_bands == len(survey_param['bands']):
                        j += 1
                        n_bands = 0
                    n_bands += 1
                    start_pos = line.index(record)
                    end_pos = line.rindex('-->')
                    file_url = line[start_pos + 2:end_pos]
                    if 'band=NULL' in line:
                        msgs.info('Skip download')
                    else:
                        band_url = line.index('band=') + len('band=')
                        # Create image name
                        if (line[band_url] == 'K') & (self.archive == 'VSA'):
                            image_name = obj_name + "_" + self.name + "_" + \
                                         line[band_url] + line[band_url+1] +\
                                         "_fov" + '{:d}'.format(self.fov)
                        else:
                            image_name = obj_name + "_" + self.name + "_" + \
                                         line[band_url] + "_fov" +\
                                         '{:d}'.format(self.fov)

                        new_entry = pd.DataFrame(data={'image_name':
                                                       image_name,
                                                       'url': file_url},
                                                 index=[0])

                        self.download_table = pd.concat([self.download_table,
                                                        new_entry],
                                                        ignore_index=True)

            out.close()

            msgs.info("Start 5s sleep to not overload server.")
            time.sleep(5)
            msgs.info("End of sleep")

    def force_photometry_params(self, header, band, filepath=None):
        """Set parameters to calculate aperture photometry for the VSA/WSA
        survey imaging.

        :param header: Image header
        :type header: astropy.io.fits.header.Header
        :param band: The filter band of the image
        :type band: str
        :param filepath: File path to the image
        :type filepath: str

        :return None
        """

        par = fits.open(filepath)

        if self.archive == 'WSA':

            ab_corr = {"Z": 0.528, "Y": 0.634, "J": 0.938,
                       "H": 1.379, "K": 1.9}
            self.exp = par['PRIMARY'].header["EXP_TIME"]
            self.zpt = par['WSAIMAGE'].header['MAGZPT']
            self.ab_corr = ab_corr[band]

        else:

            self.exp = par['GETIMAGE'].header["EXPTIME"]
            ab_corr = {"Z": 0.521, "Y": 0.618, "J": 0.937,
                       "H": 1.384, "Ks": 1.839}
            self.ab_corr = ab_corr[band]

            try:
                self.zpt = par['GETIMAGE'].header['MAGZPT']
            except:
                self.zpt = 26.7  # this is a hack for VIKING J-band

        self.back = 'back'
        self.nanomag_corr = np.power(10,
                                     0.4 * (22.5 - self.zpt - self.ab_corr))

