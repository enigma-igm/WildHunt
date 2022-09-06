#!/usr/bin/env python

import time
import os
import numpy as np

from wildhunt.surveys import imagingsurvey
from IPython import embed


# List of programmeID: VHS - 110, VVV - 120, VMC - 130, VIKING - 140, VIDEO - 150, UHS - 107, UKIDSS: LAS - 101,
# GPS - 102, GCS - 103, DXS - 104, UDS - 105
numCoords=10000  # number of coords in each bach (do not exceed 20000)

class VsaWsa(imagingsurvey.ImagingSurvey):

    def __init__(self, bands, fov, name, verbosity=1):
        """
        :param bands:
        :param fov:
        :param name:
        """

        self.database = name

        if name[0] == 'U':
            archive = 'WSA'
            if name[0:3] == 'UKI':
                if name[-3::] == 'LAS': programID='101'
                if name[-3::] == 'GPS': programID = '102'
                if name[-3::] == 'GCS': programID = '103'
                if name[-3::] == 'DXS': programID = '104'
                if name[-3::] == 'UDS': programID = '105'
                self.database = name[:-3]
            if name[0:3] == 'UHS': programID = '107'
        else:
            archive = 'VSA'
            if name[0:3] == 'VHS': programID='110'
            if name[0:3] == 'VVV': programID = '120'
            if name[0:3] == 'VMC': programID = '130'
            if name[0:3] == 'VIK': programID = '140'
            if name[0:3] == 'VID': programID = '150'
            self.database = name

        self.archive=archive
        self.programID=programID
        self.batch_size=10000

        super(VsaWsa, self).__init__(bands, fov, name, verbosity)

    def download_images(self, ra, dec, image_folder_path, n_jobs=1):
        """
        :param ra:
        :param dec:
        :param image_folder_path:
        :param n_jobs:
        :return:
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

            print('All images already exist.')
            print('Downloading aborted.')


    def retrieve_image_url_list(self):
        '''
        Download the links to the images and produce all the *_wget.sh files
        :param ra:
        :param dec:
        :param image_folder_path:
        :param n_jobs:
        :return:
        '''

        ra = self.source_table.loc[:, 'ra'].values
        dec = self.source_table.loc[:, 'dec'].values
        obj_names = self.source_table.loc[:,'obj_name'].values

        # Survey parameters
        survey_param = {'archive': self.archive, 'database': self.database,
                        'programmeID': self.programID,
                        'bands': self.bands,
                        'idPresent': 'noID', 'userX': self.fov/60, 'email': '',
                        'email1': '', 'crossHair': 'n',
                        'mode': 'wget'}

        embed()

        boundary = "--FILEUPLOAD"  # separator

        if np.size(ra) > numCoords:
            nbatch = int(np.ceil(np.size(ra) / numCoords))
        else:
            nbatch = 1

        loop = 0

        for i in range(nbatch):

            startID = numCoords * loop
            loop += 1
            uploadFile = "upload_" + str(loop) + ".txt"
            outFile = "out_" + str(loop) + ".txt"
            up = open(uploadFile, 'w+')

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
            print('Content-Disposition: form-data; name=\"startID\"\n', file=up)
            print(str(startID) + ' ', file=up)
            print(boundary, file=up)
            print(
                'Content-Disposition: form-data; name=\"fileName\"; filename=\"{}\" Content-Type: text/plain\n'.format(
                    uploadFile), file=up)

            if i != (nbatch - 1):
                n = numCoords
            else:
                n = len(ra) - startID
            for jdx in range(n):
                if dec[startID + jdx] >= 0:
                    print('{:>15},{:>15}'.format('+' + str(ra[startID + jdx]),
                                                 '+' + str(dec[startID + jdx])),
                          file=up)
                else:
                    print('{:>15},{:>15}'.format('+' + str(ra[startID + jdx]),
                                                 str(dec[startID + jdx])),
                          file=up)

            print('\n', file=up)
            print(boundary + '--', file=up)

            up.close()

            if self.archive == 'WSA':

                os.system(
                    "wget --keep-session-cookies --header=\"Content-Type: multipart/form-data;  boundary=FILEUPLOAD\""
                    " --post-file {} http://wsa.roe.ac.uk:8080/wsa/tmpMultiGetImage -O {}".format(
                        uploadFile,
                        outFile))
                url = "wget --keep-session-cookies --header=\"Content-Type: multipart/form-data;  boundary=FILEUPLOAD\"" \
                      " --post-file {} http://wsa.roe.ac.uk:8080/wsa/tmpMultiGetImage -O {}".format(uploadFile,outFile)
                embed()
            elif self.archive == 'VSA':
                os.system(
                    "wget --keep-session-cookies --header=\"Content-Type: multipart/form-data;  boundary=FILEUPLOAD\""
                    " --post-file {} http://horus.roe.ac.uk:8080/vdfs/MultiGetImage -O {}".format(
                        uploadFile,
                        outFile))
            else:
                raise NotImplementedError('Specified archive {} not '
                                          'understood. Can be "VSA" or '
                                          '"WSA".'.format(self.archive))

            out = open(outFile, 'r')

            lines = out.readlines()

            j = 0
            n_bands = 1
            for line in lines:
                record = '--http'

                if (record in line):

                    obj_name = obj_names[startID + j]
                    if n_bands == len(survey_param['bands']):
                        j += 1
                        n_bands = 0
                    n_bands += 1
                    strtPos = line.index(record)
                    endPos = line.rindex('-->')
                    fileURL = line[strtPos + 2:endPos]
                    if ('band=NULL' in line):
                        print('Skip download')
                    else:
                        band_url = line.index('band=') + len('band=')
                        # Create image name
                        image_name = obj_name + "_" + self.name + "_" + \
                                     line[band_url] + "_fov" + '{:d}'.format(self.fov)

                        self.download_table = self.download_table.append(
                            {'image_name': image_name,
                             'url': fileURL},
                            ignore_index=True)

            out.close()

            print("Start sleep")
            time.sleep(5)
            print("End sleep")
