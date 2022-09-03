#!/usr/bin/env python
"""

Main module for performing forced photometry.

"""
from IPython import  embed

import warnings
import os
import multiprocessing
from multiprocessing import Process, Queue
import numpy as np
import pandas as pd

from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy import units as u
from photutils import aperture_photometry, SkyCircularAperture, SkyCircularAnnulus
from astropy import stats
from astropy.table import Table, vstack

from wildhunt import utils, image
from wildhunt.surveys import imagingsurvey

class Forced_photometry(imagingsurvey.ImagingSurvey):

    def __init__(self, bands, fov, name, verbosity=1):
        """

        :param bands:
        :param name:
        :param fov:
        """

        super(Forced_photometry, self).__init__(bands, fov, name, verbosity)

    def forced_main(self, ra, dec, table_name, image_folder_path, radii=[1., 1.5], radius_in=7.0,
                    radius_out=10.0,epoch='J', n_jobs=1, remove=False):
        """
        Main function that calls all the other functions used to perform forced photometry
        :param ra: np.array(), right ascensions in degrees
        :param dec: np.array(), declinations in degrees
        :param image_folder_path: string, path where the images for which to perform forced photometry are stored
        :param n_jobs: int, number of forced photometry processes performed in parallel
        :return:
        """

        n_file = np.size(ra)
        n_cpu = multiprocessing.cpu_count()

        if n_jobs > n_cpu-1:
            n_jobs = n_cpu-1
        if n_jobs > n_file:
            n_jobs = n_file

        if n_jobs > 1:
            work_queue = Queue()
            processes = []
            out_tab = Table()
            for ii in range(n_file):
                work_queue.put((ra[ii], dec[ii]))

            for w in range(n_jobs):
                p = Process(target=self.mp_aperture_photometry, args=(work_queue, out_tab, w), kwargs={
                                                                                        'image_folder_path': image_folder_path,
                                                                                        'radii': radii,
                                                                                        'radius_in': radius_in,
                                                                                        'radius_out': radius_out,
                                                                                        'epoch':epoch})
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            # Merge together the different forced photometry data tables that are generated in each process
            self.save_master_table(n_jobs, table_name=table_name, remove=remove)

        else:
            final_tab = Table()
            for idx, num in enumerate(ra):
                phot_tab = self.aperture_photometry(ra[idx], dec[idx], image_folder_path, radii=radii,
                                                    radius_in=radius_in, radius_out=radius_out,epoch=epoch)
                final_tab = vstack([final_tab, phot_tab])
            final_tab.write(table_name+'_forced_photometry.csv', format='csv', overwrite=True)

    def mp_aperture_photometry(self, work_queue, out_tab, n_jobs, image_folder_path, radii=[1., 1.5],
                               radius_in=7.0, radius_out=10.0, epoch='J'):
        """
        Function that performs forced photometry in parallel by calling the aperture_photometry function
        :param image_folder_path: string, path where the images for which to perform forced photometry are stored
        :param out_tab: table where the data from forced photometry are stored
        :param n_jobs: int, number of forced photometry processes performed in parallel
        :param radii: arcesc, forced photometry aperture radius
        :param radius_in: arcesc, background extraction inner annulus radius
        :param radius_out: arcesc, background extraction outer annulus radius
        :param epoch: string, the epoch that specify the initial letter of the source names
        :return:
        """

        while not work_queue.empty():
            ra, dec = work_queue.get()
            phot = self.aperture_photometry(ra, dec, image_folder_path, radii=radii, radius_in=radius_in,
                                  radius_out=radius_out, epoch=epoch)
            out_tab = vstack([out_tab, phot])
        out_tab.write('process_' + str(n_jobs) + '_forced_photometry.csv', format='csv', overwrite=True)

    def aperture_photometry(self, ra, dec, image_folder_path, radii=[1., 1.5], radius_in=7.0, radius_out=10.0,epoch='J'):
        '''
            Perform forced photometry for a given target on images from a given imaging survey.
            It is calling the aperture_photometry from the photutils package.
            Args:
                ra: right ascension in degrees
                dec: right ascension in degrees
                image_folder_path: string, path where the images for which to perform forced photometry are stored
                :param radii: arcesc, forced photometry aperture radius
                :param radius_in: arcesc, background extraction inner annulus radius
                :param radius_out: arcesc, background extraction outer annulus radius
                :param epoch: string, the epoch that specify the initial letter of the source names

            Returns:
                astropy Table
            '''

        # open an astropy table to store the data
        obj_name = utils.coord_to_name(ra, dec, epoch=epoch)[0]
        photo_table = Table()
        photo_table['Name'] = [obj_name]
        photo_table['RA'] = [ra]
        photo_table['DEC'] = [dec]

        # get the image parameters from the corresponding survey and filter
        filters = self.bands
        survey = imagingsurvey.retrieve_survey(self.name, self.bands, self.fov)

        if survey != None:
            for band in filters:

                try:
                    image_params = survey.data_setup(obj_name,band,image_folder_path)
                    hdr = image_params.hdr
                    data = image_params.data
                    exp = image_params.exp
                    extCorr = image_params.extCorr
                    back = image_params.back
                    zpt = image_params.zpt
                    photo_table['{:}_ZP_{:}'.format(self.name, band)] = [zpt]

                    try:
                        # define WCS
                        wcs_img = wcs.WCS(hdr)
                        # define coordinates and apertures
                        position = SkyCoord(ra * u.deg, dec * u.deg, frame='fk5')
                        aperture = [SkyCircularAperture(position, r=r * u.arcsec) for r in radii]
                        pix_aperture = [aperture[i].to_pixel(wcs_img) for i in range(len(radii))]
                        back_aperture = SkyCircularAnnulus(position, r_in=radius_in * u.arcsec, r_out=radius_out * u.arcsec)

                        # estimate background
                        if back == "no_back":
                            background = np.zeros(len(radii))
                        else:
                            f_back = aperture_photometry(data, back_aperture, wcs=wcs_img)
                            background = [float(f_back['aperture_sum']) / (radius_out ** 2 - radius_in ** 2) * radii[i] ** 2 for i in
                                          range(len(radii))]
                        # compute the std from the whole image
                        ## ToDo: we might need to improve this part (remove source in the image or use variance image): look at Eduardo's method
                        mean, median, std = stats.sigma_clipped_stats(data, sigma=3.0, maxiters=5)

                        # measure the flux
                        f = aperture_photometry(data, aperture, wcs=wcs_img)
                        flux = [float(f['aperture_sum_' + str(i)]) for i in range(len(radii))]
                        # Estimate the SNR
                        SN = [(flux[i] - background[i]) / (std * np.sqrt(pix_aperture[i].area)) for i in range(len(radii))]
                        # Measure fluxes/magnitudes
                        ## ToDo: for VSA/WSA/PS1 forced photometry we compute the native fluxes and the magnitudes in Vega, while we probably want nanomaggies and AB
                        for i in range(len(radii)):
                            radii_namei = str(radii[i] * 2.0).replace('.', 'p')
                            photo_table['{:}_flux_aper_{:}'.format(band, radii_namei)] = (flux[i] - background[i]) / exp
                            photo_table['{:}_flux_aper_err_{:}'.format(band, radii_namei)] = std * np.sqrt(pix_aperture[i].area)\
                                                                                             / exp
                            photo_table['{:}_snr_aper_{:}'.format(band, radii_namei)] = SN[i]
                            if (flux[i] - background[i]) > 0.:
                                photo_table['{:}_mag_aper_{:}'.format(band, radii_namei)] = -2.5 * np.log10(
                                    photo_table['{:}_flux_aper_{:}'.format(band, radii_namei)]) + \
                                                                                            photo_table['{:}_ZP_{:}'.format(self.name, band)]\
                                                                                            - extCorr
                                photo_table['{:}_magaper_err_{:}'.format(band, radii_namei)] = (2.5 / np.log(10)) * (1.0 / SN[i])
                            else:
                                photo_table['{:}_mag_aper_{:}'.format(band, radii_namei)] = np.NAN
                                photo_table['{:}_magaper_err_{:}'.format(band, radii_namei)] = np.NAN
                            print('The {:}-band magnitude is {:}+/-{:}'.format(band, float(
                                photo_table['{:}_mag_aper_{:}'.format(band, radii_namei)]),
                                                                               float(photo_table['{:}_magaper_err_{:}'.format(band,
                                                                                radii_namei)])))
                        photo_table['success_{:}'.format(band)] = 1

                    except:
                        photo_table['{:}_ZP_{:}'.format(self.name, band)] = np.NAN
                        warnings.warn('Photometry on image ' + str(obj_name) + '.' + str(band) + '.fits failed')
                        for i in range(len(radii)):
                            radii_namei = str(radii[i] * 2.0).replace('.', 'p')
                            photo_table['{:}_flux_aper_{:}'.format(band, radii_namei)] = np.NAN
                            photo_table['{:}_flux_aper_err_{:}'.format(band, radii_namei)] = np.NAN
                            photo_table['{:}_snr_aper_{:}'.format(band, radii_namei)] = np.NAN
                            photo_table['{:}_mag_aper_{:}'.format(band, radii_namei)] = np.NAN
                            photo_table['{:}_magaper_err_{:}'.format(band, radii_namei)] = np.NAN
                        photo_table['success_{:}'.format(band)] = 0

                except:
                    warnings.warn('The image {} is corrupted and cannot be opened'.format(obj_name))

        else:
            warnings.warn('The survey {:} is not yet supported!'.format(self.name))

        return photo_table

    def save_master_table(self, n_jobs, table_name, remove=False):
        '''
        Merge the data tables generated during each forced photometry process into a single table
        Args:
            n_jobs: int, number of forced photometry processes performed in parallel
            table_name: string, name of the final output table with all the forced photometry results
            remove: bool, remove
        '''
        pd_table = pd.read_csv('process_' + str(0) + '_forced_photometry.csv')
        master_table = Table.from_pandas(pd_table)

        if remove == True: os.remove('process_' + str(0) + '_forced_photometry.csv')
        for i in range(1, n_jobs):
            pd_table = pd.read_csv('process_' + str(i) + '_forced_photometry.csv')
            par = Table.from_pandas(pd_table)
            master_table = vstack((master_table, par))
            if remove == True: os.remove('process_' + str(i) + '_forced_photometry.csv')
        master_table.write(table_name + '_forced_photometry.csv', format='csv', overwrite=True)