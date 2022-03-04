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

from scipy.optimize import curve_fit

from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy import units as u
from photutils import aperture_photometry, SkyCircularAperture, SkyCircularAnnulus, make_source_mask
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

        :param ra:
        :param dec:
        :param image_folder_path:
        :param n_jobs:
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

        :param image_folder_path:
        :param out_tab:
        :param n_jobs:
        :param radii:
        :param radius_in:
        :param radius_out:
        :param epoch:
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
            It calling the aperture_photometry from the photutils package.
            Args:
                ra:
                dec:
                image_folder_path:
                radii:
                radius_in:
                radius_out:
                epoch:

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
        survey = image.retrieve_survey(self.name, self.bands, self.fov)

        if survey != None:
            for band in filters:

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
                    ## Todo: should we subtract background for PS1 images?
                    if back == "no_back":
                        background = np.zeros(len(radii))
                    else:
                        f_back = aperture_photometry(data, back_aperture, wcs=wcs_img)
                        background = [float(f_back['aperture_sum']) / (radius_out ** 2 - radius_in ** 2) * radii[i] ** 2 for i in
                                      range(len(radii))]
                    # compute the std from the whole image
                    ## ToDo: we need to improve this part (remove source in the image or use variance image): look at Eduardo's method
                    mean, median, std = stats.sigma_clipped_stats(data, sigma=3.0, maxiters=5)
                    #std, mean, empty_flux = self.get_noiseaper(data, aperture_pixel)

                    # measure the flux
                    f = aperture_photometry(data, aperture, wcs=wcs_img)
                    flux = [float(f['aperture_sum_' + str(i)]) for i in range(len(radii))]
                    # Estimate the SNR
                    SN = [(flux[i] - background[i]) / (std * np.sqrt(pix_aperture[i].area)) for i in range(len(radii))]
                    # Measure fluxes/magnitudes
                    ## ToDo: for IR forced photometry we compute the native fluxes and the magnitudes in Vega, while we probably want nanomaggies and AB
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

        else:
            warnings.warn('The survey {:} is not yet supported!'.format(self.name))

        return photo_table

    def save_master_table(self, n_jobs, table_name, remove=False):
        '''
        Merge the different tables with the forced photometry data in one single table
        Args:
            n_jobs:
            table_name:
            remove:
        '''
        pd_table = pd.read_csv('process_' + str(0) + '_forced_photometry.csv')
        master_table = Table.from_pandas(pd_table)
        #master_table = Table(csv.open('process_' + str(0) + '_forced_photometry.csv', memmap=True)[1].data)
        if remove == True: os.remove('process_' + str(0) + '_forced_photometry.csv')
        if n_jobs != 1:
            for i in range(1, n_jobs):
                pd_table = pd.read_csv('process_' + str(i) + '_forced_photometry.csv')
                par = Table.from_pandas(pd_table)
                #par = csv.open('process_' + str(i) + '_forced_photometry.csv')
                #master_table = vstack((master_table, Table(par[1].data)))
                master_table = vstack((master_table, par))
                if remove == True: os.remove('process_' + str(i) + '_forced_photometry.csv')
        master_table.write(table_name + '_forced_photometry.csv', format='csv', overwrite=True)

    def get_noiseaper(self, data, radius):
        # print("estimating noise in aperture: ", radius)
        sources_mask = make_source_mask(data, nsigma=2.5, npixels=3,
                                        dilate_size=15, filter_fwhm=4.5)

        N = 5100
        ny, nx = data.shape
        x1 = np.int(nx * 0.09)
        x2 = np.int(nx * 0.91)
        y1 = np.int(ny * 0.09)
        y2 = np.int(ny * 0.91)
        xx = np.random.uniform(x1, x2, N)
        yy = np.random.uniform(y1, y2, N)

        mask = sources_mask[np.int_(yy), np.int_(xx)]
        xx = xx[~mask]
        yy = yy[~mask]

        positions = list(zip(xx, yy))
        apertures = CircularAperture(positions, r=radius)
        f = aperture_photometry(data, apertures, mask=sources_mask)
        f = np.ma.masked_invalid(f['aperture_sum'])
        m1 = np.isfinite(f)  # & (f!=0)
        empty_fluxes = f[m1]
        emptyapmeanflux, emptyapsigma = self.gaussian_fit_to_histogram(empty_fluxes)

        return emptyapsigma, emptyapmeanflux, empty_fluxes

    def gaussian_fit_to_histogram(self, dataset):
        """ fit a gaussian function to the histogram of the given dataset
        :param dataset: a series of measurements that is presumed to be normally
           distributed, probably around a mean that is close to zero.
        :return: mean, mu and width, sigma of the gaussian model fit.

        Taken from

        https://github.com/djones1040/PythonPhot/blob/master/PythonPhot/photfunctions.py
        """

        def gauss(x, mu, sigma):
            return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

        if np.ndim(dataset) == 2:
            musigma = np.array([gaussian_fit_to_histogram(dataset[:, i])
                                for i in range(np.shape(dataset)[1])])
            return musigma[:, 0], musigma[:, 1]

        dataset = dataset[np.isfinite(dataset)]
        ndatapoints = len(dataset)
        stdmean, stdmedian, stderr, = stats.sigma_clipped_stats(dataset, sigma=5.0)
        nhistbins = max(10, int(ndatapoints / 20))
        histbins = np.linspace(stdmedian - 5 * stderr, stdmedian + 5 * stderr,
                               nhistbins)
        yhist, xhist = np.histogram(dataset, bins=histbins)
        binwidth = np.mean(np.diff(xhist))
        binpeak = float(np.max(yhist))
        param0 = [stdmedian, stderr]  # initial guesses for gaussian mu and sigma
        xval = xhist[:-1] + (binwidth / 2)
        yval = yhist / binpeak
        try:
            minparam, cov = curve_fit(gauss, xval, yval, p0=param0)
        except RuntimeError:
            minparam = -99, -99
        mumin, sigmamin = minparam
        return mumin, sigmamin