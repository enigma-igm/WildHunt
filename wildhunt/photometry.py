#!/usr/bin/env python
"""

Main module for performing forced photometry.

"""
from IPython import  embed

import warnings
import os
import numpy as np
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
                    radius_out=10.0,epoch='J', n_jobs=1, overwrite=True):
        """

        :param ra:
        :param dec:
        :param image_folder_path:
        :param n_jobs:
        :return:
        """

        if n_jobs > 1:
            self.mp_aperture_photometry()
        else:
            final_tab = Table()
            for idx, num in enumerate(ra):
                phot_tab = self.aperture_photometry(ra[idx], dec[idx], image_folder_path, radii=radii,
                                                    radius_in=radius_in, radius_out=radius_out,epoch=epoch)
                final_tab = vstack([final_tab, phot_tab])
            final_tab.write(table_name+'_forced_photometry.csv', format='csv', overwrite=overwrite)

    def mp_aperture_photometry(self,):
        print('TO BE IMPLEMENTED')

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
        for band in filters:
            survey = image.retrieve_survey(self.name, self.bands, self.fov)

            if survey != None:

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

                    # measure the flux
                    f = aperture_photometry(data, aperture, wcs=wcs_img)
                    flux = [float(f['aperture_sum_' + str(i)]) for i in range(len(radii))]
                    # Estimate the SNR
                    SN = [(flux[i] - background[i]) / (std * np.sqrt(pix_aperture[i].area)) for i in range(len(radii))]
                    # Measure fluxes/magnitudes
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