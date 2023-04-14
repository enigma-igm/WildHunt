#!/usr/bin/env python
"""

Main module for downloading and manipulating image data.

"""

import glob
import os

import multiprocessing
from multiprocessing import Process, Queue

import numpy as np

import pandas as pd

from astropy.table import Table, vstack
from astropy import wcs, stats
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord, ICRS
from astropy.wcs.utils import proj_plane_pixel_scales

from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.anchored_artists import (AnchoredSizeBar)

from reproject.mosaicking import find_optimal_celestial_wcs
from reproject import reproject_interp

from photutils import aperture_photometry, SkyCircularAperture, SkyCircularAnnulus

import matplotlib.pyplot as plt

from wildhunt import utils
from wildhunt import pypmsgs
from wildhunt import catalog

msgs = pypmsgs.Messages()


def forced_photometry(ra, dec, survey_dicts, table_name, image_folder_path='cutouts', radii=[1.], radius_in=7.0,
                radius_out=10.0, epoch='J', n_jobs=5, remove=True):
    """Main function that calls all the other functions used to perform forced photometry
    :param ra: np.array(), right ascensions in degrees
    :param dec: np.array(), declinations in degrees
    :param survey_dicts: survey dictionaries
    :param table_name: table where the data from forced photometry are stored
    :param image_folder_path: string, path where the images are stored
    :param radii: arcesc, forced photometry aperture radius
    :param radius_in: arcesc, background extraction inner annulus radius
    :param radius_out: arcesc, background extraction outer annulus radius
    :param epoch: string, the epoch that specify the initial letter of the source names
    :param n_jobs: int, number of forced photometry processes performed in parallel
    :param remove: bool, remove the sub catalogs produced in the multiprocess forced photometry
    """

    n_file = np.size(ra)
    n_cpu = multiprocessing.cpu_count()

    if n_jobs > n_cpu - 1:
        n_jobs = n_cpu - 1
    if n_jobs > n_file:
        n_jobs = n_file

    if n_jobs > 1:
        work_queue = Queue()
        processes = []
        out_tab = Table()
        for ii in range(n_file):
            work_queue.put((ra[ii], dec[ii]))

        for n_job in range(n_jobs):
            p = Process(target=mp_get_forced_photometry, args=(work_queue, out_tab, n_job), kwargs={
                'survey_dicts': survey_dicts,
                'image_folder_path': image_folder_path,
                'radii': radii,
                'radius_in': radius_in,
                'radius_out': radius_out,
                'epoch': epoch})
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        # Merge together the different forced photometry data tables that are generated in each process
        save_master_table(n_jobs, table_name=table_name, remove=remove)

    else:
        final_tab = Table()
        for idx, num in enumerate(ra):
            phot_tab = get_aperture_photometry(ra[idx], dec[idx], survey_dicts, image_folder_path=image_folder_path,
                                           radii=radii, radius_in=radius_in, radius_out=radius_out, epoch=epoch)
            final_tab = vstack([final_tab, phot_tab])
        final_tab.write(table_name + '_forced_photometry.csv', format='csv', overwrite=True)

def mp_get_forced_photometry(work_queue, out_tab, n_jobs, survey_dicts, image_folder_path='cutouts',
                             radii=[1.], radius_in=7.0, radius_out=10.0, epoch='J'):
    ''' Get aperture photometry for one source but all bands/surveys in multiprocess
    :param out_tab: table where the data from forced photometry are stored
    :param n_jobs: int, number of forced photometry processes performed in parallel
    :param survey_dicts: survey dictionaries
    :param image_folder_path: string, path where the images are stored
    :param radii: arcesc, forced photometry aperture radius
    :param radius_in: arcesc, background extraction inner annulus radius
    :param radius_out: arcesc, background extraction outer annulus radius
    :param epoch: string, the epoch that specify the initial letter of the source names
    '''

    while not work_queue.empty():
        ra, dec = work_queue.get()
        phot = get_aperture_photometry(ra, dec, survey_dicts, image_folder_path=image_folder_path, radii=radii,
                                       radius_in=radius_in, radius_out=radius_out, epoch=epoch)
        out_tab = vstack([out_tab, phot])
    out_tab.write('process_' + str(n_jobs) + '_forced_photometry.csv', format='csv', overwrite=True)

def get_aperture_photometry(ra, dec, survey_dicts, image_folder_path='cutouts', radii=[1.], radius_in=7.0,
                            radius_out=10.0, epoch='J'):
    '''Perform forced photometry for a given target on images from a given imaging survey.

    :param ra: right ascension in degrees
    :param dec: right ascension in degrees
    :param survey_dicts: survey dictionaries
    :param image_folder_path: string, path where the images are stored
    :param radii: forced photometry aperture radius in arcsec
    :param radius_in: background extraction inner annulus radius in arcsec
    :param radius_out: background extraction outer annulus radius in arcsec
    :param epoch: string, the epoch that specify the initial letter of the source names

    Returns:
        astropy Table
    '''

    # open an astropy table to store the data
    source_name = utils.coord_to_name(ra, dec, epoch=epoch)[0]
    photo_table = Table()
    photo_table['Name'] = [source_name]
    photo_table['RA'] = [ra]
    photo_table['DEC'] = [dec]

    # Apply aperture photometry to every survey and band specified
    for survey_dict in survey_dicts:

        survey = catalog.retrieve_survey(survey_dict['survey'],
                                 survey_dict['bands'],
                                 survey_dict['fov'])
        for band in survey_dict['bands']:

            try:
                # Open the image
                image_data = Image(ra, dec, survey_dict['survey'], band, image_folder_path, fov=survey_dict['fov'])
                header = image_data.header
                data = image_data.data

                # Retrieve the important info from the header
                filepath = image_folder_path + '/' + source_name + "_" + survey_dict['survey'] + "_" + band \
                           + "_fov" + str(survey_dict['fov']) + ".fits"
                image_params = survey.force_photometry_params(header, band, filepath)
                exp = image_params.exp
                back = image_params.back
                zpt = image_params.zpt
                nanomag_corr = image_params.nanomag_corr
                ABcorr = image_params.ab_corr
                photo_table['{:}_ZP_{:}'.format(survey_dict['survey'], band)] = [zpt]

                if 'w' in band:
                    data = data.copy() * 10 ** (-image_params.ab_corr / 2.5)

                try:
                    # define WCS
                    wcs_img = wcs.WCS(header)
                    # define coordinates and apertures
                    position = SkyCoord(ra * u.deg, dec * u.deg, frame='fk5')
                    aperture = [SkyCircularAperture(position, r=r * u.arcsec) for r in radii]
                    pix_aperture = [aperture[i].to_pixel(wcs_img) for i in range(len(radii))]
                    back_aperture = SkyCircularAnnulus(position, r_in=radius_in * u.arcsec,
                                                       r_out=radius_out * u.arcsec)

                    # estimate background
                    if back == "no_back":
                        background = np.zeros(len(radii))
                    else:
                        f_back = aperture_photometry(data, back_aperture, wcs=wcs_img)
                        background = [
                            float(f_back['aperture_sum']) / (radius_out ** 2 - radius_in ** 2) * radii[i] ** 2 for i
                            in
                            range(len(radii))]
                    # compute the std from the whole image
                    ## ToDo: we might need to improve this part (remove source in the image or use variance image): look at Eduardo's method
                    mean, median, std = stats.sigma_clipped_stats(data, sigma=3.0, maxiters=5)

                    # measure the flux
                    f = aperture_photometry(data, aperture, wcs=wcs_img)
                    flux = [float(f['aperture_sum_' + str(i)]) for i in range(len(radii))]
                    # Estimate the SNR
                    SN = [(flux[i] - background[i]) / (std * np.sqrt(pix_aperture[i].area)) for i in
                          range(len(radii))]
                    # Measure fluxes/magnitudes
                    for i in range(len(radii)):
                        radii_namei = str(radii[i] * 2.0).replace('.', 'p')
                        photo_table['{:}_{:}_flux_aper_{:}'.format(survey_dict['survey'], band, radii_namei)] = \
                            (flux[i] - background[i]) / exp * nanomag_corr
                        photo_table['{:}_{:}_flux_aper_err_{:}'.format(survey_dict['survey'], band, radii_namei)] = \
                            std * np.sqrt(pix_aperture[i].area) / exp * nanomag_corr
                        photo_table['{:}_{:}_snr_aper_{:}'.format(survey_dict['survey'], band, radii_namei)] = SN[i]
                        if (flux[i] - background[i]) > 0.:
                            photo_table['{:}_{:}_mag_aper_{:}'.format(survey_dict['survey'], band, radii_namei)] = 22.5 \
                                -2.5 * np.log10(photo_table['{:}_{:}_flux_aper_{:}'.format(survey_dict['survey'], band,
                                radii_namei)])
                            photo_table['{:}_{:}_magaper_err_{:}'.format(survey_dict['survey'], band, radii_namei)] = \
                                (2.5 / np.log(10)) * (1.0 / SN[i])
                        else:
                            photo_table['{:}_{:}_mag_aper_{:}'.format(survey_dict['survey'], band, radii_namei)] = \
                                np.NAN
                            photo_table['{:}_{:}_magaper_err_{:}'.format(survey_dict['survey'], band, radii_namei)] = \
                                np.NAN
                        msgs.info('The {:}-band magnitude is {:}+/-{:}'.format(band, float(
                            photo_table['{:}_{:}_mag_aper_{:}'.format(survey_dict['survey'], band, radii_namei)]),
                                float(photo_table['{:}_{:}_magaper_err_{:}'.format(survey_dict['survey'], band,
                                                                                         radii_namei)])))
                    photo_table['{:}_success_{:}'.format(survey_dict['survey'], band)] = 1

                except:
                    photo_table['{:}_ZP_{:}'.format(survey_dict['survey'], band)] = np.NAN
                    msgs.warn('Photometry on image ' + str(source_name) + '.' + str(band) + '.fits failed')
                    for i in range(len(radii)):
                        radii_namei = str(radii[i] * 2.0).replace('.', 'p')
                        photo_table['{:}_{:}_flux_aper_{:}'.format(survey_dict['survey'], band, radii_namei)] = np.NAN
                        photo_table['{:}_{:}_flux_aper_err_{:}'.format(survey_dict['survey'], band, radii_namei)] = \
                            np.NAN
                        photo_table['{:}_{:}_snr_aper_{:}'.format(survey_dict['survey'], band, radii_namei)] = np.NAN
                        photo_table['{:}_{:}_mag_aper_{:}'.format(survey_dict['survey'], band, radii_namei)] = np.NAN
                        photo_table['{:}_{:}_magaper_err_{:}'.format(survey_dict['survey'], band, radii_namei)] = np.NAN
                    photo_table['{:}_success_{:}'.format(survey_dict['survey'], band)] = 0

            except:
                msgs.error('The image {} is corrupted and cannot be opened'.format(source_name))

    return photo_table

def save_master_table(n_jobs, table_name, remove=True):
    '''Merge the data tables generated during each forced photometry process into a single table

    :param n_jobs: int, number of forced photometry processes performed in parallel
    :param table_name: table where the data from forced photometry are stored
    :param remove: bool, remove the sub catalogs produced in the multiprocess forced photometry
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

class Image(object):

    def __init__(self, ra, dec, survey, band, image_folder_path, fov=120,
                 data=None, header=None):
        self.source_name = utils.coord_to_name(np.array([ra]),
                                               np.array([dec]),
                                               epoch="J")[0]
        self.survey = survey
        self.band = band
        self.ra = ra
        self.dec = dec
        self.image_folder_path = image_folder_path
        self.fov = fov

        self.data = data
        self.header = header

        # Open image
        if data is None and header is None:
            self.open()

    def open(self):
        """Open the image fits file.

        :return: None
        """

        # Filepath
        filepath = self.image_folder_path + '/' + self.source_name + "_" + \
                   self.survey + "_" + self.band + "*fov*.fits"

        filenames_available = glob.glob(filepath)
        file_found = False
        open_file_fov = None
        file_path = None
        if len(filenames_available) > 0:
            for filename in filenames_available:
                print(filename)

                try:
                    file_fov = int(filename.split("_")[3].split(".")[0][3:])
                except:
                    file_fov = 9999999

                if self.fov <= file_fov:
                    # hdul = fits.open(filename)
                    # data = hdul[1].data
                    # hdr = hdul[1].header

                    data, hdr = fits.getdata(filename, header=True)
                    file_found = True
                    file_path = filename
                    open_file_fov = file_fov

        if file_found:
            msgs.info("Opened {} with a fov of {} "
                      "arcseconds".format(file_path, open_file_fov))

            self.data = data
            self.header = hdr

        else:
            msgs.error("{} {}-band image of source {} in\ folder {} not "
                      "found.".format(self.survey, self.band,
                                      self.source_name,
                                      self.image_folder_path))

    def show(self, fov=None, n_sigma=3, color_map='viridis'):
        """ Show the image data.

        :param fov: Field of view
        :type: float
        :param n_sigma: Number of sigma for image color scale sigma clipping.
        :type n_sigma: int
        :param color_map: Matplotlib color map
        :type color_map:
        :return:
        """
        fig = plt.figure(figsize=(5, 5))

        subplot = 111

        # self._plot_axis(fig, subplot, fov=fov, n_sigma=n_sigma,
        #                color_map=color_map)

        self._simple_plot(fov, n_sigma=n_sigma, fig=fig, subplot=subplot,
                          color_map=color_map, north=True,)

        plt.show()

    # def _plot_axis(self, fig, subplot, fov=None, n_sigma=3,
    #               color_map='viridis'):
    #     """Plot image axis on input figure.
    #
    #     Class internal plotting routine.
    #
    #     :param fig: Input figure
    #     :type fig:
    #     :param subplot: Subplot touple (e.g., "(1,1,1)")
    #     :type subplot tuple
    #     :param fov: Field of view
    #     :type: float
    #     :param n_sigma: Number of sigma for image color scale sigma clipping.
    #     :type n_sigma: int
    #     :param color_map: Matplotlib color map
    #     :type color_map:
    #     :return:
    #     """
    #
    #     if fov is not None:
    #         cutout_data = self._get_cutout(fov=fov)
    #     else:
    #         cutout_data = None
    #
    #     if cutout_data is not None:
    #         img_data = cutout_data
    #     else:
    #         img_data = self.data
    #
    #     hdu = fits.ImageHDU(data=img_data, header=self.header)
    #
    #     axs = aplpy.FITSFigure(hdu, figure=fig,
    #                            subplot=subplot,
    #                            north=True)
    #
    #     # Sigma-clipping of the color scale
    #     mean = np.mean(img_data[~np.isnan(img_data)])
    #     std = np.std(img_data[~np.isnan(img_data)])
    #     upp_lim = mean + n_sigma * std
    #     low_lim = mean - n_sigma * std
    #     axs.show_colorscale(vmin=low_lim, vmax=upp_lim,
    #                         cmap=color_map)
    #
    #     axs.set_title(self.survey+' '+self.band)
    #
    #     return axs

    def _rotate_north_up(self):

        # Get image WCS
        wcs = WCS(self.header)

        frame = ICRS()

        new_wcs, shape = find_optimal_celestial_wcs([(self.data,
                                                   wcs)],
                                                 frame=frame)

        data, _ = reproject_interp((self.data, wcs), new_wcs,
                                             shape_out=shape)
        header = new_wcs.to_header()
        header['NAXIS1'] = shape[1]
        header['NAXIS2'] = shape[0]

        self.header = header
        self.data = data

    def _simple_plot(self, fov, n_sigma=3, fig=None, subplot=None,
                     color_map='viridis', axis=None, north=False,
                     scalebar=5*u.arcsecond, sb_pad=0.5, sb_borderpad=0.4,
                     corner='lower right', frameon=False, low_lim=None,
                     upp_lim=None, logscale=False):


        if north:
            self._rotate_north_up()

        if fov is not None:
            cutout_data = self._get_cutout(fov=fov)
        else:
            cutout_data = None

        if cutout_data is not None:
            img_data = cutout_data
        else:
            img_data = self.data

        wcs = WCS(self.header)

        axs = None
        if axis is None and fig is not None and subplot is not None:
            axs = fig.add_subplot(subplot, projection=wcs)
        elif axis is not None:
            axs = axis
        else:
            msgs.error('Neither figure and subplot tuple or figure axis '
                       'provided.')


        if isinstance(upp_lim, float) and isinstance(low_lim, float):
            msgs.info('Using user defined color scale limits.')
        else:
            msgs.info('Determining color scale limits by sigma clipping.')
            # Sigma-clipping of the color scale
            mean = np.mean(img_data[~np.isnan(img_data)])
            std = np.std(img_data[~np.isnan(img_data)])
            upp_lim = mean + n_sigma * std
            low_lim = mean - n_sigma * std

        if logscale:
            # To avoid np.NaN for negative flux values in the logNorm
            # conversion the absolute value of the minimum flux value will
            # be added for display purposes only.
            mod_img_data = img_data + abs(np.nanmin(img_data))

            axs.imshow(mod_img_data, origin='lower',
                       cmap=color_map,
                       norm=LogNorm()
                       )
        else:
            axs.imshow(img_data, origin='lower',
                       vmin=low_lim,
                       vmax=upp_lim,
                       cmap=color_map,
                       )


        if scalebar is not None:
            if isinstance(scalebar, u.Quantity):
                length = scalebar.to(u.degree).value
            elif isinstance(scalebar, u.Unit):
                length = scalebar.to(u.degree)

            self._add_scalebar(axs, length, pad=sb_pad,
                               borderpad=sb_borderpad,
                               corner=corner,
                               frameon=frameon)

        return axs

    def _add_scalebar(self, axis, length, corner='lower right',
                      pad=0.5, borderpad=0.4, frameon=False):
        # Code adapted from Aplpy

        pix_scale = proj_plane_pixel_scales(WCS(self.header))

        sx = pix_scale[0]
        sy = pix_scale[1]
        degrees_per_pixel = np.sqrt(sx * sy)

        label = '{:.1f} arcsec'.format(length*3600)

        length = length / degrees_per_pixel

        size_vertical = length/20



        artist = AnchoredSizeBar(axis.transData, length, label,
                                 corner, pad=pad, borderpad=borderpad,
                                 size_vertical=size_vertical,
                                 sep=3, frameon=frameon,
                                 fontproperties={'size': 15, 'weight': 'bold'})

        axis.add_artist(artist)

    def _get_cutout(self, fov):
        """Create a cutout from the image with a given field of view (fov)

        :param fov: Field of view for the cutout
        :type fov: float
        :return: Cutout data array
        """

        wcs_img = wcs.WCS(self.header)

        pixcrd = wcs_img.wcs_world2pix(self.ra, self.dec, 0)
        positions = (np.float(pixcrd[0]), np.float(pixcrd[1]))

        try:
            cutout_data = Cutout2D(self.data, positions, size=fov * u.arcsec,
                                   wcs=wcs_img).data
        except:
            msgs.warn("Source not in image.")
            cutout_data = None

        return cutout_data


    def get_cutout_image(self, ra, dec, fov):


        wcs_img = wcs.WCS(self.header)

        pixcrd = wcs_img.wcs_world2pix(ra, dec, 0)
        positions = (np.float(pixcrd[0]), np.float(pixcrd[1]))

        try:
            cutout = Cutout2D(self.data, positions, size=fov * u.arcsec,
                                   wcs=wcs_img, copy=True)

            header = self.header.copy()
            # Update header wcs for cutout
            header.update(cutout.wcs.to_header())

            cutout_image = Image(ra, dec, self.survey, self.band,
                                 self.image_folder_path, fov=fov,
                                 data=cutout.data, header=header)

            msgs.info("Returning generated cutout")

            return cutout_image

        except:
            msgs.warn("Cutout generation failed.")

            return None

    def calculate_aperture_photometry(self, ra, dec,
                                      aperture_radii=[1.],
                                      background_aperture=[7., 10.],
                                      ref_frame='icrs',
                                      background=True):

        img_wcs = wcs.WCS(self.header)
        img_data = self.data

        source_position = SkyCoord(ra * u.deg, dec * u.deg, frame=ref_frame)

        # Initialize source aperture
        aperture = [SkyCircularAperture(
            source_position, r=r * u.arcsec) for r in aperture_radii]
        pix_aperture = [aperture[i].to_pixel(img_wcs) for i in
                        range(len(aperture_radii))]

        # Initialize background aperture
        back_aperture = SkyCircularAnnulus(source_position,
                                           r_in=background_aperture[0] *
                                                u.arcsec,
                                           r_out=background_aperture[1] *
                                                 u.arcsec)
        # Calculate background flux level
        if not background:
            background = np.zeros(len(aperture_radii))
        else:
            background_flux = aperture_photometry(img_data,
                                         back_aperture,
                                         wcs=img_wcs)

            background_area = background_aperture[1] ** 2 - \
                              background_aperture[0] ** 2
            background_diff = background_area * aperture_radii ** 2
            background = [
                float(background_flux['aperture_sum']) / background_diff[i]
                for i in range(len(aperture_radii))]

        # TODO: Implement std calculation instead of background flux estimation
        mean, median, std = stats.sigma_clipped_stats(img_data, sigma=3.0,
                                                      maxiters=5)

        # Measure the source flux
        source_flux = aperture_photometry(img_data, aperture, wcs=img_wcs)
        flux = [float(source_flux['aperture_sum_' + str(i)]) for i in range(
            len(aperture_radii))]

        # Estimate the SNR
        snr = [(flux[i] - background[i]) /
               (std * np.sqrt(pix_aperture[i].area)) for i in
              range(len(aperture_radii))]

        return flux, snr






