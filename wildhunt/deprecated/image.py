#!/usr/bin/env python
"""

Main module for downloading and manipulating image data.

"""

import os
import glob
import math
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
from astropy.visualization import ZScaleInterval

from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from reproject.mosaicking import find_optimal_celestial_wcs
from reproject import reproject_interp

from photutils import aperture_photometry, SkyCircularAperture,\
    SkyCircularAnnulus

import matplotlib.pyplot as plt

from wildhunt import utils
from wildhunt import pypmsgs
from wildhunt import catalog

from IPython import embed

msgs = pypmsgs.Messages()


def forced_photometry(ra, dec, survey_dicts, table_name,
                      image_folder_path='cutouts', radii=[1.], radius_in=7.0,
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

def get_aperture_photometry(ra, dec, survey_dicts, image_folder_path='cutouts',
                            radii=[1.], radius_in=7.0, radius_out=10.0,
                            epoch='J'):
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
                image_data = SurveyImage(ra, dec, survey_dict['survey'], band,
                                    image_folder_path, min_fov=survey_dict[
                        'fov'])
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
                ABcorr = image_params.ABcorr
                photo_table['{:}_ZP_{:}'.format(survey_dict['survey'], band)] = [zpt]

                if 'w' in band:
                    data = data.copy() * 10 ** (-image_params.ABcorr / 2.5)

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


def get_pixelscale(hdr):
    '''
    Get pixelscale from header and return in it in arcsec/pixel
    '''

    wcs_img = wcs.WCS(hdr)
    scale = np.mean(proj_plane_pixel_scales(wcs_img)) * 3600

    return scale

def aperture_inpixels(aperture, hdr):
    '''
    receives aperture in arcsec. Returns aperture in pixels
    '''
    pixelscale = get_pixelscale(hdr)
    aperture /= pixelscale #pixels

    return aperture


def make_mult_png_fig(ra, dec, surveys, bands,
                      fovs, apertures, square_sizes, image_dir, mag_list=None,
                      magerr_list=None, sn_list=None,
                      forced_mag_list=None, forced_magerr_list=None,
                      forced_sn_list=None, n_col=3,
                      n_sigma=3, color_map_name='viridis',
                      scalebar=5 * u.arcsecond,
                      add_info_label=None, add_info_value=None):
    """Create a figure to plot cutouts for one source in all specified surveys
    and bands.

    :param ra: float
        Right Ascension of the target
    :param dec: float
        Declination of the target
     :param surveys: list of strings
        List of survey names, length has to be equal to bands and fovs
    :param bands: list of strings
        List of band names, length has to be equal to surveys and fovs
    :param fovs: list of floats
        Field of view in arcseconds of image cutouts, length has be equal to
        surveys, bands and apertures.
    :param apertures: list of floats
        List of apertures in arcseconds for forced photometry calculated,
        length has to be equal to surveys, bands and fovs
    :param square_sizes: list of floats
        List of
    :param image_dir: string
        Path to the directory where all the images are be stored
    :param mag_list: list of floats
        List of magnitudes for each survey/band
    :param magerr_list: list of floats
         List of magnitude errors for each survey/band
    :param sn_list: list of floats
         List of S/N for each survey/band
    :param forced_mag_list: list of floats
         List of forced magnitudes for each survey/band
    :param forced_magerr_list: list of floats
        List of forced magnitude errors for each survey/band
    :param forced_sn_list: list of floats
        List of forced S/N for each survey/band
    :param n_col: int
        Number of columns
    :param n_sigma: int
        Number of sigmas for the sigma-clipping routine that creates the
        boundaries for the color map.
    :param color_map_name: string
        Name of the color map
    :param add_info_value : string
        Value for additional information added to the title of the figure
    :param add_info_label : string
        Label for additional information added to the title of the figure
    :param verbosity:
        Verbosity > 0 will print verbose statements during the execution
    :return: matplotlib.figure
        Figure with the plot.
    """

    n_images = len(surveys)

    n_row = int(math.ceil(n_images / n_col))

    fig = plt.figure(figsize=(5*n_col, 5*n_row))

    fig = _make_mult_png_axes(fig, n_row, n_col, ra, dec, surveys, bands,
                              fovs, apertures, square_sizes, image_dir, mag_list,
                              magerr_list, sn_list,
                              forced_mag_list, forced_magerr_list,
                              forced_sn_list, scalebar, n_sigma,
                              color_map_name)

    coord_name = utils.coord_to_name(np.array([ra]),
                                  np.array([dec]),
                                  epoch="J")

    if add_info_label is None or add_info_value is None:
        fig.suptitle(coord_name[0])
    else:
        fig.suptitle(coord_name[0]+' '+add_info_label+'='+add_info_value)

    return fig


def _make_mult_png_axes(fig, n_row, n_col, ra, dec, surveys, bands,
                        fovs, apertures, square_sizes, image_dir, mag_list=None,
                        magerr_list=None, sn_list=None,
                        forced_mag_list=None, forced_magerr_list=None,
                        forced_sn_list=None, scalebar=5 * u.arcsecond,
                        n_sigma=3, color_map_name='viridis', show_axes=True):
    """ Create axes components to plot one source in all specified surveys
    and bands.

    :param fig: matplotlib.figure
        Figure
    :param n_row: int
        Number of rows
    :param n_col: int
        Number of columns
     :param ra: float
        Right Ascension of the target
    :param dec: float
        Declination of the target
     :param surveys: list of strings
        List of survey names, length has to be equal to bands and fovs
    :param bands: list of strings
        List of band names, length has to be equal to surveys and fovs
    :param fovs: list of floats
        Field of view in arcseconds of image cutouts, length has be equal to
        surveys, bands and apertures.
    :param apertures: list of floats
        List of apertures in arcseconds for forced photometry calculated,
        length has to be equal to surveys, bands and fovs
    :param square_sizes: list of floats
        List of
    :param image_dir: string
        Path to the directory where all the images are be stored
    :param mag_list: list of floats
        List of magnitudes for each survey/band
    :param magerr_list: list of floats
         List of magnitude errors for each survey/band
    :param sn_list: list of floats
         List of S/N for each survey/band
    :param forced_mag_list: list of floats
         List of forced magnitudes for each survey/band
    :param forced_magerr_list: list of floats
        List of forced magnitude errors for each survey/band
    :param forced_sn_list: list of floats
        List of forced S/N for each survey/band
    :param n_col: int
        Number of columns
    :param n_sigma: int
        Number of sigmas for the sigma-clipping routine that creates the
        boundaries for the color map.
    :param color_map_name: string
        Name of the color map
    :param verbosity:
        Verbosity > 0 will print verbose statements during the execution
    :return: matplotlib.figure
        Figure with the plot.
    """

    for idx, survey in enumerate(surveys):
        band = bands[idx]
        fov = fovs[idx]
        aperture = apertures[idx]
        size = square_sizes[idx]

        if mag_list is not None:
            catmag = mag_list[idx]
        else:
            catmag = None
        if magerr_list is not None:
            caterr = magerr_list[idx]
        else:
            caterr = None
        if sn_list is not None:
            catsn = sn_list[idx]
        else:
            catsn = None
        if forced_mag_list is not None:
            forced_mag = forced_mag_list[idx]
        else:
            forced_mag = None
        if forced_magerr_list is not None:
            forced_magerr = forced_magerr_list[idx]
        else:
            forced_magerr = None
        if forced_sn_list is not None:
            forced_sn = forced_sn_list[idx]
        else:
            forced_sn = None


        image = SurveyImage(ra, dec, survey, band, image_dir, min_fov=fov)

        cutout = image.get_cutout_image(ra, dec, fov)
        wcs = WCS(cutout.header)

        axs = fig.add_subplot(int(f"{n_row}{n_col}{idx + 1}"), projection=wcs)

        axs = cutout._simple_plot(n_sigma=n_sigma, fig=fig,
                                 subplot=None,
                                 color_map=color_map_name, axis=axs,
                                 north=True,
                                 scalebar=scalebar, sb_pad=0.5,
                                 sb_borderpad=0.4,
                                 corner='lower right', frameon=False,
                                 low_lim=None,
                                 upp_lim=None, logscale=False)

        axs.get_xaxis().set_visible(False)
        axs.get_yaxis().set_visible(False)

        # Plot circular aperture (forced photometry flux)
        circx, circy = wcs.wcs_world2pix(ra, dec, 0)

        aper_pix = aperture_inpixels(aperture, cutout.header)
        circle = plt.Circle((circx, circy), aper_pix, color='r', fill=False,
                            lw=1.5)
        fig.gca().add_artist(circle)

        # Plot rectangular aperture (error region)
        rect_inpixels = aperture_inpixels(size, cutout.header)
        square = plt.Rectangle((circx - rect_inpixels * 0.5,
                                circy - rect_inpixels * 0.5),
                               rect_inpixels, rect_inpixels,
                               color='r', fill=False, lw=1.5)
        fig.gca().add_artist(square)

        # Create forced photometry label
        if (forced_mag is not None):
            if (forced_sn is not None) & (forced_magerr is not None):
                forcedlabel = r'${0:s} = {1:.2f} \pm {2:.2f} (SN=' \
                              r'{3:.1f})$'.format(band + "_{forced}",
                                                  forced_mag,
                                                  forced_magerr,
                                                  forced_sn)
            elif forced_magerr is not None:
                forcedlabel = r'${0:s} = {1:.2f} \pm {2:.2f}$'.format(
                    band + "_{forced}", forced_mag, forced_magerr)
            else:
                forcedlabel = r'${0:s} = {1:.2f}$'.format(
                    band + "_{forced}", forced_mag)

            fig.gca().text(0.03, 0.16, forcedlabel, color='black',
                     weight='bold', fontsize='large',
                     bbox=dict(facecolor='white', alpha=0.6),
                     transform=fig.gca().transAxes)

        # Create catalog magnitude label
        if catmag is not None:
            if (catsn is not None) & (caterr is not None):
                maglabel = r'${0:s} = {1:.2f} \pm {2:.2f}  (SN=' \
                           r'{3:.2f})$'.format(
                    band + "_{cat}", catmag, caterr, catsn)
            elif caterr is not None:
                maglabel = r'${0:s} = {1:.2f} \pm {2:.2f}$'.format(
                    band + "_{cat}", catmag, caterr)
            else:
                maglabel = r'${0:s} = {1:.2f}$'.format(
                    band + "_{cat}", catmag)

            fig.gca().text(0.03, 0.04, maglabel, color='black',
                     weight='bold',
                     fontsize='large',
                     bbox=dict(facecolor='white', alpha=0.6),
                     transform=fig.gca().transAxes)

        fig.gca().set_title(survey + " " + band)

    return fig




class Image(object):


    def __init__(self, filename=None, data=None, header=None, exten=0,
                 fov=None, ra=None, dec=None, verbosity=1,
                 instantiate_empty=False):

        self.verbosity = verbosity

        if filename is not None and data is None and header is None:
            hdul = fits.open(filename)
            self.header = hdul[exten].header
            self.data = hdul[exten].data

        elif filename is None and data is not None and header is not None:
            self.data = data
            self.header = header

        else:
            if not instantiate_empty:
                msgs.error('You need to specify either a filename or the data '
                           'and header of your image.')
                raise ValueError()

            else:
                msgs.warn('Instantiating an empty Image object. Dangerous!')
                self.data = None
                self.header = None

        if self.data is not None:
            if ra is None or dec is None:
                ra, dec = self.get_central_coordinates_from_wcs()
                self.ra = ra
                self.dec = dec

            elif ra is not None and dec is not None:
                self.ra = ra
                self.dec = dec

        self.fov = fov   # Field of view

        self.aperture = None  # Photutils aperture for aperture photometry



    def get_central_coordinates_from_wcs(self):

        naxis_1 = self.header['NAXIS1']
        naxis_2 = self.header['NAXIS2']

        wcs_img = wcs.WCS(self.header)

        coord = wcs_img.wcs_pix2world(int(naxis_1/2), int(naxis_2/2), 0)

        ra = float(coord[0])
        dec = float(coord[1])

        return ra, dec

    def show(self, n_sigma=3, color_map='viridis'):
        """ Show the image data.

        :param n_sigma: Number of sigma for image color scale sigma clipping.
        :type n_sigma: int
        :param color_map: Matplotlib color map
        :type color_map:
        :return:
        """
        fig = plt.figure(figsize=(5, 5))

        subplot = 111

        self._simple_plot(n_sigma=n_sigma, fig=fig, subplot=subplot,
                          color_map=color_map, north=True,)

        plt.show()


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


    def _simple_plot(self, n_sigma=3, fig=None, subplot=None,
                     color_map='viridis', axis=None, north=False,
                     scalebar=5*u.arcsecond, sb_pad=0.5, sb_borderpad=0.4,
                     corner='lower right', frameon=False, low_lim=None,
                     upp_lim=None, logscale=False):
        """ Simple plot function for the image data.

        :param n_sigma: The number of sigma for the color scale.
        :type n_sigma: int
        :param fig: A matplotlib figure, which will be used to plot the
         image on
        :type fig: matplotlib.figure.Figure
        :param subplot: The subplot number.
        :type subplot: int
        :param color_map: The matplotlib color map.
        :type color_map: str
        :param axis: A matplotlib axis, which can be used as an alternative
         to plot the image on.
        :type axis: matplotlib.axes._subplots.AxesSubplot
        :param north: A boolean to indicate whether to rotate the image
         north up.
        :type north: bool
        :param scalebar: The scalebar length in arcseconds.
        :type scalebar: astropy.units.quantity.Quantity
        :param sb_pad: The scalebar padding in fraction of font size.
        :type sb_pad: float
        :param sb_borderpad: The scalebar border padding in fraction of the
         font size.
        :type sb_borderpad: float
        :param corner: The scalebar corner position.
        :type corner: str
        :param frameon: A boolean to indicate whether to add a frame around the
         scalebar.
        :type frameon: bool
        :param low_lim: The lower limit of the color scale. This overrides
         the n_sigma parameter.
        :type low_lim: float
        :param upp_lim: The upper limit of the color scale. This overrides
         the n_sigma parameter.
        :type upp_lim: float
        :param logscale: A boolean to indicate whether to use a log scale
        for the color scale.
        :type logscale: bool
        :return: matplotlib axis
        :rtype: matplotlib.axes._subplots.AxesSubplot
        """

        if north:
            self._rotate_north_up()

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
            mean = np.mean(self.data[~np.isnan(self.data)])
            std = np.std(self.data[~np.isnan(self.data)])
            upp_lim = mean + n_sigma * std
            low_lim = mean - n_sigma * std

        if logscale:
            # To avoid np.NaN for negative flux values in the logNorm
            # conversion the absolute value of the minimum flux value will
            # be added for display purposes only.
            mod_img_data = self.data + abs(np.nanmin(self.data))

            axs.imshow(mod_img_data, origin='lower',
                       cmap=color_map,
                       norm=LogNorm()
                       )
        else:
            axs.imshow(self.data, origin='lower',
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
        positions = (np.float64(pixcrd[0]), np.float64(pixcrd[1]))

        try:
            cutout_data = Cutout2D(self.data, positions, size=fov * u.arcsec,
                                   wcs=wcs_img).data
        except:
            msgs.warn("Source not in image.")
            cutout_data = None

        return cutout_data

    def get_cutout_image(self, ra, dec, fov, survey=None, band=None,
                         cutout_dir=None, save=False):
        """ Create a cutout image from the image.

        This function returns the cutout image as an image.Image object
        unless save has been set to True. In this case the cutout image will
        be saved to the specified cutout_dir directory using the survey and
        filter name as part of the file name.

        :param ra: The right ascension of the center of the cutout.
        :type ra: float
        :param dec: The declination of the center of the cutout.
        :type dec: float
        :param fov: The field of view of the cutout in arcseconds.
        :type fov: float
        :param survey: The survey name to use for the cutout.
        :type survey: str
        :param band: The filter band to use  for the cutout.
        :type band: str
        :param cutout_dir: The path to the directory to save the cutout.
        :type cutout_dir: str
        :param save: Boolean to indicate whether to save the cutout image.
        :type save: bool
        :return: Returns the cutout image as an image.Image object (
        save=False) or None (save=True).
        :rtype: image.Image
        """

        wcs_img = wcs.WCS(self.header)

        pixcrd = wcs_img.wcs_world2pix(ra, dec, 0)
        positions = (np.float64(pixcrd[0]), np.float64(pixcrd[1]))

        try:
            cutout = Cutout2D(self.data, positions, size=fov * u.arcsec,
                              wcs=wcs_img, copy=True)
        except:
            msgs.warn('Source not in image.')
            return None

        header = self.header.copy()

        # Update header wcs for cutout
        header.update(cutout.wcs.to_header())

        cutout_image = Image(data=cutout.data, header=header)

        # Save cutout image to cutout_dir
        if save:
            source_name = utils.coord_to_name(np.array([ra]),
                                              np.array([dec]),
                                              epoch="J")[0]

            cutout_path = cutout_dir + '/' + source_name + "_" + \
                          survey + "_" + band + "_fov{}.fits".format(fov)

            # Save image
            cutout_image.to_fits(cutout_path)
            if self.verbosity > 0:
                msgs.info("Cutout save to file")

            return None

        # Returning the cutout image as an Image object
        else:
            if self.verbosity > 0:
                msgs.info("Returning generated cutout")
            return cutout_image

    def to_fits(self, filename, overwrite=True):

        hdu = fits.PrimaryHDU(self.data, header=self.header)
        hdu.writeto(filename, overwrite=overwrite)

    def calculate_aperture_photometry(self, ra, dec,
                                      zero_point=None,
                                      nanomag_correction=None,
                                      ab_correction=None,
                                      exptime=1,
                                      aperture_radii=np.array([1.]),
                                      background_aperture=np.array([7., 10.]),
                                      ref_frame='icrs',
                                      background=True,
                                      band=None, survey=None):

        # Package all results into a dictionary
        if band is None:
            if self.band is None:
                msgs.warn('No filter band information provided. Setting to '
                          '"band" ')
                band = 'band'
            else:
                band = self.band

        if survey is None:
            if self.survey is None:
                msgs.warn('No survey information provided. Setting to '
                          '"survey" ')
                survey = 'survey'
            else:
                survey = self.survey

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
            background_flux = aperture_photometry(img_data, back_aperture,
                                                  wcs=img_wcs)

            background_area = background_aperture[1] ** 2 - \
                              background_aperture[0] ** 2
            background_diff = background_area * np.power(aperture_radii, 2)
            background = [
                float(background_flux['aperture_sum']) / background_diff[i]
                for i in range(len(aperture_radii))]

        # TODO: Implement std calculation instead of background flux estimation
        mean, median, std = stats.sigma_clipped_stats(img_data, sigma=3.0,
                                                      maxiters=5)

        # Measure the source flux
        source_flux = aperture_photometry(img_data, aperture, wcs=img_wcs)

        # Convert fluxes physical units (nanomaggy)
        # This includes exposure time corrections as needed for some surveys
        # (e.g., PS1)

        flux_list = []
        flux_err_list = []
        snr_list = []
        mag_list = []
        mag_err_list = []

        result_dict = {}

        for idx in range(len(aperture_radii)):

            # Calculate the flux in nanomaggies
            flux = float(source_flux['aperture_sum_' + str(idx)])
            flux = (flux - background[idx]) / exptime * nanomag_correction
            # Calculate the flux error in nanomaggies
            flux_err = std * np.sqrt(pix_aperture[idx].area) / exptime * \
                      nanomag_correction
            # Estimate the SNR
            snr = flux/flux_err

            # Calculate the magnitude for positive fluxes
            if flux > 0:
                mag = 22.5 - 2.5 * np.log10(flux)
                mag_err = (2.5 / np.log(10)) / snr
            else:
                mag = np.NaN
                mag_err = np.NaN

            flux_list.append(flux)
            flux_err_list.append(flux_err)
            snr_list.append(snr)
            mag_list.append(mag)
            mag_err_list.append(mag_err)

            survey_band = '{}_{}'.format(survey, band)

            flux_name = '{}_flux_aper_{}arcsec'.format(
                survey_band, aperture_radii[idx])
            result_dict.update({flux_name: flux})

            flux_err_name = '{}_flux_err_aper_{}arcsec'.format(
                survey_band, aperture_radii[idx])
            result_dict.update({flux_err_name: flux_err})

            snr_name = '{}_snr_aper_{}arcsec'.format(
                survey_band, aperture_radii[idx])
            result_dict.update({snr_name: snr})

            mag_name = '{}_mag_aper_{}arcsec'.format(
                survey_band, aperture_radii[idx])
            result_dict.update({mag_name: mag})

            mag_err_name = '{}_mag_err_aper_{}arcsec'.format(
                survey_band, aperture_radii[idx])
            result_dict.update({mag_err_name: mag_err})

        # Add the zero point to the result dictionary
        result_dict.update({'{}_zp'.format(survey_band): zero_point})

        # Add the AB correction to the result dictionary
        result_dict.update({'{}_ab'.format(survey_band): ab_correction})

        # Add a status flag to the result dictionary
        result_dict.update({'{}_status'.format(survey_band): 'success'})

        return result_dict


    def forced_aperture_photometry(self):

        pass

    def get_coordinate_bounds(self):
        """Get the RA and Dec minimum and maximum values for an image.

        :return: The minimum and maximum values in RA and Dec for the image.
        :rtype: tuple
        """

        image_wcs = WCS(self.header)

        img_ra_bounds = []
        img_dec_bounds = []
        for x in [0, self.data.shape[1] - 1]:
            for y in [0, self.data.shape[0] - 1]:
                ra, dec = image_wcs.wcs_pix2world(x, y, 0)
                img_ra_bounds.append(ra)
                img_dec_bounds.append(dec)

        return np.min(img_ra_bounds), np.max(img_ra_bounds), \
               np.min(img_dec_bounds), np.max(img_dec_bounds)




class SurveyImage(Image):


    def __init__(self, ra, dec, survey, band, image_dir, min_fov,
                 data=None, header=None, verbosity=1,
                 instantiate_empty=False):

        self.ra = ra
        self.dec = dec
        self.survey = survey
        self.band = band
        self.image_dir = image_dir
        self.fov = min_fov
        self.verbosity = verbosity

        self.source_name = utils.coord_to_name(np.array([ra]),
                                               np.array([dec]),
                                               epoch="J")[0]

        if data is None and header is None:
            msgs.info('Trying to open from image directory')
            data, header = self.open()
        elif data is not None and header is not None:
            msgs.info('User supplied image header and data')

        super(SurveyImage, self).__init__(data=data, header=header,
                                          fov=min_fov,
                                          instantiate_empty=instantiate_empty)

    def open(self):
        """Open the survey image fits file

        :return: None
        """

        # Filepath
        filepath = self.image_dir + '/' + self.source_name + "_" + \
                   self.survey + "_" + self.band + "*fov*.fits"

        filenames_available = glob.glob(filepath)
        file_found = False
        open_file_fov = None
        file_path = None
        if len(filenames_available) > 0:
            for filename in filenames_available:

                try:
                    file_fov = int(filename.split("_")[3].split(".")[0][3:])
                except:
                    file_fov = 9999999

                if self.fov <= file_fov:

                    data, header = fits.getdata(filename, header=True)
                    file_found = True
                    file_path = filename
                    open_file_fov = file_fov

        if file_found:
            msgs.info("Opened {} with a fov of {} "
                      "arcseconds".format(file_path, open_file_fov))

            return data, header

        else:
            msgs.warn("{} {}-band image of source {} with a minimum FOV of "
                       "{} in folder {} not found.".format(self.survey, self.band,
                                       self.source_name, self.fov,
                                       self.image_dir))

            return None, None

    def get_aperture_photometry(self, aperture_radii=np.array([1.]),
                                background_aperture=np.array([7., 10.]),
                                ref_frame='icrs'):

        survey = catalog.retrieve_survey(self.survey,
                                         self.band,
                                         self.fov)

        filepath = self.image_dir + '/' + self.source_name + "_" + \
                   self.survey + "_" + self.band + "_fov{}.fits".format(
                   self.fov)

        # Retrieve survey specific information
        image_params = survey.force_photometry_params(self.header,
                                                      self.band,
                                                      filepath)
        exptime = image_params.exp
        back = image_params.back
        zero_point = image_params.zpt
        nanomag_corr = image_params.nanomag_corr
        ab_correction = image_params.ABcorr

        if back == 'no_back':
            background = False
        else:
            background = True

        result = self.calculate_aperture_photometry(
            self.ra, self.dec, zero_point=zero_point,
            nanomag_correction=nanomag_corr, ab_correction=ab_correction,
            exptime=exptime, aperture_radii=aperture_radii,
            background_aperture=background_aperture, ref_frame=ref_frame,
            background=background)

        return result