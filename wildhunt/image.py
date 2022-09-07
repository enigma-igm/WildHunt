#!/usr/bin/env python
"""

Main module for downloading and manipulating image data.

"""

import glob

import numpy as np

import aplpy

from astropy.table import Table
from astropy import wcs, stats
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales

from mpl_toolkits.axes_grid1.anchored_artists import (AnchoredEllipse,
                                                      AnchoredSizeBar)

from astropy.coordinates import ICRS

from reproject.mosaicking import find_optimal_celestial_wcs

from reproject import reproject_interp

from photutils import aperture_photometry, SkyCircularAperture, SkyCircularAnnulus

import matplotlib.pyplot as plt

from wildhunt import utils
from wildhunt import pypmsgs
from wildhunt.catalog import retrieve_survey

from IPython import embed

msgs = pypmsgs.Messages()

def mp_get_forced_photometry(ra, dec, survey_dict):
    # Get aperture photometry for one source but all bands/surveys


    # return photometry for each source but all filters/surveys (a row in a
    # ra/dec table
    pass

def get_aperture_photometry(ra, dec, survey_dicts, image_folder_path='cutouts', radii=[1.], radius_in=7.0,
                            radius_out=10.0, epoch='J'):
    '''Perform forced photometry for a given target on images from a given imaging survey.

    :param ra: right ascension in degrees
    :param dec: right ascension in degrees
    :param survey_dicts: survey dictionaries
    :param image_folder_path: string, path where the images for which to perform forced photometry are stored
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

        survey = retrieve_survey(survey_dict['survey'],
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
                extCorr = image_params.extCorr
                back = image_params.back
                zpt = image_params.zpt
                photo_table['{:}_ZP_{:}'.format(survey_dict['survey'], band)] = [zpt]

                if 'w' in band:
                    data = data.copy() * 10 ** (-image_params.corr / 2.5)

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
                    ## ToDo: for VSA/WSA/PS1 forced photometry we compute the native fluxes and the magnitudes in Vega, while we probably want nanomaggies and AB
                    for i in range(len(radii)):
                        radii_namei = str(radii[i] * 2.0).replace('.', 'p')
                        photo_table['{:}_{:}_flux_aper_{:}'.format(survey_dict['survey'], band, radii_namei)] = \
                            (flux[i] - background[i]) / exp
                        photo_table['{:}_{:}_flux_aper_err_{:}'.format(survey_dict['survey'], band, radii_namei)] = \
                            std * np.sqrt(pix_aperture[i].area) / exp
                        photo_table['{:}_{:}_snr_aper_{:}'.format(survey_dict['survey'], band, radii_namei)] = SN[i]
                        if (flux[i] - background[i]) > 0.:
                            photo_table['{:}_{:}_mag_aper_{:}'.format(survey_dict['survey'], band, radii_namei)] = \
                                -2.5 * np.log10(photo_table['{:}_{:}_flux_aper_{:}'.format(survey_dict['survey'], band,
                                radii_namei)]) + photo_table['{:}_ZP_{:}'.format(survey_dict['survey'],
                                                                                                band)] - extCorr
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
    embed()
    return photo_table


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

        subplot = (1, 1, 1)

        self._plot_axis(fig, subplot, fov=fov, n_sigma=n_sigma,
                       color_map=color_map)

        plt.show()

    def _plot_axis(self, fig, subplot, fov=None, n_sigma=3,
                  color_map='viridis'):
        """Plot image axis on input figure.

        Class internal plotting routine.

        :param fig: Input figure
        :type fig:
        :param subplot: Subplot touple (e.g., "(1,1,1)")
        :type subplot tuple
        :param fov: Field of view
        :type: float
        :param n_sigma: Number of sigma for image color scale sigma clipping.
        :type n_sigma: int
        :param color_map: Matplotlib color map
        :type color_map:
        :return:
        """

        if fov is not None:
            cutout_data = self.get_cutout(fov=fov)
        else:
            cutout_data = None

        if cutout_data is not None:
            img_data = cutout_data
        else:
            img_data = self.data

        hdu = fits.ImageHDU(data=img_data, header=self.header)

        axs = aplpy.FITSFigure(hdu, figure=fig,
                               subplot=subplot,
                               north=True)

        # Sigma-clipping of the color scale
        mean = np.mean(img_data[~np.isnan(img_data)])
        std = np.std(img_data[~np.isnan(img_data)])
        upp_lim = mean + n_sigma * std
        low_lim = mean - n_sigma * std
        axs.show_colorscale(vmin=low_lim, vmax=upp_lim,
                            cmap=color_map)

        axs.set_title(self.survey+' '+self.band)

        return axs

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
                     scalebar=5*u.arcsecond, sb_pad=0.5, sb_borderpad=0.4):

        if north:
            self._rotate_north_up()

        if fov is not None:
            cutout_data = self.get_cutout(fov=fov)
        else:
            cutout_data = None

        if cutout_data is not None:
            img_data = cutout_data
        else:
            img_data = self.data

        wcs = WCS(self.header)

        if axis is None and fig is not None and subplot is not None:
            axs = fig.add_subplot(subplot, projection=wcs)
        elif axis is not None:
            axs = axis
        else:
            msgs.error('Neither figure and subplot tuple or figure axis '
                       'provided.')

        # Sigma-clipping of the color scale
        mean = np.mean(img_data[~np.isnan(img_data)])
        std = np.std(img_data[~np.isnan(img_data)])
        upp_lim = mean + n_sigma * std
        low_lim = mean - n_sigma * std

        axs.imshow(img_data, origin='lower', vmin=low_lim,
                   vmax=upp_lim, cmap=color_map)

        if scalebar is not None:
            if isinstance(scalebar, u.Quantity):
                length = scalebar.to(u.degree).value
            elif isinstance(scalebar, u.Unit):
                length = scalebar.to(u.degree)

            self._add_scalebar(axis, length, pad=sb_pad,
                               borderpad=sb_borderpad)

        return axs

    def _add_scalebar(self, axis, length, corner='lower right',
                      pad=0.5, borderpad=0.4):
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
                                 sep=3, frameon=False,
                                 fontproperties={'size':15, 'weight':'bold'})

        axis.add_artist(artist)

    def get_cutout(self, fov):
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


