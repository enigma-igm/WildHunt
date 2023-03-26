#!/usr/bin/env python
"""

Main module for downloading and manipulating image data.

"""

import os
import glob
import math

import numpy as np
import pandas as pd

import string
from astropy import wcs, stats
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord, ICRS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.visualization import ZScaleInterval

from matplotlib.patches import Circle, Ellipse, Rectangle
from matplotlib.collections import PatchCollection
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


def get_pixelscale(hdr):
    """ Returns the pixel scale in arcsec/pixel

    :param hdr: Fits header of an image
    :type hdr: astropy.io.fits.header.Header
    :return:
    """
    wcs_img = wcs.WCS(hdr)
    scale = np.mean(proj_plane_pixel_scales(wcs_img)) * 3600

    return scale


def aperture_inpixels(aperture, hdr):
    """ Converts aperture size from arcsec to pixels

    :param aperture: Aperture size in arcsec.
    :type aperture: float
    :param hdr: Fits header of an image
    :type hdr: astropy.io.fits.header.Header
    :return:
    """

    pixelscale = get_pixelscale(hdr)
    aperture /= pixelscale # convert to pixels

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
    :param scalebar: Scalebar size in arcseconds
    :type scalebar: astropy.units.quantity.Quantity
    :param add_info_value : string
        Value for additional information added to the title of the figure
    :param add_info_label : string
        Label for additional information added to the title of the figure
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
                        n_sigma=3, color_map_name='viridis'):
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
        img_wcs = WCS(cutout.header)

        axs = fig.add_subplot(int(f"{n_row}{n_col}{idx + 1}"),
                              projection=img_wcs)

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
        circx, circy = img_wcs.wcs_world2pix(ra, dec, 0)

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
        if forced_mag is not None:
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
                           weight='bold', fontsize='large',
                           bbox=dict(facecolor='white', alpha=0.6),
                           transform=fig.gca().transAxes)

        fig.gca().set_title(survey + " " + band)

    return fig


class Image(object):
    """ Class to handle astronomical images in the framework of wild_hunt.
    """

    def __init__(self, filename=None, data=None, header=None, exten=0,
                 ra=None, dec=None, survey=None, band=None, verbosity=1,
                 instantiate_empty=False):
        """ Initialize an image.

        A filename for a fits image can be specified, alternatively the data
        and header of a fits image can be specified directly.

        If no filename or data/header are provided and instantiate_empty is
        set to True, an empty Image object is created. This is dangerous and
        should only be used if the user knows what they are doing.

        :param filename: The filename of an image (fits format) to be loaded.
        :type filename: str
        :param data: Image data.
        :type data: numpy.ndarray
        :param header: Image header.
        :type header: astropy.io.fits.header.Header
        :param exten: Fits extension to be loaded, holding the science image.
        :type exten: int
        :param ra: Right ascension of the image center in degrees. If this
         is None and the image has a WCS, the central coordinates will be
         extracted from the header.
        :type ra: float
        :param dec: Declination of the image center in degrees. If this
         is None and the image has a WCS, the central coordinates will be
         extracted from the header.
        :type dec: float
        :param survey: Name of the survey the image is from.
        :type survey: str
        :param band: Name of the band the image is from.
        :type band: str
        :param verbosity: Verbosity level. Default: 1.
        :type verbosity: int
        :param instantiate_empty: Boolean to indicate whether to allow
         instantiation of an empty Image object. Default: False.
        :type instantiate_empty: bool
        """

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

        # Set band and survey variables
        self.band = band
        self.survey = survey

    def get_central_coordinates_from_wcs(self):
        """ Get the central coordinates of the image from the WCS information

        :return: Right Ascention and Declination of the image center in
         degrees.
        :rtype: tuple
        """

        naxis_1 = self.header['NAXIS1']
        naxis_2 = self.header['NAXIS2']

        wcs_img = wcs.WCS(self.header)

        coord = wcs_img.wcs_pix2world(int(naxis_1/2), int(naxis_2/2), 0)

        ra = float(coord[0])
        dec = float(coord[1])

        return ra, dec

    def show(self, n_sigma=3, color_map='viridis'):
        """ Show the image data using matplotlib.

        :param n_sigma: Number of sigma for image color scale sigma clipping.
        :type n_sigma: int
        :param color_map: Matplotlib color map
        :type color_map: str
        :return:
        """
        fig = plt.figure(figsize=(5, 5))

        subplot = 111

        self._simple_plot(n_sigma=n_sigma, fig=fig, subplot=subplot,
                          color_map=color_map, north=True)

        plt.show()

    def _rotate_north_up(self):
        """ Rotate the image so that North is up.

        :return: None
        """

        # Get image WCS
        img_wcs = WCS(self.header)

        frame = ICRS()

        new_wcs, shape = find_optimal_celestial_wcs([(self.data,
                                                      img_wcs)],
                                                    frame=frame)

        data, _ = reproject_interp((self.data, img_wcs), new_wcs,
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
            else:
                raise TypeError('Scalebar must be a Quantity or a Unit.')

            self._add_scalebar(axs, length, pad=sb_pad,
                               borderpad=sb_borderpad,
                               corner=corner,
                               frameon=frameon)

        return axs

    # TODO: Document and clean up!
    def finding_chart(self, fov, target_aperture=5, color_scale='zscale',
                      n_sigma=3, color_map='Greys', scalebar=0.2*u.arcmin,
                      sb_pad=0.5, sb_borderpad=0.4, corner='lower right',
                      frameon=True, low_lim=None, upp_lim=None, offset_df=None,
                      offset_focus=False, offset_id=0,
                      offset_ra_column_name='offset_ra',
                      offset_dec_column_name='offset_dec',
                      offset_mag_column_name='mag_z',
                      offset_id_column_name='offset_shortname',
                      label_position='bottom'):

        if offset_focus:
            im_ra = offset_df.loc[offset_id, offset_ra_column_name]
            im_dec = offset_df.loc[offset_id, offset_dec_column_name]
        else:
            im_ra = self.ra
            im_dec = self.dec

        self._rotate_north_up()

        chart_img = self.get_cutout_image(im_ra, im_dec, fov)
        self.data = chart_img.data
        self.header = chart_img.header

        # Setting up the figure
        fig = plt.figure(figsize=(12, 12))
        if offset_df is not None:
            fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
        else:
            fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

        wcs = WCS(self.header)
        axs = fig.add_subplot(111, projection=wcs)

        if color_scale == 'zscale':
            zscale = ZScaleInterval()
            low_lim, upp_lim = zscale.get_limits(self.data)
        else:
            if isinstance(upp_lim, float) and isinstance(low_lim, float):
                msgs.info('Using user defined color scale limits.')
            else:
                msgs.info('Determining color scale limits by sigma clipping.')

                # Sigma-clipping of the color scale
                mean = np.mean(self.data[~np.isnan(self.data)])
                std = np.std(self.data[~np.isnan(self.data)])
                upp_lim = mean + n_sigma * std
                low_lim = mean - n_sigma * std

        # Plot the image
        axs.imshow(self.data, origin='lower',
                   vmin=low_lim,
                   vmax=upp_lim,
                   cmap=color_map)

        # Plot the scalebar
        if scalebar is not None:
            if isinstance(scalebar, u.Quantity):
                length = scalebar.to(u.degree).value
            elif isinstance(scalebar, u.Unit):
                length = scalebar.to(u.degree)
            else:
                raise TypeError('Scalebar must be a Quantity or a Unit.')

            self._add_scalebar(axs, length, pad=sb_pad,
                               borderpad=sb_borderpad,
                               corner=corner,
                               frameon=frameon,
                               fontsize=12, fontweight='normal')

        # Plot a circular aperture around the target
        self._add_aperture_circle(axs, self.ra, self.dec, target_aperture)

        # Plot offset stars
        if offset_df is not None and offset_ra_column_name is not None and \
            offset_dec_column_name is not None and offset_mag_column_name is \
                not None and offset_id_column_name is not None:
            # Mark the offsets on the finding chart
            self.mark_offset_stars(axs, offset_df, offset_id,
                                   offset_ra_column_name,
                                   offset_dec_column_name,
                                   aperture_radius=2,
                                   label_position=label_position)
            # Add the offset star info box
            self.plot_offset_info_box(axs, offset_df,
                                      offset_ra_column_name,
                                      offset_dec_column_name,
                                      offset_mag_column_name)

        # Add title
        c = SkyCoord(ra=self.ra, dec=self.dec, unit=(u.degree, u.degree))
        title = 'RA= {0} ; DEC = {1}'.format(
            c.ra.to_string(precision=3, sep=":", unit=u.hour),
            c.dec.to_string(precision=3, sep=":", unit=u.degree,
                            alwayssign=True))
        plt.title(title, fontsize=20)

        axs.grid(color='white', ls='dotted')

        axs.set_xlabel('Right Ascension', fontsize=20)
        axs.set_ylabel('Declination', fontsize=20)

        return fig

    # TODO: Document and clean up!
    def mark_offset_stars(self, axs, offset_df, offset_id=0,
                          offset_ra_column_name='offset_ra',
                          offset_dec_column_name='offset_dec',
                          aperture_radius=4,
                          label_position='bottom'):

        position_dict = {"left": [8, 0], "right": [-8, 0], "top": [0, 5],
                         "bottom": [0, -5], "topleft": [8, 5]}

        ra_pos, dec_pos = position_dict[label_position]

        for num, idx in enumerate(offset_df.index):

            ra_off = offset_df.loc[idx, offset_ra_column_name]
            dec_off = offset_df.loc[idx, offset_dec_column_name]
            # Plot circular apertures
            self._add_aperture_circle(axs, ra_off, dec_off, aperture_radius,
                                      edgecolor='blue')
            # Create the labels
            letters = list(string.ascii_uppercase)
            label = letters[num]
            img_wcs = WCS(self.header)
            x, y = img_wcs.wcs_world2pix(ra_off, dec_off, 0)
            label_fac = 5
            if num == offset_id:
                axs.text(x+ra_pos*label_fac, y+dec_pos*label_fac, label,
                         color='blue',
                         size='x-large',
                         verticalalignment='center', family='serif',)

            else:
                axs.text(x+ra_pos*label_fac, y+dec_pos*label_fac, label,
                         color='blue',
                         size='large',
                         verticalalignment='center', family='serif')

    # TODO: Document and clean up!
    def plot_offset_info_box(self, axs, offset_df, ra_column_name,
                             dec_column_name, mag_column_name):

        target_info = 'Target: RA={:.4f}, DEC={:.4f}'.format(self.ra, self.dec)

        info_list = [target_info]

        for num, idx in enumerate(offset_df.index):

            if True:
                ra_off = offset_df.loc[idx, ra_column_name]
                dec_off = offset_df.loc[idx, dec_column_name]

                # Set position angles and separations (East of North)
                pos_angle = offset_df.loc[idx, 'pos_angle']
                separation = offset_df.loc[idx, 'separation']
                dra = offset_df.loc[idx, 'dra_offset']
                ddec = offset_df.loc[idx, 'ddec_offset']
                mag = offset_df.loc[idx, mag_column_name]

                info = '{}: RA={:.4f}, DEC={:.4f}, {}={:.2f}, PosAngle={' \
                       ':.2f}'.format(string.ascii_uppercase[num],
                                      ra_off,
                                      dec_off, mag_column_name,
                                      mag, pos_angle)
                info_off = 'Sep={:.2f}, Dra={:.2f}, ' \
                           'Ddec={:.2f}'.format(separation, dra, ddec)
                info_list.append(info+' '+info_off)

        boxdict = dict(facecolor='white', alpha=1.0, edgecolor='none')
        axs.text(.02, -0.15, "\n".join(info_list), transform=axs.transAxes,
                fontsize='small',
                bbox=boxdict)

    def _add_scalebar(self, axis, length, corner='lower right',
                      pad=0.5, borderpad=0.4, fontsize=15,
                      fontweight='bold', frameon=False):
        """ Add a scalebar to the image.

        Function adopted from Aplpy.

        :param axis: The matplotlib axis to add the scalebar to.
        :type axis: matplotlib.axes._subplots.AxesSubplot
        :param length: The length of the scalebar in degrees.
        :param corner: The corner position of the scalebar.
        :type corner: str
        :param pad: The padding of the scalebar in fraction of the font size.
        :type pad: float
        :param borderpad: The border padding of the scalebar in fraction of
         the font size.
        :type borderpad: float
        :param frameon: Boolean to indicate whether to add a frame around the
         scalebar.
        :return: None
        """

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
                                 fontproperties={'size': fontsize,
                                                 'weight': fontweight})

        axis.add_artist(artist)

    def _add_aperture_ellipse(self, axis, ra, dec, a, b, angle,
                              edgecolor='red', linewidth=2):
        """ Add an aperture ellipse to the image.

        :param axis: The matplotlib axis.
        :type axis: matplotlib.axes._subplots.AxesSubplot
        :param ra: The right ascension of the center of the ellipse.
        :type ra: float
        :param dec: The declination of the center of the ellipse.
        :type dec: float
        :param a: The major axis (2 * semi-major axis) in pixels
        :type a: float
        :param b: The minor axis (2 * semi-minor axis) in pixels
        :type b: float
        :param angle: The position angle of the major axis in degrees.
        :type angle: float
        :param edgecolor: The color of the ellipse (default: red).
        :type edgecolor: str
        :param linewidth: The line width of the ellipse (default: 2).
        :type linewidth: int
        :return: None
        """

        artist = Ellipse(xy=(WCS(self.header).wcs_world2pix(ra, dec, 0)),
                         width=a, height=b, angle=angle,
                         edgecolor=edgecolor, linewidth=linewidth,
                         facecolor='None')

        axis.add_artist(artist)

    def _add_aperture_circle(self, axis, ra, dec, radius, edgecolor='red',
                             linewidth=2):
        """ Add a circular aperture to the image.

        :param axis: The matplotlib axis.
        :type axis: matplotlib.axes._subplots.AxesSubplot
        :param ra: The right ascension of the center of the circle.
        :type ra: float
        :param dec: The declination of the center of the circle.
        :type dec: float
        :param radius: The radius of the circle in arcseconds.
        :type radius: float
        :param edgecolor: The color of the circle (default: red).
        :type edgecolor: str
        :param linewidth: The line width of the circle (default: 2).
        :type linewidth: int
        :return: None
        """

        img_wcs = WCS(self.header)
        radius = radius / 3600  # Convert to degrees
        radius_pix = radius / img_wcs.proj_plane_pixel_scales()[0].value

        artist = Circle(xy=(img_wcs.wcs_world2pix(ra, dec, 0)),
                        radius=radius_pix, edgecolor=edgecolor,
                        linewidth=linewidth, facecolor='None')

        axis.add_artist(artist)

    def add_aperture_rectangle(self, axis, ra, dec, width, height, angle=0,
                               frame='world',
                               edgecolor='red', linewidth=2):
        """ Add a rectangular aperture to the image.

        This function is adapted from aplpy: https://github.com/aplpy/aplpy

        It correctly rotates the
        rectangle around its center position, see discussion here
        https://github.com/aplpy/aplpy/pull/327

        :param axis: The matplotlib axis.
        :type axis: matplotlib.axes._subplots.AxesSubplot
        :param ra: The right ascension of the center of the rectangle.
        :type ra: float
        :param dec: The declination of the center of the rectangle.
        :type dec: float
        :param width: The width of the rectangle in pixels/arcseconds.
        :type width: float
        :param height: The height of the rectangle in pixels/arcseconds.
        :type height: float
        :param angle: The position angle of the rectangle in degrees.
        :type angle: float
        :param frame: The coordinate frame for the width and height dimensions.
         Can be 'pixel' or 'world' (default: 'world').
        :type frame: str
        :param edgecolor: The color of the circle (default: red).
        :type edgecolor: str
        :param linewidth: The line width of the circle (default: 2).
        :type linewidth: int
        :return:
        """

        img_wcs = WCS(self.header)

        if frame == 'pixel':
            pix_x, pix_y = ra, dec
            pix_w = width
            pix_h = height
            transform = axis.transData
        else:
            pix_x, pix_y = img_wcs.wcs_world2pix(ra, dec, 0)
            pix_scale = img_wcs.proj_plane_pixel_scales()
            sx, sy = pix_scale[0].value, pix_scale[1].value

            pix_w = width / sx / 3600
            pix_h = height / sy / 3600
            transform = axis.transData

        xp = pix_x - pix_w / 2.
        yp = pix_y - pix_h / 2.
        radeg = np.pi / 180
        xr = (xp - pix_x) * np.cos((angle) * radeg) - (yp - pix_y) * np.sin(
            (angle) * radeg) + pix_x
        yr = (xp - pix_x) * np.sin((angle) * radeg) + (yp - pix_y) * np.cos(
            (angle) * radeg) + pix_y

        artist = Rectangle(xy=(xr, yr), width=pix_w, height=pix_h, angle=angle,
                           edgecolor=edgecolor, linewidth=linewidth,
                           facecolor='None', transform=transform)

        axis.add_artist(artist)

    def _add_slit(self, axis, ra, dec, width, length, angle, edgecolor='red',
                  linewidth=2):
        """ Add a slit rectangle to the image.

        :param axis: The matplotlib axis.
        :type axis: matplotlib.axes._subplots.AxesSubplot
        :param ra: The right ascension of the center of the slit.
        :type ra: float
        :param dec: The declination of the center of the slit.
        :type dec: float
        :param width: The length of the slit in arcseconds.
        :type width: float
        :param length: The length of the slit in arcseconds.
        :type length: float
        :param angle: The position angle of the slit in degrees.
        :type angle: float
        :param edgecolor: The color of the slit (default: red).
        :type edgecolor: str
        :param linewidth: The line width of the slit (default: 2).
        :type linewidth: int
        :return: None
        """

        self.add_aperture_rectangle(axis, ra, dec, width, length, angle=angle,
                                    frame='world', edgecolor=edgecolor,
                                    linewidth=linewidth)

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

    def to_fits(self, filepath, overwrite=True):
        """ Save the image to a fits file.

        :param filepath: Filepath of the fits file to save the image to.
        :param overwrite: Boolean to indicate whether to overwrite the file
         if it already exists.
        :return: None
        """

        hdu = fits.PrimaryHDU(self.data, header=self.header)
        hdu.writeto(filepath, overwrite=overwrite)

    def calculate_aperture_photometry(self, ra, dec, nanomag_correction,
                                      zero_point=None,
                                      ab_correction=None,
                                      exptime_norm=1,
                                      aperture_radii=np.array([1.]),
                                      background_aperture=np.array([7., 10.]),
                                      ref_frame='icrs',
                                      background=True,
                                      band=None, survey=None):
        """ Calculate the aperture photometry of a source in the image.

        The correction factor to convert from image flux/counts to nanomaggies
        is a required argument. This should include zero point and AB
        correction factors.

        An example of how to calculate the correction factor is shown below:
        nanomag_correction = np.power(10, 0.4 * (22.5 - zero_point -
        ab_correction))

        The zero point and AB correction factor keyword arguments are not
        used in the photometric calculation but are included in the output.

        :param ra: The Right Ascension of the source in decimal degrees.
        :type ra: float
        :param dec: The Declination of the source in decimal degrees.
        :type dec: float
        :param nanomag_correction: The correction factor to convert from
        image flux/counts to nanomaggies. This should include zero point and
         AB correction.
        :type nanomag_correction: float
        :param zero_point: The photometric zero point of the image in
         magnitudes.
        :type zero_point: float
        :param ab_correction: The AB correction factor to convert from
         survey filter band magnitude to AB magnitude.
        :type ab_correction: float
        :param exptime_norm: The exposure time normalization factor. This
         should be set to the exposure time of the image in seconds if the
         values of the image are in counts instead of fluxes. Default: 1.
        :type exptime_norm: float
        :param aperture_radii: List of aperture radii in arcseconds to
        calculate the forced photometry for. Default: [1.]
        :type aperture_radii: np.ndarray
        :param background_aperture: The inner and outer radii of the background
         annulus in arcseconds. Default: [7., 10.]
        :type background_aperture: np.ndarray
        :param ref_frame: The WCS reference frame to use for the coordinates
         of the catalog sources. Default: 'icrs'
        :type ref_frame: string
        :param background: Boolean to indicate whether to calculate the
         local background level in an annulus around the source and subtract
         it. Default: True
        :param band: The filter band name of the image used for the key
        names of the result dictionary. If not provided, the band name
        will be set to 'band'.
        :type band: str
        :param survey:
        :return: Dictionary containing the aperture photometry results.
        :rtype: dict
        """

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

        survey_band = '{}_{}'.format(survey, band)

        for idx in range(len(aperture_radii)):

            # Calculate the flux in nanomaggies
            flux = float(source_flux['aperture_sum_' + str(idx)])
            flux = (flux - background[idx]) / exptime_norm * nanomag_correction
            # Calculate the flux error in nanomaggies
            flux_err = std * np.sqrt(pix_aperture[idx].area) / exptime_norm * \
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
    """ A class to handles astronomical images directly related to specific
    imaging surveys.

    This class dervies from the Image class.

    It adds the functionality to the Image class to automatically open survey
    images downloaded using the Catalog and ImagingSurvey classes. These
    downloaded images have a specific naming convention that the SurveyImage
    class operates on.

    It further expands the functionality of the Image class to calculate
    aperture photometry by interacting with the ImagingSurvey class, which
    contains the survey specific information needed to perform the source
    flux measurements.

    """

    def __init__(self, ra, dec, survey, band, image_dir, min_fov,
                 data=None, header=None, verbosity=1,
                 instantiate_empty=False):
        """ Initialize the SurveyImage class.

        :param ra: The RA coordinate of the image center in decimal degrees.
         For survey images this should also be the RA position of a specific
         source of interest.
        :type ra: float
        :param dec: The Dec coordinate of the image center in decimal degrees.
         For survey images this should also be the declination position of a
         specific source of interest.
        :type dec: float
        :param survey: The survey name the image is from.
        :type survey: str
        :param band: The survey filter band of the image.
        :type band: str
        :param image_dir: The directory where the image is located.
        :type image_dir: str
        :param min_fov: The minimum field of view the image should have in
         arcseconds.
        :type min_fov: float
        :param data: The image data array.
        :type data: numpy.ndarray
        :param header: The image header.
        :type header: astropy.io.fits.Header
        :param verbosity: Verbosity level.
        :type verbosity: int
        :param instantiate_empty:
        :param instantiate_empty: Boolean to indicate whether to allow
         instantiation of an empty Image object. Default: False.
        :type instantiate_empty: bool
        """

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
                                          ra=ra, dec=dec, survey=survey,
                                          band=band, verbosity=verbosity,
                                          instantiate_empty=instantiate_empty)

    def open(self):
        """This function opens an image from the image directory
        based on the RA, Dec, survey, and band and minimum field of view.

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
                      "{} in folder {} not found.".format(self.survey,
                                                          self.band,
                                                          self.source_name,
                                                          self.fov,
                                                          self.image_dir))

            return None, None

    def get_aperture_photometry(self, aperture_radii=np.array([1.]),
                                background_aperture=np.array([7., 10.]),
                                ref_frame='icrs'):
        """ Perform aperture photometry on the SurveyImage at the RA and Dec
        position initialized with the class.

        :param aperture_radii: List of aperture radii in arcseconds to
        calculate the forced photometry for. Default: [1.]
        :type aperture_radii: np.ndarray
        :param background_aperture: The inner and outer radii of the background
         annulus in arcseconds. Default: [7., 10.]
        :type background_aperture: np.ndarray
        :param ref_frame: The WCS reference frame to use for the coordinates
         of the catalog sources. Default: 'icrs'
        :type ref_frame: string
        :return:
        """

        survey = catalog.retrieve_survey(self.survey,
                                         [self.band],
                                         self.fov)

        filepath = self.image_dir + '/' + self.source_name + "_" + \
                  self.survey + "_" + self.band + "_fov{}.fits".format(
                   self.fov)

        # Retrieve survey specific information
        survey.force_photometry_params(self.header, self.band, filepath)
        exptime = survey.exp
        background = survey.back
        zero_point = survey.zpt
        nanomag_corr = survey.nanomag_corr
        ab_correction = survey.ab_corr

        result = self.calculate_aperture_photometry(
            self.ra, self.dec, zero_point=zero_point,
            nanomag_correction=nanomag_corr, ab_correction=ab_correction,
            exptime_norm=exptime, aperture_radii=aperture_radii,
            background_aperture=background_aperture, ref_frame=ref_frame,
            background=background)

        return result
