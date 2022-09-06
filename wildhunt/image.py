#!/usr/bin/env python
"""

Main module for downloading and manipulating image data.

"""

import glob
import numpy as np

import aplpy
from astropy import wcs
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
from astropy.visualization import ZScaleInterval
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales

from mpl_toolkits.axes_grid1.anchored_artists import (AnchoredEllipse,
                                                      AnchoredSizeBar)

from astropy.coordinates import ICRS

from reproject.mosaicking import find_optimal_celestial_wcs

from reproject import reproject_interp

import matplotlib.pyplot as plt

from wildhunt import utils
from wildhunt import pypmsgs

from IPython import embed

msgs = pypmsgs.Messages()

def mp_get_forced_photometry(ra, dec, survey_dict):
    # Get aperture photometry for one source but all bands/surveys


    # return photometry for each source but all filters/surveys (a row in a
    # ra/dec table
    pass


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


    def get_aperture_photometry(self, ra, dec, survey, band):
        # This function calculates aperture photometry on the image

        # Possible return a catalog entry
        pass


        result_dict = {'ap_flux_2': None}

        return result_dict


