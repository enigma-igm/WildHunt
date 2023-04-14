#!/usr/bin/env python

""" Python module to generate multi-band image plots for a given source.

These functions are used in the inspector GUI.

"""

import math
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from wildhunt.utils import coord_to_name
from wildhunt import image as whim


def make_mult_png_fig(ra, dec, surveys, bands,
                      fovs, apertures, square_sizes, image_dir, mag_list=None,
                      magerr_list=None, sn_list=None,
                      forced_mag_list=None, forced_magerr_list=None,
                      forced_sn_list=None, n_col=3,
                      n_sigma=3, color_map_name='gray',
                      scalebar=5 * u.arcsecond,
                      ell_a=None, ell_b=None, ell_theta=None, ell_color='red',
                      ell_display=None,
                      add_info_label=None, add_info_value=None,
                      add_cross=False, cross_color='red'):
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
                              forced_sn_list, scalebar,
                              ell_a, ell_b, ell_theta, ell_color,
                              ell_display,
                              n_sigma,
                              color_map_name,
                              add_cross=add_cross, cross_color='red')

    coord_name = coord_to_name(np.array([ra]),
                                  np.array([dec]),
                                  epoch="J")

    if add_info_label is None or add_info_value is None:
        fig.suptitle(coord_name[0])
    else:
        fig.suptitle(coord_name[0]+' '+add_info_label+'='+add_info_value)

    return fig


def _make_mult_png_axes(fig, n_row, n_col, ra, dec, surveys, bands,
                        fovs, apertures, square_sizes, image_dir, ID=None, mag_list=None,
                        magerr_list=None, sn_list=None,
                        forced_mag_list=None, forced_magerr_list=None,
                        forced_sn_list=None, scalebar=5 * u.arcsecond,
                        ell_a=None, ell_b=None, ell_theta=None, ell_color='red',
                        ell_display=None, ell_factor=np.array([1]),
                        add_cross=False, cross_color='red',
                        n_sigma=3, color_map_name='gray', show_axes=True):
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

        image = whim.SurveyImage(ra, dec, survey, band, image_dir, min_fov=fov,
                            ID=ID, instantiate_empty=True)

        if image.data is not None:

            cutout = image.get_cutout_image(ra, dec, fov)
            wcs = WCS(cutout.header)

            # axs = fig.add_subplot(int(f"{n_row}{n_col}{idx + 1}"), projection=wcs)
            axs = fig.add_subplot(n_row, n_col, idx + 1, projection=wcs)

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

            if type(add_cross) in [list, np.ndarray]:
                add_cross_idx = add_cross[idx]
            elif type(add_cross) == bool:
                add_cross_idx = add_cross

            if add_cross_idx:
                xpix, ypix = wcs.all_world2pix(ra, dec, 0)
                axs.plot(xpix, ypix, '+', color=cross_color, markersize=15)

            if ell_a is None or ell_b is None or ell_theta is None:
                # Plot circular aperture (forced photometry flux)
                (yy, xx) = cutout.data.shape
                circx = (xx * 0.5)  # + 1
                circy = (yy * 0.5)  # + 1
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

            elif ell_a is not None and ell_b is not None and ell_theta is not None:

                if ell_display[idx]:

                    for jdx, fac in enumerate(ell_factor):

                        if type(ell_color) == list:
                            ell_color_jdx = ell_color[jdx]

                            cutout._add_aperture_ellipse(
                                axs, ra, dec, ell_a*fac, ell_b*fac,
                                ell_theta,
                                edgecolor=ell_color_jdx)

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
                         weight='bold',
                         fontsize='large',
                         bbox=dict(facecolor='white', alpha=0.6),
                         transform=fig.gca().transAxes)

            fig.gca().set_title(survey + " " + band)

        else:
            # axs = fig.add_subplot(int(f"{n_row}{n_col}{idx + 1}"))
            axs = fig.add_subplot(n_row, n_col, idx + 1)
            axs.axis('off')

    return fig

def aperture_inpixels(aperture, hdr):
    """ Return the aperture size in pixels.

    :param aperture: The aperture size in arcsec
    :type aperture: float
    :param hdr: The header of the image
    :type hdr: astropy.io.fits.header.Header
    :return: The aperture size in pixels
    :rtype: float
    """
    pixelscale = get_pixelscale(hdr)
    aperture /= pixelscale  # pixels

    return aperture

def get_pixelscale(hdr):
    """ Return the pixel scale of the image in arcsec/pixel.

    :param hdr: The header of the image
    :type hdr: astropy.io.fits.header.Header
    :return: The pixel scale of the image in arcsec/pixel
    :rtype: float
    """
    wcs_img = WCS(hdr)
    scale = np.mean(proj_plane_pixel_scales(wcs_img)) * 3600

    return scale