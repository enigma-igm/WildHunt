#!/usr/bin/env python


import math
import glob
import aplpy
import numpy as np

from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata.utils import Cutout2D
from astropy.wcs.utils import proj_plane_pixel_scales

import matplotlib.pyplot as plt

from wildhunt import utils, catalog
import wildhunt.image as whim

from IPython import embed

def plot_source_images(ra, dec, survey_dicts, fov, auto_download=True,
                       n_col=5, image_folder_path='cutouts'):
    """Plot image cutouts for all specified surveys/bands for a single
    source defined by ra, dec, with auto_download."""

    n_images = np.sum([len(dict['bands']) for dict in survey_dicts])

    print(n_images)


    n_row = int(math.ceil(n_images / n_col))

    fig = plt.figure(figsize=(5*n_col, 5*n_row))

    fig.subplots_adjust(left=0.17, bottom=0.15, right=0.95, top=0.95)

    survey_list = []
    band_list = []
    for dict in survey_dicts:
        for band in dict['bands']:
            survey_list.append(dict['survey'])
            band_list.append(band)

    print(survey_list, band_list)
    for idx, survey in enumerate(survey_list):

        img = whim.Image(ra, dec, survey, band_list[idx], image_folder_path,
                         fov=fov)

        axs = img._plot_axis(fig, subplot=(n_row, n_col, idx+1), fov=20)

        # axs = img._simple_plot(fig, subplot=(n_row, n_col, idx+1), fov=10)
        # embed()
        # axs.get_xaxis().set_visible(False)
        # axs.get_yaxis().set_visible(False)

    plt.show()



# ------------------------------------------------------------------------------
#  Plotting functions for image_cutouts (OLD ROUTINES, MAYB BE UPDATED)
# ------------------------------------------------------------------------------

def open_image(filename, ra, dec, fov, image_folder_path, verbosity=0):

    """Opens an image defined by the filename with a fov of at least the
    specified size (in arcseonds).

    :param filename:
    :param ra:
    :param dec:
    :param fov:
    :param image_folder_path:
    :param verbosity:
    :return:
    """

    filenames_available = glob.glob(filename)
    file_found = False
    open_file_fov = None
    file_path = None
    if len(filenames_available) > 0:
        for filename in filenames_available:

            try:
                file_fov = int(filename.split("_")[3].split(".")[0][3:])
            except:
                file_fov = 9999999

            if fov <= file_fov:
                data, hdr = fits.getdata(filename, header=True)
                file_found = True
                file_path =filename
                open_file_fov = file_fov

    if file_found:
        if verbosity > 0:
            print("Opened {} with a fov of {} "
                  "arcseconds".format(file_path, open_file_fov))

        return data, hdr, file_path

    else:
        if verbosity > 0:
            print("File {} in folder {} not found. Target with RA {}"
                  " and Decl {}".format(filename, image_folder_path,
                                        ra, dec))
        return None, None, None


def make_mult_png_fig(ra, dec, surveys, bands,
                  fovs, apertures, square_sizes, image_folder_path, mag_list=None,
                  magerr_list=None, sn_list=None,
                  forced_mag_list=None, forced_magerr_list=None,
                  forced_sn_list=None, n_col=3,
                  n_sigma=3, color_map_name='viridis',
                  add_info_label=None, add_info_value=None, verbosity=0):
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
    :param image_folder_path: string
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
                  fovs, apertures, square_sizes, image_folder_path, mag_list,
                  magerr_list, sn_list,
                  forced_mag_list, forced_magerr_list,
                  forced_sn_list, n_sigma, color_map_name, verbosity)

    coord_name = utils.coord_to_name(np.array([ra]),
                                  np.array([dec]),
                                  epoch="J")
    if add_info_label is None or add_info_value is None:
        fig.suptitle(coord_name[0])
    else:
        fig.suptitle(coord_name[0]+' '+add_info_label+'='+add_info_value)

    return fig


def _make_mult_png_axes(fig, n_row, n_col, ra, dec, surveys, bands,
                  fovs, apertures, square_sizes, image_folder_path, mag_list=None,
                  magerr_list=None, sn_list=None,
                  forced_mag_list=None, forced_magerr_list=None,
                  forced_sn_list=None,
                  n_sigma=3, color_map_name='viridis', verbosity=0):
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
    :param image_folder_path: string
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

        # Get the correct filename, accept larger fovs
        coord_name = utils.coord_to_name(np.array([ra]), np.array([dec]),
                                      epoch="J")

        filename = image_folder_path + '/' + coord_name[0] + "_" + survey + "_" + \
                   band + "*fov*.fits"

        data, hdr, file_path = open_image(filename, ra, dec, fov,
                                       image_folder_path,
                               verbosity)

        if data is not None and hdr is not None:
            file_found = True
        else:
            file_found = False

        # Old plotting routine to modify, currently it only plots images for
        # surveys and bands that it could open, no auto download implemented
        if file_found:
            wcs_img = wcs.WCS(hdr)

            pixcrd = wcs_img.wcs_world2pix(ra, dec, 0)
            positions = (np.float(pixcrd[0]), np.float(pixcrd[1]))
            overlap = True

            if verbosity >= 4:
                print("[DIAGNOSTIC] Image file shape {}".format(data.shape))

            try:
                img_stamp = Cutout2D(data, positions, size=fov * u.arcsec,
                                     wcs=wcs_img)

                if verbosity >= 4:
                    print("[DIAGNOSTIC] Cutout2D file shape {}".format(
                        img_stamp.shape))

            except:
                print("Source not in image")
                overlap = False
                img_stamp = None


            if img_stamp is not None:

                if overlap:
                    img_stamp = img_stamp.data

                hdu = fits.ImageHDU(data=img_stamp, header=hdr)

                axs = aplpy.FITSFigure(hdu, figure=fig,
                                       subplot=(n_row, n_col, idx + 1),
                                       north=True)

                # Check if input color map name is a color map, else use viridis
                try:
                    cm = plt.get_cmap(color_map_name)
                except ValueError:
                    print('Color map argument is not a color map. Setting '
                          'default: viridis')
                    cm = plt.get_cmap('viridis')
                    color_map_name = 'viridis'

                # Sigma-clipping of the color scale
                mean = np.mean(img_stamp[~np.isnan(img_stamp)])
                std = np.std(img_stamp[~np.isnan(img_stamp)])
                upp_lim = mean + n_sigma * std
                low_lim = mean - n_sigma * std
                axs.show_colorscale(vmin=low_lim, vmax=upp_lim,
                                    cmap=color_map_name)

                # Plot circular aperture (forced photometry flux)
                (yy, xx) = img_stamp.shape
                circx = (xx * 0.5)  # + 1
                circy = (yy * 0.5)  # + 1
                aper_pix = aperture_inpixels(aperture, hdr)
                circle = plt.Circle((circx, circy), aper_pix, color='r', fill=False,
                                    lw=1.5)
                fig.gca().add_artist(circle)

                # Plot rectangular aperture (error region)
                rect_inpixels = aperture_inpixels(size, hdr)
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


def aperture_inpixels(aperture, hdr):
    '''
    receives aperture in arcsec. Returns aperture in pixels
    '''
    pixelscale = get_pixelscale(hdr)
    aperture /= pixelscale #pixels

    return aperture

def get_pixelscale(hdr):
    '''
    Get pixelscale from header and return in it in arcsec/pixel
    '''

    wcs_img = wcs.WCS(hdr)
    scale = np.mean(proj_plane_pixel_scales(wcs_img)) * 3600

    return scale

def generate_cutout_images(ra_sources, dec_sources, survey_dicts, imgsize = 30, download_images=True, n_col = 6,
                           image_folder_path='cutouts', n_jobs=1, epoch='J'):



    if download_images == True:
        for survey_dict in survey_dicts:
            survey = catalog.retrieve_survey(survey_dict['survey'],
                                     survey_dict['bands'],
                                     survey_dict['fov'])
            survey.download_images(ra_sources, dec_sources, image_folder_path, n_jobs)

    obj_names = utils.coord_to_name(ra_sources, dec_sources, epoch=epoch)

    for obj_name, ra, dec in zip(obj_names, ra_sources, dec_sources):

        cutout_names = []
        band_names = []

        for survey_dict in survey_dicts:
            for band in survey_dict['bands']:

                image = image_folder_path + '/' + obj_name + "_" + survey_dict['survey'] + "_" + band + "_fov" + \
                         '{:d}.fits'.format(survey_dict['fov'])

                pos = SkyCoord(ra * u.deg, dec * u.deg, frame='fk5')

                if band in ['Y','J','H','K']:
                    wcs_img = wcs.WCS(fits.getheader(image, 1))
                    data = fits.getdata(image, 1)
                else:
                    wcs_img = wcs.WCS(fits.getheader(image, 0))
                    data = fits.getdata(image, 0)

                size = (imgsize * u.arcsec, imgsize * u.arcsec)
                cutout = Cutout2D(data, pos, size, wcs=wcs_img)

                hdu = fits.PrimaryHDU(cutout.data)
                hdu.header.update(cutout.wcs.to_header())
                cutout_name = image_folder_path + '/cutout_' + obj_name + "_" + survey_dict['survey'] + "_" + band \
                              + "_fov" + '{}.fits'.format(str(imgsize))
                hdu.writeto(cutout_name, overwrite=True)

                cutout_names.append('cutout_' + obj_name + "_" + survey_dict['survey'] + "_" + band \
                              + "_fov" + '{}.fits'.format(str(imgsize)))
                band_names.append(band)

        thumbnail(obj_name, ra, dec, cutout_names, band_names, n_col = n_col, image_folder_path = image_folder_path)

def thumbnail(obj_name, ra, dec, cutouts, bands, n_col = 6, smooth=None, north=True, size=30, pmin=5.0, pmax=95.0,
              show_circle=True, interpolation='nearest', cmap='gist_yarg', aspect='auto',image_folder_path='cutouts'):
    '''Making thumbnails for a list of given bands

    '''

    n_images = len(cutouts)
    n_row = int(math.ceil(n_images / n_col))
    fig = plt.figure(figsize=(5*n_col, 5*n_row))

    title = obj_name
    plt.title(title, loc='center', fontsize=25)
    plt.axis('off')

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0)
    for idx, cutout in enumerate(cutouts):
        try:

            si = aplpy.FITSFigure(image_folder_path + '/' + cutout,
                                  north=north, figure=fig, subplot=(n_row, n_col, idx + 1))
            si.recenter(ra, dec, radius=size / 2 / 3600.)
            if show_circle:
                si.show_circles(ra, dec, radius=size / 2 / 5 / 3600., lw=4, color='cyan')
            if smooth is not None:
                if isinstance(smooth, int):
                    this_smooth=smooth
                else:
                    this_smooth = smooth[idx]
                if this_smooth==0:
                    this_smooth=None
            else:
                this_smooth=None
            si.show_colorscale(interpolation=interpolation, aspect=aspect, cmap=cmap, stretch='linear', pmin=pmin,
                               pmax=pmax, smooth=this_smooth)

            si.add_label(0.1, 0.85, bands[idx], relative=True, horizontalalignment='left',
                             color='r', size=45)

            si.axis_labels.hide()
            si.tick_labels.hide()
            si.ticks.hide()
        except:
            print('Skiping {:}'.format(cutout))

        plt.savefig(image_folder_path + '/' + obj_name + '.png', dpi=300)

    plt.close('all')