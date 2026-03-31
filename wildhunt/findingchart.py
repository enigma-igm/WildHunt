#!/usr/bin/env python

import aplpy
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from wildhunt.utilities import general_utils as general_utils

# Define some colors
red = (238/255., 102/255., 119/255.)
red = 'red'
blue = (0/255., 68/255., 136/255.)
blue = 'blue'
cyan = (51/255., 187/255., 238/255.)
teal = (0, 153/255., 136/255.)
green = (34/255., 136/255., 51/255.)

def make_finding_charts(table, ra_column_name, dec_column_name,
                        target_column_name, survey, band,
                        aperture, fov, image_folder_path,
                        offset_table=None,
                        offset_id=0,
                        offset_focus=False,
                        offset_ra_column_name=None,
                        offset_dec_column_name=None,
                        pos_angle_column_name=None,
                        offset_mag_column_name=None,
                        offset_id_column_name=None,
                        # offset_finding_chart=True,
                        label_position='bottom',
                        slit_width=None,
                        slit_length=None,
                        format ='pdf',
                        auto_download=False, verbosity=0):

    """Create and save finding charts plots for all targets in the input table.

    :param table: pandas.core.frame.DataFrame
        Dataframe with targets to plot finding charts for
    :param ra_column_name: string
        Right ascension column name
    :param dec_column_name: string
        Declination column name
    :param target_column_name: string
        Name of the target identifier column
    :param survey: string
        Survey name
    :param band: string
        Passband name
    :param aperture: float
        Aperture to plot in arcseconds
    :param fov: float
        Field of view in arcseconds
    :param image_folder_path: string
        Path to where the image will be stored
    :param offset_table: pandas.core.frame.DataFrame
        Pandas dataframe with offset star information for all targets
    :param offset_id: int
        Integer indicating the primary offset from the offset table
    :param offset_focus: boolean
        Boolean to indicate whether offset star will be in the center or not
    :param offset_ra_column_name: string
        Offset star dataframe right ascension column name
    :param offset_dec_column_name: string
        Offset star dataframe declination column name
    :param pos_angle_column_name: string
        Offset star dataframe position angle column name
    :param offset_mag_column_name: string
        Offset star dataframe magnitude column name
    :param offset_id_column_name: string
        Offset star dataframe identifier column name
    :param label_position: string
        String that defines the label position for the offset stars.
        Possible label positions are ["left", "right", "top", "bottom",
         "topleft"]
    :param slit_width: float
        Slit width in arcseconds.
    :param slit_length: float
        Slit length in arcseconds
    :param format: string
        A string indicating in which format the finding charts are save.
        Possible formats: 'pdf', 'png'
    :param auto_download: boolean
        Boolean to indicate whether images should be automatically downloaded.
    :param verbosity:
        Verbosity > 0 will print verbose statements during the execution.
    """

    surveys = [survey]
    bands = [band]
    fovs = [fov]

    print(offset_table)
    print(table)

    for idx in table.index:
        ra = table.loc[idx, ra_column_name]
        dec = table.loc[idx, dec_column_name]
        target_name = table.loc[idx, target_column_name]

        if offset_table is not None:
            offset_target = offset_table.query('{}=="{}"'.format(target_column_name,
                            target_name))

            # Set position angle
            if len(offset_target) > 0:
                if pos_angle_column_name is not None:
                    position_angle = offset_target.loc[offset_target.index[0],
                                                   pos_angle_column_name]
                else:
                    target_coords = SkyCoord(ra=ra, dec=dec,
                                             unit=(u.deg, u.deg),
                                             frame='icrs')
                    offset_coords = SkyCoord(ra=offset_target.loc[:,
                                                offset_ra_column_name].values,
                                             dec=offset_target.loc[:,
                                                 offset_dec_column_name].values,
                                             unit=(u.deg, u.deg),
                                             frame='icrs')
                    # Calculate position angles(East of North)
                    pos_angles = offset_coords.position_angle(target_coords).to(
                        u.deg)
                    # Take position angle to offset_id star in list
                    position_angle = pos_angles[offset_id].to(u.deg).value

            else:
                position_angle = 0
                offset_target = None
        else:
            offset_target = None
            position_angle = 0

        if offset_target is not None:
            offset_target.reset_index(inplace=True, drop=True)

        print(offset_target)

        # if auto_download:
        #     if offset_focus:
        #         survey_dict = [{'survey': survey,
        #                         'bands': ['z'],
        #                         'fov':fov}]
        #
        #         image.get_images()
        #
        #         ct.get_photometry(offset_target.loc[[0]],
        #                              offset_ra_column_name,
        #                              offset_dec_column_name,
        #                              surveys,
        #                              bands,
        #                              image_folder_path,
        #                              fovs,
        #                              # n_jobs=1,
        #                              verbosity=verbosity)
        #     else:
        #         survey_dict = [{'survey': survey,
        #                         'bands': ['z'],
        #                         'fov':fov}]
        #
        #         image.get_images(np.array([table.loc[idx, ra_column_name]]),
        #                          np.array([table.loc[idx, dec_column_name]]),
        #                          image_folder_path,
        #                          survey_dict=survey_dict)


        fig = make_finding_chart(ra, dec, survey, band, aperture, fov,
                                 image_folder_path,
                                 offset_df=offset_target,
                                 offset_id=offset_id,
                                 offset_focus=offset_focus,
                                 offset_ra_column_name=offset_ra_column_name,
                                 offset_dec_column_name=offset_dec_column_name,
                                 offset_mag_column_name=offset_mag_column_name,
                                 offset_id_column_name=offset_id_column_name,
                                 label_position=label_position,
                                 slit_width=slit_width,
                                 slit_length=slit_length,
                                 position_angle=position_angle,
                                 verbosity=verbosity)

        if format == 'pdf':
            fig.save('fc_{}.pdf'.format(target_name), transparent=False)
        if format == 'png':
            fig.save('fc_{}.png'.format(target_name), transparent=False)

        print('{} created'.format('fc_{}'.format(target_name)))


def make_finding_chart(ra, dec, survey, band, aperture, fov,
                       image_folder_path,
                       offset_df=None,
                       offset_id=0,
                       offset_focus=False,
                       offset_ra_column_name=None,
                       offset_dec_column_name=None,
                       offset_mag_column_name=None,
                       offset_id_column_name=None,
                       label_position='bottom',
                       slit_width=5, slit_length=60,
                       position_angle=None, verbosity=2):

    """Make the finding chart figure and return it.

    This is an internal function, but can be used to create one finding chart.

    :param ra: float
        Right ascension of the target in decimal degrees
    :param dec: float
        Declination of the target in decimal degrees
    :param survey: string
        Survey name
    :param band: string
        Passband name
    :param aperture: float
        Size of the plotted aperture in arcseconds
    :param fov: float
        Field of view in arcseconds
    :param image_folder_path: string
        Path to where the image will be stored
    :param offset_df: pandas.core.frame.DataFrame
        Pandas dataframe with offset star information
    :param offset_id: int
        Integer indicating the primary offset from the offset table
    :param offset_focus: boolean
        Boolean to indicate whether offset star will be in the center or not
    :param offset_ra_column_name: string
        Offset star dataframe right ascension column name
    :param offset_dec_column_name: string
        Offset star dataframe declination column name
    :param offset_mag_column_name: string
        Offset star dataframe magnitude column name
    :param offset_id_column_name: string
        Offset star dataframe identifier column name
    :param label_position: string
        String that defines the label position for the offset stars.
        Possible label positions are ["left", "right", "top", "bottom",
         "topleft"]
    :param slit_width: float
        Slit width in arcseconds.
    :param slit_length: float
        Slit length in arcseconds
    :param position_angle:
        Position angle for the observation.
    :param verbosity:
        Verbosity > 0 will print verbose statements during the execution.
    :return: matplotlib.figure
        Return the matplotlib figure of the finding chart.
    """


    if offset_focus:
        im_ra = offset_df.loc[offset_id, offset_ra_column_name]
        im_dec = offset_df.loc[offset_id, offset_dec_column_name]
    else:
        im_ra = ra
        im_dec = dec

    coord_name = general_utils.coord_to_name(np.array([im_ra]), np.array([im_dec]),
                                  epoch="J")

    filename = image_folder_path + '/' + coord_name[0] + "_" + survey + "_" + \
               band + "*.fits"

    data, hdr, file_path = plotting.open_image(filename, im_ra, im_dec,
                                      fov,
                                      image_folder_path,
                                      verbosity=verbosity)

    # # Reproject data if position angle is specified
    # if position_angle != 0:
    #     hdr['CRPIX1'] = int(hdr['NAXIS1'] / 2.)
    #     hdr['CRPIX2'] = int(hdr['NAXIS2'] / 2.)
    #     hdr['CRVAL1'] = im_ra
    #     hdr['CRVAL2'] = im_dec
    #
    #     new_hdr = hdr.copy()
    #
    #     pa_rad = np.deg2rad(position_angle)
    #
    #     # TODO: Note that the rotation definition here reflects one axis
    #     # TODO: to make sure that it is a rotated version of north up east left
    #     # TODO: both 001 components have a negative sign!
    #     new_hdr['PC001001'] = -np.cos(pa_rad)
    #     new_hdr['PC001002'] = np.sin(pa_rad)
    #     new_hdr['PC002001'] = np.sin(pa_rad)
    #     new_hdr['PC002002'] = np.cos(pa_rad)
    #
    #     from reproject import reproject_interp
    #
    #     data, footprint = reproject_interp((data, hdr),
    #                                        new_hdr,
    #                                        shape_out=[hdr['NAXIS1'],
    #                                                   hdr['NAXIS2']])
    #     hdr = new_hdr

    if data is not None:
        # Plotting routine from here on.
        hdu = fits.PrimaryHDU(data, hdr)

        # De-rotate image along the position angle
        fig = aplpy.FITSFigure(hdu)

        if fov is not None:
            fig.recenter(im_ra, im_dec, radius=fov / 3600. * 0.5)

        try:
            zscale = ZScaleInterval()
            z1, z2 = zscale.get_limits(data)
            fig.show_grayscale(vmin=z1, vmax=z2)
        except Exception as e:
            print('Exception encountered: {}'.format(str(e)))
            fig.show_grayscale(pmin=10, pmax=99)

        fig.add_scalebar(fov/4/3600., '{:.1f} arcmin'.format(fov/4/60.),
                         color='black',
                         # font='serif',
                         linewidth=4
                         )

        if slit_length is not None and slit_width is not None:

            if position_angle == 0:
                _plot_slit(fig, im_ra, im_dec, slit_length, slit_width,
                           position_angle)
            else:
                _plot_slit(fig, im_ra, im_dec, slit_length, slit_width,
                           0)

        if offset_df is not None and offset_ra_column_name is not None and \
            offset_dec_column_name is not None and offset_mag_column_name is \
            not None and offset_id_column_name is not None:
            print("[INFO] Generating offsets for {}".format(filename))

            _plot_offset_stars(fig, ra, dec, offset_df, fov,
                               offset_id,
                               offset_ra_column_name,
                               offset_dec_column_name,
                               offset_mag_column_name,
                               offset_id_column_name,
                               label_position=label_position)

            _plot_info_box(fig, ra, dec, offset_df, offset_ra_column_name,
                           offset_dec_column_name, offset_mag_column_name)

        fig.show_circles(xw=ra, yw=dec, radius=aperture / 3600.,
                         edgecolor=red,
                         alpha=1, lw=3)

        fig.axis_labels.set_xtext('Right Ascension')
        fig.axis_labels.set_ytext('Declination')

        c = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree))
        title = 'RA= {0} ; DEC = {1}'.format(
            c.ra.to_string(precision=3, sep=":", unit=u.hour),
            c.dec.to_string(precision=3, sep=":", unit=u.degree, alwayssign=True))
        plt.title(title)

        fig.add_grid()
        fig.grid.show()

        fig.set_theme('publication')

        return fig




def _plot_slit(fig, ra, dec, slit_length, slit_width, position_angle):
    # slit_label = 'PA=${0:.2f}$deg\n \n'.format(position_angle)
    # slit_label += 'width={0:.1f}"; length={1:.1f}"'.format(
    #     slit_width, slit_length)

    fig = show_rectangles(fig, ra, dec, slit_width / 3600., slit_length / 3600.,
                        edgecolor='w', lw=1.0, angle=position_angle,
                        coords_frame='world')



position_dict = {"left": [8, 0], "right": [-8, 0], "top": [0, 5],
                 "bottom": [0, -5], "topleft": [8, 5]}

def _plot_offset_stars(fig, ra, dec, offset_df, fov, offset_id,
                       ra_column_name,
                       dec_column_name,
                       mag_column_name,
                       id_column_name,
                       label_position="left"):

    # Check if star is in image

    radius = fov / 25. / 3600.

    ra_pos, dec_pos = position_dict[label_position]

    fig.show_circles(xw=offset_df.loc[offset_id, ra_column_name],
                     yw=offset_df.loc[offset_id, dec_column_name],
                     radius=radius * 0.5,
                     edgecolor=blue,
                     lw=3)

    # fig.show_rectangles(offset_df.drop(offset_id)[ra_column_name],
    #                     offset_df.drop(offset_id)[dec_column_name],
    #                     radius, radius, edgecolor=blue, lw=1)

    abc_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

    for num, idx in enumerate(offset_df.index):

        if idx == 0:

            ra_off = offset_df.loc[idx, ra_column_name]
            dec_off = offset_df.loc[idx, dec_column_name]

            target_coords = SkyCoord(ra=ra, dec=dec,
                                     unit=(u.deg, u.deg),
                                     frame='icrs')
            offset_coords = SkyCoord(ra=ra_off,
                                     dec=dec_off, unit=(u.deg, u.deg),
                                     frame='icrs')

            separation = offset_coords.separation(target_coords).to(u.arcsecond)

            label = '{}'.format(abc_dict[num])

            if separation.value <= fov/2.:
                if idx == offset_id:
                    fig.add_label(ra_off + ra_pos * 5 / 3600. / 3.,
                                  dec_off + dec_pos * 5 / 3600. / 3., label,
                                  color=blue, size='x-large',
                                  verticalalignment='center', family='serif')

                else:
                    fig.add_label(ra_off + ra_pos * radius/5.,
                                  dec_off + dec_pos * radius/5., label,
                              color=blue, size='large',
                              verticalalignment='center', family='serif')



def _plot_info_box(fig, ra, dec, offset_df, ra_column_name, dec_column_name,
                       mag_column_name,):


    target_info = 'Target: RA={:.4f}, DEC={:.4f}'.format(ra, dec)

    info_list = [target_info]

    abc_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4:'E'}

    for num, idx in enumerate(offset_df.index):

        if idx == 0:


            ra_off = offset_df.loc[idx, ra_column_name]
            dec_off = offset_df.loc[idx, dec_column_name]

            target_coords = SkyCoord(ra=ra, dec=dec,
                                     unit=(u.deg, u.deg),
                                     frame='icrs')
            offset_coords = SkyCoord(ra=ra_off,
                                     dec=dec_off, unit=(u.deg, u.deg),
                                     frame='icrs')
            # Calculate position angles and separations (East of North)
            pos_angles = offset_coords.position_angle(target_coords).to(u.deg)
            separations = offset_coords.separation(target_coords).to(u.arcsecond)
            dra, ddec = offset_coords.spherical_offsets_to(target_coords)

            mag = offset_df.loc[idx, mag_column_name]
            info = '{}:\t RA={:.4f}, DEC={:.4f}, {}={:.2f}, PosAngle={' \
                   ':.2f}'.format(abc_dict[num],
                                                              ra_off,
                                                      dec_off, mag_column_name,
                                                                                 mag, pos_angles)
            info_off = 'Sep={:.2f}, Dra={:.2f}, ' \
                       'Ddec={:.2f}'.format(separations, dra.to(
                'arcsecond'), ddec.to('arcsecond'))
            info_list.append(info)
            info_list.append(info_off)


    ax = plt.gca()
    boxdict = dict(facecolor='white', alpha=1.0, edgecolor='none')
    ax.text(.02, 0.02, "\n".join(info_list), transform=ax.transAxes,
            fontsize='small',
            bbox=boxdict)


def show_rectangles(fig, xw, yw, width, height, angle=0, layer=False,
                    zorder=None, coords_frame='world', **kwargs):
    """
    Overlay rectangles on the current plot.

    ATTENTION! THIS IS A MODIFIED VERSION OF THE ORIGINAL APLPY ROUTINE THAT
    CORRECTLY ROTATES THE RECTANGLE AROUND ITS CENTER POSITION.
    see https://github.com/aplpy/aplpy/pull/327

    Parameters
    ----------
    xw : list or `~numpy.ndarray`
        The x positions of the centers of the rectangles (in world coordinates)
    yw : list or `~numpy.ndarray`
        The y positions of the centers of the rectangles (in world coordinates)
    width : int or float or list or `~numpy.ndarray`
        The width of the rectangle (in world coordinates)
    height : int or float or list or `~numpy.ndarray`
        The height of the rectangle (in world coordinates)
    angle : int or float or list or `~numpy.ndarray`, optional
        rotation in degrees (anti-clockwise). Default
        angle is 0.0.
    layer : str, optional
        The name of the rectangle layer. This is useful for giving
        custom names to layers (instead of rectangle_set_n) and for
        replacing existing layers.
    coords_frame : 'pixel' or 'world'
        The reference frame in which the coordinates are defined. This is
        used to interpret the values of ``xw``, ``yw``, ``width``, and
        ``height``.
    kwargs
        Additional keyword arguments (such as facecolor, edgecolor, alpha,
        or linewidth) are passed to Matplotlib
        :class:`~matplotlib.collections.PatchCollection` class, and can be
        used to control the appearance of the rectangles.
    """

    xw, yw, width, height, angle = aplpy.core.uniformize_1d(xw, yw, width,
                                                      height, angle)

    if 'facecolor' not in kwargs:
        kwargs.setdefault('facecolor', 'none')

    if layer:
        fig.remove_layer(layer, raise_exception=False)

    if coords_frame not in ['pixel', 'world']:
        raise ValueError("coords_frame should be set to 'pixel' or 'world'")

    # While we could plot the shape using the get_transform('world') mode
    # from WCSAxes, the issue is that the rotation angle is also measured in
    # world coordinates so will not be what the user is expecting. So we
    # allow the user to specify the reference frame for the coordinates and
    # for the rotation.

    if coords_frame == 'pixel':
        x, y = xw, yw
        w = width
        h = height
        a = angle
        transform = fig.ax.transData
    else:
        x, y = fig.world2pixel(xw, yw)
        pix_scale = aplpy.core.proj_plane_pixel_scales(fig._wcs)
        sx, sy = pix_scale[fig.x], pix_scale[fig.y]
        w = width / sx
        h = height / sy
        a = angle
        transform = fig.ax.transData

    # x = x - w / 2.
    # y = y - h / 2.
    #
    # patches = []
    # for i in range(len(x)):
    #     patches.append(Rectangle((x[i], y[i]), width=w[i], height=h[i],
    #                               angle=a[i]))

    xp = x - w / 2.
    yp = y - h / 2.
    radeg = np.pi / 180
    xr = (xp - x) * np.cos((angle) * radeg) - (yp - y) * np.sin(
        (angle) * radeg) + x
    yr = (xp - x) * np.sin((angle) * radeg) + (yp - y) * np.cos(
        (angle) * radeg) + y

    patches = []
    for i in range(len(xr)):
        patches.append(
            Rectangle((xr[i], yr[i]), width=w[i], height=h[i], angle=a[i]))

    # Due to bugs in matplotlib, we need to pass the patch properties
    # directly to the PatchCollection rather than use match_original.
    p = PatchCollection(patches, transform=transform, **kwargs)

    if zorder is not None:
        p.zorder = zorder
    c = fig.ax.add_collection(p)

    if layer:
        rectangle_set_name = layer
    else:
        fig._rectangle_counter += 1
        rectangle_set_name = 'rectangle_set_' + str(fig._rectangle_counter)

    fig._layers[rectangle_set_name] = c

    return fig