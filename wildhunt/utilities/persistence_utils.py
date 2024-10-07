#!/usr/bin/env python

# =========================================================================== #
# ===================== Functions for persistence check ===================== #
# =========================================================================== #

import os
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from wildhunt import image as whimg
from wildhunt import pypmsgs
from wildhunt.utilities import euclid_utils as eu
from wildhunt.utilities import general_utils as whut
from wildhunt.utilities import query_utils as whqu

msgs = pypmsgs.Messages()


# =========================================================================== #

ChunkedEncodingError = requests.exceptions.ChunkedEncodingError
ProtocolError = requests.urllib3.exceptions.ProtocolError

# =========================================================================== #

VERBOSE = 0

# =========================================================================== #


# TODO: Remove the query from this
# TODO: Add a parameter to the query for the search radius
def download_persistence_input_tbl_single_obj(ra, dec, user, fov=30 / 3600.0):
    """Fetch and return the persistence input table for a single astronomical object.

    This function constructs a SQL query to retrieve data from the
    `sedm.calibrated_frame` table for a specified astronomical object,
    based on its right ascension (RA) and declination (DEC) coordinates.
    It filters the results to include only entries from the 'NISP' instrument
    within the field of view (FOV) of the given coordinates. The results are
    returned as a pandas DataFrame containing relevant observational details.

    :param ra: The right ascension of the target object in degrees.
    :type ra: float
    :param dec: The declination of the target object in degrees.
    :type dec: float
    :param user: The user object for authentication to access the database.
    :type user: User
    :return: A DataFrame containing the persistence input details for the target object.
    :rtype: pandas.DataFrame
    """

    query_calib = (
        "SELECT observation_stack.instrument_name, observation_stack.observation_id, "
        "observation_stack.pointing_id, observation_stack.frame_seq, observation_stack.filter_name, "
        "observation_stack.file_name, observation_stack.ra, observation_stack.dec, "
        "observation_stack.calibrated_frame_oid, observation_stack.file_path "
        "FROM sedm.calibrated_frame AS observation_stack WHERE (instrument_name='NISP') "
        "AND ((observation_stack.fov IS NOT NULL AND "
        f"INTERSECTS(CIRCLE('ICRS',{ra},{dec},{fov}),observation_stack.fov)=1)) "
        "ORDER BY observation_id ASC"
    )

    query_mosaic = (
        "SELECT mosaic_product.file_name, mosaic_product.mosaic_product_oid, mosaic_product.tile_index, "
        "mosaic_product.instrument_name, mosaic_product.filter_name, mosaic_product.category, "
        "mosaic_product.second_type, mosaic_product.ra, mosaic_product.dec, mosaic_product.technique, "
        "mosaic_product.file_path FROM sedm.mosaic_product WHERE ((mosaic_product.fov IS NOT NULL AND "
        f"INTERSECTS(CIRCLE('ICRS',{ra},{dec},{fov}), "
        "mosaic_product.fov)=1)) ORDER BY mosaic_product.tile_index ASC"
    )

    query_dict = {"calib": query_calib, "mosaic": query_mosaic}
    output_dict = {}

    for img_type, query in query_dict.items():
        tbl = pd.read_csv(
            StringIO(whqu.sync_query(query=query, user=user, savepath=None))
        )

        # make access url with placeholder
        eu.build_cutout_access_urls(tbl)

        # update the placeholders
        eu.complete_cutout_access_urls(tbl, ra, dec, fov)

        output_dict[img_type] = tbl

    return output_dict


# =========================================================================== #


def generate_persistence_input_df(ras, decs, user):
    """Download and merge persistence input tables for multiple astronomical objects.

    This function iterates over arrays of right ascension (RA) and declination (DEC)
    coordinates, retrieving persistence input data for each object by calling the
    `download_persistence_input_tbl_single_obj` function. The resulting DataFrames are
    merged into a single DataFrame, ensuring that duplicates are removed based on the
    'calibrated_frame_oid'. The function also generates image access URLs for the merged
    table and returns the combined DataFrame along with a dictionary of individual tables.

    :param ras: An array of right ascension values for the target objects in degrees.
    :type ras: iterable of float
    :param decs: An array of declination values for the target objects in degrees.
    :type decs: iterable of float
    :param user: The user object for authentication to access the database.
    :type user: User
    :return: A tuple containing:
             - A DataFrame with the merged persistence input data, excluding the 'file_path' column.
             - A dictionary mapping each object (identified by RA and DEC) to its corresponding DataFrame.
    :rtype: tuple (pandas.DataFrame, dict)
    """
    tbls = {}

    for ra, dec in zip(ras, decs):
        # for now get only the calibrated images. If we then need the mosaic too, we just
        #  handle it here
        tbl_dict = download_persistence_input_tbl_single_obj(ra, dec, user)

        calib_input = tbl_dict["calib"]
        mosaic_input = tbl_dict["mosaic"]

        tbls[f"{ra}_{dec}"] = calib_input

    # note! For mostic we need to drop the duplicates using tile_idx, most likely
    merged_tbl = pd.concat(tbls.values()).drop_duplicates(
        "calibrated_frame_oid", ignore_index=True
    )

    eu.build_image_access_urls(merged_tbl)

    return merged_tbl.drop("file_path", axis=1), tbls


# =========================================================================== #


def create_persistence_qa_plot(
    ra, dec, result_df, cutout_dir, output_dir, mer_df=None, stack_df=None
):
    """Create a QA plot for the persistence check.

    :param ra: Right ascension in degrees
    :type ra: float
    :param dec: Declination in degrees
    :type dec: float
    :param result_df: DataFrame with the result data
    :type result_df: pandas.DataFrame
    :param cutout_dir: Path to the directory where the cutouts will be saved
    :type cutout_dir: str
    :param output_dir: Path to the directory where the output will be saved
    :type output_dir: str
    :param mer_df: DataFrame with the MER cutout data
    :type mer_df: pandas.DataFrame
    :param stack_df: DataFrame with the STACKED cutout data
    :type stack_df: pandas.DataFrame
    :return: None
    """

    # Reduce result_df to ra and dec in question with a delta of 1 arcsec
    delta_ra = 0.000278
    delta_dec = 0.000278
    df = result_df.query(
        "{} < ra < {} and {} < dec < {}".format(
            ra - delta_ra, ra + delta_ra, dec - delta_dec, dec + delta_dec
        )
    )
    df.query("processed == True and img_extension == img_extension", inplace=True)

    # Group result dataframe by observation id
    obs_id_gb = df.groupby("observation_id")

    for obs_id, df in obs_id_gb:
        plot_persistence_cutouts(
            ra,
            dec,
            obs_id,
            df,
            cutout_dir,
            output_dir,
            mer_df=mer_df,
            stack_df=stack_df,
        )


# =========================================================================== #


def plot_persistence_cutouts(
    ra, dec, obs_id, calib_df, cutout_dir, output_dir, mer_df=None, stack_df=None
):
    """Plot the cutouts of the persistence check.

    :param ra: Right ascension in degrees
    :type ra: float
    :param dec: Declination in degrees
    :type dec: float
    :param obs_id: Observation ID
    :type obs_id: int
    :param calib_df: DataFrame with the calibrated frame data
    :type calib_df: pandas.DataFrame
    :param cutout_dir: Path to the directory where the cutouts will be saved
    :type cutout_dir: str
    :param output_dir: Path to the directory where the output will be saved
    :type output_dir: str
    :param mer_df: DataFrame with the MER cutout data
    :type mer_df: pandas.DataFrame
    :param stack_df: DataFrame with the STACKED cutout data
    :type stack_df: pandas.DataFrame
    :return: None

    """

    # Plot setup
    cols = 4
    if mer_df is not None:
        cols += 1
    if stack_df is not None:
        cols += 1

    fig = plt.figure(figsize=(10, 2 * cols))
    fig.subplots_adjust(hspace=0.2)

    row_counter = 0

    # TODO: Add the MER mosaic cutouts
    if mer_df is not None:
        row_counter += 1

    # TODO: Add the STACKED cutouts
    if stack_df is not None:
        row_counter += 1

    # Add the NIR calibrated cutouts
    filter_dict = {"NIR_Y": 0, "NIR_J": 1, "NIR_H": 2}

    for _, row in calib_df.iterrows():
        filter_idx = filter_dict[row["filter_name"]]
        seq_idx = row["frame_seq"]
        filename = row["file_name"].replace(".fits", "_cutout.fits")

        file_path = os.path.join(cutout_dir, filename)

        cutout = whimg.Image(filename=file_path)

        ax_idx = seq_idx + filter_idx * 4 + 1
        ax = fig.add_subplot(3, cols, ax_idx)

        cutout._simple_plot(n_sigma=3, axis=ax, frameon=True, scalebar=None)

        cutout._add_aperture_circle(ax, ra, dec, 1)

        ax.set_title(
            "{} \n".format(row["filter_name"])
            + r" Flux {:.3f}+-{:.3f} $\mu$Jy".format(
                row["flux_aper_1.0"], row["flux_err_aper_1.0"]
            )
        )

        # Remove axis labels
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )

    # Save the plot
    plt.tight_layout()
    if VERBOSE:
        msgs.info(f"Current ra, dec: {ra}, {dec}")

    source_name = whut.coord_to_name([ra], [dec], epoch="J")[0]
    plt.savefig(
        os.path.join(
            output_dir, "{}_{}_persistence_cutouts.pdf".format(source_name, obs_id)
        )
    )


# =========================================================================== #


def check_persistence(
    ra,
    dec,
    calib_df,
    img_dir,
    cutout_dir,
    output_dir,
    cutout_fov=20,
    aperture_radii=np.array([1.0, 2.0]),
):
    """Check the persistence of a source in the Euclid images.

    :param ra: Right ascension in degrees
    :type ra: float
    :param dec: Declination in degrees
    :type dec: float
    :param calib_df: DataFrame with the calibrated frames
    :type calib_df: pandas.DataFrame
    :param img_dir: Path to the directory with the images
    :type img_dir: str
    :param cutout_dir: Path to the directory where the cutouts will be saved
    :type cutout_dir: str
    :param output_dir: Path to the directory where the output will be saved
    :type output_dir: str
    :param cutout_fov: Field of view of the cutout in arcsec. Defaults to 20 arcsec.
    :type cutout_fov: float
    :param aperture_radii: Radii of the apertures in arcsec. Defaults to [1., 2.]
    :type aperture_radii: numpy.ndarray
    :return: None

    """

    # Instantiate the SkyCoord object
    coord = SkyCoord(ra, dec, unit="deg", frame="icrs")

    # Downselect the calibrated frames to the NISP frames
    calib_columns = [
        "instrument_name",
        "observation_id",
        "pointing_id",
        "frame_seq",
        "filter_name",
        "file_name",
    ]

    result_df = calib_df.query('instrument_name == "NISP"')[calib_columns]
    result_df.loc[:, "ra"] = ra
    result_df.loc[:, "dec"] = dec
    result_df.loc[:, "processed"] = False
    result_df.loc[:, "img_extension"] = None

    for idx in result_df.index:
        file_path = os.path.join(img_dir, result_df.loc[idx, "file_name"])
        band = result_df.loc[idx, "filter_name"]
        survey = "Euclid-WIDE"

        # Test if the image exists
        if not os.path.exists(file_path):
            msgs.warn(f"File {file_path} does not exist.")
            continue
        else:
            msgs.info(f"Processing file {file_path}")
            result_df.loc[idx, "processed"] = True

        #  Cycle through extension to find the correct extension with the source
        hdul = fits.open(file_path)

        try:
            photfnu = hdul[0].header["PHOTFNU"]
            photrelex = hdul[0].header["PHRELEX"]
        except KeyError:
            photfnu = None
            photrelex = None

        for hdu in [h for h in hdul if "SCI" in h.name]:
            header = hdu.header
            wcs = WCS(header)

            # Test if the coordinate is within the image
            x, y = wcs.world_to_pixel(coord)

            # If source in extension, then
            if 0 < x < header["NAXIS1"] and 0 < y < header["NAXIS2"]:
                msgs.info("Coordinate is within the image")
                msgs.info(f"Extension: {hdu.name}")

                result_df.loc[idx, "img_extension"] = hdu.name

                # Load the image
                img = whimg.Image(file_path, exten=hdu.name, survey=survey, band=band)
                # Create a cutout image
                cutout = img.get_cutout_image(
                    coord.ra.value, coord.dec.value, cutout_fov
                )
                # Save the cutout image
                cutout_name = result_df.loc[idx, "file_name"].replace(
                    ".fits", "_cutout.fits"
                )
                cutout_path = os.path.join(cutout_dir, cutout_name)
                cutout.to_fits(cutout_path)

                # Calculate forced aperture photometry

                # nanomag_correction = 1 # ToDo: Implement the conversion to physical units
                zp_ab = img.header["ZPAB"]
                zp_ab_err = img.header["ZPABE"]
                gain = img.header["GAIN"]
                photreldt = img.header["PHRELDT"]

                msgs.info(f"ZPAB: {zp_ab}, {band}")

                nanomag_correction = np.power(10, 0.4 * (22.5 - zp_ab)) * gain

                phot_result = img.calculate_aperture_photometry(
                    coord.ra.value,
                    coord.dec.value,
                    nanomag_correction,
                    aperture_radii=aperture_radii,
                    exptime_norm=1,
                    background_aperture=np.array([7, 10.0]),
                )

                # Create a metric that flags persistence
                # ToDo!!!

                # Save the results to the result DataFrame
                prefix = "{}_{}".format(survey, band)
                for radius in aperture_radii:
                    raw_flux = phot_result[
                        "{}_raw_aper_sum_{:.1f}arcsec".format(prefix, radius)
                    ]
                    raw_flux_err = phot_result[
                        "{}_raw_aper_sum_err_{:.1f}arcsec".format(prefix, radius)
                    ]
                    # flux = phot_result['{}_flux_aper_{:.1f}arcsec'.format(prefix, radius)]
                    # flux_err = phot_result['{}_flux_err_aper_{:.1f}arcsec'.format(prefix, radius)]
                    # snr = phot_result['{}_snr_aper_{:.1f}arcsec'.format(prefix, radius)]
                    # abmag = phot_result['{}_mag_aper_{:.1f}arcsec'.format(prefix, radius)]
                    # abmag_err = phot_result['{}_mag_err_aper_{:.1f}arcsec'.format(prefix, radius)]

                    # Calculation according to
                    # https://apceuclidccweb-pp.in2p3.fr/Documentation/NIR/NIR_AbsolutePhotometry/develop/0.5.0/md_README.html
                    snr = raw_flux / raw_flux_err
                    abmag = -2.5 * np.log10(raw_flux) + zp_ab

                    if photfnu is not None and photrelex is not None:
                        flux = (
                            raw_flux / 87.2248 * photfnu * photrelex * photreldt
                        )  # flux in micro Jy
                        flux_err = (
                            raw_flux_err / 87.2248 * photfnu * photrelex * photreldt
                        )
                    else:
                        # only needed for cutouts
                        flux = 10 ** (-((abmag - 8.9) / 2.5)) * 1e6
                        flux_err = flux / snr

                    result_df.loc[idx, "raw_aper_sum_{:.1f}".format(radius)] = raw_flux
                    result_df.loc[idx, "flux_aper_{:.1f}".format(radius)] = flux
                    result_df.loc[idx, "flux_err_aper_{:.1f}".format(radius)] = flux_err
                    result_df.loc[idx, "snr_aper_{:.1f}".format(radius)] = snr
                    result_df.loc[idx, "abmag_aper_{:.1f}".format(radius)] = abmag
                    result_df.loc[idx, "abmag_err_aper_{:.1f}".format(radius)] = (
                        zp_ab_err
                    )

    # Create the persistence plot
    create_persistence_qa_plot(ra, dec, result_df, cutout_dir, output_dir)

    coord_name = whut.coord_to_name([ra], [dec], epoch="J")[0]
    table_name = "{}_persistence_check.csv".format(coord_name)
    result_df.to_csv(os.path.join(output_dir, table_name), index=False)
