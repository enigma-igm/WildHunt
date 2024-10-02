#!/usr/bin/env python
import logging
import os
from http.client import IncompleteRead
from pathlib import Path
from urllib.error import HTTPError

import numpy as np
import pandas as pd
import requests
from astropy import units
from astropy.coordinates import SkyCoord
from tqdm import tqdm

from wildhunt import pypmsgs
from wildhunt.utilities import download_utils as whdu
from wildhunt.utilities import persistence_utils as whpu

# =========================================================================== #

msgs = pypmsgs.Messages()

ChunkedEncodingError = requests.exceptions.ChunkedEncodingError
ProtocolError = requests.urllib3.exceptions.ProtocolError

logger = logging.getLogger(__name__)

# =========================================================================== #
# ============== General notes about the tables on the archive ============== #
# =========================================================================== #


# random notes from the archive (Updated 20240723)
# sedm.aux_stacked -> auxiliary data (PSF and other data model)
# sedm.calibrated_detectors -> information about the detectors (including zeropoints)
# sedm.calibrated_frame -> contains information about calibrated images (including paths)
#   This seem to also include the MER catalogue in the future (F-006)
# sedm.column_values -> unclear
# sedm.combined_spectra and sedm.combined_spectra_product -> spectral products, unclear in what shape or form
# sedm.mosaic_product -> mosaic including filters from different surveys - unclear what their use or utility is
# sedm.observation_mode -> useful for the IDs in any observation mode column -> see CSV file
# sedm.observation_mosaic -> id match for the mosaic - if needed, this needs some digging for the id match
# sedm.observation_stack -> all the info about the stacked images
#   This seem to be what we actually want to use!


# =========================================================================== #


# For certificate warning see here:
# https://stackoverflow.com/questions/30405867/how-to-get-python-requests-to-trust-a-self-signed-ssl-certificate
# as far as I understand strongly recommended but not strictly needed


# =========================================================================== #


if os.environ.get("WILDHUNT_LOCALPATH") is None:
    LOCAL_PATH = Path.home()
else:
    LOCAL_PATH = Path(os.environ.get("WILDHUNT_LOCALPATH"))

CERT_KEY = LOCAL_PATH / "eas-esac-esa-int-chain.pem"
if not CERT_KEY.exists():
    CERT_KEY = False


# =========================================================================== #


# Note: for VIS we want calibrated QUAD frame, the non quad version is already
#  missing some source that is present in the quad
product_type_dict = {
    "calib": ["DpdVisCalibratedQuadFrame", "DpdNirCalibratedFrame"],
    "stacked": ["DpdVisStackedFrame", "DpdNirStackedFrame"],
    "mosaic": ["DpdMerBksMosaic", "DpdMerBksMosaic"],  # same for VIS and NIR
}

# =========================================================================== #
# ==================== SAS OTF's table related functions ==================== #
# =========================================================================== #


def load_full_table_from_sas(
    user,
    query_table,
    fname="",
    tbl_in=None,
    use_local_tbl=False,
    sync_query=True,
):
    """Load a catalogue from a specified file or query the EUCLID TAP server.

    This function attempts to load a catalogue either from a specified CSV file
    or by querying the EUCLID TAP server. If both a filename and a DataFrame
    are provided, the function will ignore the DataFrame and re-download the
    catalogue.

    :param fname: The path to the CSV file from which to load the catalogue.
    :type fname: str
    :param tbl_in: An optional DataFrame containing the catalogue data.
    :type tbl_in: pandas.DataFrame or None
    :param query_table: The name of the table to query if downloading is required.
    :type query_table: str
    :param user: The user object containing session cookies for authentication.
                 If None, a new user object will be created.
    :type user: User or None
    :param use_local_tbl: If True, load the catalogue from the local CSV file
                          if it exists; otherwise, download the table.
    :type use_local_tbl: bool
    :raises ValueError: If both a file name and a DataFrame are provided.
    :raises IOError: If there is an error reading the CSV file or downloading the table.
    :return: A tuple containing the loaded DataFrame and the user object.
    :rtype: (pandas.DataFrame, User)
    """
    user._check_for_login()

    if fname != "" and tbl_in is not None:
        msgs.warn(
            "Received both a table object and a path."
            f"Ignoring input table object and redownloading to {fname}."
        )

    # either read from file (and by default query and download it)
    # or directly pass a table
    if fname != "":
        fname = Path(fname)

        if fname.exists() and use_local_tbl:
            msgs.info("Requested usa of local table. Table exists, loading it.")
            tbl = pd.read_csv(fname)
        else:
            msgs.info(f"Querying archive and saving results to {fname}.")
            tbl = whdu.download_full_table_from_sas(
                query_table, user, fname, sync_query=sync_query
            )
    elif tbl_in is not None:
        tbl = tbl_in.copy()
    else:
        msgs.error(
            "Either provide a path to load a catalogue from, or the table itself."
        )

    return tbl, user


# =========================================================================== #


def parse_sas_catalogue(tbl_in, inplace=False, force=False):
    """Parse a catalogue DataFrame to add and rename columns for easier access.

    This function processes an input DataFrame, adding necessary columns for
    cutout access URLs and image access URLs, and renames existing columns
    to conform to standard naming conventions. It can operate in place or
    return a modified copy of the input DataFrame.

    :param tbl_in: The input DataFrame containing the catalogue data.
    :type tbl_in: pandas.DataFrame
    :param inplace: If True, modify the input DataFrame directly; otherwise, return a new copy.
    :type inplace: bool
    :param force: If True, force the addition of new columns even if they already exist.
    :type force: bool
    :return: The modified DataFrame with added and renamed columns.
    :rtype: pandas.DataFrame
    """
    if inplace:
        tbl = tbl_in
    else:
        tbl = tbl_in.copy()

    if "cutout_access_url" not in tbl.columns or force:
        msgs.info("Added `cutout_access_url` column.")
        # names are the same for both stack and calib, which is what we'll want to use most of the time
        build_cutout_access_urls(tbl)

    # TODO: (Future us!) if this gets very slow (unlikely), merge the two loops
    if "image_access_url" not in tbl.columns or force:
        msgs.info("Added `image_access_url` column.")
        build_image_access_urls(tbl)

    # needed for ivoa_score
    if "s_ra" in tbl.columns:
        msgs.info("Renaming columns s_ra, s_dec, dataproduct_subtype and filter.")
        tbl.rename(
            {
                "s_ra": "ra",
                "s_dec": "dec",
                "dataproduct_subtype": "product_type",
                "filter": "filter_",
            },
            axis=1,
            inplace=True,
        )

    # needed for stack and calib
    if "filter_name" in tbl.columns:
        msgs.info("Renaming column filter_name.")
        tbl.rename(
            {"filter_name": "filter_"},
            axis=1,
            inplace=True,
        )

    return tbl


# =========================================================================== #
# ============================= Cutout functions ============================ #
# =========================================================================== #


def build_cutout_access_url(
    path,
    filename,
    ra,
    dec,
    fov,
    obsid=None,
    tileidx=None,
    search_type="CIRCLE",
):
    """Helper function for `build_cutout_access_url`.
    Construct a cutout access URL based on the provided path and filename of the full image.

    This function builds a URL that allows access to cutout images. It constructs
    the first part of the URL by extracting the collection name from the given path.
    The function requires either an observation ID (`obsid`) or a tile index (`tileidx`),
    but not both, and will generate the appropriate URL parameters accordingly.

    :param path: The directory path where the files are located.
    :type path: str
    :param filename: The name of the file for which the URL is being constructed.
    :type filename: str
    :param obsid: The observation ID; must be None if `tileidx` is provided.
    :type obsid: str or None
    :param tileidx: The tile index; must be None if `obsid` is provided.
    :type tileidx: str or None
    :return: The constructed cutout access URL.
    :rtype: str
    """
    assert not (
        obsid is not None and tileidx is not None
    ), "Either `obsid` or `tileindex` must be None"

    # the url to be send the request to is always in the form of (where {} indicates a variable:
    #  https://easotf.esac.esa.int/sas-cutout/cutout?
    #  filepath={filepath}/{filename}&collection={collection}&obsid={obsid}
    #  &POS=CIRCLE,187.89,29.54,0.0333333333333333
    base = "https://easotf.esac.esa.int/sas-cutout/cutout"

    # get collection from path itself
    collection = path.split("/")[-2]

    # there should really be no way to have both not None
    if obsid is None and tileidx is not None:
        params = f"filepath={path}/{filename}&collection=MER&tileindex={tileidx}"
    elif obsid is not None and tileidx is None:
        params = f"filepath={path}/{filename}&collection={collection}&obsid={obsid}"
    else:
        raise ValueError(
            "Both obsid and tileidx were not none. Something went wrong, "
            "please report on GitHub"
        )

    search = f"POS={search_type},{ra},{dec},{fov}"
    return f"{base}?{params}?{search}"


# =========================================================================== #


def build_cutout_access_urls(tbl, ra, dec, fov, search_type="CIRCLE"):
    """Generate cutout access URLs for files in the given DataFrame.

    This function iterates through the provided DataFrame, constructing cutout access
    URLs for the files identified by their paths and names. The function handles two
    scenarios: if the observation IDs are present, it uses them; otherwise, it falls
    back on tile indexes to generate the URLs. The resulting URLs are added to a new
    column in the DataFrame. Note: this modifies the dataframe in place!

    There is a distinction in how urls are generated: for cutouts service requires
    a different set of parameters mosaic and calibrated/stacked images.

    :param tbl: The input DataFrame containing the file paths, file names, and
                either observation IDs or tile indexes.
    :type tbl: pandas.DataFrame
    :return: None; modifies the input DataFrame in place by adding a column for
             cutout access URLs.
    """
    # unifies the access url for both the mer tiles and everything else
    access_urls = []

    # MER wants the TILEINDEX instead of OBSID, switch the column here
    if "observation_id" in tbl.columns:
        for _p, _f, _o in zip(
            tbl["file_path"], tbl["file_name"], tbl["observation_id"]
        ):
            access_urls.append(
                build_cutout_access_url(
                    _p.strip(),
                    _f.strip(),
                    ra,
                    dec,
                    fov,
                    obsid=_o,
                    search_type=search_type,
                )
            )
    else:
        for _p, _f, _ti in zip(tbl["file_path"], tbl["file_name"], tbl["tile_index"]):
            access_urls.append(
                build_cutout_access_url(
                    _p.strip(),
                    _f.strip(),
                    ra,
                    dec,
                    fov,
                    tileidx=_ti,
                    search_type=search_type,
                )
            )

    tbl["cutout_access_url"] = access_urls


# =========================================================================== #
# ============================= Image functions ============================= #
# =========================================================================== #


def build_image_access_urls(tbl):
    """Create image access URLs for files listed in the DataFrame.

    This function generates image access URLs for each file name in the provided
    DataFrame. It constructs the URLs using a predefined base address and includes
    the file names as parameters. The resulting URLs are added to a new column in
    the DataFrame. Note: this modifies the dataframe in place!

    :param tbl: The input DataFrame containing the file names for which image access
                URLs are to be generated.
    :type tbl: pandas.DataFrame
    :return: None; modifies the input DataFrame in place by adding a column for
             image access URLs.
    """
    base = "https://easotf.esac.esa.int/sas-dd/data"

    tbl["image_access_url"] = [
        f"{base}?file_name={_fn}&release=sedm&RETRIEVAL_TYPE=FILE"
        for _fn in tbl["file_name"]
    ]


# =========================================================================== #


def download_images_from_sas(
    df,
    user,
    img_outpath,
    img_outname,
    download_function=whdu.download_with_progress_bar,
    url_column="cutout_access_url",
):
    """Download images from the SAS using information from the provided DataFrame.

    This function iterates through a DataFrame containing image information and
    downloads images from the specified SAS (Science Archive Service) URLs.
    The function allows the user to specify an output path and file name for the
    downloaded images, with options for verbosity and progress tracking during
    the download process.

    :param df: The DataFrame containing image details, including the access URLs and
               filenames for the images to be downloaded.
    :type df: pandas.DataFrame
    :param user: The user object for authentication to access the SAS.
    :type user: User
    :param img_outpath: The directory path where the downloaded images will be saved.
    :type img_outpath: str or pathlib.Path
    :param img_outname: The base name for the downloaded images; if None, the filenames
                        from the DataFrame will be used.
    :type img_outname: str or None
    :param verbose: If True, enable verbose logging for download progress. Defaults to True.
    :type verbose: bool
    :param donwload_function: The function to use for downloading images (default is
                             `download_with_progress_bar`).
    :type donwload_function: callable
    :return: None; downloads images by saving them to the specified output path.
    """
    # TODO: redirect errors to tqdm
    for _, row in (iter := tqdm(df.iterrows())):
        if img_outname is None:
            current_img_outname = row["file_name"]
        else:
            current_img_outname = img_outname + f"_{_}"

        iter.set_description(f"Downloading {current_img_outname} to {img_outpath}.")

        try:
            download_function(row[url_column], user, img_outpath / current_img_outname)
        except (IncompleteRead, HTTPError, AttributeError, ValueError) as err:
            msgs.warn(f"Download error encountered: {err}")
            logger.info(f"Download of {current_img_outname} unsuccessful")


# =========================================================================== #
# =============== Sorting functions: get closest images and ================= #
# =============== generate the cutout url for local tables  ================= #
# =========================================================================== #


@units.quantity_input()
def get_closest_image_using_local_tbl(
    ra: units.deg,
    dec: units.deg,
    cat,
    band,
    ra_cat="ra",
    dec_cat="dec",
):
    """Retrieve the closest image URLs based on given coordinates from a catalogue.
    Should provide the same output as the OTF, but using a single query, or a local table.
    Useful if there is not available internet connection, for example.

    This function identifies the closest images to the specified right ascension (RA)
    and declination (DEC) coordinates from a provided catalogue. It calculates the
    angular separation between the target coordinates and the catalogue entries to
    determine proximity. If no images are found within a specified threshold, a warning
    is issued and an empty list is returned. The function can also handle different
    column names for RA and DEC if specified.

    :param ra: The right ascension of the target location.
    :type ra: astropy.units.Quantity (degrees)
    :param dec: The declination of the target location.
    :type dec: astropy.units.Quantity (degrees)
    :param cat: The catalogue DataFrame containing images and their coordinates.
    :type cat: pandas.DataFrame
    :param band: The band of observation to filter images.
    :type band: str
    :param ra_cat: The name of the column in `cat` containing RA coordinates.
                   Defaults to 'ra'.
    :type ra_cat: str
    :param dec_cat: The name of the column in `cat` containing DEC coordinates.
                    Defaults to 'dec'.
    :type dec_cat: str
    :return: A tuple containing the DataFrame of closest images and the minimum distance
             to the target coordinates in arcseconds.
    :rtype: tuple (pandas.DataFrame, astropy.units.Quantity)
    """
    # this takes the closes images to the target
    target_coord = SkyCoord(ra, dec, frame="icrs")
    cat_coord = SkyCoord(
        cat[ra_cat].to_numpy() * ra.unit,
        cat[dec_cat].to_numpy() * dec.unit,
        frame="icrs",
    )
    dist = target_coord.separation(cat_coord).to(units.arcsec)
    if len(dist) == 0:
        return pd.DataFrame(data={"cutout_access_url": []}), None

    # probably too much
    # TODO: Is there a better way to determine the distance?
    elif dist.min() > 1.0 * units.deg:
        msgs.warn(
            f"Images in band {band} are all farther than 1 deg "
            f"for the object at ra, dec: {ra.value:.4f}, {dec.value:.4f}."
        )
        return pd.DataFrame(data={"cutout_access_url": []}), dist.min()

    # Add distance as column and sort df by distance
    # TODO: Check that this works for mosaic and stack
    # probably not, I bet I'll need to change the key for the query for MER
    cat["dist"] = dist.value
    sorted_cat = cat.sort_values("dist", ignore_index=True)

    # get the first observation_id, return the catalogue
    out = sorted_cat.query(f"observation_id == {sorted_cat['observation_id'][0]}")
    return out, out["dist"].values


# =========================================================================== #


@units.quantity_input()
def get_closest_image_using_sas_otf(
    ra: units.deg,
    dec: units.deg,
    cat,
    band,
    ra_cat="ra",
    dec_cat="dec",
):
    """Fetch the closest image URLs from the SAS OTF service based on given coordinates.

    This function is intended to retrieve the closest image URLs from the
    SAS (Science Archive Service) On-The-Fly (OTF) service for a specified
    right ascension (RA) and declination (DEC) coordinate. However, the
    implementation is not currently provided, resulting in a NotImplementedError.

    :param ra: The right ascension of the target location.
    :type ra: astropy.units.Quantity (degrees)
    :param dec: The declination of the target location.
    :type dec: astropy.units.Quantity (degrees)
    :param cat: The catalogue DataFrame containing images and their coordinates.
    :type cat: pandas.DataFrame
    :param band: The band of observation to filter images.
    :type band: str
    :param ra_cat: The name of the column in `cat` containing RA coordinates.
                   Defaults to 'ra'.
    :type ra_cat: str
    :param dec_cat: The name of the column in `cat` containing DEC coordinates.
                    Defaults to 'dec'.
    :type dec_cat: str
    :raises NotImplementedError: This function has not been implemented yet.
    """
    # FIXME!
    raise NotImplementedError


# =========================================================================== #


@units.quantity_input()
def get_download_urls_of_closest_images(
    ra: units.deg,
    dec: units.deg,
    fov: units.arcsec,
    cat,
    band,
    search_function,  # either get_closest_image_using_local_tbl or get_closest_image_using_local_tbl
    search_type="CIRCLE",
    ra_cat="ra",
    dec_cat="dec",
):
    """Generate download URLs for cutout images based on given coordinates and search parameters.

    This function retrieves the closest cutout image URLs based on the specified
    right ascension (RA) and declination (DEC) coordinates. It also constructs
    download URLs using the provided `side` and `search_type` parameters. The
    function converts the `side` from arcseconds to degrees and gathers the
    necessary URLs for downloading.

    :param ra: The right ascension of the target location.
    :type ra: astropy.units.Quantity (degrees)
    :param dec: The declination of the target location.
    :type dec: astropy.units.Quantity (degrees)
    :param side: The size of the cutout image in arcseconds.
    :type side: astropy.units.Quantity (arcseconds)
    :param cat: The catalogue DataFrame containing image data.
    :type cat: pandas.DataFrame
    :param band: The band of observation for filtering images.
    :type band: str
    :param search_type: The type of search to perform, defaults to "CIRCLE".
    :type search_type: str
    :return: A list of download URLs for the cutout images.
    :rtype: list of str
    """
    fov = fov.to(units.deg).value

    # this is now a dataframe
    closest_images = search_function(
        ra,
        dec,
        cat,
        band,
        ra_cat=ra_cat,
        dec_cat=dec_cat,
    )[0]

    return build_cutout_access_urls(
        closest_images, ra.value, dec.value, fov, search_type
    )


# =========================================================================== #
# ========================== High-level functions =========================== #
# =========================================================================== #


@units.quantity_input()
def generate_wildhunt_download_df(
    ra_arr: units.deg,
    dec_arr: units.deg,
    fov: units.arcsec,
    cat,
    requested_band,
    search_function=get_closest_image_using_local_tbl,  # TODO: Change this once the other function is implemented
    search_type="CIRCLE",
    ra_cat="ra",
    dec_cat="dec",
):
    """Create a DataFrame of download URLs for cutout images based on input coordinate arrays.

    This function constructs a DataFrame containing download URLs for cutout images
    corresponding to arrays of right ascension (RA) and declination (DEC) coordinates.
    It filters the input catalogue based on the requested band and constructs the
    necessary URLs for each coordinate pair, organizing the results into a structured DataFrame.

    :param ra_arr: An array of right ascension values for the target locations.
    :type ra_arr: astropy.units.Quantity (degrees)
    :param dec_arr: An array of declination values for the target locations.
    :type dec_arr: astropy.units.Quantity (degrees)
    :param side: The size of the cutout images in arcseconds.
    :type side: astropy.units.Quantity (arcseconds)
    :param cat: The catalogue DataFrame containing relevant image information.
    :type cat: pandas.DataFrame
    :param requested_band: The band of observation to filter images.
    :type requested_band: str
    :return: A DataFrame containing columns for RA, DEC, download URLs, and filter information.
    :rtype: pandas.DataFrame
    """
    # transition layer to wildhunt
    filtered_cat = cat.query(f"filter_ == '{requested_band}'").reset_index(drop=True)

    urls_, filter_, ras, decs = [], [], [], []
    for _ra, _dec in zip(ra_arr, dec_arr):
        urls = get_download_urls_of_closest_images(
            _ra,
            _dec,
            fov,
            filtered_cat,
            requested_band,
            search_function,
            search_type=search_type,
            ra_cat=ra_cat,
            dec_cat=dec_cat,
        )
        bands = [requested_band] * len(urls)

        # build columns for the dataframe
        urls_.append(urls)
        filter_.append(bands)
        ras.append([_ra.value] * len(urls))
        decs.append([_dec.value] * len(urls))

    return pd.DataFrame(
        data={
            "ra": np.hstack(ras),
            "dec": np.hstack(decs),
            "url": np.hstack(urls_),
            "filter": np.hstack(filter_),
        }
    )


# =========================================================================== #


def download_cutouts(obj_ra, obj_dec, img_urls, cutout_outpath, user):
    """Download cutout images from given URLs and save them to the specified folder.

    This function downloads cutout images from a list of URLs and saves them to a
    designated folder. It handles the file naming based on the provided right ascension
    (RA) and declination (DEC) values and checks for successful HTTP responses.
    The user can provide login credentials; if not, a user object will be instantiated.

    :param obj_ra: The right ascension of the target object for naming downloaded files.
    :type obj_ra: float
    :param obj_dec: The declination of the target object for naming downloaded files.
    :type obj_dec: float
    :param img_urls: A list of URLs for downloading the cutout images.
    :type img_urls: list of str
    :param folder: The folder path where the images will be saved.
    :type folder: pathlib.Path
    :param user: (Optional) A user object for authentication; if None, a new user will be created.
    :type user: User or None
    :param verbose: Level of verbosity for log messages; higher numbers yield more output.
    :type verbose: int
    :return: None; downloads images and saves them to the specified folder.
    """
    cutout_outpath = Path(cutout_outpath)
    user._check_for_login()

    for img_url in img_urls:
        logger.info(f"Image URL: {img_url}")

        if "NIR" in img_url:
            band = img_url.split("IMAGE_")[1][0]
        else:
            band = "VIS"

        fname_out = cutout_outpath / f"{obj_ra:.5f}_{obj_dec:.5f}_{band}.fits"

        whdu.download_without_progress_bar(
            img_url, user, fname_out, check_for_existing=False
        )


# =========================================================================== #


# TODO: Clean up from here
def prepare_sas_catalogue(
    cat_path,
    sas_catalogue,
    user,
    product_type,
    product_type_dict,
    use_local_tbl=False,  # default in any case, better make it explicit
):
    """Load and filter a SAS catalogue based on specified product types.

    This function retrieves a catalogue from the specified SAS source, optionally
    using a local table for efficiency. It then filters the catalogue to include
    only those data products matching the specified product type(s). The function
    also renames columns as needed for consistent handling across different tables.

    :param cat_path: The path to the catalogue file to be loaded.
    :type cat_path: str or pathlib.Path
    :param sas_catalogue: The name of the SAS catalogue to query.
    :type sas_catalogue: str
    :param user: The user object for authentication to access the catalogue.
    :type user: User
    :param product_type: The type of product to filter in the catalogue.
    :type product_type: str
    :param product_type_dict: A dictionary mapping product types to their corresponding values.
    :type product_type_dict: dict
    :param use_local_tbl: If True, utilize a local table for loading the catalogue instead
                          of querying the SAS; defaults to False.
    :type use_local_tbl: bool
    :return: A DataFrame containing the filtered catalogue data products of the specified type.
    :rtype: pandas.DataFrame
    :raises ValueError: If no data product of the specified type is found in the catalogue.
    """
    cat = load_catalogue(
        fname=cat_path,
        query_table=sas_catalogue,
        user=user,
        use_local_tbl=use_local_tbl,
    )[0]

    # select only the products that you want
    # parse_sas_catalogue just does some renaming to make sure that
    # different tables can be handles with the same code
    parse_sas_catalogue(cat, inplace=True)

    cat_data_prod = cat.query(
        f"product_type == '{product_type_dict[product_type][0]}' "
        + f"or product_type == '{product_type_dict[product_type][1]}'"
    ).reset_index()

    if cat_data_prod.empty:
        msgs.error(
            f"No data product of type {product_type} found. Are you using the correct table?"
        )
        raise ValueError
    else:
        return cat_data_prod


# =========================================================================== #


@units.quantity_input()
def download_all_images(
    ra: units.deg,
    dec: units.deg,
    user,
    cat_outpath,
    img_outpath,
    img_outname=None,
    img_type="calib",
    tile_cat=None,
):
    """Download all images related to a specified set of coordinates from the SAS.

    This function retrieves and downloads images associated with the provided
    right ascension (RA) and declination (DEC) coordinates. The `img_type` parameter
    specifies both the table to query and the data product used in the download
    process. If a tile catalogue is not provided, it is created based on the
    specified `img_type`. The images are grouped by identifier to optimize
    the download process.

    :param ra: An array of right ascension values for the target locations.
    :type ra: astropy.units.Quantity (degrees)
    :param dec: An array of declination values for the target locations.
    :type dec: astropy.units.Quantity (degrees)
    :param user: The user object for authentication to access the SAS.
    :type user: User
    :param cat_outpath: The directory path where the catalogue will be saved.
    :type cat_outpath: str or pathlib.Path
    :param img_outpath: The directory path where the downloaded images will be saved.
    :type img_outpath: str or pathlib.Path
    :param img_outname: (Optional) The base name for the downloaded images; if None,
                        the filenames from the DataFrame will be used.
    :type img_outname: str or None
    :param img_type: The type of image to download; should be either 'mosaic' or 'calib'.
    :type img_type: str
    :param verbose: If True, enable verbose logging for download progress. Defaults to False.
    :type verbose: bool
    :param tile_cat: (Optional) A previously prepared tile catalogue; if None, a new
                     catalogue will be created.
    :type tile_cat: pandas.DataFrame or None
    :return: None; performs the download and saves the relevant catalogue.
    :raises ValueError: If `img_type` is neither 'mosaic' nor 'calib'.
    """

    cat_outpath = Path(cat_outpath)
    img_outpath = Path(img_outpath)

    # Restric image type
    if img_type not in ["mosaic", "calib"]:
        raise ValueError("`img_type` should be either 'mosaic' or 'calib'")

    # download new tile catalogue and process it
    if tile_cat is None:
        tile_cat = prepare_sas_catalogue(
            cat_outpath,
            img_type,
            user,
            img_type,
            product_type_dict,
            use_local_tbl=False,
        )

    # download all images - the logic behind this is to group by identifier
    #  to minimize the number of images to download
    df = None

    for ra_, dec_ in zip(ra, dec):
        partial = get_closest_image_url(
            ra_,
            dec_,
            tile_cat,
            "[VIS, Y, J, H]",
            query_sas_otf=False,
        )[0]

        df = partial if df is None else pd.concat([df, partial], ignore_index=True)

    # this gets all 4 bands at the same time
    unique_identifier = (
        "mosaic_product_oid" if img_type == "mosaic" else "calibrated_frame_oid"
    )
    df_urls = df.drop_duplicates(unique_identifier, ignore_index=True)

    # actually download the images
    download_images_from_sas(df_urls, user, img_outpath, img_outname)

    # and save the catalogue just in case it is needed for anything
    df_urls.to_csv(img_outpath / "Euclid_urls_vetted.csv")


# =========================================================================== #


# TODO: This should use units as well
def persistance_pipeline(
    ras,
    decs,
    user,
    output_full_img_dir,
    output_cutout_dir,
    output_persistence_check_dir,
    download_function=whdu.download_with_progress_bar,
    **kwargs,
):
    """Execute the persistence pipeline for multiple astronomical objects.

    This function orchestrates the workflow for handling persistence data for
    multiple astronomical objects identified by their right ascension (RA)
    and declination (DEC) coordinates. It first retrieves the necessary input
    data by downloading persistence tables. The pipeline then downloads
    images from the SAS (Science Archive Service) and subsequently performs
    persistence checks for each specified object. The function accepts various
    parameters for configuration and outputs.

    :param ras: An array of right ascension values for the target objects in degrees.
    :type ras: iterable of float
    :param decs: An array of declination values for the target objects in degrees.
    :type decs: iterable of float
    :param user: The user object for authentication to access the database.
    :type user: User
    :param output_full_img_dir: The directory path where full images will be saved.
    :type output_full_img_dir: str or pathlib.Path
    :param output_cutout_dir: The directory path where cutout images will be saved.
    :type output_cutout_dir: str or pathlib.Path
    :param output_persistence_check_dir: The directory path for persistence check results.
    :type output_persistence_check_dir: str or pathlib.Path
    :param verbose: If True, enable verbose logging for the pipeline process.
    :type verbose: bool
    :param kwargs: Additional keyword arguments for flexibility in processing.
    :return: None; performs image downloading and persistence checking without returning values.
    """
    downloaded_table, dict_input_tbl = whpu.generate_persistence_input_df(
        ras, decs, user
    )

    msgs.info(f"Starting download of {downloaded_table.shape[0]} images!")

    download_images_from_sas(
        downloaded_table,
        user,
        output_full_img_dir,
        None,
        download_function=download_function,
    )

    for ra, dec in zip(ras, decs):
        whpu.check_persistence(
            ra,
            dec,
            dict_input_tbl[f"{ra}_{dec}"],
            output_full_img_dir,
            output_cutout_dir,
            output_persistence_check_dir,
            **kwargs,
        )
