# utlities to collect all the queries needed to query the EUCLID archive
import re

from astropy import units

# =========================================================================== #


# ivoa_obscore seems to contain file paths for every single object in the archive,
# but it is very, very slow to download as there is a WHERE involved.
# stacked images are available, but it appears only a subset of those are available
#  from the archive (Deep fields? Unsure...);
# Calib images are dithered so a bit of a mess in terms of filename

# For cutouts, it is highly recommended to run this with either `stack` or `mosaic`

# =========================================================================== #


def query_full_sas_image_tbl(name):
    """Generate an SQL query based on the specified type of data to retrieve.

    This function returns a pre-defined SQL SELECT statement according to the
    given `name`. The valid options are 'ivoa_obscore', 'stack', 'calib', and
    'mosaic'. If an invalid option is provided, a ValueError is raised.

    :param name: The type of data to query. Should be one of 'ivoa_obscore', 'stack', 'calib', or 'mosaic'.
    :type name: str
    :return: An SQL SELECT statement as a string.
    :rtype: str
    :raises ValueError: If the provided name is not a valid option.
    """
    name = name.lower()

    if name not in ["ivoa_obscore", "stacked", "calib", "mosaic"]:
        raise ValueError(
            "[Error] Valid options are `'ivoa_obscore', 'stacked', 'calib', 'mosaic'`"
        )

    if name == "ivoa_obscore":
        out = """SELECT s_ra, s_dec, t_exptime, obs_id, obs_collection, cutout_access_url,
               dataproduct_subtype, dataproduct_type, filter, instrument_name
               FROM ivoa.obscore WHERE t_exptime > 0"""
    elif name == "stacked":
        out = """SELECT ra, dec, duration AS t_exp, file_name, file_path, filter_name,
               instrument_name, observation_id, observation_stack_oid, product_type,
               release_name FROM sedm.observation_stack"""
    elif name == "calib":
        out = """SELECT ra, dec, duration AS t_exp, file_name, file_path, filter_name,
               instrument_name, observation_id, calibrated_frame_oid, product_type,
               release_name FROM sedm.calibrated_frame"""
    elif name == "mosaic":
        out = """SELECT ra, dec, file_name, file_path, filter_name,
               instrument_name, tile_index, mosaic_product_oid, product_type,
               release_name FROM sedm.mosaic_product"""

    return re.sub(r"\n +", " ", out)


# =========================================================================== #


@units.quantity_input()
def query_sas_image_tbl_by_coord(
    name,
    ra: units.deg,
    dec: units.deg,
    search_radius: units.deg,
):
    """Generate an SQL query based on the specified type of data to retrieve.

    This function returns a pre-defined SQL SELECT statement according to the
    given `name`. The valid options are 'stacked', 'calib', and
    'mosaic'. If an invalid option is provided, a ValueError is raised.

    :param name: The type of data to query. Should be one of 'ivoa_obscore', 'stack', 'calib', or 'mosaic'.
    :type name: str
    :return: An SQL SELECT statement as a string.
    :rtype: str
    :raises ValueError: If the provided name is not a valid option.
    """
    name = name.lower()

    if name not in ["stacked", "calib", "mosaic"]:
        raise ValueError("[Error] Valid options are `'stacked', 'calib', 'mosaic'`")

    if name == "stacked":
        out = f"""SELECT ra, dec, duration AS t_exp, file_name, file_path, filter_name,
                instrument_name, observation_id, observation_stack_oid, product_type,
                release_name FROM sedm.observation_stack AS os
                WHERE (product_type like '%Stacked%') AND 
                (os.fov IS NOT NULL AND INTERSECTS(CIRCLE('ICRS', {ra.value}, {dec.value}, {search_radius.to(units.deg).value}), os.fov) = 1) 
                ORDER BY observation_id ASC"""
    elif name == "calib":
        out = f"""SELECT ra, dec, duration AS t_exp, file_name, file_path, filter_name,
                instrument_name, observation_id, calibrated_frame_oid, product_type,
                release_name FROM sedm.calibrated_frame AS cf
                WHERE (product_type like '%Calibrated%') AND 
                (cf.fov IS NOT NULL AND INTERSECTS(CIRCLE('ICRS', {ra.value}, {dec.value}, {search_radius.to(units.deg).value}), cf.fov) = 1) 
                ORDER BY observation_id ASC"""
    elif name == "mosaic":
        out = f"""SELECT ra, dec, file_name, file_path, filter_name,
                instrument_name, tile_index, mosaic_product_oid, product_type,
                release_name FROM sedm.mosaic_product AS mp
                WHERE (mp.environment='DR1')
                AND (mp.category='SCIENCE')
                AND (mp.fov IS NOT NULL AND INTERSECTS(CIRCLE('ICRS', {ra.value}, {dec.value}, {search_radius.to(units.deg).value}), mp.fov) = 1)
                ORDER BY mp.tile_index ASC"""

    return re.sub(r"\n +", " ", out)


# =========================================================================== #


@units.quantity_input()
def query_sas_catalogue_tbl_by_coord(
    name,
    ra: units.deg,
    dec: units.deg,
    search_radius: units.deg,
):
    """Generate an SQL query based on the specified type of data to retrieve.

    This function returns a pre-defined SQL SELECT statement according to the
    given `name`. The valid options are 'stacked', 'calib', and
    'mosaic'. If an invalid option is provided, a ValueError is raised.

    :param name: The type of data to query. Should be one of 'ivoa_obscore', 'stack', 'calib', or 'mosaic'.
    :type name: str
    :return: An SQL SELECT statement as a string.
    :rtype: str
    :raises ValueError: If the provided name is not a valid option.
    """
    name = name.lower()

    if name not in ["stacked", "calib", "mosaic"]:
        raise ValueError("[Error] Valid options are `'stacked', 'calib', 'mosaic'`")

    if name == "stacked":
        out = f"""SELECT frame_catalog.file_path, frame_catalog.file_name, 
                frame_catalog.catalog_oid,  
                frame_catalog.observation_id, frame_catalog.instrument_name, 
                frame_catalog.filter_name, frame_catalog.ra, frame_catalog.dec, 
                frame_catalog.obs_time, frame_catalog.product_type, 
                frame_catalog.product_id, frame_catalog.data_set_release 
                FROM sedm.frame_catalog 
                WHERE (product_type like '%Stacked%') AND 
                (frame_catalog.fov IS NOT NULL AND 
                INTERSECTS(CIRCLE('ICRS', {ra.value}, {dec.value}, {search_radius.to(units.deg).value}), frame_catalog.fov)=1) 
                ORDER BY observation_id ASC"""
    elif name == "calib":
        out = f"""SELECT frame_catalog.file_path, frame_catalog.file_name, 
                frame_catalog.catalog_oid,  
                frame_catalog.observation_id, frame_catalog.instrument_name, 
                frame_catalog.filter_name, frame_catalog.ra, frame_catalog.dec, 
                frame_catalog.obs_time, frame_catalog.product_type, 
                frame_catalog.product_id, frame_catalog.data_set_release 
                FROM sedm.frame_catalog 
                WHERE (product_type like '%Calibrated%') AND 
                (frame_catalog.fov IS NOT NULL AND 
                INTERSECTS(CIRCLE('ICRS', {ra.value}, {dec.value}, {search_radius.to(units.deg).value}), frame_catalog.fov)=1) 
                ORDER BY observation_id ASC"""
    elif name == "mosaic":
        out = f"""SELECT basic_download_data.basic_download_data_oid, 
                basic_download_data.product_type, basic_download_data.product_id,
                basic_download_data.file_path AS file_path,
                CAST(basic_download_data.file_name_list as text) AS file_name_list,
                CAST(basic_download_data.observation_id_list as text) AS observation_id_list,
                CAST(basic_download_data.tile_index_list as text) AS tile_index_list,
                CAST(basic_download_data.patch_id_list as text) AS patch_id_list,
                CAST(basic_download_data.filter_name as text) AS filter_name,
                basic_download_data.data_set_release FROM sedm.basic_download_data
                WHERE (environment='DR1')
                AND (product_type='dpdMerFinalCatalog')
                AND (basic_download_data.fov IS NOT NULL AND INTERSECTS(CIRCLE('ICRS', {ra.value}, {dec.value}, {search_radius.to(units.deg).value}), basic_download_data.fov)=1)
                ORDER BY observation_id_list ASC"""

    return re.sub(r"\n +", " ", out)
