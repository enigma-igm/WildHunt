# utlities to collect all the queries needed to query the EUCLID archive
import re

# =========================================================================== #


# ivoa_obscore seems to contain file paths for every single object in the archive,
# but it is very, very slow to download as there is a WHERE involved.
# stacked images are available, but it appears only a subset of those are available
#  from the archive (Deep fields? Unsure...);
# Calib images are dithered so a bit of a mess in terms of filename

# For cutouts, it is highly recommended to run this with either `stack` or `mosaic`

# =========================================================================== #


def query_full_table(name):
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

    if name not in ["ivoa_obscore", "stack", "calib", "mosaic"]:
        raise ValueError(
            "[Error] Valid options are `'ivoa_obscore', 'stack', 'calib', 'mosaic'`"
        )

    if name == "ivoa_obscore":
        out = """SELECT s_ra, s_dec, t_exptime, obs_id, obs_collection, cutout_access_url,
               dataproduct_subtype, dataproduct_type, filter, instrument_name
               FROM ivoa.obscore WHERE t_exptime > 0"""
    elif name == "stack":
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
