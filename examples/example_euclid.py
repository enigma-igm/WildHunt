#!/usr/bin/env python

import os
import time

from astropy import units

from wildhunt import catalog, pypmsgs
from wildhunt.user import User
from wildhunt.utilities import euclid_utils as weu

msgs = pypmsgs.Messages()


# download Euclid images
def example_download_cutouts():
    # TODO: Check if this still works
    t0 = time.time()

    cat = catalog.Catalog(
        "example",
        "RA",
        "DEC",
        "Name",
        datapath="/Users/francesco/repo/WildHunt/examples/data/Euclid_sources.csv",
    )

    survey_dict = [
        {"survey": "Euclid", "bands": ["VIS", "Y", "J", "H"], "fov": 10},
    ]

    cat.get_survey_images("/Users/francesco/.tmp/cutouts", survey_dict, n_jobs=3)
    msgs.info(f"Took {time.time() - t0:.1f}s to download the requested cutouts.")


def example_download_all_images():
    # For testing purposes only the coordinates are duplicated.
    ra = [149.7848750, 149.784875] * units.deg
    dec = [2.0673917, 2.06739] * units.deg

    # Currently the following image types are implemented
    # - 'calib' for calibrated images
    # - 'stacked' for stacked images
    # - 'mosaic' for MER mosaics
    data_product_type = ["img", "mosaic"]

    # Set the output paths
    queried_tbl_outpath = "/Users/francesco/.tmp/euclid_downloads"
    retrieved_data_outpath = f"/Users/francesco/.tmp/euclid_downloads/{data_product_type[0]}/{data_product_type[1]}"

    if not os.path.exists(queried_tbl_outpath):
        os.makedirs(queried_tbl_outpath)
    if not os.path.exists(retrieved_data_outpath):
        os.makedirs(retrieved_data_outpath)

    # user for SAS login
    user = User()
    user.sas_login()

    weu.download_data_for_all_bands(
        ra,
        dec,
        user,
        queried_tbl_outpath,
        retrieved_data_outpath,
        search_function=weu.get_closest_data_product_using_sas,
        data_product_type=data_product_type,
    )


def example_download_all_catalogues():
    # For testing purposes only the coordinates are duplicated.
    ra = [149.7848750, 149.784875] * units.deg
    dec = [2.0673917, 2.06739] * units.deg

    # Currently the following catalogue types are implemented
    # - 'calib' for calibrated catalogues
    # - 'stacked' for stacked catalogues
    # - 'mosaic' for MER catalogues
    data_product_type = ["cat", "mosaic"]

    # Set the output paths
    queried_tbl_outpath = "/Users/francesco/.tmp/euclid_downloads"
    retrieved_data_outpath = f"/Users/francesco/.tmp/euclid_downloads/{data_product_type[0]}/{data_product_type[1]}"

    if not os.path.exists(queried_tbl_outpath):
        os.makedirs(queried_tbl_outpath)
    if not os.path.exists(retrieved_data_outpath):
        os.makedirs(retrieved_data_outpath)

    # user for SAS login
    user = User()
    user.sas_login()

    weu.download_data_for_all_bands(
        ra,
        dec,
        user,
        queried_tbl_outpath,
        retrieved_data_outpath,
        weu.get_closest_data_product_using_sas,
        data_product_type=data_product_type,
    )


if __name__ == "__main__":
    # set the correct euclid environment -- defaults to IDR
    # from wildhunt.config import set_euclid_env
    # set_euclid_env("OTF")

    # example_download_cutouts()
    # example_download_all_images()
    example_download_all_catalogues()
