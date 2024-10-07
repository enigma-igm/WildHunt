#!/usr/bin/env python

import os
import time

from astropy import units

from wildhunt import catalog, pypmsgs
from wildhunt.user import User
from wildhunt.utilities import euclid_utils as eu

msgs = pypmsgs.Messages()


# download Euclid images
def example_download_cutouts():
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

    # Set the output paths
    cat_outpath = "/Users/francesco/.tmp/euclid_downloads"
    img_outpath = "/Users/francesco/.tmp/euclid_downloads/img"

    if not os.path.exists(cat_outpath):
        os.makedirs(cat_outpath)
    if not os.path.exists(img_outpath):
        os.makedirs(img_outpath)

    # Currently the following image types are implemented
    # - 'calib' for calibrated images
    # - 'mosaic' for MER mosaics
    img_type = "mosaic"
    user = User()
    user.sasotf_login()

    eu.download_all_images(ra, dec, user, cat_outpath, img_outpath, img_type=img_type)


if __name__ == "__main__":
    example_download_cutouts()

    example_download_all_images()
